#!/usr/bin/env python3
"""
Optuna / TPE over the same grouped discrete perturbation space as the GFN,
with cache-first evaluation and built-in sweep support over:
- beta
- tau_quantile
- threshold_temperature
- seed

Behavior:
- Source of truth is scalar LOSS stored in the JSON cache at --reward_cache_path
- Reward is ALWAYS recomputed explicitly from stored loss
- On a cache miss, evaluate live once through CropSimulatorProxy, then write the
  new LOSS back into the main JSON cache
- The live proxy uses a separate internal cache file by default, so it does not
  conflict with the main state_key cache

Example:
python tpe_optimization.py \
  --reward_cache_path precomputed/reward_table_sf0.15.json \
  --step_fraction 0.15 \
  --n_cycles 1 \
  --decay_factor 0.5 \
  --n_trials 100 \
  --reward_mode thresholded_sigmoid \
  --loss_norm q10q90 \
  --q_low 0.10 \
  --q_high 0.90 \
  --tau_quantile 0.05 \
  --threshold_temperature 0.04 \
  --betas 1 4 8 \
  --seeds 0 1 2 \
  --reward_epsilon 1e-3 \
  --wandb_group tpe_beta_seed_sweep_sf015
"""

from __future__ import annotations

import argparse
import atexit
import copy
import itertools
import json
import random
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
import wandb

from gflownet.envs.greenhouse.constants import (
    BASELINE_PARAMETERS,
    GROUP_ORDER,
    INITIAL_CONDITIONS,
    PARAMETER_BOUNDS,
    PERTURBATION_SCHEME,
)
from gflownet.proxy.greenhouse.cropSimulatorProxy_thresholded import CropSimulatorProxy


def apply_perturbation(
    current_params: Dict[str, float],
    group_name: str,
    action_name: str,
    step_fraction: float,
) -> None:
    """Same grouped update logic as the GFN environment."""
    action = PERTURBATION_SCHEME[group_name][action_name]
    for param_name, direction in action.items():
        if direction == 0:
            continue
        lo, hi = PARAMETER_BOUNDS[param_name]
        val = current_params[param_name]
        current_params[param_name] = float(
            np.clip(val + direction * step_fraction * (hi - lo), lo, hi)
        )


def build_config(config: Dict[str, float], normalize: bool = False) -> List[float]:
    parameters = [0.0] * len(BASELINE_PARAMETERS)
    for i, k in enumerate(sorted(BASELINE_PARAMETERS)):
        parameters[i] = float(
            config.get(k, INITIAL_CONDITIONS.get(k, BASELINE_PARAMETERS[k]))
        )

    if normalize:
        for i, k in enumerate(sorted(BASELINE_PARAMETERS)):
            lo, hi = PARAMETER_BOUNDS.get(k, (0, 0))
            parameters[i] = 0.5 if lo == hi else (parameters[i] - lo) / (hi - lo)
    return parameters


def action_seq_to_key(action_seq: List[str]) -> str:
    return "|".join(action_seq)


def _slug(x: object) -> str:
    s = str(x)
    s = s.replace(".", "p")
    s = s.replace("-", "m")
    s = re.sub(r"[^A-Za-z0-9_]+", "_", s)
    return s


def _as_loss(value) -> float:
    """
    Accept cache entries of the form:
      key -> loss
      key -> {"loss": ..., "params": ...}
    """
    if isinstance(value, dict):
        if "loss" not in value:
            raise KeyError(f"Cache entry dict is missing 'loss': {value}")
        return float(value["loss"])
    return float(value)


def normalize_losses(
    losses: np.ndarray,
    mode: str,
    q_low: float,
    q_high: float,
    reward_anchor: Optional[float],
    reward_scale: Optional[float],
) -> Tuple[np.ndarray, Dict[str, float]]:
    losses = np.asarray(losses, dtype=np.float64)

    if mode == "none":
        anchor = 0.0 if reward_anchor is None else float(reward_anchor)
        scale = 1.0 if reward_scale is None else max(float(reward_scale), 1e-12)
        return (losses - anchor) / scale, {"anchor": anchor, "scale": scale}

    if mode != "q10q90":
        raise ValueError(f"Unsupported loss_norm: {mode}")

    anchor = float(np.quantile(losses, q_low)) if reward_anchor is None else float(reward_anchor)
    if reward_scale is None:
        high = float(np.quantile(losses, q_high))
        scale = max(high - anchor, 1e-12)
    else:
        scale = max(float(reward_scale), 1e-12)

    return (losses - anchor) / scale, {"anchor": anchor, "scale": scale}


def recompute_rewards_from_losses(
    losses_by_key: Dict[str, float],
    reward_mode: str,
    beta: float,
    loss_norm: str,
    q_low: float,
    q_high: float,
    reward_anchor: Optional[float],
    reward_scale: Optional[float],
    reward_tau: Optional[float],
    tau_quantile: float,
    threshold_temperature: float,
    reward_epsilon: float,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Explicitly rebuild rewards from scalar losses.
    """
    keys = list(losses_by_key.keys())
    losses = np.array([float(losses_by_key[k]) for k in keys], dtype=np.float64)

    loss_normed, meta = normalize_losses(
        losses=losses,
        mode=loss_norm,
        q_low=q_low,
        q_high=q_high,
        reward_anchor=reward_anchor,
        reward_scale=reward_scale,
    )

    if reward_mode == "softmin":
        rewards = np.exp(-float(beta) * loss_normed)
        tau = None
    elif reward_mode == "thresholded_sigmoid":
        tau = (
            float(np.quantile(loss_normed, tau_quantile))
            if reward_tau is None
            else float(reward_tau)
        )
        temp = max(float(threshold_temperature), 1e-12)
        sig = 1.0 / (1.0 + np.exp((loss_normed - tau) / temp))
        rewards = float(reward_epsilon) + (1.0 - float(reward_epsilon)) * (sig ** float(beta))
    else:
        raise ValueError(f"Unsupported reward_mode: {reward_mode}")

    reward_by_key = {k: float(r) for k, r in zip(keys, rewards)}
    meta.update(
        {
            "reward_mode": reward_mode,
            "beta": float(beta),
            "tau": None if tau is None else float(tau),
            "tau_quantile": float(tau_quantile),
            "threshold_temperature": float(threshold_temperature),
            "reward_epsilon": float(reward_epsilon),
        }
    )
    return reward_by_key, meta


@dataclass
class TrialResult:
    trial_step: int
    state_key: str
    reward: float
    loss: float
    best_reward_so_far: float
    best_loss_so_far: float
    action_seq: List[str]
    params: Dict[str, float]
    cache_hit: bool


class BOBaseline:
    def __init__(self, args: argparse.Namespace):
        self.args = args

        self.cache_path = Path(args.reward_cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache = self._load_cache(self.cache_path)

        # The live proxy uses a separate internal cache file to avoid conflicts with
        # the main state_key JSON cache format.
        if args.proxy_reward_cache_path is None:
            stem = self.cache_path.stem
            suffix = self.cache_path.suffix or ".json"
            self.proxy_cache_path = self.cache_path.with_name(f"{stem}_live_proxy_cache{suffix}")
        else:
            self.proxy_cache_path = Path(args.proxy_reward_cache_path)
            self.proxy_cache_path.parent.mkdir(parents=True, exist_ok=True)

        self.proxy: Optional[CropSimulatorProxy] = None  # lazy init on miss only
        self.best_reward = -np.inf
        self.best_loss = np.inf
        self.results: List[TrialResult] = []

        self.cache_misses = 0
        self.cache_hits = 0
        self.cache_dirty = 0

        atexit.register(self._flush_cache)

    @staticmethod
    def _load_cache(path: Path) -> Dict[str, object]:
        if not path.exists():
            return {}
        with open(path, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Cache file must contain a JSON object: {path}")
        return data

    def _flush_cache(self) -> None:
        if self.cache_dirty <= 0:
            return
        tmp = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        with open(tmp, "w") as f:
            json.dump(self.cache, f, indent=2)
        tmp.replace(self.cache_path)
        self.cache_dirty = 0
        print(f"[cache] saved {self.cache_path}")

    def _maybe_save_cache(self) -> None:
        if self.cache_dirty >= self.args.cache_save_every:
            self._flush_cache()

    def _get_proxy(self) -> CropSimulatorProxy:
        if self.proxy is None:
            self.proxy = CropSimulatorProxy(
                reward_cache_path=str(self.proxy_cache_path),
                reward_mode=self.args.reward_mode,
                beta=self.args.beta,
                loss_norm=self.args.loss_norm,
                q_low=self.args.q_low,
                q_high=self.args.q_high,
                reward_anchor=self.args.reward_anchor,
                reward_scale=self.args.reward_scale,
                reward_tau=self.args.reward_tau,
                tau_quantile=self.args.tau_quantile,
                threshold_temperature=self.args.threshold_temperature,
                reward_epsilon=self.args.reward_epsilon,
                cache_save_every=self.args.cache_save_every,
                device=self.args.device,
                float_precision=self.args.float_precision,
            )
        return self.proxy

    def _losses_by_key(self) -> Dict[str, float]:
        return {k: _as_loss(v) for k, v in self.cache.items()}

    def _reward_for_key(self, key: str) -> float:
        reward_by_key, _ = recompute_rewards_from_losses(
            losses_by_key=self._losses_by_key(),
            reward_mode=self.args.reward_mode,
            beta=self.args.beta,
            loss_norm=self.args.loss_norm,
            q_low=self.args.q_low,
            q_high=self.args.q_high,
            reward_anchor=self.args.reward_anchor,
            reward_scale=self.args.reward_scale,
            reward_tau=self.args.reward_tau,
            tau_quantile=self.args.tau_quantile,
            threshold_temperature=self.args.threshold_temperature,
            reward_epsilon=self.args.reward_epsilon,
        )
        return float(reward_by_key[key])

    def evaluate(self, state_key: str, param_values: Dict[str, float]) -> tuple[float, float, bool]:
        # Cache hit: use stored loss, recompute reward explicitly.
        if state_key in self.cache:
            self.cache_hits += 1
            loss = _as_loss(self.cache[state_key])
            reward = self._reward_for_key(state_key)
            return reward, loss, True

        # Cache miss: evaluate live once through the proxy, then store LOSS only.
        self.cache_misses += 1
        proxy = self._get_proxy()

        proxy_config = build_config(param_values, normalize=False)
        cache_key = proxy._make_cache_key(proxy_config)

        _ = float(proxy([proxy_config]).item())

        if cache_key not in proxy.reward_cache:
            raise KeyError(
                f"Proxy evaluated config but did not populate reward_cache for key={cache_key}"
            )

        live_cache_value = proxy.reward_cache[cache_key]
        live_loss = _as_loss(live_cache_value)

        self.cache[state_key] = {
            "params": {k: float(v) for k, v in param_values.items()},
            "loss": float(live_loss),
        }
        self.cache_dirty += 1

        reward = self._reward_for_key(state_key)
        self._maybe_save_cache()
        return reward, float(live_loss), False

    def objective(self, trial: optuna.Trial) -> float:
        current_params = {
            k: float(INITIAL_CONDITIONS.get(k, BASELINE_PARAMETERS[k]))
            for k in BASELINE_PARAMETERS
        }
        action_seq: List[str] = []

        for cycle in range(self.args.n_cycles):
            cycle_step_fraction = self.args.step_fraction * (self.args.decay_factor ** cycle)
            for group_name in GROUP_ORDER:
                available_actions = list(PERTURBATION_SCHEME[group_name].keys())
                action = trial.suggest_categorical(
                    f"{group_name}_cycle{cycle}",
                    available_actions,
                )
                action_seq.append(action)
                apply_perturbation(current_params, group_name, action, cycle_step_fraction)

        state_key = action_seq_to_key(action_seq)
        reward, loss, cache_hit = self.evaluate(state_key, current_params)

        self.best_reward = max(self.best_reward, reward)
        self.best_loss = min(self.best_loss, loss)

        rec = TrialResult(
            trial_step=trial.number,
            state_key=state_key,
            reward=reward,
            loss=loss,
            best_reward_so_far=self.best_reward,
            best_loss_so_far=self.best_loss,
            action_seq=action_seq,
            params={k: float(v) for k, v in current_params.items()},
            cache_hit=cache_hit,
        )
        self.results.append(rec)

        trial.set_user_attr("state_key", state_key)
        trial.set_user_attr("cache_hit", cache_hit)
        trial.set_user_attr("loss", loss)

        log_dict = {
            "trial_step": trial.number,
            "state_key": state_key,
            "reward": reward,
            "loss": loss,
            "best_reward_so_far": self.best_reward,
            "best_loss_so_far": self.best_loss,
            "unique_states_so_far": len({r.state_key for r in self.results}),
            "cache_hit": int(cache_hit),
            "cache_hits_total": self.cache_hits,
            "cache_misses_total": self.cache_misses,
        }
        for cycle in range(self.args.n_cycles):
            for gi, group_name in enumerate(GROUP_ORDER):
                idx = cycle * len(GROUP_ORDER) + gi
                log_dict[f"{group_name}_cycle{cycle}"] = action_seq[idx]
        for k, v in current_params.items():
            log_dict[k] = v

        wandb.log(log_dict)
        return reward

    def save_outputs(self, output_dir: Path, study: optuna.Study) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        for r in self.results:
            row = {
                "trial_step": r.trial_step,
                "state_key": r.state_key,
                "reward": r.reward,
                "loss": r.loss,
                "best_reward_so_far": r.best_reward_so_far,
                "best_loss_so_far": r.best_loss_so_far,
                "cache_hit": int(r.cache_hit),
            }
            for i, action in enumerate(r.action_seq):
                row[f"action_{i}"] = action
            row.update({f"param__{k}": v for k, v in r.params.items()})
            rows.append(row)

        df = pd.DataFrame(rows).sort_values("trial_step")
        csv_path = output_dir / "optuna_trials.csv"
        df.to_csv(csv_path, index=False)

        summary = {
            "n_trials": len(df),
            "best_reward": float(df["reward"].max()) if len(df) else None,
            "best_loss": float(df["loss"].min()) if len(df) else None,
            "mean_reward": float(df["reward"].mean()) if len(df) else None,
            "median_reward": float(df["reward"].median()) if len(df) else None,
            "unique_states": int(df["state_key"].nunique()) if len(df) else 0,
            "cache_hits": int(self.cache_hits),
            "cache_misses": int(self.cache_misses),
            "best_params": study.best_params if len(df) else None,
        }
        with open(output_dir / "optuna_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        artifact = wandb.Artifact(
            name=f"optuna-trials-{wandb.run.id}",
            type="optuna_results",
            metadata=summary,
        )
        artifact.add_file(str(csv_path))
        artifact.add_file(str(output_dir / "optuna_summary.json"))
        wandb.log_artifact(artifact)

        print(f"[saved] {csv_path}")
        print(f"[saved] {output_dir / 'optuna_summary.json'}")

        self._flush_cache()

        if self.proxy is not None and hasattr(self.proxy, "save_final_cache"):
            try:
                self.proxy.save_final_cache()
            except Exception as e:
                print(f"[warn] proxy.save_final_cache() failed: {e}")


def build_sweep_grid(args: argparse.Namespace) -> List[dict]:
    betas = args.betas if args.betas is not None else [args.beta]
    tau_quantiles = args.tau_quantiles if args.tau_quantiles is not None else [args.tau_quantile]
    temperatures = (
        args.threshold_temperatures
        if args.threshold_temperatures is not None
        else [args.threshold_temperature]
    )
    seeds = args.seeds if args.seeds is not None else [args.seed]

    grid = []
    for beta, tau_q, temp, seed in itertools.product(
        betas, tau_quantiles, temperatures, seeds
    ):
        grid.append(
            {
                "beta": float(beta),
                "tau_quantile": float(tau_q),
                "threshold_temperature": float(temp),
                "seed": int(seed),
            }
        )
    return grid


def config_to_name(cfg: dict) -> str:
    return (
        f"seed{_slug(cfg['seed'])}"
        f"_b{_slug(cfg['beta'])}"
        f"_tauq{_slug(cfg['tau_quantile'])}"
        f"_T{_slug(cfg['threshold_temperature'])}"
    )


def run_one_study(base_args: argparse.Namespace, overrides: dict) -> dict:
    args = copy.deepcopy(base_args)
    for k, v in overrides.items():
        setattr(args, k, v)

    random.seed(args.seed)
    np.random.seed(args.seed)

    config_name = config_to_name(overrides)
    base_name = args.study_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{base_name}_{config_name}"

    wandb.init(
        mode=args.wandb_mode,
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        name=run_name,
        job_type="optuna_sweep_member",
        config=vars(args),
        reinit="finish_previous",
    )
    wandb.define_metric("*", step_metric="trial_step")

    baseline = BOBaseline(args)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        study_name=run_name,
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(baseline.objective, n_trials=args.n_trials)

    out_dir = Path(args.output_dir) / config_name / wandb.run.id
    baseline.save_outputs(out_dir, study)

    result = {
        "config_name": config_name,
        "seed": int(args.seed),
        "beta": float(args.beta),
        "tau_quantile": float(args.tau_quantile),
        "threshold_temperature": float(args.threshold_temperature),
        "best_reward": float(study.best_value),
        "best_loss": float(baseline.best_loss),
        "cache_hits": int(baseline.cache_hits),
        "cache_misses": int(baseline.cache_misses),
        "best_params": study.best_params,
        "wandb_run_id": wandb.run.id,
        "output_dir": str(out_dir),
    }

    wandb.summary["seed"] = int(args.seed)
    wandb.summary["best_reward"] = result["best_reward"]
    wandb.summary["best_loss"] = result["best_loss"]
    wandb.summary["cache_hits"] = result["cache_hits"]
    wandb.summary["cache_misses"] = result["cache_misses"]
    wandb.finish()

    return result


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    ap.add_argument("--n_cycles", type=int, default=1)
    ap.add_argument("--step_fraction", type=float, default=0.15)
    ap.add_argument("--decay_factor", type=float, default=0.5)
    ap.add_argument("--n_trials", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)

    ap.add_argument("--reward_cache_path", default="precomputed/reward_table_sf0.15.json")
    ap.add_argument(
        "--reward_mode",
        default="thresholded_sigmoid",
        choices=["thresholded_sigmoid", "softmin"],
    )
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--loss_norm", default="q10q90", choices=["q10q90", "none"])
    ap.add_argument("--q_low", type=float, default=0.10)
    ap.add_argument("--q_high", type=float, default=0.90)
    ap.add_argument("--reward_anchor", type=float, default=None)
    ap.add_argument("--reward_scale", type=float, default=None)
    ap.add_argument("--reward_tau", type=float, default=None)
    ap.add_argument("--tau_quantile", type=float, default=0.10)
    ap.add_argument("--threshold_temperature", type=float, default=0.08)
    ap.add_argument("--reward_epsilon", type=float, default=1e-3)
    ap.add_argument("--cache_save_every", type=int, default=25)

    ap.add_argument("--betas", type=float, nargs="+", default=None)
    ap.add_argument("--tau_quantiles", type=float, nargs="+", default=None)
    ap.add_argument("--threshold_temperatures", type=float, nargs="+", default=None)

    ap.add_argument("--proxy_reward_cache_path", default=None)

    ap.add_argument("--wandb_project", default="optuna-crop-calibration")
    ap.add_argument("--wandb_entity", default=None)
    ap.add_argument("--wandb_mode", default="online")
    ap.add_argument("--wandb_group", default=None)
    ap.add_argument("--output_dir", default="bo_outputs")
    ap.add_argument("--study_name", default=None)

    ap.add_argument("--device", default="cpu")
    ap.add_argument("--float_precision", type=int, default=32)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    grid = build_sweep_grid(args)

    print(f"[sweep] running {len(grid)} configuration(s)")
    all_results = []

    for cfg in grid:
        print(
            f"[sweep] seed={cfg['seed']} "
            f"beta={cfg['beta']} "
            f"tau_quantile={cfg['tau_quantile']} "
            f"T={cfg['threshold_temperature']}"
        )
        result = run_one_study(args, cfg)
        all_results.append(result)

    summary_dir = Path(args.output_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)

    summary_json = summary_dir / "sweep_summary.json"
    with open(summary_json, "w") as f:
        json.dump(all_results, f, indent=2)

    if all_results:
        df = pd.DataFrame(
            [
                {
                    **{k: v for k, v in r.items() if k != "best_params"},
                    "best_params_json": json.dumps(r["best_params"], sort_keys=True),
                }
                for r in all_results
            ]
        ).sort_values(["beta", "tau_quantile", "threshold_temperature", "seed"])

        summary_csv = summary_dir / "sweep_summary.csv"
        df.to_csv(summary_csv, index=False)

        group_cols = ["beta", "tau_quantile", "threshold_temperature"]
        agg = (
            df.groupby(group_cols, as_index=False)
            .agg(
                mean_best_reward=("best_reward", "mean"),
                std_best_reward=("best_reward", "std"),
                mean_best_loss=("best_loss", "mean"),
                std_best_loss=("best_loss", "std"),
                mean_cache_hits=("cache_hits", "mean"),
                mean_cache_misses=("cache_misses", "mean"),
                n_runs=("seed", "count"),
            )
            .sort_values(["mean_best_reward", "mean_best_loss"], ascending=[False, True])
        )
        agg_csv = summary_dir / "sweep_summary_aggregated.csv"
        agg.to_csv(agg_csv, index=False)

        print("\nCompleted sweep members:")
        for _, row in df.iterrows():
            print(
                f"  {row['config_name']}"
                f" | best_reward={row['best_reward']:.6f}"
                f" | best_loss={row['best_loss']:.6f}"
                f" | cache_hits={int(row['cache_hits'])}"
                f" | cache_misses={int(row['cache_misses'])}"
            )

        print("\nAggregated by reward config:")
        for _, row in agg.iterrows():
            print(
                f"  beta={row['beta']}"
                f" tau_q={row['tau_quantile']}"
                f" T={row['threshold_temperature']}"
                f" | mean_best_reward={row['mean_best_reward']:.6f}"
                f" | mean_best_loss={row['mean_best_loss']:.6f}"
                f" | n={int(row['n_runs'])}"
            )

        print(f"\n[saved] {summary_json}")
        print(f"[saved] {summary_csv}")
        print(f"[saved] {agg_csv}")


if __name__ == "__main__":
    main()
