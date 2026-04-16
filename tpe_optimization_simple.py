#!/usr/bin/env python3
"""
Simplified Optuna / TPE baseline over the grouped discrete perturbation space.

Supported reward modes
----------------------
- softmin
- thresholded_sigmoid
- context_kth_exp_simple

For the contextual mode:
    1) compute percentile-rank loss per context in [0, 1]
    2) take the k-th smallest contextual rank
    3) reward = exp(-beta * kth_rank)

This script is intentionally aligned with the simplified contextual proxy.
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

try:
    from gflownet.proxy.greenhouse.cropSimulatorProxy_contextual_simple import CropSimulatorProxy
except Exception:
    from gflownet.proxy.greenhouse.cropSimulatorProxy_contextual import CropSimulatorProxy


SCALAR_REWARD_MODES = {
    "softmin",
    "thresholded_sigmoid",
}

CONTEXT_REWARD_MODES = {
    "context_kth_exp_simple",
}


def is_contextual_mode(reward_mode: str) -> bool:
    return reward_mode in CONTEXT_REWARD_MODES


def apply_perturbation(
    current_params: Dict[str, float],
    group_name: str,
    action_name: str,
    step_fraction: float,
) -> None:
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


def _entry_from_cache_value(value) -> Dict[str, object]:
    """
    Normalize cache entries to:
      {"loss": float, "loss_by_context": dict[str,float], "params": dict|None}
    """
    if isinstance(value, dict):
        if "loss" in value:
            loss = float(value["loss"])
        elif "scalar_loss" in value:
            loss = float(value["scalar_loss"])
        else:
            raise KeyError(f"Cache entry dict is missing 'loss': {value}")
        loss_by_context = {
            str(k): float(v) for k, v in value.get("loss_by_context", {}).items()
        }
        params = value.get("params")
        return {
            "loss": loss,
            "loss_by_context": loss_by_context,
            "params": params,
        }

    return {"loss": float(value), "loss_by_context": {}, "params": None}


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


def recompute_scalar_rewards_from_entries(
    entries_by_key: Dict[str, Dict[str, object]],
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
    keys = list(entries_by_key.keys())
    losses = np.array([float(entries_by_key[k]["loss"]) for k in keys], dtype=np.float64)

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
        raise ValueError(f"Unsupported scalar reward_mode: {reward_mode}")

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


def build_context_rank_stats_from_entries(
    entries_by_key: Dict[str, Dict[str, object]],
    team_ids: List[str],
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for team in team_ids:
        vals = []
        for entry in entries_by_key.values():
            lbc = entry.get("loss_by_context", {})
            if team in lbc:
                vals.append(float(lbc[team]))
        arr = np.asarray(sorted(vals), dtype=float)
        if arr.size > 0:
            out[team] = arr
    return out


def percentile_rank(sorted_arr: np.ndarray, value: float) -> float:
    n = int(sorted_arr.size)
    if n <= 1:
        return 0.0
    idx = int(np.searchsorted(sorted_arr, value, side="left"))
    return float(idx / (n - 1))


def recompute_contextual_rewards_from_entries(
    entries_by_key: Dict[str, Dict[str, object]],
    context_team_ids: List[str],
    context_top_k: int,
    beta: float,
    context_fallback_to_scalar: bool,
) -> Tuple[Dict[str, Optional[float]], Dict[str, object]]:
    keys = list(entries_by_key.keys())

    all_teams = sorted(
        {
            str(team)
            for entry in entries_by_key.values()
            for team in entry.get("loss_by_context", {}).keys()
        }
    )
    team_ids = list(context_team_ids) if context_team_ids else all_teams
    rank_arrays = build_context_rank_stats_from_entries(entries_by_key, team_ids)

    reward_by_key: Dict[str, Optional[float]] = {}
    kth_rank_by_key: Dict[str, Optional[float]] = {}

    for key in keys:
        loss_by_context = {
            str(team): float(v)
            for team, v in entries_by_key[key].get("loss_by_context", {}).items()
        }

        vals = []
        for team in team_ids:
            if team not in loss_by_context:
                continue
            if team not in rank_arrays:
                continue
            vals.append(percentile_rank(rank_arrays[team], float(loss_by_context[team])))

        if not vals:
            reward_by_key[key] = None if context_fallback_to_scalar else 1e-12
            kth_rank_by_key[key] = None
            continue

        vals_sorted = sorted(float(v) for v in vals)
        k = min(max(int(context_top_k), 1), len(vals_sorted))
        kth_rank = float(vals_sorted[k - 1])
        kth_rank_by_key[key] = kth_rank
        reward_by_key[key] = float(np.exp(-float(beta) * kth_rank))

    valid_rewards = [float(v) for v in reward_by_key.values() if v is not None]
    valid_kth = [float(v) for v in kth_rank_by_key.values() if v is not None]

    meta = {
        "reward_mode": "context_kth_exp_simple",
        "context_top_k": int(context_top_k),
        "beta": float(beta),
        "context_team_ids": list(team_ids),
        "kth_rank_min": None if not valid_kth else float(np.min(valid_kth)),
        "kth_rank_max": None if not valid_kth else float(np.max(valid_kth)),
        "reward_min": None if not valid_rewards else float(np.min(valid_rewards)),
        "reward_max": None if not valid_rewards else float(np.max(valid_rewards)),
        "context_fallback_to_scalar": bool(context_fallback_to_scalar),
    }
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

        if args.proxy_reward_cache_path is None:
            stem = self.cache_path.stem
            suffix = self.cache_path.suffix or ".json"
            self.proxy_cache_path = self.cache_path.with_name(f"{stem}_live_proxy_cache{suffix}")
        else:
            self.proxy_cache_path = Path(args.proxy_reward_cache_path)
            self.proxy_cache_path.parent.mkdir(parents=True, exist_ok=True)

        self.proxy: Optional[CropSimulatorProxy] = None
        self.best_reward = -np.inf
        self.best_loss = np.inf
        self.results: List[TrialResult] = []

        self.cache_misses = 0
        self.cache_hits = 0
        self.cache_dirty = 0

        atexit.register(self._flush_cache)

    @staticmethod
    def _load_cache(path: Path) -> Dict[str, Dict[str, object]]:
        if not path.exists():
            return {}
        with open(path, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Cache file must contain a JSON object: {path}")
        return {k: _entry_from_cache_value(v) for k, v in data.items()}

    def _flush_cache(self) -> None:
        if self.cache_dirty <= 0:
            return
        tmp = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        serializable = {}
        for k, entry in self.cache.items():
            out = {"loss": float(entry["loss"])}
            if entry.get("loss_by_context"):
                out["loss_by_context"] = {
                    str(team): float(v) for team, v in entry["loss_by_context"].items()
                }
            if entry.get("params") is not None:
                out["params"] = entry["params"]
            serializable[k] = out
        with open(tmp, "w") as f:
            json.dump(serializable, f, indent=2)
        tmp.replace(self.cache_path)
        self.cache_dirty = 0
        print(f"[cache] saved {self.cache_path}")

    def _maybe_save_cache(self) -> None:
        if self.cache_dirty >= self.args.cache_save_every:
            self._flush_cache()

    def _get_proxy(self) -> CropSimulatorProxy:
        if self.proxy is None:
            kwargs = dict(
                reward_cache_path=str(self.proxy_cache_path),
                reward_mode=self.args.reward_mode,
                beta=self.args.beta,
                cache_save_every=self.args.cache_save_every,
                loss_norm=self.args.loss_norm,
                q_low=self.args.q_low,
                q_high=self.args.q_high,
                reward_anchor=self.args.reward_anchor,
                reward_scale=self.args.reward_scale,
                reward_tau=self.args.reward_tau,
                tau_quantile=self.args.tau_quantile,
                threshold_temperature=self.args.threshold_temperature,
                reward_epsilon=self.args.reward_epsilon,
                context_top_k=self.args.context_top_k,
                context_team_ids=self.args.context_team_ids,
                context_fallback_to_scalar=self.args.context_fallback_to_scalar,
                scalar_fallback_reward_mode=self.args.scalar_fallback_reward_mode,
                device=self.args.device,
                float_precision=self.args.float_precision,
            )
            self.proxy = CropSimulatorProxy(**kwargs)
        return self.proxy

    def _reward_for_key(self, key: str) -> float:
        if is_contextual_mode(self.args.reward_mode):
            reward_by_key, _ = recompute_contextual_rewards_from_entries(
                entries_by_key=self.cache,
                context_team_ids=self.args.context_team_ids,
                context_top_k=self.args.context_top_k,
                beta=self.args.beta,
                context_fallback_to_scalar=self.args.context_fallback_to_scalar,
            )

            if any(v is None for v in reward_by_key.values()):
                scalar_rewards, _ = recompute_scalar_rewards_from_entries(
                    entries_by_key=self.cache,
                    reward_mode=self.args.scalar_fallback_reward_mode,
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
                for kk, vv in list(reward_by_key.items()):
                    if vv is None:
                        reward_by_key[kk] = float(scalar_rewards[kk])

            return float(reward_by_key[key])

        reward_by_key, _ = recompute_scalar_rewards_from_entries(
            entries_by_key=self.cache,
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
        if state_key in self.cache:
            self.cache_hits += 1
            entry = self.cache[state_key]
            loss = float(entry["loss"])
            reward = self._reward_for_key(state_key)
            return reward, loss, True

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
        entry = _entry_from_cache_value(live_cache_value)

        self.cache[state_key] = {
            "params": {k: float(v) for k, v in param_values.items()},
            "loss": float(entry["loss"]),
            "loss_by_context": {
                str(team): float(v) for team, v in entry.get("loss_by_context", {}).items()
            },
        }
        self.cache_dirty += 1

        reward = self._reward_for_key(state_key)
        self._maybe_save_cache()
        return reward, float(entry["loss"]), False

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
            "reward_mode": self.args.reward_mode,
            "best_reward": float(df["reward"].max()) if len(df) else None,
            "best_loss": float(df["loss"].min()) if len(df) else None,
            "mean_reward": float(df["reward"].mean()) if len(df) else None,
            "median_reward": float(df["reward"].median()) if len(df) else None,
            "unique_states": int(df["state_key"].nunique()) if len(df) else 0,
            "cache_hits": int(self.cache_hits),
            "cache_misses": int(self.cache_misses),
            "best_params": study.best_params if len(df) else None,
            "context_top_k": int(self.args.context_top_k),
            "beta": float(self.args.beta),
            "context_team_ids": list(self.args.context_team_ids),
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
    seeds = args.seeds if args.seeds is not None else [args.seed]

    if args.reward_mode == "context_kth_exp_simple":
        ks = args.context_top_ks if args.context_top_ks is not None else [args.context_top_k]
        betas = args.betas if args.betas is not None else [args.beta]
        grid = []
        for k, beta, seed in itertools.product(ks, betas, seeds):
            grid.append(
                {
                    "context_top_k": int(k),
                    "beta": float(beta),
                    "seed": int(seed),
                }
            )
        return grid

    betas = args.betas if args.betas is not None else [args.beta]
    tau_quantiles = args.tau_quantiles if args.tau_quantiles is not None else [args.tau_quantile]
    temperatures = (
        args.threshold_temperatures
        if args.threshold_temperatures is not None
        else [args.threshold_temperature]
    )

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


def config_to_name(args: argparse.Namespace) -> str:
    if args.reward_mode == "context_kth_exp_simple":
        return (
            f"{_slug(args.reward_mode)}"
            f"_seed{_slug(args.seed)}"
            f"_k{_slug(args.context_top_k)}"
            f"_b{_slug(args.beta)}"
        )

    return (
        f"{_slug(args.reward_mode)}"
        f"_seed{_slug(args.seed)}"
        f"_b{_slug(args.beta)}"
        f"_tauq{_slug(args.tau_quantile)}"
        f"_T{_slug(args.threshold_temperature)}"
    )


def group_cols_for_mode(reward_mode: str) -> List[str]:
    if reward_mode == "context_kth_exp_simple":
        return ["context_top_k", "beta"]
    return ["beta", "tau_quantile", "threshold_temperature"]


def run_one_study(base_args: argparse.Namespace, overrides: dict) -> dict:
    args = copy.deepcopy(base_args)
    for k, v in overrides.items():
        setattr(args, k, v)

    random.seed(args.seed)
    np.random.seed(args.seed)

    config_name = config_to_name(args)
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
        "reward_mode": str(args.reward_mode),
        "best_reward": float(study.best_value),
        "best_loss": float(baseline.best_loss),
        "cache_hits": int(baseline.cache_hits),
        "cache_misses": int(baseline.cache_misses),
        "best_params": study.best_params,
        "wandb_run_id": wandb.run.id,
        "output_dir": str(out_dir),
        "context_top_k": int(args.context_top_k),
        "beta": float(args.beta),
        "tau_quantile": float(args.tau_quantile),
        "threshold_temperature": float(args.threshold_temperature),
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

    ap.add_argument("--reward_cache_path", default="precomputed_contextual/reward_table_sf0.15.json")
    ap.add_argument(
        "--reward_mode",
        default="context_kth_exp_simple",
        choices=sorted(SCALAR_REWARD_MODES | CONTEXT_REWARD_MODES),
    )

    ap.add_argument("--beta", type=float, default=4.0)
    ap.add_argument("--betas", type=float, nargs="+", default=None)

    ap.add_argument("--loss_norm", default="q10q90", choices=["q10q90", "none"])
    ap.add_argument("--q_low", type=float, default=0.10)
    ap.add_argument("--q_high", type=float, default=0.90)

    ap.add_argument("--reward_anchor", type=float, default=None)
    ap.add_argument("--reward_scale", type=float, default=None)
    ap.add_argument("--reward_tau", type=float, default=None)
    ap.add_argument("--tau_quantile", type=float, default=0.10)
    ap.add_argument("--tau_quantiles", type=float, nargs="+", default=None)
    ap.add_argument("--threshold_temperature", type=float, default=0.08)
    ap.add_argument("--threshold_temperatures", type=float, nargs="+", default=None)
    ap.add_argument("--reward_epsilon", type=float, default=1e-3)
    ap.add_argument("--cache_save_every", type=int, default=25)

    ap.add_argument("--context_top_k", type=int, default=3)
    ap.add_argument("--context_top_ks", type=int, nargs="+", default=None)
    ap.add_argument(
        "--context_team_ids",
        nargs="+",
        default=["Reference", "Digilog", "IUACAAS", "Automatoes", "TheAutomators", "AICU"],
    )
    ap.add_argument(
        "--no_context_fallback_to_scalar",
        dest="context_fallback_to_scalar",
        action="store_false",
        help="If set, context_kth_exp_simple requires loss_by_context and will not fall back to scalar reward.",
    )
    ap.set_defaults(context_fallback_to_scalar=True)
    ap.add_argument(
        "--scalar_fallback_reward_mode",
        default="softmin",
        choices=sorted(SCALAR_REWARD_MODES),
    )

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
        if args.reward_mode == "context_kth_exp_simple":
            print(
                f"[sweep] seed={cfg['seed']} "
                f"k={cfg['context_top_k']} "
                f"beta={cfg['beta']}"
            )
        else:
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
        )

        group_cols = group_cols_for_mode(args.reward_mode)
        sort_cols = [c for c in group_cols + ["seed"] if c in df.columns]
        df = df.sort_values(sort_cols)

        summary_csv = summary_dir / "sweep_summary.csv"
        df.to_csv(summary_csv, index=False)

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
            pieces = [f"{col}={row[col]}" for col in group_cols]
            print(
                f"  {' '.join(pieces)}"
                f" | mean_best_reward={row['mean_best_reward']:.6f}"
                f" | mean_best_loss={row['mean_best_loss']:.6f}"
                f" | n={int(row['n_runs'])}"
            )

        print(f"\n[saved] {summary_json}")
        print(f"[saved] {summary_csv}")
        print(f"[saved] {agg_csv}")


if __name__ == "__main__":
    main()
