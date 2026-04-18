#!/usr/bin/env python3
from __future__ import annotations

import argparse
import atexit
import copy
import itertools
import json
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
import wandb

from gflownet.envs.greenhouse.constants_unique_actions_vanthoor import (
    BASELINE_PARAMETERS,
    GROUP_ORDER,
    INITIAL_CONDITIONS,
    PARAMETER_BOUNDS,
    PERTURBATION_SCHEME,
)
from fmu.pool.batch_contextual import evaluate_all


SCALAR_REWARD_MODES = {"softmin", "thresholded_sigmoid"}
CONTEXT_REWARD_MODES = {
    "context_cvar_exp_simple",
    "context_cvar_blend_exp_simple",
}


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


def action_seq_to_key(action_seq: List[str]) -> str:
    return "|".join(action_seq)


def _entry_from_cache_value(value) -> Dict[str, object]:
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


def build_context_quantile_stats_from_entries(
    entries_by_key: Dict[str, Dict[str, object]],
    team_ids: List[str],
    q_low: float,
    q_high: float,
) -> Dict[str, Tuple[float, float]]:
    out: Dict[str, Tuple[float, float]] = {}
    for team in team_ids:
        vals = []
        for entry in entries_by_key.values():
            lbc = entry.get("loss_by_context", {})
            if team in lbc:
                vals.append(float(lbc[team]))
        arr = np.asarray(vals, dtype=float)
        if arr.size > 0:
            lo = float(np.quantile(arr, q_low))
            hi = float(np.quantile(arr, q_high))
            out[team] = (lo, max(hi - lo, 1e-12))
    return out


def recompute_contextual_cvar_rewards_from_entries(
    entries_by_key: Dict[str, Dict[str, object]],
    context_team_ids: List[str],
    context_tail_count: int,
    beta: float,
    q_low: float,
    q_high: float,
    context_cvar_lambda: Optional[float] = None,
    context_fallback_to_scalar: bool = True,
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
    quant_stats = build_context_quantile_stats_from_entries(
        entries_by_key, team_ids, q_low=q_low, q_high=q_high
    )

    reward_by_key: Dict[str, Optional[float]] = {}
    agg_by_key: Dict[str, Optional[float]] = {}

    for key in keys:
        loss_by_context = {
            str(team): float(v)
            for team, v in entries_by_key[key].get("loss_by_context", {}).items()
        }

        vals = []
        for team in team_ids:
            if team not in loss_by_context:
                continue
            if team not in quant_stats:
                continue
            lo, scale = quant_stats[team]
            vals.append((float(loss_by_context[team]) - lo) / scale)

        if not vals:
            reward_by_key[key] = None if context_fallback_to_scalar else 1e-12
            agg_by_key[key] = None
            continue

        vals_sorted = sorted(float(v) for v in vals)
        m = min(max(int(context_tail_count), 1), len(vals_sorted))
        tail_mean = float(np.mean(vals_sorted[-m:]))

        if context_cvar_lambda is None:
            agg = tail_mean
        else:
            mean_all = float(np.mean(vals_sorted))
            agg = (1.0 - float(context_cvar_lambda)) * mean_all + float(context_cvar_lambda) * tail_mean

        agg_by_key[key] = agg
        reward_by_key[key] = float(np.exp(-float(beta) * agg))

    meta = {
        "context_tail_count": int(context_tail_count),
        "context_cvar_lambda": None if context_cvar_lambda is None else float(context_cvar_lambda),
        "beta": float(beta),
        "q_low": float(q_low),
        "q_high": float(q_high),
        "context_team_ids": list(team_ids),
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


class SimpleTPEBaseline:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.cache_path = Path(args.reward_cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache = self._load_cache(self.cache_path)

        self.best_reward = -np.inf
        self.best_loss = np.inf
        self.results: List[TrialResult] = []

        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_dirty = 0
        self.start_time = time.perf_counter()
        self.last_trial_time = self.start_time

        atexit.register(self._flush_cache)

    @staticmethod
    def _load_cache(path: Path) -> Dict[str, Dict[str, object]]:
        if not path.exists():
            return {}
        with open(path, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Cache file must contain a JSON object: {path}")
        out = {}
        for k, v in data.items():
            entry = _entry_from_cache_value(v)
            # safeguard: do not keep broken contextual rows
            if entry.get("loss_by_context"):
                out[k] = entry
        return out

    def _flush_cache(self) -> None:
        if self.cache_dirty <= 0:
            return
        tmp = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        serializable = {}
        for k, entry in self.cache.items():
            if not entry.get("loss_by_context"):
                continue
            out = {"loss": float(entry["loss"])}
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

    def _reward_for_key(self, key: str, extra_entries: Optional[Dict[str, Dict[str, object]]] = None) -> float:
        entries = dict(self.cache)
        if extra_entries:
            entries.update(extra_entries)

        if self.args.reward_mode == "context_cvar_exp_simple":
            reward_by_key, _ = recompute_contextual_cvar_rewards_from_entries(
                entries_by_key=entries,
                context_team_ids=self.args.context_team_ids,
                context_tail_count=self.args.context_tail_count,
                beta=self.args.beta,
                q_low=self.args.q_low,
                q_high=self.args.q_high,
                context_cvar_lambda=None,
                context_fallback_to_scalar=self.args.context_fallback_to_scalar,
            )
        else:
            reward_by_key, _ = recompute_contextual_cvar_rewards_from_entries(
                entries_by_key=entries,
                context_team_ids=self.args.context_team_ids,
                context_tail_count=self.args.context_tail_count,
                beta=self.args.beta,
                q_low=self.args.q_low,
                q_high=self.args.q_high,
                context_cvar_lambda=self.args.context_cvar_lambda,
                context_fallback_to_scalar=self.args.context_fallback_to_scalar,
            )

        if any(v is None for v in reward_by_key.values()):
            scalar_rewards, _ = recompute_scalar_rewards_from_entries(
                entries_by_key=entries,
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

    def evaluate(self, state_key: str, state_tuple: Tuple[str, ...], param_values: Dict[str, float]) -> tuple[float, float, bool]:
        if state_key in self.cache:
            self.cache_hits += 1
            entry = self.cache[state_key]
            loss = float(entry["loss"])
            reward = self._reward_for_key(state_key)
            return reward, loss, True

        self.cache_misses += 1

        losses, details = evaluate_all(
            states={state_tuple: param_values},
            fmu_path=self.args.fmu_path,
            team_ids=self.args.context_team_ids,
            data_dir=self.args.data_dir,
            n_workers=self.args.n_workers,
            timeout=self.args.timeout,
            verbose=self.args.verbose_eval,
            loss_type=self.args.loss_type,
            huber_delta=self.args.huber_delta,
            relative_floor_frac=self.args.relative_floor_frac,
            relative_floor_abs=self.args.relative_floor_abs,
            return_details=True,
        )

        entry = {
            "loss": 1e6,
            "loss_by_context": {},
            "params": {k: float(v) for k, v in param_values.items()},
        }

        if state_tuple in losses:
            entry["loss"] = float(losses[state_tuple])
        lbc = details.get("loss_by_context", {}) if isinstance(details, dict) else {}
        if state_tuple in lbc:
            entry["loss_by_context"] = {
                str(team): float(v) for team, v in lbc[state_tuple].items()
            }

        extra_entries = {state_key: entry}
        reward = self._reward_for_key(state_key, extra_entries=extra_entries)

        if entry["loss_by_context"]:
            self.cache[state_key] = entry
            self.cache_dirty += 1
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
        state_tuple = tuple(action_seq)
        reward, loss, cache_hit = self.evaluate(state_key, state_tuple, current_params)

        self.best_reward = max(self.best_reward, reward)
        self.best_loss = min(self.best_loss, loss)

        trial.set_user_attr("state_key", state_key)
        trial.set_user_attr("cache_hit", cache_hit)
        trial.set_user_attr("loss", loss)

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
        wandb.log(log_dict)
        self._print_progress(trial.number, reward, loss, cache_hit)
        return reward

    def _print_progress(self, trial_num: int, reward: float, loss: float, cache_hit: bool) -> None:
        if self.args.log_every <= 0:
            return
        if ((trial_num + 1) % self.args.log_every) != 0 and (trial_num + 1) != self.args.n_trials:
            return

        now = time.perf_counter()
        trial_elapsed = now - self.last_trial_time
        total_elapsed = now - self.start_time
        done = trial_num + 1
        hit_rate = self.cache_hits / max(self.cache_hits + self.cache_misses, 1)
        print(
            f"[progress] trial={done}/{self.args.n_trials} "
            f"reward={reward:.6f} loss={loss:.6f} best_reward={self.best_reward:.6f} "
            f"best_loss={self.best_loss:.6f} cache_hit={int(cache_hit)} "
            f"hit_rate={hit_rate:.3f} trial_s={trial_elapsed:.2f} total_s={total_elapsed:.2f}",
            flush=True,
        )
        self.last_trial_time = now

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
            "context_tail_count": int(self.args.context_tail_count),
            "context_cvar_lambda": float(self.args.context_cvar_lambda),
            "beta": float(self.args.beta),
            "context_team_ids": list(self.args.context_team_ids),
            "n_workers": int(self.args.n_workers),
        }
        with open(output_dir / "optuna_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Keep this optional to avoid wandb artifact staging issues on fragile environments.
        if self.args.log_artifacts:
            try:
                artifact = wandb.Artifact(
                    name=f"optuna-trials-{wandb.run.id}",
                    type="optuna_results",
                    metadata=summary,
                )
                artifact.add_file(str(csv_path))
                artifact.add_file(str(output_dir / "optuna_summary.json"))
                wandb.log_artifact(artifact)
            except Exception as e:
                print(f"[warn] wandb artifact logging failed: {e}")

        print(f"[saved] {csv_path}")
        print(f"[saved] {output_dir / 'optuna_summary.json'}")
        self._flush_cache()


def build_sweep_grid(args: argparse.Namespace) -> List[dict]:
    seeds = args.seeds if args.seeds is not None else [args.seed]
    if args.reward_mode == "context_cvar_blend_exp_simple":
        tail_counts = args.context_tail_counts if args.context_tail_counts is not None else [args.context_tail_count]
        betas = args.betas if args.betas is not None else [args.beta]
        lambdas = args.context_cvar_lambdas if args.context_cvar_lambdas is not None else [args.context_cvar_lambda]
        grid = []
        for tail_count, beta, lam, seed in itertools.product(tail_counts, betas, lambdas, seeds):
            grid.append(
                {
                    "context_tail_count": int(tail_count),
                    "beta": float(beta),
                    "context_cvar_lambda": float(lam),
                    "seed": int(seed),
                }
            )
        return grid

    tail_counts = args.context_tail_counts if args.context_tail_counts is not None else [args.context_tail_count]
    betas = args.betas if args.betas is not None else [args.beta]
    grid = []
    for tail_count, beta, seed in itertools.product(tail_counts, betas, seeds):
        grid.append(
            {
                "context_tail_count": int(tail_count),
                "beta": float(beta),
                "seed": int(seed),
            }
        )
    return grid


def config_to_name(args: argparse.Namespace) -> str:
    if args.reward_mode == "context_cvar_blend_exp_simple":
        return (
            f"{args.reward_mode}"
            f"_seed{args.seed}"
            f"_tail{args.context_tail_count}"
            f"_lam{args.context_cvar_lambda}"
            f"_b{args.beta}"
        )
    return (
        f"{args.reward_mode}"
        f"_seed{args.seed}"
        f"_tail{args.context_tail_count}"
        f"_b{args.beta}"
    )


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

    baseline = SimpleTPEBaseline(args)
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
        "context_tail_count": int(args.context_tail_count),
        "context_cvar_lambda": float(args.context_cvar_lambda),
        "beta": float(args.beta),
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
    ap.add_argument("--fmu_path", default="fmu/FMU/tomato.fmu")
    ap.add_argument("--data_dir", default="data/greenhouse/secondEdition")

    ap.add_argument("--n_cycles", type=int, default=2)
    ap.add_argument("--step_fraction", type=float, default=0.3)
    ap.add_argument("--decay_factor", type=float, default=0.5)
    ap.add_argument("--n_trials", type=int, default=100)
    ap.add_argument("--n_workers", type=int, default=6)
    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)

    ap.add_argument(
        "--reward_mode",
        default="context_cvar_blend_exp_simple",
        choices=sorted(CONTEXT_REWARD_MODES),
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
    ap.add_argument("--threshold_temperature", type=float, default=0.08)
    ap.add_argument("--reward_epsilon", type=float, default=1e-3)
    ap.add_argument("--cache_save_every", type=int, default=25)

    ap.add_argument("--context_tail_count", type=int, default=2)
    ap.add_argument("--context_tail_counts", type=int, nargs="+", default=None)
    ap.add_argument("--context_cvar_lambda", type=float, default=0.25)
    ap.add_argument("--context_cvar_lambdas", type=float, nargs="+", default=None)
    ap.add_argument(
        "--context_team_ids",
        nargs="+",
        default=["Reference", "Digilog", "IUACAAS", "Automatoes", "TheAutomators", "AICU"],
    )
    ap.add_argument(
        "--no_context_fallback_to_scalar",
        dest="context_fallback_to_scalar",
        action="store_false",
    )
    ap.set_defaults(context_fallback_to_scalar=True)
    ap.add_argument(
        "--scalar_fallback_reward_mode",
        default="softmin",
        choices=sorted(SCALAR_REWARD_MODES),
    )

    ap.add_argument("--loss_type", default="absolute_relative", choices=["absolute_relative", "huber_relative", "rse"])
    ap.add_argument("--huber_delta", type=float, default=1.0)
    ap.add_argument("--relative_floor_frac", type=float, default=0.05)
    ap.add_argument("--relative_floor_abs", type=float, default=1e-6)
    ap.add_argument("--verbose_eval", action="store_true")

    ap.add_argument("--reward_cache_path", default="precomputed_cvar/tpe_twocycle_cache.json")
    ap.add_argument("--log_every", type=int, default=1)
    ap.add_argument("--wandb_project", default="optuna-crop-calibration")
    ap.add_argument("--wandb_entity", default=None)
    ap.add_argument("--wandb_mode", default="online")
    ap.add_argument("--wandb_group", default=None)
    ap.add_argument("--output_dir", default="bo_outputs")
    ap.add_argument("--study_name", default=None)
    ap.add_argument("--log_artifacts", action="store_true")
    ap.add_argument("--device", default="cpu")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    grid = build_sweep_grid(args)

    print(f"[sweep] running {len(grid)} configuration(s)")
    print(
        f"[config] n_cycles={args.n_cycles} step_fraction={args.step_fraction} "
        f"decay_factor={args.decay_factor} n_workers={args.n_workers} log_every={args.log_every}"
    )

    all_results = []
    for cfg in grid:
        print(
            f"[sweep] seed={cfg['seed']} "
            f"tail={cfg['context_tail_count']} "
            f"{'lambda=' + str(cfg['context_cvar_lambda']) + ' ' if 'context_cvar_lambda' in cfg else ''}"
            f"beta={cfg['beta']}"
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
        summary_csv = summary_dir / "sweep_summary.csv"
        df.to_csv(summary_csv, index=False)
        print(f"[saved] {summary_json}")
        print(f"[saved] {summary_csv}")


if __name__ == "__main__":
    main()
