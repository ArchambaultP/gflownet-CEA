#!/usr/bin/env python3
"""
Optuna / TPE baseline over the same grouped discrete perturbation space as the GFN.

Key changes relative to the older baseline:
- uses the thresholded-beta proxy
- searches the same grouped action space / cycle structure as the GFN
- logs explicit state_key + chosen actions to wandb
- saves a trials CSV/JSON for downstream comparison
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

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
        parameters[i] = float(config.get(k, INITIAL_CONDITIONS.get(k, BASELINE_PARAMETERS[k])))

    if normalize:
        for i, k in enumerate(sorted(BASELINE_PARAMETERS)):
            lo, hi = PARAMETER_BOUNDS.get(k, (0, 0))
            parameters[i] = 0.5 if lo == hi else (parameters[i] - lo) / (hi - lo)
    return parameters


def action_seq_to_key(action_seq: List[str]) -> str:
    return "|".join(action_seq)


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


class BOBaseline:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.proxy = CropSimulatorProxy(
            reward_cache_path=args.reward_cache_path,
            reward_mode=args.reward_mode,
            beta=args.beta,
            loss_norm=args.loss_norm,
            q_low=args.q_low,
            q_high=args.q_high,
            reward_anchor=args.reward_anchor,
            reward_scale=args.reward_scale,
            reward_tau=args.reward_tau,
            tau_quantile=args.tau_quantile,
            threshold_temperature=args.threshold_temperature,
            reward_epsilon=args.reward_epsilon,
            cache_save_every=args.cache_save_every,
            device=args.device,
            float_precision=args.float_precision,
        )
        self.best_reward = -np.inf
        self.best_loss = np.inf
        self.results: List[TrialResult] = []

    def evaluate(self, param_values: Dict[str, float]) -> tuple[float, float]:
        proxy_config = build_config(param_values, normalize=False)
        cache_key = self.proxy._make_cache_key(proxy_config)
        reward = float(self.proxy([proxy_config]).item())
        loss = float(self.proxy.reward_cache[cache_key])
        return reward, loss

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
        reward, loss = self.evaluate(current_params)

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
        }
        for cycle in range(self.args.n_cycles):
            for gi, group_name in enumerate(GROUP_ORDER):
                idx = cycle * len(GROUP_ORDER) + gi
                log_dict[f"{group_name}_cycle{cycle}"] = action_seq[idx]
        for k, v in current_params.items():
            log_dict[k] = v
        wandb.log(log_dict)

        return reward

    def save_outputs(self, output_dir: Path) -> None:
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


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_cycles", type=int, default=1)
    ap.add_argument("--step_fraction", type=float, default=0.15)
    ap.add_argument("--decay_factor", type=float, default=0.5)
    ap.add_argument("--n_trials", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--reward_cache_path", default="precomputed/reward_table_sf0.15.json")
    ap.add_argument("--reward_mode", default="thresholded_sigmoid", choices=["thresholded_sigmoid", "softmin"])
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--loss_norm", default="q10q90", choices=["q10q90", "none"])
    ap.add_argument("--q_low", type=float, default=0.10)
    ap.add_argument("--q_high", type=float, default=0.90)
    ap.add_argument("--reward_anchor", default=None)
    ap.add_argument("--reward_scale", default=None)
    ap.add_argument("--reward_tau", default=None)
    ap.add_argument("--tau_quantile", type=float, default=0.10)
    ap.add_argument("--threshold_temperature", type=float, default=0.08)
    ap.add_argument("--reward_epsilon", type=float, default=1e-3)
    ap.add_argument("--cache_save_every", type=int, default=100)

    ap.add_argument("--wandb_project", default="optuna-crop-calibration")
    ap.add_argument("--wandb_entity", default=None)
    ap.add_argument("--wandb_mode", default="online")
    ap.add_argument("--output_dir", default="bo_outputs")
    ap.add_argument("--study_name", default=None)

    ap.add_argument("--device", default="cpu")
    ap.add_argument("--float_precision", type=int, default=32)
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    run_name = args.study_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    wandb.init(
        mode=args.wandb_mode,
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config=vars(args),
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

    out_dir = Path(args.output_dir) / wandb.run.id
    baseline.save_outputs(out_dir)

    print(f"Best reward: {study.best_value:.6f}")
    print(f"Best actions: {study.best_params}")
    if hasattr(baseline.proxy, "save_final_cache"):
        baseline.proxy.save_final_cache()

    wandb.finish()


if __name__ == "__main__":
    main()
