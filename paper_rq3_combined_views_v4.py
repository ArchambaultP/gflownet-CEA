#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb

from fmu.pool.batch import evaluate_all
from fmu.tomato_controller import TomatoController
from gflownet.envs.greenhouse.constants import (
    BASELINE_PARAMETERS,
    GROUP_ORDER,
    INITIAL_CONDITIONS,
    PARAMETER_BOUNDS,
    PERTURBATION_SCHEME,
)
from gflownet.envs.greenhouse.sim_env import CropSimEnv
from gflownet.proxy.greenhouse.cropSimulatorProxy import CropSimulatorProxy


COLORS = {
    "Observed": "black",
    "Initial": "#7f7f7f",
    "TPE": "#1f77b4",
    "GFN": "#ff7f0e",
    "Random": "#2ca02c",
}

DEFAULT_TEAM_IDS = [
    "AICU",
    "Automatoes",
    "Digilog",
    "IUACAAS",
    "Reference",
    "TheAutomators",
]
DEFAULT_FMU_ENV = "GREENHOUSE_FMU_PATH"
DEFAULT_DATA_ENV = "GREENHOUSE_DATA_DIR"


def _cache_metadata_dict(args):
    return {
        "n_cycles": int(args.n_cycles),
        "step_fraction": float(args.step_fraction),
        "decay_factor": float(args.decay_factor),
        "beta": "" if args.beta is None else str(args.beta),
        "loss_type": str(args.loss_type),
        "huber_delta": float(args.huber_delta),
        "relative_floor_frac": float(args.relative_floor_frac),
        "relative_floor_abs": float(args.relative_floor_abs),
    }


def _cache_row_matches(row: dict, meta: dict) -> bool:
    try:
        return (
            int(row.get("n_cycles")) == int(meta["n_cycles"])
            and abs(float(row.get("step_fraction")) - float(meta["step_fraction"])) <= 1e-15
            and abs(float(row.get("decay_factor")) - float(meta["decay_factor"])) <= 1e-15
            and str(row.get("beta", "")) == str(meta["beta"])
            and str(row.get("loss_type")) == str(meta["loss_type"])
            and abs(float(row.get("huber_delta")) - float(meta["huber_delta"])) <= 1e-15
            and abs(float(row.get("relative_floor_frac")) - float(meta["relative_floor_frac"])) <= 1e-15
            and abs(float(row.get("relative_floor_abs")) - float(meta["relative_floor_abs"])) <= 1e-15
        )
    except Exception:
        return False


def load_loss_cache(cache_path: Path, meta: dict) -> Dict[str, float]:
    if not cache_path.exists():
        return {}
    try:
        df = pd.read_csv(cache_path)
    except Exception:
        return {}
    out = {}
    for row in df.to_dict(orient="records"):
        if _cache_row_matches(row, meta):
            out[str(row["state_key"])] = float(row["loss"])
    return out


def append_loss_cache(cache_path: Path, loss_map: Dict[str, float], meta: dict):
    if not loss_map:
        return
    rows = []
    for key, loss in loss_map.items():
        rows.append({"state_key": key, "loss": float(loss), **meta})
    new_df = pd.DataFrame(rows)
    if cache_path.exists():
        try:
            old_df = pd.read_csv(cache_path)
            df = pd.concat([old_df, new_df], ignore_index=True)
            df = df.drop_duplicates(
                subset=[
                    "state_key",
                    "n_cycles",
                    "step_fraction",
                    "decay_factor",
                    "beta",
                    "loss_type",
                    "huber_delta",
                    "relative_floor_frac",
                    "relative_floor_abs",
                ],
                keep="last",
            )
        except Exception:
            df = new_df
    else:
        df = new_df
    df.to_csv(cache_path, index=False)


def extract_forward_state_dict(ckpt):
    if "forward" in ckpt and isinstance(ckpt["forward"], dict):
        return ckpt["forward"]
    if "state_dict" in ckpt:
        sd = ckpt["state_dict"]
        cand = {k[len("forward."):]: v for k, v in sd.items() if k.startswith("forward.")}
        if cand:
            return cand
    if any(str(k).endswith("weight") for k in ckpt.keys()):
        return ckpt
    raise KeyError("Could not find forward policy state_dict in checkpoint")


def build_mlp_from_state_dict(state_dict):
    weight_keys = [k for k in state_dict if k.endswith("weight")]
    if not weight_keys:
        raise ValueError("No linear weights found in state_dict")

    def key_order(k):
        parts = k.split(".")
        out = []
        for p in parts:
            try:
                out.append((0, int(p)))
            except ValueError:
                out.append((1, p))
        return out

    weight_keys = sorted(weight_keys, key=key_order)
    layers = []
    for i, wk in enumerate(weight_keys):
        bk = wk[:-6] + "bias"
        W = state_dict[wk]
        b = state_dict[bk]
        out_dim, in_dim = W.shape
        layer = nn.Linear(in_dim, out_dim)
        layer.weight.data.copy_(W)
        layer.bias.data.copy_(b)
        layers.append(layer)
        if i < len(weight_keys) - 1:
            layers.append(nn.ReLU())
    model = nn.Sequential(*layers)
    model.eval()
    return model


def download_gfn_checkpoint(project: str, run_id: str, alias: str = "final") -> Path:
    api = wandb.Api()
    artifact = None
    for try_alias in [alias, "latest"]:
        try:
            artifact = api.artifact(f"{project}/ckpt-{run_id}:{try_alias}")
            break
        except wandb.errors.CommError:
            continue
    if artifact is None:
        raise RuntimeError(f"Could not find checkpoint artifact for {project}/ckpt-{run_id}")
    root = Path(tempfile.mkdtemp(prefix=f"wandb_ckpt_{run_id}_"))
    artifact_dir = Path(artifact.download(root=str(root)))
    ckpt = artifact_dir / "final.ckpt"
    if not ckpt.exists():
        ckpts = list(artifact_dir.rglob("*.ckpt"))
        if not ckpts:
            raise FileNotFoundError("No .ckpt file found in downloaded artifact")
        ckpt = ckpts[0]
    return ckpt


def fetch_tpe_trial_sequence(project: str, run_id: str) -> List[str]:
    api = wandb.Api()
    artifact_name = f"{project}/optuna-trials-{run_id}:latest"
    tmp_dir = None
    try:
        artifact = api.artifact(artifact_name)
        tmp_dir = tempfile.mkdtemp(prefix=f"optuna_trials_{run_id}_")
        artifact_dir = artifact.download(root=tmp_dir)
        csv_path = Path(artifact_dir) / "optuna_trials.csv"
        if not csv_path.exists():
            candidates = list(Path(artifact_dir).rglob("optuna_trials.csv"))
            if candidates:
                csv_path = candidates[0]
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            if "trial_step" in df.columns and "state_key" in df.columns:
                df = df.sort_values("trial_step").drop_duplicates("trial_step")
                seq = [str(x) for x in df["state_key"].tolist()]
                print(f"[TPE] loaded {len(seq)} trials from artifact {artifact_name}")
                return seq
    except Exception as e:
        print(f"[TPE] artifact lookup failed for {run_id}: {e}")
    finally:
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    run = api.run(f"{project}/{run_id}")
    rows = []
    for row in run.scan_history():
        if "trial_step" not in row or "state_key" not in row:
            continue
        rows.append({"trial_step": int(row["trial_step"]), "state_key": str(row["state_key"])})
    if not rows:
        raise RuntimeError(
            "No TPE trial sequence could be reconstructed: missing optuna_trials artifact and "
            "no usable wandb history rows with both trial_step and state_key."
        )
    df = pd.DataFrame(rows).sort_values("trial_step").drop_duplicates("trial_step")
    seq = list(df["state_key"])
    print(f"[TPE] loaded {len(seq)} trials from wandb history fallback for {run_id}")
    return seq


def action_value_to_token_map(env):
    return {v: k for k, v in env.pert2id.items()}


def sample_gfn_sequences(model, env: CropSimEnv, n_samples: int, device: str = "cpu", seed: int = 0) -> List[str]:
    model = model.to(device)
    rng = np.random.default_rng(seed)
    action_space = env.get_action_space()
    id2tok = action_value_to_token_map(env)
    out = []

    with torch.no_grad():
        for _ in range(n_samples):
            prefix = [()]
            toks = []
            max_steps = env.n_groups * env.n_cycles + 2
            for _step in range(max_steps):
                x = env.states2policy([prefix]).to(device)
                logits = model(x)[0]
                mask_invalid = env.get_mask_invalid_actions_forward(state=prefix, done=False)
                valid_actions = [a for a, m in zip(action_space, mask_invalid) if not m]
                if not valid_actions:
                    break

                valid_indices = []
                valid_logits = []
                for a, m in zip(action_space, mask_invalid):
                    if not m:
                        valid_indices.append(a)
                        idx = env.action2idx[a] if hasattr(env, "action2idx") else action_space.index(a)
                        valid_logits.append(float(logits[idx].item()))
                valid_logits = np.asarray(valid_logits, dtype=float)
                valid_logits = valid_logits - np.max(valid_logits)
                probs = np.exp(valid_logits)
                probs /= probs.sum()

                chosen = valid_indices[int(rng.choice(len(valid_indices), p=probs))]
                if hasattr(env, "eos") and chosen == env.eos:
                    break

                toks.append(id2tok[chosen])
                prefix = prefix + [(
                    1 + (len(prefix) - 1) // env.n_groups,
                    (len(prefix) - 1) % env.n_groups,
                    chosen,
                )]
            out.append("|".join(toks))
    return out


def sample_random_sequences(env: CropSimEnv, n_samples: int, seed: int = 0) -> List[str]:
    rng = np.random.default_rng(seed)
    action_space = env.get_action_space()
    id2tok = action_value_to_token_map(env)
    out = []

    for _ in range(n_samples):
        prefix = [()]
        toks = []
        max_steps = env.n_groups * env.n_cycles + 2
        for _step in range(max_steps):
            mask_invalid = env.get_mask_invalid_actions_forward(state=prefix, done=False)
            valid_actions = [a for a, m in zip(action_space, mask_invalid) if not m]
            if not valid_actions:
                break

            chosen = valid_actions[int(rng.integers(len(valid_actions)))]
            if hasattr(env, "eos") and chosen == env.eos:
                break

            toks.append(id2tok[chosen])
            prefix = prefix + [(
                1 + (len(prefix) - 1) // env.n_groups,
                (len(prefix) - 1) % env.n_groups,
                chosen,
            )]
        out.append("|".join(toks))
    return out


def apply_perturbation(current_params: Dict[str, float], group_name: str, action_name: str, step_fraction: float) -> None:
    action = PERTURBATION_SCHEME[group_name][action_name]
    for param_name, direction in action.items():
        if direction == 0:
            continue
        lo, hi = PARAMETER_BOUNDS[param_name]
        val = current_params[param_name]
        current_params[param_name] = float(np.clip(val + direction * step_fraction * (hi - lo), lo, hi))


def state_key_to_params(state_key: str, n_cycles: int, step_fraction: float, decay_factor: float) -> Dict[str, float]:
    toks = [t.strip() for t in state_key.split("|")]
    expected = n_cycles * len(GROUP_ORDER)
    if len(toks) != expected:
        raise ValueError(f"Expected {expected} actions in state_key, got {len(toks)}: {state_key}")

    current_params = {k: float(INITIAL_CONDITIONS.get(k, BASELINE_PARAMETERS[k])) for k in BASELINE_PARAMETERS}

    idx = 0
    for cycle in range(n_cycles):
        cycle_step_fraction = step_fraction * (decay_factor ** cycle)
        for group_name in GROUP_ORDER:
            action = toks[idx]
            idx += 1
            apply_perturbation(current_params, group_name, action, cycle_step_fraction)
    return current_params


def hamming_distance(a: str, b: str) -> int:
    ta = a.split("|")
    tb = b.split("|")
    if len(ta) != len(tb):
        raise ValueError("Cannot compare state keys of different lengths")
    return sum(x != y for x, y in zip(ta, tb))


def mean_pairwise_hamming_of_top_unique(seq: List[str], loss_map: Dict[str, float], top_k: int) -> float:
    unique = sorted(set(seq), key=lambda k: (loss_map.get(k, np.inf), k))
    top = unique[:top_k]
    if len(top) < 2:
        return 0.0
    dists = []
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            dists.append(hamming_distance(top[i], top[j]))
    return float(np.mean(dists))


def median_top_k_losses_of_unique(seq: List[str], loss_map: Dict[str, float], top_k: int) -> float:
    unique = sorted(set(seq), key=lambda k: (loss_map.get(k, np.inf), k))
    vals = [loss_map[k] for k in unique if k in loss_map][:top_k]
    if not vals:
        return float("nan")
    return float(np.median(vals))


def summarize_sequence(seq: List[str], loss_map: Dict[str, float], diversity_top_k: int, top_loss_k: int) -> dict:
    evaluated = [loss_map[k] for k in seq if k in loss_map]
    return {
        "best_final_loss": float(np.min(evaluated)) if evaluated else np.nan,
        "mean_sampled_loss": float(np.mean(evaluated)) if evaluated else np.nan,
        "median_sampled_loss": float(np.median(evaluated)) if evaluated else np.nan,
        "median_top_loss_k": median_top_k_losses_of_unique(seq, loss_map, top_loss_k),
        "unique_states": len(set(seq)),
        "mean_hamming_top_unique": mean_pairwise_hamming_of_top_unique(seq, loss_map, diversity_top_k),
    }


def aggregate_rows(rows: List[dict], method: str):
    frame = pd.DataFrame(rows)
    return {
        "Method": method,
        "Best final loss mean": float(frame["best_final_loss"].mean()),
        "Best final loss std": float(frame["best_final_loss"].std(ddof=0)),
        "Median top-loss-k mean": float(frame["median_top_loss_k"].mean()),
        "Median top-loss-k std": float(frame["median_top_loss_k"].std(ddof=0)),
        "Unique states mean": float(frame["unique_states"].mean()),
        "Unique states std": float(frame["unique_states"].std(ddof=0)),
        "Mean Hamming top-unique mean": float(frame["mean_hamming_top_unique"].mean()),
        "Mean Hamming top-unique std": float(frame["mean_hamming_top_unique"].std(ddof=0)),
    }


def plot_seed_aggregated_bars(out_path: Path, summary_rows: List[dict], diversity_top_k: int):
    metrics = [
        ("Best final loss mean", "Best loss"),
        ("Unique states mean", "Unique states"),
        ("Mean Hamming top-unique mean", f"Mean Hamming of top-{diversity_top_k} unique states"),
    ]
    methods = [r["Method"] for r in summary_rows]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, (metric_key, title) in zip(axes, metrics):
        vals = [r[metric_key] for r in summary_rows]
        err_key = metric_key.replace(" mean", " std")
        errs = [r[err_key] for r in summary_rows]
        x = np.arange(len(methods))
        ax.bar(x, vals, yerr=errs, capsize=4, color=[COLORS[m] for m in methods])
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.set_title(title)
        ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_seed_points(out_path: Path, per_seed_rows: List[dict], diversity_top_k: int):
    frame = pd.DataFrame(per_seed_rows)
    metrics = [
        ("best_final_loss", "Best loss"),
        ("unique_states", "Unique states"),
        ("mean_hamming_top_unique", f"Mean Hamming of top-{diversity_top_k} unique states"),
    ]
    methods = ["TPE", "GFN", "Random"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    rng = np.random.default_rng(0)
    for ax, (col, title) in zip(axes, metrics):
        for i, method in enumerate(methods):
            vals = frame.loc[frame["method"] == method, col].to_numpy(dtype=float)
            jitter = rng.normal(0, 0.04, size=len(vals))
            ax.scatter(np.full_like(vals, i, dtype=float) + jitter, vals, color=COLORS[method], alpha=0.9)
            if len(vals) > 0:
                ax.hlines(vals.mean(), i - 0.18, i + 0.18, colors="black", linewidth=2)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods)
        ax.set_title(title)
        ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_absolute_performance_bars(out_path: Path, summary_rows: List[dict], top_loss_k: int):
    metrics = [
        ("Best final loss mean", "Best loss"),
        ("Median top-loss-k mean", f"Median of top-{top_loss_k} losses"),
    ]
    methods = [r["Method"] for r in summary_rows]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    for ax, (metric_key, title) in zip(axes, metrics):
        vals = [r[metric_key] for r in summary_rows]
        err_key = metric_key.replace(" mean", " std")
        errs = [r[err_key] for r in summary_rows]
        x = np.arange(len(methods))
        ax.bar(x, vals, yerr=errs, capsize=4, color=[COLORS[m] for m in methods])
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.set_title(title)
        ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_absolute_performance_points(out_path: Path, per_seed_rows: List[dict], top_loss_k: int):
    frame = pd.DataFrame(per_seed_rows)
    metrics = [
        ("best_final_loss", "Best loss"),
        ("median_top_loss_k", f"Median of top-{top_loss_k} losses"),
    ]
    methods = ["TPE", "GFN", "Random"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    rng = np.random.default_rng(1)
    for ax, (col, title) in zip(axes, metrics):
        for i, method in enumerate(methods):
            vals = frame.loc[frame["method"] == method, col].to_numpy(dtype=float)
            jitter = rng.normal(0, 0.04, size=len(vals))
            ax.scatter(np.full_like(vals, i, dtype=float) + jitter, vals, color=COLORS[method], alpha=0.9)
            if len(vals) > 0:
                ax.hlines(vals.mean(), i - 0.18, i + 0.18, colors="black", linewidth=2)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods)
        ax.set_title(title)
        ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _as_scalar(v):
    if isinstance(v, (list, tuple)):
        return float(v[-1]) if v else 0.0
    if hasattr(v, "__len__"):
        return float(v[-1]) if len(v) > 0 else 0.0
    return float(v)


def simulate_dm_harvest_series(params: Dict[str, float], team: str, fmu_path: str, data_dir: str):
    team_data = CropSimulatorProxy.get_team_obs_dataset(data_dir, team)
    ctrl_data = CropSimulatorProxy.get_team_control_dataset(data_dir, team)
    input_trace = CropSimulatorProxy.compute_trace(ctrl_data, delta="30min")
    setpoints = (team_data.index - team_data.index.min())[1:].total_seconds().tolist()
    init = {**BASELINE_PARAMETERS, **INITIAL_CONDITIONS, **params}
    controller = TomatoController(
        fmu_path,
        start_time=0,
        stop_time=86400.0 * 200,
        step_size=120.0,
        logger=None,
    )
    sim_out = controller.simulate(input_trace, setpoints, init_conds=init)
    n = min(len(sim_out), len(team_data))
    try:
        planting_time = ctrl_data.index.min()
        x_days = ((team_data.index[:n] - planting_time).total_seconds() / 86400.0).to_numpy(dtype=float)
    except Exception:
        x_days = np.arange(n, dtype=float)
    obs_y = team_data["DM_harvest_obs"].iloc[:n].to_numpy(dtype=float)
    sim_y = np.array([_as_scalar(output["C_harvest"]) for _, output in sim_out[:n]], dtype=float)
    return x_days, obs_y, sim_y


def plot_team_grid(out_path: Path, best_params: Dict[str, Dict[str, float]], fmu_path: str, data_dir: str, team_ids: List[str]):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=False, sharey=False)
    axes = axes.flatten()
    for ax, team in zip(axes, team_ids):
        obs_x, obs_y, init_y = simulate_dm_harvest_series(best_params["Initial"], team, fmu_path, data_dir)
        _, _, tpe_y = simulate_dm_harvest_series(best_params["TPE"], team, fmu_path, data_dir)
        _, _, gfn_y = simulate_dm_harvest_series(best_params["GFN"], team, fmu_path, data_dir)
        _, _, rnd_y = simulate_dm_harvest_series(best_params["Random"], team, fmu_path, data_dir)
        ax.plot(obs_x, obs_y, color=COLORS["Observed"], linewidth=2, label="Observed")
        ax.plot(obs_x, init_y, color=COLORS["Initial"], linewidth=1.8, linestyle="--", label="Initial")
        ax.plot(obs_x, tpe_y, color=COLORS["TPE"], linewidth=1.8, label="Best TPE")
        ax.plot(obs_x, gfn_y, color=COLORS["GFN"], linewidth=1.8, label="Best GFN")
        ax.plot(obs_x, rnd_y, color=COLORS["Random"], linewidth=1.8, label="Best Random")
        ax.set_title(team)
        ax.set_xlabel("Days after planting")
        ax.set_ylabel("DM harvest")
        ax.grid(alpha=0.25)
        ax.locator_params(axis="x", nbins=5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gfn_project", required=True)
    ap.add_argument("--gfn_run_ids", nargs="+", required=True)
    ap.add_argument("--bo_project", required=True)
    ap.add_argument("--bo_run_ids", nargs="+", required=True)
    ap.add_argument("--n_cycles", type=int, default=2)
    ap.add_argument("--step_fraction", type=float, required=True)
    ap.add_argument("--decay_factor", type=float, default=1.0)
    ap.add_argument("--retrieval_budget", type=int, default=100)
    ap.add_argument("--train_budget", type=int, default=5000)
    ap.add_argument("--beta", type=float, default=None)
    ap.add_argument("--diversity_top_k", type=int, default=20)
    ap.add_argument("--top_loss_k", type=int, default=10)
    ap.add_argument("--artifact_alias", default="final")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--sample_seed_base", type=int, default=0)
    ap.add_argument("--output_dir", default="paper_rq3_outputs")
    ap.add_argument("--cache_name", default="evaluated_state_losses_cache.csv")
    ap.add_argument("--ignore_cache", action="store_true")
    ap.add_argument("--fmu_path", default=None)
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--n_workers", type=int, default=16)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--loss_type", default="huber_relative", choices=["huber_relative", "rse", "absolute_relative"])
    ap.add_argument("--huber_delta", type=float, default=1.0)
    ap.add_argument("--relative_floor_frac", type=float, default=0.05)
    ap.add_argument("--relative_floor_abs", type=float, default=1e-6)
    args = ap.parse_args()

    if len(args.gfn_run_ids) != len(args.bo_run_ids):
        raise ValueError("You must provide the same number of GFN and BO/TPE run ids.")
    if args.fmu_path is None:
        args.fmu_path = os.environ.get(DEFAULT_FMU_ENV)
    if args.data_dir is None:
        args.data_dir = os.environ.get(DEFAULT_DATA_ENV)
    if args.fmu_path is None:
        raise ValueError(f"No FMU path configured. Pass --fmu_path or set {DEFAULT_FMU_ENV}.")
    if args.data_dir is None:
        raise ValueError(f"No greenhouse data directory configured. Pass --data_dir or set {DEFAULT_DATA_ENV}.")

    n_seeds = len(args.gfn_run_ids)
    total_budget = args.train_budget + args.retrieval_budget
    beta_tag = "" if args.beta is None else f"_beta{args.beta}"
    out_dir = Path(args.output_dir) / f"{args.n_cycles}cycle_sf{args.step_fraction}_budget{args.retrieval_budget}_train{args.train_budget}{beta_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    gfn_retrieval_seqs = []
    tpe_retrieval_seqs = []
    tpe_total_seqs = []
    random_retrieval_seqs = []
    random_total_seqs = []

    for i, gfn_run_id in enumerate(args.gfn_run_ids):
        ckpt_path = download_gfn_checkpoint(args.gfn_project, gfn_run_id, alias=args.artifact_alias)
        tmp_root = ckpt_path.parent.parent if ckpt_path.parent.name == "files" else ckpt_path.parent
        try:
            ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
            model = build_mlp_from_state_dict(extract_forward_state_dict(ckpt))
            env = CropSimEnv(n_cycles=args.n_cycles, step_fraction=args.step_fraction, precomputed=False, device=args.device)
            gfn_seq = sample_gfn_sequences(model, env, args.retrieval_budget, device=args.device, seed=args.sample_seed_base + i)
            rnd_retrieval = sample_random_sequences(env, args.retrieval_budget, seed=args.sample_seed_base + 1000 + i)
            rnd_total = sample_random_sequences(env, total_budget, seed=args.sample_seed_base + 2000 + i)
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)
        gfn_retrieval_seqs.append(gfn_seq)
        random_retrieval_seqs.append(rnd_retrieval)
        random_total_seqs.append(rnd_total)

    for bo_run_id in args.bo_run_ids:
        seq_full = fetch_tpe_trial_sequence(args.bo_project, bo_run_id)
        eff_retrieval = min(args.retrieval_budget, len(seq_full))
        eff_total = min(total_budget, len(seq_full))
        if eff_retrieval < args.retrieval_budget:
            print(f"[WARN] BO/TPE run {bo_run_id} provides only {len(seq_full)} completed trial records; using {eff_retrieval} for retrieval.")
        if eff_total < total_budget:
            print(f"[WARN] BO/TPE run {bo_run_id} provides only {len(seq_full)} completed trial records; using {eff_total} for total-budget comparison.")
        tpe_retrieval_seqs.append(seq_full[:eff_retrieval])
        tpe_total_seqs.append(seq_full[:eff_total])

    unique_keys = set()
    for seq in gfn_retrieval_seqs + tpe_retrieval_seqs + tpe_total_seqs + random_retrieval_seqs + random_total_seqs:
        unique_keys.update(seq)
    unique_keys = sorted(unique_keys)

    cache_meta = _cache_metadata_dict(args)
    beta_cache_tag = "all" if args.beta is None else str(args.beta)
    cache_stem = Path(args.cache_name).stem
    cache_suffix = Path(args.cache_name).suffix or ".csv"
    cache_path = out_dir / f"{cache_stem}_beta{beta_cache_tag}{cache_suffix}"
    cached_loss_map = {} if args.ignore_cache else load_loss_cache(cache_path, cache_meta)
    missing_keys = [key for key in unique_keys if key not in cached_loss_map]

    print(f"[cache] loaded {len(cached_loss_map)} matching evaluated states from {cache_path}" if cache_path.exists() and not args.ignore_cache else "[cache] no matching cache loaded")
    print(f"[cache] need to evaluate {len(missing_keys)} new states")

    new_loss_map = {}
    if missing_keys:
        states = {}
        for key in missing_keys:
            combo = tuple(key.split("|"))
            states[combo] = state_key_to_params(key, n_cycles=args.n_cycles, step_fraction=args.step_fraction, decay_factor=args.decay_factor)

        losses_result = evaluate_all(
            states=states,
            fmu_path=args.fmu_path,
            team_ids=DEFAULT_TEAM_IDS,
            data_dir=args.data_dir,
            n_workers=args.n_workers,
            timeout=args.timeout,
            verbose=True,
            loss_type=args.loss_type,
            huber_delta=args.huber_delta,
            relative_floor_frac=args.relative_floor_frac,
            relative_floor_abs=args.relative_floor_abs,
        )
        if isinstance(losses_result, tuple):
            losses_tuple = losses_result[0]
        else:
            losses_tuple = losses_result
        new_loss_map = {"|".join(k) if isinstance(k, tuple) else k: float(v) for k, v in losses_tuple.items()}
        append_loss_cache(cache_path, new_loss_map, cache_meta)

    loss_map = dict(cached_loss_map)
    loss_map.update(new_loss_map)

    per_seed_equal_rows = []
    per_seed_total_rows = []
    for i in range(n_seeds):
        g_eq = summarize_sequence(gfn_retrieval_seqs[i], loss_map, args.diversity_top_k, args.top_loss_k)
        g_eq["method"] = "GFN"
        g_eq["seed_index"] = i
        per_seed_equal_rows.append(g_eq)

        t_eq = summarize_sequence(tpe_retrieval_seqs[i], loss_map, args.diversity_top_k, args.top_loss_k)
        t_eq["method"] = "TPE"
        t_eq["seed_index"] = i
        per_seed_equal_rows.append(t_eq)

        r_eq = summarize_sequence(random_retrieval_seqs[i], loss_map, args.diversity_top_k, args.top_loss_k)
        r_eq["method"] = "Random"
        r_eq["seed_index"] = i
        per_seed_equal_rows.append(r_eq)

        g_tot = summarize_sequence(gfn_retrieval_seqs[i], loss_map, args.diversity_top_k, args.top_loss_k)
        g_tot["method"] = "GFN"
        g_tot["seed_index"] = i
        per_seed_total_rows.append(g_tot)

        t_tot = summarize_sequence(tpe_total_seqs[i], loss_map, args.diversity_top_k, args.top_loss_k)
        t_tot["method"] = "TPE"
        t_tot["seed_index"] = i
        per_seed_total_rows.append(t_tot)

        r_tot = summarize_sequence(random_total_seqs[i], loss_map, args.diversity_top_k, args.top_loss_k)
        r_tot["method"] = "Random"
        r_tot["seed_index"] = i
        per_seed_total_rows.append(r_tot)

    pd.DataFrame(per_seed_equal_rows).to_csv(out_dir / "rq3_equal_budget_per_seed_metrics.csv", index=False)
    pd.DataFrame(per_seed_total_rows).to_csv(out_dir / "rq3_total_budget_per_seed_metrics.csv", index=False)

    equal_summary_rows = [
        aggregate_rows([r for r in per_seed_equal_rows if r["method"] == "TPE"], "TPE"),
        aggregate_rows([r for r in per_seed_equal_rows if r["method"] == "GFN"], "GFN"),
        aggregate_rows([r for r in per_seed_equal_rows if r["method"] == "Random"], "Random"),
    ]
    total_summary_rows = [
        aggregate_rows([r for r in per_seed_total_rows if r["method"] == "TPE"], "TPE"),
        aggregate_rows([r for r in per_seed_total_rows if r["method"] == "GFN"], "GFN"),
        aggregate_rows([r for r in per_seed_total_rows if r["method"] == "Random"], "Random"),
    ]

    pd.DataFrame(equal_summary_rows).to_csv(out_dir / "rq3_equal_budget_results_table.csv", index=False)
    pd.DataFrame(total_summary_rows).to_csv(out_dir / "rq3_total_budget_results_table.csv", index=False)

    plot_seed_aggregated_bars(out_dir / "rq3_equal_budget_bars.png", equal_summary_rows, args.diversity_top_k)
    plot_seed_points(out_dir / "rq3_equal_budget_points.png", per_seed_equal_rows, args.diversity_top_k)
    plot_absolute_performance_bars(out_dir / "rq3_total_budget_bars.png", total_summary_rows, args.top_loss_k)
    plot_absolute_performance_points(out_dir / "rq3_total_budget_points.png", per_seed_total_rows, args.top_loss_k)

    def best_state_from_method(seqs: List[List[str]]) -> str:
        candidates = []
        for seq in seqs:
            for key in seq:
                if key in loss_map:
                    candidates.append(key)
        if not candidates:
            raise RuntimeError("No evaluated states available to choose a best state.")
        return min(set(candidates), key=lambda k: (loss_map[k], k))

    best_state_keys_equal = {
        "TPE": best_state_from_method(tpe_retrieval_seqs),
        "GFN": best_state_from_method(gfn_retrieval_seqs),
        "Random": best_state_from_method(random_retrieval_seqs),
    }
    best_state_keys_total = {
        "TPE": best_state_from_method(tpe_total_seqs),
        "GFN": best_state_from_method(gfn_retrieval_seqs),
        "Random": best_state_from_method(random_total_seqs),
    }
    best_params_total = {
        "Initial": {k: float(INITIAL_CONDITIONS.get(k, BASELINE_PARAMETERS[k])) for k in BASELINE_PARAMETERS},
        "TPE": state_key_to_params(best_state_keys_total["TPE"], args.n_cycles, args.step_fraction, args.decay_factor),
        "GFN": state_key_to_params(best_state_keys_total["GFN"], args.n_cycles, args.step_fraction, args.decay_factor),
        "Random": state_key_to_params(best_state_keys_total["Random"], args.n_cycles, args.step_fraction, args.decay_factor),
    }
    plot_team_grid(out_dir / "rq3_best_state_team_grid_total_budget.png", best_params_total, args.fmu_path, args.data_dir, DEFAULT_TEAM_IDS)

    summary = {
        "n_seeds": n_seeds,
        "n_cycles": args.n_cycles,
        "step_fraction": args.step_fraction,
        "decay_factor": args.decay_factor,
        "retrieval_budget": args.retrieval_budget,
        "train_budget": args.train_budget,
        "beta": args.beta,
        "total_budget": total_budget,
        "diversity_top_k": args.diversity_top_k,
        "top_loss_k": args.top_loss_k,
        "fmu_path": args.fmu_path,
        "data_dir": args.data_dir,
        "n_unique_states_requested": len(unique_keys),
        "n_states_loaded_from_cache": len(cached_loss_map),
        "n_new_states_evaluated": len(new_loss_map),
        "cache_path": str(cache_path),
        "best_state_keys_equal": best_state_keys_equal,
        "best_state_keys_total": best_state_keys_total,
        "best_state_losses_equal": {k: float(loss_map[v]) for k, v in best_state_keys_equal.items()},
        "best_state_losses_total": {k: float(loss_map[v]) for k, v in best_state_keys_total.items()},
        "equal_summary_rows": equal_summary_rows,
        "total_summary_rows": total_summary_rows,
    }
    with open(out_dir / "rq3_combined_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved combined RQ3 outputs to {out_dir}")
    print(f"  - cache: {cache_path}")
    print("  - rq3_equal_budget_bars.png")
    print("  - rq3_equal_budget_points.png")
    print("  - rq3_total_budget_bars.png")
    print("  - rq3_total_budget_points.png")
    print("  - rq3_best_state_team_grid_total_budget.png")
    print("  - rq3_equal_budget_results_table.csv")
    print("  - rq3_total_budget_results_table.csv")
    print("  - rq3_equal_budget_per_seed_metrics.csv")
    print("  - rq3_total_budget_per_seed_metrics.csv")
    print("  - rq3_combined_summary.json")


if __name__ == "__main__":
    main()
