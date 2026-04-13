#!/usr/bin/env python3
from __future__ import annotations

"""
Seed-aware RQ3 figure generator for larger non-enumerable grouped perturbation spaces.

Design:
- Variance comes from independently trained seeds, not synthetic repeats.
- GFN is evaluated on post-training samples only.
- TPE and Random are evaluated on total_budget = train_budget + post_train_budget
  to match the total simulator cost of "train the GFN, then sample from it".
- Sampled states are evaluated through the project's existing batch.evaluate_all path.

Outputs:
- rq3_seed_aggregated_bars.png
- rq3_seed_points.png
- rq3_results_table.csv
- rq3_per_seed_metrics.csv
- rq3_summary.json
- evaluated_state_losses_cache.csv
"""

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb

from gflownet.envs.greenhouse.constants import (
    BASELINE_PARAMETERS,
    GROUP_ORDER,
    INITIAL_CONDITIONS,
    PARAMETER_BOUNDS,
    PERTURBATION_SCHEME,
)
from gflownet.envs.greenhouse.sim_env import CropSimEnv


COLORS = {
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


from fmu.pool.batch import evaluate_all



def _safe_mean(values: Iterable[float], default: float = 1e6) -> float:
    values = list(values)
    if not values:
        return float(default)
    return float(np.mean(values))


def _cache_metadata_dict(args):
    return {
        "n_cycles": int(args.n_cycles),
        "step_fraction": float(args.step_fraction),
        "decay_factor": float(args.decay_factor),
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
        rows.append({
            "state_key": key,
            "loss": float(loss),
            **meta,
        })
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

    # First choice: the artifact explicitly saved by bayesian_optimization_thresholded.py
    #   name = f"optuna-trials-{wandb.run.id}"
    # which contains optuna_trials.csv with one row per completed trial.
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

    # Fallback: wandb history rows with both trial_step and state_key.
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


def apply_perturbation(current_params: Dict[str, float], group_name: str, action_name: str, step_fraction: float) -> None:
    action = PERTURBATION_SCHEME[group_name][action_name]
    for param_name, direction in action.items():
        if direction == 0:
            continue
        lo, hi = PARAMETER_BOUNDS[param_name]
        val = current_params[param_name]
        current_params[param_name] = float(
            np.clip(val + direction * step_fraction * (hi - lo), lo, hi)
        )


def state_key_to_params(state_key: str, n_cycles: int, step_fraction: float, decay_factor: float) -> Dict[str, float]:
    toks = [t.strip() for t in state_key.split("|")]
    expected = n_cycles * len(GROUP_ORDER)
    if len(toks) != expected:
        raise ValueError(f"Expected {expected} actions in state_key, got {len(toks)}: {state_key}")

    current_params = {
        k: float(INITIAL_CONDITIONS.get(k, BASELINE_PARAMETERS[k]))
        for k in BASELINE_PARAMETERS
    }

    idx = 0
    for cycle in range(n_cycles):
        cycle_step_fraction = step_fraction * (decay_factor ** cycle)
        for group_name in GROUP_ORDER:
            action = toks[idx]
            idx += 1
            apply_perturbation(current_params, group_name, action, cycle_step_fraction)
    return current_params


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


def hamming_distance(a: str, b: str) -> int:
    ta = a.split("|")
    tb = b.split("|")
    if len(ta) != len(tb):
        raise ValueError("Cannot compare state keys of different lengths")
    return sum(x != y for x, y in zip(ta, tb))


def _pairwise_hamming_stats(seq: List[str]):
    uniq = list(dict.fromkeys(seq))
    if len(uniq) < 2:
        return {"mean_hamming": 0.0, "std_hamming": 0.0}
    dists = []
    for i in range(len(uniq)):
        for j in range(i + 1, len(uniq)):
            dists.append(hamming_distance(uniq[i], uniq[j]))
    arr = np.asarray(dists, dtype=float)
    return {"mean_hamming": float(arr.mean()), "std_hamming": float(arr.std())}


def summarize_sequence(seq: List[str], loss_map: Dict[str, float]):
    evaluated = [loss_map[k] for k in seq if k in loss_map]
    if not evaluated:
        return {
            "best_final_loss": np.nan,
            "mean_sampled_loss": np.nan,
            "median_sampled_loss": np.nan,
            "unique_states": 0,
        }
    return {
        "best_final_loss": float(np.min(evaluated)),
        "mean_sampled_loss": float(np.mean(evaluated)),
        "median_sampled_loss": float(np.median(evaluated)),
        "unique_states": len(set(seq)),
    }


def aggregate_rows(rows: List[dict], method: str):
    frame = pd.DataFrame(rows)
    return {
        "Method": method,
        "Best final loss mean": float(frame["best_final_loss"].mean()),
        "Best final loss std": float(frame["best_final_loss"].std(ddof=0)),
        "Mean sampled loss mean": float(frame["mean_sampled_loss"].mean()),
        "Mean sampled loss std": float(frame["mean_sampled_loss"].std(ddof=0)),
        "Median sampled loss mean": float(frame["median_sampled_loss"].mean()),
        "Median sampled loss std": float(frame["median_sampled_loss"].std(ddof=0)),
        "Unique states mean": float(frame["unique_states"].mean()),
        "Unique states std": float(frame["unique_states"].std(ddof=0)),
        "Mean pairwise Hamming mean": float(frame["mean_hamming"].mean()),
        "Mean pairwise Hamming std": float(frame["mean_hamming"].std(ddof=0)),
    }


def plot_seed_aggregated_bars(out_path: Path, summary_rows: List[dict]):
    metrics = [
        ("Best final loss mean", "Best final loss"),
        ("Unique states mean", "Unique states"),
        ("Mean pairwise Hamming mean", "Mean pairwise Hamming"),
    ]
    methods = [r["Method"] for r in summary_rows]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))

    for ax, (metric_key, title) in zip(axes, metrics):
        vals = [r[metric_key] for r in summary_rows]
        err_key = metric_key.replace(" mean", " std")
        errs = [r[err_key] for r in summary_rows]
        x = np.arange(len(methods))
        ax.bar(
            x,
            vals,
            yerr=errs,
            capsize=4,
            color=[COLORS[m] for m in methods],
        )
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.set_title(title)
        ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_seed_points(out_path: Path, per_seed_rows: List[dict]):
    frame = pd.DataFrame(per_seed_rows)
    metrics = [
        ("best_final_loss", "Best final loss"),
        ("unique_states", "Unique states"),
        ("mean_hamming", "Mean pairwise Hamming"),
    ]
    methods = ["TPE", "GFN", "Random"]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gfn_project", required=True)
    ap.add_argument("--gfn_run_ids", nargs="+", required=True)
    ap.add_argument("--bo_project", required=True)
    ap.add_argument("--bo_run_ids", nargs="+", required=True)
    ap.add_argument("--n_cycles", type=int, default=2)
    ap.add_argument("--step_fraction", type=float, required=True)
    ap.add_argument("--decay_factor", type=float, default=1.0, help="Per-cycle step multiplier. Leave at 1.0 if your setup used the same step size in all cycles.")
    ap.add_argument("--train_budget", type=int, required=True)
    ap.add_argument("--post_train_budget", type=int, default=100)
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

    if args.fmu_path is None:
        args.fmu_path = os.environ.get(DEFAULT_FMU_ENV)
    if args.data_dir is None:
        args.data_dir = os.environ.get(DEFAULT_DATA_ENV)

    if args.fmu_path is None:
        raise ValueError(
            f"No FMU path configured. Pass --fmu_path or set the {DEFAULT_FMU_ENV} environment variable."
        )
    if args.data_dir is None:
        raise ValueError(
            f"No greenhouse data directory configured. Pass --data_dir or set the {DEFAULT_DATA_ENV} environment variable."
        )

    if len(args.gfn_run_ids) != len(args.bo_run_ids):
        raise ValueError("You must provide the same number of GFN and BO/TPE run ids.")
    n_seeds = len(args.gfn_run_ids)
    total_budget = args.train_budget + args.post_train_budget

    out_dir = Path(args.output_dir) / f"{args.n_cycles}cycle_sf{args.step_fraction}"
    out_dir.mkdir(parents=True, exist_ok=True)

    gfn_seqs = []
    tpe_seqs = []
    random_seqs = []

    for i, gfn_run_id in enumerate(args.gfn_run_ids):
        ckpt_path = download_gfn_checkpoint(args.gfn_project, gfn_run_id, alias=args.artifact_alias)
        tmp_root = ckpt_path.parent.parent if ckpt_path.parent.name == "files" else ckpt_path.parent
        try:
            ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
            model = build_mlp_from_state_dict(extract_forward_state_dict(ckpt))
            env = CropSimEnv(n_cycles=args.n_cycles, step_fraction=args.step_fraction, precomputed=False, device=args.device)
            gfn_seq = sample_gfn_sequences(
                model,
                env,
                args.post_train_budget,
                device=args.device,
                seed=args.sample_seed_base + i,
            )
            rand_seq = sample_random_sequences(
                env,
                total_budget,
                seed=args.sample_seed_base + 1000 + i,
            )
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)

        gfn_seqs.append(gfn_seq)
        random_seqs.append(rand_seq)

    for i, bo_run_id in enumerate(args.bo_run_ids):
        seq_full = fetch_tpe_trial_sequence(args.bo_project, bo_run_id)
        effective = min(total_budget, len(seq_full))
        if effective < total_budget:
            print(
                f"[WARN] BO/TPE run {bo_run_id} provides only {len(seq_full)} completed trial records; "
                f"using {effective} instead of the requested {total_budget}."
            )
        tpe_seqs.append(seq_full[:effective])

    unique_keys = set()
    for seq in gfn_seqs + tpe_seqs + random_seqs:
        unique_keys.update(seq)
    unique_keys = sorted(unique_keys)

    cache_meta = _cache_metadata_dict(args)
    cache_path = out_dir / args.cache_name
    cached_loss_map = {} if args.ignore_cache else load_loss_cache(cache_path, cache_meta)

    missing_keys = [key for key in unique_keys if key not in cached_loss_map]
    print(f"[cache] loaded {len(cached_loss_map)} matching evaluated states from {cache_path}" if cache_path.exists() and not args.ignore_cache else "[cache] no matching cache loaded")
    print(f"[cache] need to evaluate {len(missing_keys)} new states")

    new_loss_map = {}
    if missing_keys:
        states = {}
        for key in missing_keys:
            combo = tuple(key.split("|"))
            states[combo] = state_key_to_params(
                key,
                n_cycles=args.n_cycles,
                step_fraction=args.step_fraction,
                decay_factor=args.decay_factor,
            )

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

    per_seed_rows = []
    for i in range(n_seeds):
        g = summarize_sequence(gfn_seqs[i], loss_map)
        g.update(_pairwise_hamming_stats(gfn_seqs[i]))
        g["method"] = "GFN"
        g["seed_index"] = i
        per_seed_rows.append(g)

        t = summarize_sequence(tpe_seqs[i], loss_map)
        t.update(_pairwise_hamming_stats(tpe_seqs[i]))
        t["method"] = "TPE"
        t["seed_index"] = i
        per_seed_rows.append(t)

        r = summarize_sequence(random_seqs[i], loss_map)
        r.update(_pairwise_hamming_stats(random_seqs[i]))
        r["method"] = "Random"
        r["seed_index"] = i
        per_seed_rows.append(r)

    pd.DataFrame(per_seed_rows).to_csv(out_dir / "rq3_per_seed_metrics.csv", index=False)

    summary_rows = [
        aggregate_rows([r for r in per_seed_rows if r["method"] == "TPE"], "TPE"),
        aggregate_rows([r for r in per_seed_rows if r["method"] == "GFN"], "GFN"),
        aggregate_rows([r for r in per_seed_rows if r["method"] == "Random"], "Random"),
    ]
    pd.DataFrame(summary_rows).to_csv(out_dir / "rq3_results_table.csv", index=False)

    plot_seed_aggregated_bars(out_dir / "rq3_seed_aggregated_bars.png", summary_rows)
    plot_seed_points(out_dir / "rq3_seed_points.png", per_seed_rows)

    summary = {
        "n_seeds": n_seeds,
        "fmu_path": args.fmu_path,
        "data_dir": args.data_dir,
        "n_cycles": args.n_cycles,
        "step_fraction": args.step_fraction,
        "decay_factor": args.decay_factor,
        "train_budget": args.train_budget,
        "post_train_budget": args.post_train_budget,
        "total_budget_for_tpe_and_random": total_budget,
        "n_unique_states_requested": len(unique_keys),
        "n_states_loaded_from_cache": len(cached_loss_map),
        "n_new_states_evaluated": len(new_loss_map),
        "cache_path": str(cache_path),
        "summary_rows": summary_rows,
    }
    with open(out_dir / "rq3_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved seed-aware RQ3 outputs to {out_dir}")
    print(f"  - cache: {cache_path}")
    print("  - rq3_seed_aggregated_bars.png")
    print("  - rq3_seed_points.png")
    print("  - rq3_results_table.csv")
    print("  - rq3_per_seed_metrics.csv")
    print("  - rq3_summary.json")


if __name__ == "__main__":
    main()
