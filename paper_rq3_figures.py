#!/usr/bin/env python3
from __future__ import annotations

"""
Seed-aware RQ3 figure generator for larger non-enumerable grouped perturbation spaces.

Design:
- Variance comes from independently trained seeds, not synthetic repeats.
- GFN is evaluated on post-training samples only.
- TPE and Random are evaluated on total_budget = train_budget + post_train_budget
  to match the total simulator cost of "train the GFN, then sample from it".
- All sampled states are evaluated live with the simulator.

Outputs:
- rq3_seed_aggregated_bars.png
- rq3_seed_points.png
- rq3_results_table.csv
- rq3_per_seed_metrics.csv
- rq3_summary.json
"""

import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
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


_SINGLE_EVAL_SCRIPT = r"""
import os
import pickle
import sys
import numpy as np

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

args_file, result_file = sys.argv[1], sys.argv[2]

with open(args_file, 'rb') as f:
    (
        key,
        params,
        fmu_path,
        team,
        data_dir,
        loss_type,
        huber_delta,
        relative_floor_frac,
        relative_floor_abs,
    ) = pickle.load(f)

from fmu.tomato_controller import TomatoController
from gflownet.envs.greenhouse.constants import BASELINE_PARAMETERS, INITIAL_CONDITIONS
from gflownet.proxy.greenhouse.cropSimulatorProxy import CropSimulatorProxy


def _as_scalar(v):
    if isinstance(v, (list, tuple)):
        return float(v[-1]) if v else 0.0
    if hasattr(v, '__len__'):
        return float(v[-1]) if len(v) > 0 else 0.0
    return float(v)


def _series_floor(values, relative_floor_frac, relative_floor_abs):
    arr = np.asarray(values, dtype=float)
    arr = np.abs(arr[np.isfinite(arr)])
    arr = arr[arr > 0]
    if arr.size == 0:
        scale = 1.0
    else:
        scale = float(np.percentile(arr, 90))
    return max(float(relative_floor_abs), float(relative_floor_frac) * scale)


def _huber(e, delta):
    ae = abs(float(e))
    if ae <= delta:
        return 0.5 * ae * ae
    return float(delta) * (ae - 0.5 * float(delta))


def _point_loss(e, loss_type, huber_delta):
    if loss_type == 'huber_relative':
        return _huber(e, huber_delta)
    if loss_type == 'absolute_relative':
        return abs(float(e))
    return float(e) * float(e)


team_data = CropSimulatorProxy.get_team_obs_dataset(data_dir, team)
input_trace = CropSimulatorProxy.compute_trace(
    CropSimulatorProxy.get_team_control_dataset(data_dir, team), delta='30min')
setpoints = (team_data.index - team_data.index.min())[1:].total_seconds().tolist()

init = {**BASELINE_PARAMETERS, **INITIAL_CONDITIONS, **params}
controller = TomatoController(
    fmu_path, start_time=0, stop_time=86400.0 * 200,
    step_size=120.0, logger=None)

sim_out = controller.simulate(input_trace, setpoints, init_conds=init)

DM_floor = _series_floor(team_data['DM_harvest_obs'].values, relative_floor_frac, relative_floor_abs)
N_floor = _series_floor(team_data['N_harvest_per_m2'].values, relative_floor_frac, relative_floor_abs)

point_losses = []
for idx, (_, output) in enumerate(sim_out):
    y_DM = float(team_data['DM_harvest_obs'].iloc[idx])
    y_N = float(team_data['N_harvest_per_m2'].iloc[idx])
    y_hat_DM = _as_scalar(output['C_harvest'])
    y_hat_N = _as_scalar(output['N_harvest'])

    if y_DM > 0:
        e_dm = (float(y_hat_DM) - y_DM) / max(abs(y_DM), DM_floor)
        point_losses.append(_point_loss(e_dm, loss_type, huber_delta))

    if y_N > 0:
        e_n = (float(y_hat_N) - y_N) / max(abs(y_N), N_floor)
        point_losses.append(_point_loss(e_n, loss_type, huber_delta))

with open(result_file, 'wb') as f:
    pickle.dump(
        {
            'key': key,
            'team': team,
            'point_losses': point_losses,
        },
        f,
    )
"""


def _normalize_combo_key(key):
    if isinstance(key, tuple):
        return key
    if isinstance(key, str) and "|" in key:
        return tuple(key.split("|"))
    return key


def _safe_mean(values: Iterable[float], default: float = 1e6) -> float:
    values = list(values)
    if not values:
        return float(default)
    return float(np.mean(values))


def evaluate_all_fixed(
    states=None,
    terminal_states=None,
    fmu_path=None,
    team_ids=None,
    data_dir=None,
    n_workers=16,
    timeout=180,
    verbose=False,
    loss_type="huber_relative",
    huber_delta=1.0,
    relative_floor_frac=0.05,
    relative_floor_abs=1e-6,
    temp_root=None,
):
    if states is None:
        states = terminal_states
    if states is None:
        raise ValueError("One of `states` or `terminal_states` must be provided.")
    if team_ids is None:
        raise ValueError("team_ids must be provided")

    work = [("|".join(combo), params) for combo, params in states.items()]
    jobs = [(key, params, team) for key, params in work for team in team_ids]

    temp_root = temp_root or os.environ.get("BATCH_EVAL_TMPDIR")
    if temp_root:
        os.makedirs(temp_root, exist_ok=True)

    total_done = 0
    team_point_losses: Dict[str, Dict[str, List[float]]] = {}

    with tempfile.TemporaryDirectory(prefix="batch_eval_", dir=temp_root) as tmp_dir:
        for wave_start in range(0, len(jobs), n_workers):
            wave = jobs[wave_start:wave_start + n_workers]
            procs = []
            wave_files = []

            for i, (key, params, team) in enumerate(wave):
                idx = wave_start + i
                args_file = os.path.join(tmp_dir, f"args_{idx}.pkl")
                result_file = os.path.join(tmp_dir, f"result_{idx}.pkl")
                wave_files.extend([args_file, result_file])
                with open(args_file, "wb") as f:
                    pickle.dump(
                        (
                            key,
                            params,
                            fmu_path,
                            team,
                            data_dir,
                            loss_type,
                            huber_delta,
                            relative_floor_frac,
                            relative_floor_abs,
                        ),
                        f,
                    )

                p = subprocess.Popen(
                    [sys.executable, "-c", _SINGLE_EVAL_SCRIPT, args_file, result_file],
                    env={
                        **os.environ,
                        "OPENBLAS_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "1",
                        "OMP_NUM_THREADS": "1",
                        "PYTHONPATH": os.pathsep.join(sys.path),
                        **({"TMPDIR": temp_root} if temp_root else {}),
                    },
                    cwd=os.getcwd(),
                )
                procs.append((p, result_file))

            for p, result_file in procs:
                try:
                    p.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()
                    continue

                if p.returncode != 0:
                    continue

                try:
                    with open(result_file, "rb") as f:
                        payload = pickle.load(f)
                    key = payload["key"]
                    team = payload["team"]
                    point_losses = list(payload.get("point_losses", []))
                    team_point_losses.setdefault(key, {})[team] = point_losses
                    total_done += 1
                except Exception:
                    pass

            for fp in wave_files:
                try:
                    if os.path.exists(fp):
                        os.remove(fp)
                except OSError:
                    pass

            if verbose:
                print(f"  {total_done}/{len(jobs)} team evaluations done")

        losses = {}
        for key, team_map in team_point_losses.items():
            per_team_means = [_safe_mean(vals) for vals in team_map.values() if vals]
            losses[_normalize_combo_key(key)] = _safe_mean(per_team_means)

        return losses


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
    run = api.run(f"{project}/{run_id}")
    rows = []
    for row in run.scan_history():
        if "trial_step" not in row or "state_key" not in row:
            continue
        rows.append({"trial_step": int(row["trial_step"]), "state_key": str(row["state_key"])})
    if not rows:
        raise RuntimeError("No TPE trial history with trial_step/state_key found in wandb run")
    df = pd.DataFrame(rows).sort_values("trial_step").drop_duplicates("trial_step")
    return list(df["state_key"])


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

    ap.add_argument("--fmu_path", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--team_ids", nargs="+", required=True)
    ap.add_argument("--n_workers", type=int, default=16)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--loss_type", default="huber_relative", choices=["huber_relative", "rse", "absolute_relative"])
    ap.add_argument("--huber_delta", type=float, default=1.0)
    ap.add_argument("--relative_floor_frac", type=float, default=0.05)
    ap.add_argument("--relative_floor_abs", type=float, default=1e-6)
    args = ap.parse_args()

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
            print(f"[WARN] BO/TPE run {bo_run_id} has only {len(seq_full)} trials; using {effective} instead of {total_budget}.")
        tpe_seqs.append(seq_full[:effective])

    unique_keys = set()
    for seq in gfn_seqs + tpe_seqs + random_seqs:
        unique_keys.update(seq)
    unique_keys = sorted(unique_keys)

    states = {}
    for key in unique_keys:
        combo = tuple(key.split("|"))
        states[combo] = state_key_to_params(
            key,
            n_cycles=args.n_cycles,
            step_fraction=args.step_fraction,
            decay_factor=args.decay_factor,
        )

    losses_tuple = evaluate_all_fixed(
        states=states,
        fmu_path=args.fmu_path,
        team_ids=args.team_ids,
        data_dir=args.data_dir,
        n_workers=args.n_workers,
        timeout=args.timeout,
        loss_type=args.loss_type,
        huber_delta=args.huber_delta,
        relative_floor_frac=args.relative_floor_frac,
        relative_floor_abs=args.relative_floor_abs,
        verbose=True,
    )
    loss_map = {"|".join(k) if isinstance(k, tuple) else k: float(v) for k, v in losses_tuple.items()}

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
        "n_cycles": args.n_cycles,
        "step_fraction": args.step_fraction,
        "decay_factor": args.decay_factor,
        "train_budget": args.train_budget,
        "post_train_budget": args.post_train_budget,
        "total_budget_for_tpe_and_random": total_budget,
        "n_unique_states_evaluated": len(unique_keys),
        "summary_rows": summary_rows,
    }
    with open(out_dir / "rq3_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved seed-aware RQ3 outputs to {out_dir}")
    print("  - rq3_seed_aggregated_bars.png")
    print("  - rq3_seed_points.png")
    print("  - rq3_results_table.csv")
    print("  - rq3_per_seed_metrics.csv")
    print("  - rq3_summary.json")


if __name__ == "__main__":
    main()
