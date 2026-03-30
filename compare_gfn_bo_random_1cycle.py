#!/usr/bin/env python3
"""
Compare a trained GFN, a BO run from wandb, and random search on the 1-cycle enumerable space.

Important interpretation:
- BO and random are true evaluation-budget baselines.
- If your GFN was trained on a fully enumerated reward table, the fairest comparison is
  "given B candidate draws, which method retrieves better states?" not end-to-end simulator cost.
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path

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

PERT_NAMES = sorted({p for group in PERTURBATION_SCHEME.values() for p in group.keys()})
PERT2ID = {p: i for i, p in enumerate(PERT_NAMES)}
N_GROUPS = len(GROUP_ORDER)
N_PARAMS = len(BASELINE_PARAMETERS)


def apply_perturbation(step_fraction, group_name, perturb_name, values=None):
    params = {} if values is None else dict(values)
    for p, direction in PERTURBATION_SCHEME[group_name][perturb_name].items():
        lo, hi = PARAMETER_BOUNDS[p]
        val = params.get(p, INITIAL_CONDITIONS.get(p, BASELINE_PARAMETERS[p]))
        params[p] = float(np.clip(val + direction * step_fraction * (hi - lo), lo, hi))
    return params


def build_config_from_action_seq(action_seq, step_fraction, decay_factor=0.5, normalize=False):
    config = {}
    for op_idx, action in enumerate(action_seq):
        cycle = op_idx // len(GROUP_ORDER) + 1
        group_name = GROUP_ORDER[op_idx % len(GROUP_ORDER)]
        sf = step_fraction * (decay_factor ** (cycle - 1))
        config = apply_perturbation(sf, group_name, action, config)

    parameters = [0.0] * len(BASELINE_PARAMETERS)
    for i, k in enumerate(sorted(BASELINE_PARAMETERS)):
        parameters[i] = float(config.get(k, INITIAL_CONDITIONS.get(k, BASELINE_PARAMETERS[k])))
    if normalize:
        for i, k in enumerate(sorted(BASELINE_PARAMETERS)):
            lo, hi = PARAMETER_BOUNDS[k]
            parameters[i] = 0.5 if lo == hi else (parameters[i] - lo) / (hi - lo)
    return parameters


def build_policy_input(step_num, step_fraction, chosen_pert_ids, current_config_norm):
    n_operations = len(GROUP_ORDER)
    vec = [-1.0] * (2 + n_operations + N_PARAMS)
    vec[0] = float(step_num)
    vec[1] = float(step_fraction)
    for i, pid in enumerate(chosen_pert_ids, start=1):
        vec[i] = float(pid)
    vec[2 + n_operations:] = current_config_norm
    return vec


def get_valid_action_ids(group_idx):
    group_name = GROUP_ORDER[group_idx]
    return [PERT2ID[p] for p in PERTURBATION_SCHEME[group_name].keys()]


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


def exact_gfn_distribution(model, step_fraction, device="cpu"):
    from itertools import product

    model = model.to(device)
    probs = {}
    grids = [list(PERTURBATION_SCHEME[g].keys()) for g in GROUP_ORDER]

    for combo in product(*grids):
        chosen_ids = []
        log_p = 0.0
        for op_idx, action_name in enumerate(combo):
            group_idx = op_idx % len(GROUP_ORDER)
            config_norm = build_config_from_action_seq(combo[:op_idx], step_fraction, normalize=True) if op_idx > 0 else build_config_from_action_seq([], step_fraction, normalize=True)
            x = torch.tensor(
                [build_policy_input(op_idx + 1, step_fraction, chosen_ids, config_norm)],
                dtype=torch.float32,
                device=device,
            )
            logits = model(x)[0]
            valid_ids = get_valid_action_ids(group_idx)
            mask = torch.tensor([pid in valid_ids for pid in range(logits.shape[0])], dtype=torch.bool, device=device)
            masked = logits.clone()
            masked[~mask] = float("-inf")
            log_probs = torch.log_softmax(masked, dim=0)
            log_p += float(log_probs[PERT2ID[action_name]].item())
            chosen_ids.append(PERT2ID[action_name])
        probs["|".join(combo)] = math.exp(log_p)

    z = sum(probs.values())
    return {k: v / z for k, v in probs.items()}


def load_reward_table(path):
    with open(path) as f:
        raw = json.load(f)
    keys = list(raw.keys())
    losses = {k: float(raw[k]["loss"]) for k in keys}
    rewards = {k: float(raw[k].get("reward", 0.0)) for k in keys}
    return keys, losses, rewards


def fetch_bo_trial_sequence(project, run_id):
    api = wandb.Api()
    run = api.run(f"{project}/{run_id}")
    rows = []
    for row in run.scan_history():
        if "trial_step" not in row or "state_key" not in row:
            continue
        rows.append({
            "trial_step": int(row["trial_step"]),
            "state_key": str(row["state_key"]),
        })
    if not rows:
        raise RuntimeError("No BO trial history with trial_step/state_key found in wandb run")
    df = pd.DataFrame(rows).sort_values("trial_step").drop_duplicates("trial_step")
    return list(df["state_key"])


def download_gfn_checkpoint(project, run_id, alias="final"):
    api = wandb.Api()
    artifact = api.artifact(f"{project}/ckpt-{run_id}:{alias}")
    root = Path(tempfile.mkdtemp(prefix=f"wandb_ckpt_{run_id}_"))
    artifact_dir = Path(artifact.download(root=str(root)))
    ckpt = artifact_dir / "final.ckpt"
    if not ckpt.exists():
        ckpts = list(artifact_dir.rglob("*.ckpt"))
        if not ckpts:
            raise FileNotFoundError("No .ckpt file found in downloaded artifact")
        ckpt = ckpts[0]
    return ckpt


def summarize_sequence(seq, losses, topk_sets):
    best_curve = []
    best = np.inf
    seen = set()
    topk_unique = {k: 0 for k in topk_sets}
    first_hit = {k: None for k in topk_sets}

    for i, key in enumerate(seq, start=1):
        if key not in losses:
            continue
        seen.add(key)
        best = min(best, losses[key])
        best_curve.append(best)
        for k, topset in topk_sets.items():
            if key in topset:
                if first_hit[k] is None:
                    first_hit[k] = i
                topk_unique[k] = len(seen & topset)

    return {
        "best_curve": best_curve,
        "best_final_loss": float(best) if best_curve else np.nan,
        "unique_states": len(seen),
        "topk_unique": topk_unique,
        "first_hit": first_hit,
    }


def monte_carlo_from_distribution(dist, budget, n_repeats, losses, topk_sets, rng):
    keys = np.array(list(dist.keys()))
    probs = np.array([dist[k] for k in keys], dtype=float)
    probs /= probs.sum()

    curves = []
    finals = []
    uniqs = []
    topk = {k: [] for k in topk_sets}
    firsts = {k: [] for k in topk_sets}
    for _ in range(n_repeats):
        seq = list(rng.choice(keys, size=budget, replace=True, p=probs))
        s = summarize_sequence(seq, losses, topk_sets)
        curves.append(s["best_curve"])
        finals.append(s["best_final_loss"])
        uniqs.append(s["unique_states"])
        for k in topk_sets:
            topk[k].append(s["topk_unique"][k])
            firsts[k].append(s["first_hit"][k] if s["first_hit"][k] is not None else np.nan)

    curve_arr = np.array(curves, dtype=float)
    return {
        "best_curve_mean": curve_arr.mean(axis=0).tolist(),
        "best_curve_std": curve_arr.std(axis=0).tolist(),
        "best_final_loss_mean": float(np.mean(finals)),
        "best_final_loss_std": float(np.std(finals)),
        "unique_states_mean": float(np.mean(uniqs)),
        "topk_unique_mean": {k: float(np.mean(v)) for k, v in topk.items()},
        "first_hit_mean": {k: float(np.nanmean(v)) for k, v in firsts.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reward_table", required=True)
    ap.add_argument("--gfn_project", required=True)
    ap.add_argument("--gfn_run_id", required=True)
    ap.add_argument("--bo_project", required=True)
    ap.add_argument("--bo_run_id", required=True)
    ap.add_argument("--step_fraction", type=float, required=True)
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--artifact_alias", default="final")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n_repeats", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output_dir", default="compare_gfn_bo_random")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    keys, losses, rewards = load_reward_table(args.reward_table)
    sorted_by_loss = sorted(keys, key=lambda k: losses[k])
    topk_sets = {1: set(sorted_by_loss[:1]), 5: set(sorted_by_loss[:5]), 10: set(sorted_by_loss[:10]), 20: set(sorted_by_loss[:20])}

    ckpt_path = download_gfn_checkpoint(args.gfn_project, args.gfn_run_id, alias=args.artifact_alias)
    ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    forward_sd = extract_forward_state_dict(ckpt)
    model = build_mlp_from_state_dict(forward_sd)
    p_gfn = exact_gfn_distribution(model, step_fraction=args.step_fraction, device=args.device)

    bo_seq = fetch_bo_trial_sequence(args.bo_project, args.bo_run_id)[: args.budget]
    bo_summary = summarize_sequence(bo_seq, losses, topk_sets)

    rng = np.random.default_rng(args.seed)
    random_keys = np.array(keys)
    random_curves = []
    random_finals = []
    random_uniqs = []
    random_topk = {k: [] for k in topk_sets}
    random_firsts = {k: [] for k in topk_sets}
    for _ in range(args.n_repeats):
        seq = list(rng.choice(random_keys, size=args.budget, replace=False))
        s = summarize_sequence(seq, losses, topk_sets)
        random_curves.append(s["best_curve"])
        random_finals.append(s["best_final_loss"])
        random_uniqs.append(s["unique_states"])
        for k in topk_sets:
            random_topk[k].append(s["topk_unique"][k])
            random_firsts[k].append(s["first_hit"][k] if s["first_hit"][k] is not None else np.nan)

    random_curve_arr = np.array(random_curves, dtype=float)
    random_summary = {
        "best_curve_mean": random_curve_arr.mean(axis=0).tolist(),
        "best_curve_std": random_curve_arr.std(axis=0).tolist(),
        "best_final_loss_mean": float(np.mean(random_finals)),
        "best_final_loss_std": float(np.std(random_finals)),
        "unique_states_mean": float(np.mean(random_uniqs)),
        "topk_unique_mean": {k: float(np.mean(v)) for k, v in random_topk.items()},
        "first_hit_mean": {k: float(np.nanmean(v)) for k, v in random_firsts.items()},
    }

    gfn_summary = monte_carlo_from_distribution(p_gfn, args.budget, args.n_repeats, losses, topk_sets, rng)

    plt.figure(figsize=(6, 4))
    x = np.arange(1, args.budget + 1)
    plt.plot(x, bo_summary["best_curve"], label="BO")
    plt.plot(x, gfn_summary["best_curve_mean"], label="GFN")
    plt.fill_between(
        x,
        np.array(gfn_summary["best_curve_mean"]) - np.array(gfn_summary["best_curve_std"]),
        np.array(gfn_summary["best_curve_mean"]) + np.array(gfn_summary["best_curve_std"]),
        alpha=0.2,
    )
    plt.plot(x, random_summary["best_curve_mean"], label="Random")
    plt.fill_between(
        x,
        np.array(random_summary["best_curve_mean"]) - np.array(random_summary["best_curve_std"]),
        np.array(random_summary["best_curve_mean"]) + np.array(random_summary["best_curve_std"]),
        alpha=0.2,
    )
    plt.xlabel("Evaluation / draw budget")
    plt.ylabel("Best loss so far")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "best_loss_vs_budget.png", dpi=200)
    plt.close()

    summary = {
        "budget": args.budget,
        "bo": {
            "best_final_loss": bo_summary["best_final_loss"],
            "unique_states": bo_summary["unique_states"],
            "topk_unique": bo_summary["topk_unique"],
            "first_hit": bo_summary["first_hit"],
        },
        "gfn": gfn_summary,
        "random": random_summary,
    }
    with open(out_dir / "comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"[saved] {out_dir / 'comparison_summary.json'}")
    print(f"[saved] {out_dir / 'best_loss_vs_budget.png'}")


if __name__ == "__main__":
    main()
