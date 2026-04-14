#!/usr/bin/env python3
from __future__ import annotations

"""
Paper-oriented figure generator for the enumerable 1-cycle setting.

Focus:
- RQ1: mode discovery against the exact reward-defined target distribution
- RQ2: retrieval performance and distributional fidelity

This script combines ideas from:
- simple_wandb_landscape.py
- compare_gfn_bo_random_1cycle.py

Key design choices:
- uses the exact stored reward table for the enumerable setting
- reconstructs the exact GFN-induced terminal-state distribution from the checkpoint
- fetches the TPE/Optuna baseline sequence directly from wandb history
- compares GFN, TPE, and random search under the same 1-cycle retrieval budget
- exports figures and CSV/JSON summaries suitable for the paper
"""

import argparse
import csv
import json
import math
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb

from gflownet.envs.greenhouse.constants import GROUP_ORDER, PERTURBATION_SCHEME
from gflownet.envs.greenhouse.sim_env import CropSimEnv


DISPLAY_ALIASES = {
    "increase": "↑",
    "decrease": "↓",
    "higher_sensitivity": "sens+",
    "lower_sensitivity": "sens−",
    "none": "none",
    "shift_warm": "warm",
    "shift_cold": "cold",
    "widen_optimum": "widen",
    "narrow_optimum": "narrow",
    "more_fruit_growth": "fruit+",
    "more_veg_growth": "veg+",
    "lower_resp_cost": "resp−",
    "higher_resp_cost": "resp+",
}

GROUP_ORDERS = [
    ["none", "increase", "decrease"],
    ["none", "increase", "decrease", "higher_sensitivity", "lower_sensitivity"],
    ["none", "shift_warm", "shift_cold", "widen_optimum", "narrow_optimum"],
    ["none", "increase", "decrease", "higher_sensitivity", "lower_sensitivity"],
    ["none", "more_fruit_growth", "more_veg_growth", "lower_resp_cost", "higher_resp_cost", "higher_sensitivity", "lower_sensitivity"],
]


def normalize_token(tok: str) -> str:
    return tok.strip()


def load_reward_table(path: str):
    with open(path) as f:
        raw = json.load(f)
    keys = list(raw.keys())
    losses = {k: float(raw[k]["loss"]) for k in keys}
    rewards = {k: float(raw[k].get("reward", 0.0)) for k in keys}
    return keys, losses, rewards


def normalize_prob_dict(d: Dict[str, float]) -> Dict[str, float]:
    z = float(sum(d.values()))
    if z <= 0:
        raise ValueError("Cannot normalize distribution with non-positive total mass.")
    return {k: float(v / z) for k, v in d.items()}


def stable_softmin(losses: Dict[str, float], beta: float, norm: str = "q10q90", q_low: float = 0.1, q_high: float = 0.9):
    keys = list(losses.keys())
    arr = np.array([losses[k] for k in keys], dtype=float)
    if norm == "none":
        L = arr.copy()
    elif norm == "q10q90":
        lo = float(np.quantile(arr, q_low))
        hi = float(np.quantile(arr, q_high))
        scale = max(hi - lo, 1e-12)
        L = (arr - lo) / scale
    else:
        raise ValueError(norm)
    logw = -beta * L
    logw -= np.max(logw)
    w = np.exp(logw)
    w /= w.sum()
    return {k: float(v) for k, v in zip(keys, w)}


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
                return [str(x) for x in df["state_key"].tolist()]
    except Exception:
        pass
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
        raise RuntimeError("No TPE trial history with trial_step/state_key found in wandb run")
    df = pd.DataFrame(rows).sort_values("trial_step").drop_duplicates("trial_step")
    return list(df["state_key"])


def compute_exact_gfn_distribution(model, env: CropSimEnv, state_keys: List[str], device: str = "cpu") -> Dict[str, float]:
    model = model.to(device)
    action_space = env.get_action_space()
    probs = {}

    with torch.no_grad():
        for state_key in state_keys:
            toks = [normalize_token(t) for t in state_key.split("|")]
            prefix = [()]
            logp = 0.0

            for tok in toks:
                x = env.states2policy([prefix]).to(device)
                logits = model(x)[0]

                mask_invalid = env.get_mask_invalid_actions_forward(state=prefix, done=False)
                valid_mask = torch.tensor([not m for m in mask_invalid], dtype=torch.bool, device=device)

                masked_logits = logits.clone()
                masked_logits[~valid_mask] = float("-inf")
                log_softmax = torch.log_softmax(masked_logits, dim=0)

                action_value = env.pert2id[tok]
                action_idx = env.action2idx[action_value] if hasattr(env, "action2idx") else action_space.index(action_value)
                logp += float(log_softmax[action_idx].item())

                prefix = prefix + [(
                    1 + (len(prefix) - 1) // env.n_groups,
                    (len(prefix) - 1) % env.n_groups,
                    action_value,
                )]

            try:
                mask_invalid = env.get_mask_invalid_actions_forward(state=prefix, done=False)
                valid_actions = [a for a, m in zip(action_space, mask_invalid) if not m]
                if len(valid_actions) == 1 and hasattr(env, "eos") and valid_actions[0] == env.eos:
                    x = env.states2policy([prefix]).to(device)
                    logits = model(x)[0]
                    valid_mask = torch.tensor([not m for m in mask_invalid], dtype=torch.bool, device=device)
                    masked_logits = logits.clone()
                    masked_logits[~valid_mask] = float("-inf")
                    log_softmax = torch.log_softmax(masked_logits, dim=0)
                    eos_idx = env.action2idx[env.eos]
                    logp += float(log_softmax[eos_idx].item())
            except Exception:
                pass

            probs[state_key] = math.exp(logp)

    return normalize_prob_dict(probs)


def aggregate_mass(state_keys: List[str], prob_dict: Dict[str, float]):
    shape = (len(GROUP_ORDERS[0]), len(GROUP_ORDERS[1]), len(GROUP_ORDERS[3]), len(GROUP_ORDERS[2]))
    arr = np.zeros(shape, float)
    idx_maps = [{tok: i for i, tok in enumerate(order)} for order in GROUP_ORDERS]
    for k in state_keys:
        s = tuple(normalize_token(t) for t in k.split("|"))
        prob = prob_dict.get(k, 0.0)
        i = idx_maps[0][s[0]]
        j = idx_maps[1][s[1]]
        x = idx_maps[2][s[2]]
        y = idx_maps[3][s[3]]
        arr[i, j, y, x] += prob
    return arr


def plot_faceted_mass(arr, title, out_png, vmin=None, vmax=None, cmap="viridis", cbar_label="Probability mass"):
    g1_order, g2_order, g3_order, g4_order, _ = GROUP_ORDERS
    if vmin is None:
        vmin = float(np.min(arr))
    if vmax is None:
        vmax = float(np.max(arr))
    fig, axes = plt.subplots(len(g1_order), len(g2_order), figsize=(18, 11), squeeze=False)
    im = None
    for i, g1 in enumerate(g1_order):
        for j, g2 in enumerate(g2_order):
            ax = axes[i][j]
            im = ax.imshow(arr[i, j], aspect="auto", origin="lower", vmin=vmin, vmax=vmax, cmap=cmap)
            ax.set_title(f"Photosynthesis = {DISPLAY_ALIASES[g2]}")
            if i == len(g1_order) - 1:
                ax.set_xticks(range(len(g3_order)))
                ax.set_xticklabels([DISPLAY_ALIASES[t] for t in g3_order], rotation=45, ha="right")
                ax.set_xlabel("Temp. Inhibition")
            else:
                ax.set_xticks(range(len(g3_order)))
                ax.set_xticklabels([])
            if j == 0:
                ax.set_yticks(range(len(g4_order)))
                ax.set_yticklabels([DISPLAY_ALIASES[t] for t in g4_order])
                ax.set_ylabel(f"Canopy = {DISPLAY_ALIASES[g1]}\nTemp. & Dev.")
            else:
                ax.set_yticks(range(len(g4_order)))
                ax.set_yticklabels([])
    fig.suptitle(
        title + "\nrows = Canopy, cols = Photosynthesis, x = Temp. Inhibition, y = Temp. & Dev., Biomass summed",
        y=0.98,
    )
    fig.subplots_adjust(left=0.06, right=0.90, bottom=0.08, top=0.90, wspace=0.25, hspace=0.35)
    cax = fig.add_axes([0.92, 0.15, 0.015, 0.68])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(cbar_label)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def topk_curve_from_prob(prob_dict: Dict[str, float]):
    vals = np.array(sorted(prob_dict.values(), reverse=True), dtype=float)
    return np.arange(1, len(vals) + 1), vals, np.cumsum(vals)


def build_neighbor_graph(state_keys: List[str]) -> Dict[str, List[str]]:
    by_tokens = {k: k.split("|") for k in state_keys}
    action_lists = {g: list(PERTURBATION_SCHEME[g].keys()) for g in GROUP_ORDER}
    key_set = set(state_keys)
    neighbors = {}
    for k, toks in by_tokens.items():
        nbrs = []
        for i, group_name in enumerate(GROUP_ORDER):
            cur = toks[i]
            for alt in action_lists[group_name]:
                if alt == cur:
                    continue
                cand = toks.copy()
                cand[i] = alt
                cand_key = "|".join(cand)
                if cand_key in key_set:
                    nbrs.append(cand_key)
        neighbors[k] = nbrs
    return neighbors


def compute_modes_and_basins(target_prob: Dict[str, float], neighbors: Dict[str, List[str]]):
    keys = list(target_prob.keys())

    def better(a: str, b: str) -> bool:
        pa, pb = target_prob[a], target_prob[b]
        if pa > pb + 1e-15:
            return True
        if abs(pa - pb) <= 1e-15 and a < b:
            return True
        return False

    sink_of = {}
    for start in keys:
        cur = start
        seen = set()
        while True:
            if cur in sink_of:
                sink = sink_of[cur]
                break
            seen.add(cur)
            better_neighbors = [n for n in neighbors[cur] if better(n, cur)]
            if not better_neighbors:
                sink = cur
                break
            better_neighbors.sort(key=lambda x: (-target_prob[x], x))
            nxt = better_neighbors[0]
            if nxt in seen:
                sink = nxt
                break
            cur = nxt
        for s in seen:
            sink_of[s] = sink

    basins = defaultdict(list)
    for k, s in sink_of.items():
        basins[s].append(k)

    mode_rows = []
    for mode_key, members in basins.items():
        mode_rows.append({
            "mode_key": mode_key,
            "mode_peak_prob": float(target_prob[mode_key]),
            "mode_basin_mass": float(sum(target_prob[m] for m in members)),
            "basin_size": int(len(members)),
            "members": members,
        })
    mode_rows.sort(key=lambda r: (-r["mode_basin_mass"], -r["mode_peak_prob"], r["mode_key"]))
    basin_of = {m: mode_key for mode_key, members in basins.items() for m in members}
    return mode_rows, basin_of


def add_loss_ranks_to_modes(mode_rows, losses: Dict[str, float]):
    sorted_by_loss = sorted(losses, key=lambda k: losses[k])
    rank = {k: i + 1 for i, k in enumerate(sorted_by_loss)}
    for r in mode_rows:
        r["mode_loss"] = float(losses[r["mode_key"]])
        r["mode_center_rank"] = int(rank[r["mode_key"]])
        r["best_member_loss"] = float(min(losses[m] for m in r["members"]))
    return mode_rows


def save_csv(path, rows):
    if not rows:
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def plot_target_vs_gfn_mode_mass_figure(out_path: Path, mode_rows: List[dict], p_gfn: Dict[str, float], top_n_modes: int):
    rows = mode_rows[:top_n_modes]
    labels = [f"M{i+1}" for i in range(len(rows))]
    target_vals = [r["mode_basin_mass"] for r in rows]
    gfn_vals = [float(sum(p_gfn.get(m, 0.0) for m in r["members"])) for r in rows]
    x = np.arange(len(rows))
    w = 0.36
    plt.figure(figsize=(7.0, 4.2))
    plt.bar(x - w / 2, target_vals, width=w, label="Target basin mass")
    plt.bar(x + w / 2, gfn_vals, width=w, label="GFN basin mass")
    plt.xticks(x, labels)
    plt.xlabel("Top target basins")
    plt.ylabel("Probability mass")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def summarize_sequence(seq: List[str], losses: Dict[str, float], rewards: Dict[str, float], topk_sets: Dict[int, set], top_mode_keys=None, basin_of=None):
    best_loss_curve = []
    best_reward_curve = []
    best_loss = np.inf
    best_reward = -np.inf
    seen = set()
    topk_unique = {k: 0 for k in topk_sets}
    first_hit = {k: None for k in topk_sets}
    mode_hit_curve = []
    seen_modes = set()

    for i, key in enumerate(seq, start=1):
        if key not in losses:
            continue
        seen.add(key)
        best_loss = min(best_loss, losses[key])
        best_reward = max(best_reward, rewards[key])
        best_loss_curve.append(best_loss)
        best_reward_curve.append(best_reward)
        for k, topset in topk_sets.items():
            if key in topset:
                if first_hit[k] is None:
                    first_hit[k] = i
                topk_unique[k] = len(seen & topset)
        if top_mode_keys is not None and basin_of is not None:
            mode = basin_of.get(key)
            if mode in top_mode_keys:
                seen_modes.add(mode)
            mode_hit_curve.append(len(seen_modes))

    return {
        "best_loss_curve": best_loss_curve,
        "best_reward_curve": best_reward_curve,
        "best_final_loss": float(best_loss) if best_loss_curve else np.nan,
        "best_final_reward": float(best_reward) if best_reward_curve else np.nan,
        "unique_states": len(seen),
        "topk_unique": topk_unique,
        "first_hit": first_hit,
        "mode_hit_curve": mode_hit_curve,
        "mode_hits_final": len(seen_modes) if top_mode_keys is not None else None,
    }


def _safe_nanmean(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return None
    return float(np.nanmean(arr))


def monte_carlo_from_distribution(dist, budget, n_repeats, losses, rewards, topk_sets, rng, top_mode_keys=None, basin_of=None):
    keys = np.array(list(dist.keys()))
    probs = np.array([dist[k] for k in keys], dtype=float)
    probs /= probs.sum()

    loss_curves = []
    reward_curves = []
    mode_curves = []
    final_losses = []
    final_rewards = []
    uniqs = []
    topk = {k: [] for k in topk_sets}
    firsts = {k: [] for k in topk_sets}
    hit_rates = {k: [] for k in topk_sets}
    mode_hits_final = []

    for _ in range(n_repeats):
        seq = list(rng.choice(keys, size=budget, replace=True, p=probs))
        s = summarize_sequence(seq, losses, rewards, topk_sets, top_mode_keys=top_mode_keys, basin_of=basin_of)
        loss_curves.append(s["best_loss_curve"])
        reward_curves.append(s["best_reward_curve"])
        final_losses.append(s["best_final_loss"])
        final_rewards.append(s["best_final_reward"])
        uniqs.append(s["unique_states"])
        if top_mode_keys is not None:
            mode_curves.append(s["mode_hit_curve"])
            mode_hits_final.append(s["mode_hits_final"])
        for k in topk_sets:
            topk[k].append(s["topk_unique"][k])
            firsts[k].append(s["first_hit"][k] if s["first_hit"][k] is not None else np.nan)
            hit_rates[k].append(0 if s["first_hit"][k] is None else 1)

    out = {
        "best_loss_curve_mean": np.array(loss_curves, dtype=float).mean(axis=0).tolist(),
        "best_loss_curve_std": np.array(loss_curves, dtype=float).std(axis=0).tolist(),
        "best_reward_curve_mean": np.array(reward_curves, dtype=float).mean(axis=0).tolist(),
        "best_reward_curve_std": np.array(reward_curves, dtype=float).std(axis=0).tolist(),
        "best_final_loss_mean": float(np.mean(final_losses)),
        "best_final_loss_std": float(np.std(final_losses)),
        "best_final_reward_mean": float(np.mean(final_rewards)),
        "best_final_reward_std": float(np.std(final_rewards)),
        "unique_states_mean": float(np.mean(uniqs)),
        "topk_unique_mean": {k: float(np.mean(v)) for k, v in topk.items()},
        "topk_unique_std": {k: float(np.std(v)) for k, v in topk.items()},
        "first_hit_mean": {k: _safe_nanmean(v) for k, v in firsts.items()},
        "hit_rate": {k: float(np.mean(v)) for k, v in hit_rates.items()},
    }
    if top_mode_keys is not None and mode_curves:
        mode_curve_arr = np.array(mode_curves, dtype=float)
        out["mode_hit_curve_mean"] = mode_curve_arr.mean(axis=0).tolist()
        out["mode_hit_curve_std"] = mode_curve_arr.std(axis=0).tolist()
        out["mode_hits_final_mean"] = float(np.mean(mode_hits_final))
        out["mode_hits_final_std"] = float(np.std(mode_hits_final))
    return out


def pad_curve(curve: List[float], target_len: int):
    if len(curve) == 0:
        return [np.nan] * target_len
    if len(curve) >= target_len:
        return curve[:target_len]
    return curve + [curve[-1]] * (target_len - len(curve))


def plot_rq2_retrieval_curves(out_path: Path, budget: int, tpe_summary, gfn_summary, random_summary):
    x = np.arange(1, budget + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    axes[0].plot(x, np.array(pad_curve(tpe_summary["best_loss_curve"], budget)), label="TPE", linewidth=2)
    axes[0].plot(x, np.array(gfn_summary["best_loss_curve_mean"]), label="GFN", linewidth=2)
    axes[0].fill_between(
        x,
        np.array(gfn_summary["best_loss_curve_mean"]) - np.array(gfn_summary["best_loss_curve_std"]),
        np.array(gfn_summary["best_loss_curve_mean"]) + np.array(gfn_summary["best_loss_curve_std"]),
        alpha=0.2,
    )
    axes[0].plot(x, np.array(random_summary["best_loss_curve_mean"]), label="Random", linewidth=2)
    axes[0].fill_between(
        x,
        np.array(random_summary["best_loss_curve_mean"]) - np.array(random_summary["best_loss_curve_std"]),
        np.array(random_summary["best_loss_curve_mean"]) + np.array(random_summary["best_loss_curve_std"]),
        alpha=0.2,
    )
    axes[0].set_xlabel("Retrieval budget")
    axes[0].set_ylabel("Best-so-far loss")
    axes[0].set_title("Retrieval performance (loss)")
    axes[0].grid(alpha=0.3)
    axes[0].legend(frameon=False)

    axes[1].plot(x, np.array(pad_curve(tpe_summary["best_reward_curve"], budget)), label="TPE", linewidth=2)
    axes[1].plot(x, np.array(gfn_summary["best_reward_curve_mean"]), label="GFN", linewidth=2)
    axes[1].fill_between(
        x,
        np.array(gfn_summary["best_reward_curve_mean"]) - np.array(gfn_summary["best_reward_curve_std"]),
        np.array(gfn_summary["best_reward_curve_mean"]) + np.array(gfn_summary["best_reward_curve_std"]),
        alpha=0.2,
    )
    axes[1].plot(x, np.array(random_summary["best_reward_curve_mean"]), label="Random", linewidth=2)
    axes[1].fill_between(
        x,
        np.array(random_summary["best_reward_curve_mean"]) - np.array(random_summary["best_reward_curve_std"]),
        np.array(random_summary["best_reward_curve_mean"]) + np.array(random_summary["best_reward_curve_std"]),
        alpha=0.2,
    )
    axes[1].set_xlabel("Retrieval budget")
    axes[1].set_ylabel("Best-so-far reward")
    axes[1].set_title("Retrieval performance (reward)")
    axes[1].grid(alpha=0.3)
    axes[1].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_rq2_topk_recovery(out_path: Path, tpe_summary, gfn_summary, random_summary, ks=(5, 10, 20)):
    ks = list(ks)
    x = np.arange(len(ks))
    w = 0.26
    tpe_vals = [tpe_summary["topk_unique"].get(k, 0) for k in ks]
    gfn_vals = [gfn_summary["topk_unique_mean"].get(k, 0.0) for k in ks]
    gfn_errs = [gfn_summary["topk_unique_std"].get(k, 0.0) for k in ks]
    rnd_vals = [random_summary["topk_unique_mean"].get(k, 0.0) for k in ks]
    rnd_errs = [random_summary["topk_unique_std"].get(k, 0.0) for k in ks]

    plt.figure(figsize=(6.0, 4.2))
    plt.bar(x - w, tpe_vals, width=w, label="TPE")
    plt.bar(x, gfn_vals, width=w, yerr=gfn_errs, capsize=3, label="GFN")
    plt.bar(x + w, rnd_vals, width=w, yerr=rnd_errs, capsize=3, label="Random")
    plt.xticks(x, [f"Top-{k}" for k in ks])
    plt.ylabel("Unique top-k states found")
    plt.xlabel("Reference set")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_rq2_distributional_fidelity(out_path: Path, p_target: Dict[str, float], p_gfn: Dict[str, float], topk_max: int = 100):
    k_t, sp_t, cum_t = topk_curve_from_prob(p_target)
    _, sp_g, cum_g = topk_curve_from_prob(p_gfn)
    kmax = min(topk_max, len(k_t))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].plot(k_t[:kmax], cum_t[:kmax], lw=2, label="Target")
    axes[0].plot(k_t[:kmax], cum_g[:kmax], lw=2, label="GFN")
    axes[0].set_xlabel("Top-k states")
    axes[0].set_ylabel("Cumulative mass")
    axes[0].set_title("Cumulative top-k state mass")
    axes[0].grid(alpha=0.3)
    axes[0].legend(frameon=False)

    axes[1].plot(k_t[:kmax], sp_t[:kmax], lw=2, label="Target")
    axes[1].plot(k_t[:kmax], sp_g[:kmax], lw=2, label="GFN")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Rank")
    axes[1].set_ylabel("Probability")
    axes[1].set_title("Probability-rank comparison")
    axes[1].grid(alpha=0.3)
    axes[1].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def distributional_metrics(p_target: Dict[str, float], p_gfn: Dict[str, float], losses: Dict[str, float]):
    keys = list(p_target.keys())
    t = np.array([p_target[k] for k in keys], dtype=float)
    g = np.array([p_gfn[k] for k in keys], dtype=float)

    eps = 1e-12
    t_eps = np.maximum(t, eps)
    g_eps = np.maximum(g, eps)
    m = 0.5 * (t_eps + g_eps)

    js = 0.5 * np.sum(t_eps * np.log(t_eps / m)) + 0.5 * np.sum(g_eps * np.log(g_eps / m))
    l1 = float(np.sum(np.abs(t - g)))
    tv = 0.5 * l1
    cosine = float(np.dot(t, g) / (np.linalg.norm(t) * np.linalg.norm(g)))
    corr = float(np.corrcoef(t, g)[0, 1])

    loss_vec = np.array([losses[k] for k in keys], dtype=float)
    exp_loss_target = float(np.sum(t * loss_vec))
    exp_loss_gfn = float(np.sum(g * loss_vec))

    return {
        "js_divergence": float(js),
        "l1_distance": l1,
        "total_variation": float(tv),
        "cosine_similarity": cosine,
        "pearson_correlation": corr,
        "expected_loss_target": exp_loss_target,
        "expected_loss_gfn": exp_loss_gfn,
    }


def make_summary_table_csv(out_path: Path, tpe_summary, gfn_summary, random_summary, dist_metrics):
    rows = [
        {
            "Method": "TPE",
            "Best final loss": tpe_summary["best_final_loss"],
            "Best final reward": tpe_summary["best_final_reward"],
            "Unique top-10": tpe_summary["topk_unique"].get(10, np.nan),
            "Unique top-20": tpe_summary["topk_unique"].get(20, np.nan),
            "Distinct top-5 basins hit": tpe_summary.get("mode_hits_final", np.nan),
            "JS divergence to target": np.nan,
        },
        {
            "Method": "GFN",
            "Best final loss": gfn_summary["best_final_loss_mean"],
            "Best final reward": gfn_summary["best_final_reward_mean"],
            "Unique top-10": gfn_summary["topk_unique_mean"].get(10, np.nan),
            "Unique top-20": gfn_summary["topk_unique_mean"].get(20, np.nan),
            "Distinct top-5 basins hit": gfn_summary.get("mode_hits_final_mean", np.nan),
            "JS divergence to target": dist_metrics["js_divergence"],
        },
        {
            "Method": "Random",
            "Best final loss": random_summary["best_final_loss_mean"],
            "Best final reward": random_summary["best_final_reward_mean"],
            "Unique top-10": random_summary["topk_unique_mean"].get(10, np.nan),
            "Unique top-20": random_summary["topk_unique_mean"].get(20, np.nan),
            "Distinct top-5 basins hit": random_summary.get("mode_hits_final_mean", np.nan),
            "JS divergence to target": np.nan,
        },
    ]
    pd.DataFrame(rows).to_csv(out_path, index=False)



def aggregate_seed_summaries(seed_summaries: List[dict], budget: int):
    if not seed_summaries:
        raise ValueError("No seed summaries provided.")
    loss_curves = np.array([pad_curve(s["best_loss_curve"], budget) for s in seed_summaries], dtype=float)
    reward_curves = np.array([pad_curve(s["best_reward_curve"], budget) for s in seed_summaries], dtype=float)
    mode_curves = np.array([pad_curve(s["mode_hit_curve"], budget) for s in seed_summaries], dtype=float)
    out = {
        "best_loss_curve_mean": loss_curves.mean(axis=0).tolist(),
        "best_loss_curve_std": loss_curves.std(axis=0).tolist(),
        "best_reward_curve_mean": reward_curves.mean(axis=0).tolist(),
        "best_reward_curve_std": reward_curves.std(axis=0).tolist(),
        "best_final_loss_mean": float(np.mean([s["best_final_loss"] for s in seed_summaries])),
        "best_final_loss_std": float(np.std([s["best_final_loss"] for s in seed_summaries])),
        "best_final_reward_mean": float(np.mean([s["best_final_reward"] for s in seed_summaries])),
        "best_final_reward_std": float(np.std([s["best_final_reward"] for s in seed_summaries])),
        "unique_states_mean": float(np.mean([s["unique_states"] for s in seed_summaries])),
        "unique_states_std": float(np.std([s["unique_states"] for s in seed_summaries])),
        "topk_unique_mean": {},
        "topk_unique_std": {},
        "first_hit_mean": {},
        "hit_rate": {},
        "mode_hit_curve_mean": mode_curves.mean(axis=0).tolist(),
        "mode_hit_curve_std": mode_curves.std(axis=0).tolist(),
        "mode_hits_final_mean": float(np.mean([s["mode_hits_final"] for s in seed_summaries])),
        "mode_hits_final_std": float(np.std([s["mode_hits_final"] for s in seed_summaries])),
    }
    ks = sorted(seed_summaries[0]["topk_unique"].keys())
    for k in ks:
        vals = [s["topk_unique"][k] for s in seed_summaries]
        out["topk_unique_mean"][k] = float(np.mean(vals))
        out["topk_unique_std"][k] = float(np.std(vals))
        firsts = [s["first_hit"][k] if s["first_hit"][k] is not None else np.nan for s in seed_summaries]
        out["first_hit_mean"][k] = _safe_nanmean(firsts)
        out["hit_rate"][k] = float(np.mean([0 if s["first_hit"][k] is None else 1 for s in seed_summaries]))
    return out


def make_summary_table_csv_seeded(out_path: Path, tpe_summary, gfn_summary, random_summary, dist_metrics_seed_rows):
    dist_df = pd.DataFrame(dist_metrics_seed_rows)
    gfn_js_mean = float(dist_df["js_divergence"].mean())
    gfn_js_std = float(dist_df["js_divergence"].std(ddof=0))
    rows = [
        {
            "Method": "TPE",
            "Best final loss mean": tpe_summary["best_final_loss_mean"],
            "Best final loss std": tpe_summary["best_final_loss_std"],
            "Best final reward mean": tpe_summary["best_final_reward_mean"],
            "Best final reward std": tpe_summary["best_final_reward_std"],
            "Unique top-10 mean": tpe_summary["topk_unique_mean"].get(10, np.nan),
            "Unique top-10 std": tpe_summary["topk_unique_std"].get(10, np.nan),
            "Distinct top-5 basins hit mean": tpe_summary.get("mode_hits_final_mean", np.nan),
            "Distinct top-5 basins hit std": tpe_summary.get("mode_hits_final_std", np.nan),
            "JS divergence to target mean": np.nan,
            "JS divergence to target std": np.nan,
        },
        {
            "Method": "GFN",
            "Best final loss mean": gfn_summary["best_final_loss_mean"],
            "Best final loss std": gfn_summary["best_final_loss_std"],
            "Best final reward mean": gfn_summary["best_final_reward_mean"],
            "Best final reward std": gfn_summary["best_final_reward_std"],
            "Unique top-10 mean": gfn_summary["topk_unique_mean"].get(10, np.nan),
            "Unique top-10 std": gfn_summary["topk_unique_std"].get(10, np.nan),
            "Distinct top-5 basins hit mean": gfn_summary.get("mode_hits_final_mean", np.nan),
            "Distinct top-5 basins hit std": gfn_summary.get("mode_hits_final_std", np.nan),
            "JS divergence to target mean": gfn_js_mean,
            "JS divergence to target std": gfn_js_std,
        },
        {
            "Method": "Random",
            "Best final loss mean": random_summary["best_final_loss_mean"],
            "Best final loss std": random_summary["best_final_loss_std"],
            "Best final reward mean": random_summary["best_final_reward_mean"],
            "Best final reward std": random_summary["best_final_reward_std"],
            "Unique top-10 mean": random_summary["topk_unique_mean"].get(10, np.nan),
            "Unique top-10 std": random_summary["topk_unique_std"].get(10, np.nan),
            "Distinct top-5 basins hit mean": random_summary.get("mode_hits_final_mean", np.nan),
            "Distinct top-5 basins hit std": random_summary.get("mode_hits_final_std", np.nan),
            "JS divergence to target mean": np.nan,
            "JS divergence to target std": np.nan,
        },
    ]
    pd.DataFrame(rows).to_csv(out_path, index=False)


def plot_target_vs_gfn_mode_mass_seeded(out_path: Path, mode_rows: List[dict], gfn_dists: List[Dict[str, float]], top_n_modes: int):
    rows = mode_rows[:top_n_modes]
    labels = [f"M{i+1}" for i in range(len(rows))]
    target_vals = [r["mode_basin_mass"] for r in rows]
    gfn_arr = np.array(
        [[float(sum(dist.get(m, 0.0) for m in r["members"])) for r in rows] for dist in gfn_dists],
        dtype=float,
    )
    gfn_vals = gfn_arr.mean(axis=0)
    gfn_errs = gfn_arr.std(axis=0)
    x = np.arange(len(rows))
    w = 0.36
    plt.figure(figsize=(7.0, 4.2))
    plt.bar(x - w / 2, target_vals, width=w, label="Target basin mass", color=COLORS["TPE"])
    plt.bar(x + w / 2, gfn_vals, width=w, yerr=gfn_errs, capsize=3, label="GFN basin mass (mean ± std)", color=COLORS["GFN"])
    plt.xticks(x, labels)
    plt.xlabel("Top target basins")
    plt.ylabel("Probability mass")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close(fig=None)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reward_table", required=True)
    ap.add_argument("--gfn_project", required=True)
    ap.add_argument("--gfn_run_ids", nargs="+", required=True)
    ap.add_argument("--bo_project", required=True, help="wandb entity/project for the TPE baseline")
    ap.add_argument("--bo_run_ids", nargs="+", required=True, help="wandb run ids for the TPE baseline")
    ap.add_argument("--step_fraction", type=float, required=True)
    ap.add_argument("--budget", type=int, required=True, help="Post-training retrieval budget for the enumerable setting")
    ap.add_argument("--artifact_alias", default="final")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--sample_seed_base", type=int, default=0)
    ap.add_argument("--output_dir", default="paper_rq12_outputs")
    ap.add_argument("--top_n_modes", type=int, default=5)
    ap.add_argument("--topk_max", type=int, default=100)
    ap.add_argument("--allow_softmin_target", action="store_true")
    ap.add_argument("--beta", type=float, default=None, help="Used only with --allow_softmin_target and for output tagging")
    ap.add_argument("--loss_norm", choices=["none", "q10q90"], default="q10q90")
    ap.add_argument("--q_low", type=float, default=0.1)
    ap.add_argument("--q_high", type=float, default=0.9)
    ap.add_argument("--linear_gap_y", action="store_true", help="Use linear instead of log scale on the loss-gap panel")
    ap.add_argument("--target_power_gamma", type=float, default=0.5, help="Power-law color normalization for the stored-reward target heatmap (1.0 keeps linear scale)")
    args = ap.parse_args()

    if len(args.gfn_run_ids) != len(args.bo_run_ids):
        raise ValueError("You must provide the same number of GFN and BO/TPE run ids.")
    n_seeds = len(args.gfn_run_ids)

    beta_tag = "" if args.beta is None else f"_beta{args.beta}"
    out_dir = Path(args.output_dir) / f"sf{args.step_fraction}_budget{args.budget}{beta_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    keys, losses, rewards = load_reward_table(args.reward_table)
    p_target, target_meta = load_target_distribution(
        losses,
        rewards,
        require_stored_reward=not args.allow_softmin_target,
        beta=args.beta,
        norm=args.loss_norm,
        q_low=args.q_low,
        q_high=args.q_high,
    )

    neighbors = build_neighbor_graph(keys)
    target_mode_rows, basin_of = compute_modes_and_basins(p_target, neighbors)
    target_mode_rows = add_loss_ranks_to_modes(target_mode_rows, losses)
    top_mode_keys = [r["mode_key"] for r in target_mode_rows[: args.top_n_modes]]

    p_gfn_seeds = []
    for run_id in args.gfn_run_ids:
        ckpt_path = download_gfn_checkpoint(args.gfn_project, run_id, alias=args.artifact_alias)
        tmp_root = ckpt_path.parent.parent if ckpt_path.parent.name == "files" else ckpt_path.parent
        try:
            ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
            model = build_mlp_from_state_dict(extract_forward_state_dict(ckpt))
            env = CropSimEnv(n_cycles=1, step_fraction=args.step_fraction, precomputed=True, device=args.device)
            p_gfn_seeds.append(compute_exact_gfn_distribution(model, env, keys, device=args.device))
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)

    p_gfn_mean = {k: float(np.mean([dist[k] for dist in p_gfn_seeds])) for k in keys}

    tpe_seqs_full = [fetch_tpe_trial_sequence(args.bo_project, run_id) for run_id in args.bo_run_ids]
    effective_budget = min([args.budget] + [len(seq) for seq in tpe_seqs_full])
    if effective_budget < args.budget:
        print(f"[WARN] Requested budget={args.budget}, but at least one TPE run has fewer steps. Using {effective_budget}.")

    sorted_by_loss = sorted(keys, key=lambda k: losses[k])
    topk_sets = {1: set(sorted_by_loss[:1]), 5: set(sorted_by_loss[:5]), 10: set(sorted_by_loss[:10]), 20: set(sorted_by_loss[:20])}
    global_best_loss = losses[sorted_by_loss[0]]
    state_keys_arr = np.array(keys)

    tpe_seed_summaries = []
    gfn_seed_summaries = []
    random_seed_summaries = []
    dist_metric_rows = []

    for i in range(n_seeds):
        tpe_seq = tpe_seqs_full[i][:effective_budget]
        tpe_seed_summaries.append(
            summarize_sequence(tpe_seq, losses, rewards, topk_sets, top_mode_keys=top_mode_keys, basin_of=basin_of)
        )

        rng = np.random.default_rng(args.sample_seed_base + i)
        gfn_keys = np.array(list(p_gfn_seeds[i].keys()))
        gfn_probs = np.array([p_gfn_seeds[i][k] for k in gfn_keys], dtype=float)
        gfn_probs /= gfn_probs.sum()
        gfn_seq = list(rng.choice(gfn_keys, size=effective_budget, replace=True, p=gfn_probs))
        gfn_seed_summaries.append(
            summarize_sequence(gfn_seq, losses, rewards, topk_sets, top_mode_keys=top_mode_keys, basin_of=basin_of)
        )

        rnd_seq = list(rng.choice(state_keys_arr, size=effective_budget, replace=False))
        random_seed_summaries.append(
            summarize_sequence(rnd_seq, losses, rewards, topk_sets, top_mode_keys=top_mode_keys, basin_of=basin_of)
        )

        row = distributional_metrics(p_target, p_gfn_seeds[i], losses)
        row["seed_index"] = i
        row["gfn_run_id"] = args.gfn_run_ids[i]
        dist_metric_rows.append(row)

    tpe_summary = aggregate_seed_summaries(tpe_seed_summaries, effective_budget)
    gfn_summary = aggregate_seed_summaries(gfn_seed_summaries, effective_budget)
    random_summary = aggregate_seed_summaries(random_seed_summaries, effective_budget)

    arr_target = aggregate_mass(keys, p_target)
    arr_gfn_mean = aggregate_mass(keys, p_gfn_mean)
    target_vmax = float(arr_target.max())
    gfn_vmax = float(arr_gfn_mean.max())
    diff = arr_gfn_mean - arr_target
    diff_abs = float(np.max(np.abs(diff)))

    target_norm = None
    if args.target_power_gamma is not None and abs(args.target_power_gamma - 1.0) > 1e-12:
        target_norm = mcolors.PowerNorm(gamma=args.target_power_gamma, vmin=0.0, vmax=target_vmax)

    plot_faceted_mass(
        arr_target,
        "Stored reward-defined target distribution",
        out_dir / "rq1_target_distribution_mass_heatmap.png",
        vmin=0.0,
        vmax=target_vmax,
        norm=target_norm,
    )
    plot_faceted_mass(
        arr_gfn_mean,
        "Exact GFN distribution across seeds (mean)",
        out_dir / "rq1_gfn_distribution_mass_heatmap.png",
        vmin=0.0,
        vmax=gfn_vmax,
    )
    plot_faceted_mass(
        diff,
        "Mean GFN − target mass difference",
        out_dir / "rq1_mass_difference_heatmap.png",
        vmin=-diff_abs,
        vmax=diff_abs,
        cmap="coolwarm",
        cbar_label="Mass difference",
    )
    plot_target_vs_gfn_mode_mass_seeded(
        out_dir / "rq1_target_vs_gfn_top_mode_masses.png",
        target_mode_rows,
        p_gfn_seeds,
        top_n_modes=args.top_n_modes,
    )

    gfn_mode_rows, _ = compute_modes_and_basins(p_gfn_mean, neighbors)
    gfn_mode_rows = add_loss_ranks_to_modes(gfn_mode_rows, losses)
    save_csv(out_dir / "rq1_target_modes.csv", [
        {
            "mode_key": r["mode_key"],
            "mode_peak_prob": r["mode_peak_prob"],
            "mode_basin_mass": r["mode_basin_mass"],
            "basin_size": r["basin_size"],
            "mode_loss": r["mode_loss"],
            "mode_center_rank": r["mode_center_rank"],
            "best_member_loss": r["best_member_loss"],
        } for r in target_mode_rows
    ])
    save_csv(out_dir / "rq1_gfn_modes.csv", [
        {
            "mode_key": r["mode_key"],
            "mode_peak_prob": r["mode_peak_prob"],
            "mode_basin_mass": r["mode_basin_mass"],
            "basin_size": r["basin_size"],
            "mode_loss": r["mode_loss"],
            "mode_center_rank": r["mode_center_rank"],
            "best_member_loss": r["best_member_loss"],
        } for r in gfn_mode_rows
    ])

    pd.DataFrame(dist_metric_rows).to_csv(out_dir / "rq2_distributional_metrics_per_seed.csv", index=False)

    rq1_summary = {
        "gfn_run_ids": args.gfn_run_ids,
        "bo_run_ids": args.bo_run_ids,
        "step_fraction": args.step_fraction,
        "effective_budget": effective_budget,
        "n_seeds": n_seeds,
        "beta": args.beta,
        **target_meta,
        "n_target_modes": len(target_mode_rows),
        "n_gfn_modes_mean_distribution": len(gfn_mode_rows),
        "top_target_mode_keys": top_mode_keys,
        "target_top3_basin_mass": float(sum(r["mode_basin_mass"] for r in target_mode_rows[:3])),
        "gfn_mass_on_top3_target_basins_mean_distribution": float(sum(sum(p_gfn_mean.get(m, 0.0) for m in r["members"]) for r in target_mode_rows[:3])),
    }
    with open(out_dir / "rq1_mode_summary.json", "w") as f:
        json.dump(rq1_summary, f, indent=2)

    plot_rq2_retrieval_curves(
        out_dir / "rq2_retrieval_curves.png",
        effective_budget,
        global_best_loss,
        tpe_summary,
        gfn_summary,
        random_summary,
        log_gap_y=not args.linear_gap_y,
    )
    plot_rq2_topk_recovery(out_dir / "rq2_topk_recovery.png", tpe_summary, gfn_summary, random_summary, ks=(5, 10, 20))
    plot_rq2_probability_rank(out_dir / "rq2_probability_rank.png", p_target, p_gfn_mean, topk_max=args.topk_max)

    rq2_summary = {
        "effective_budget": effective_budget,
        "n_seeds": n_seeds,
        "tpe": tpe_summary,
        "gfn": gfn_summary,
        "random": random_summary,
        "distributional_fidelity_per_seed": dist_metric_rows,
    }
    with open(out_dir / "rq2_retrieval_summary.json", "w") as f:
        json.dump(rq2_summary, f, indent=2)

    make_summary_table_csv_seeded(out_dir / "rq12_results_table.csv", tpe_summary, gfn_summary, random_summary, dist_metric_rows)

    print(f"Saved seeded RQ1/RQ2 figures to {out_dir}")
    print(f"Target source: {target_meta['target_source']}")
    print("  - rq1_target_distribution_mass_heatmap.png")
    print("  - rq1_gfn_distribution_mass_heatmap.png")
    print("  - rq1_mass_difference_heatmap.png")
    print("  - rq1_target_vs_gfn_top_mode_masses.png")
    print("  - rq2_retrieval_curves.png")
    print("  - rq2_topk_recovery.png")
    print("  - rq2_probability_rank.png")
    print("  - rq12_results_table.csv")


if __name__ == "__main__":
    main()
