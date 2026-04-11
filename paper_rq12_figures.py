#!/usr/bin/env python3
from __future__ import annotations

"""
Paper-oriented figure generator for the enumerable 1-cycle setting.

Focus:
- RQ1: mode discovery against the exact stored reward-defined target distribution
- RQ2: retrieval performance and distributional fidelity

Key changes relative to the earlier draft:
- by default, REQUIRES stored rewards in the reward table (to avoid silently switching to a softmin target)
- uses explicit colors for line/bar fills
- uses optimality gap on the loss panel for a more interpretable retrieval comparison
- replaces cumulative top-k state mass with a more direct fidelity view:
  rank-probability + target-vs-GFN probability scatter
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
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb

from gflownet.envs.greenhouse.constants import GROUP_ORDER, PERTURBATION_SCHEME
from gflownet.envs.greenhouse.sim_env import CropSimEnv


COLORS = {
    "TPE": "#1f77b4",
    "GFN": "#ff7f0e",
    "Random": "#2ca02c",
}

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


def load_target_distribution(losses, rewards, require_stored_reward=True, beta=None, norm="q10q90", q_low=0.1, q_high=0.9):
    reward_sum = float(sum(rewards.values()))
    if reward_sum > 0:
        return normalize_prob_dict(rewards), {"target_source": "stored_reward"}
    if require_stored_reward:
        raise ValueError(
            "The reward table does not contain positive stored rewards. "
            "For paper figures in the enumerable setting, use a reward table with stored reward values, "
            "or rerun with --allow_softmin_target and --beta explicitly."
        )
    if beta is None:
        raise ValueError("No stored rewards found. Pass --beta when allowing a softmin target.")
    return stable_softmin(losses, beta=beta, norm=norm, q_low=q_low, q_high=q_high), {
        "target_source": "softmin_from_loss",
        "beta": beta,
        "loss_norm": norm,
        "q_low": q_low,
        "q_high": q_high,
    }


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


def plot_faceted_mass(arr, title, out_png, vmin=None, vmax=None, cmap="viridis", cbar_label="Probability mass", norm=None):
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
            if norm is None:
                im = ax.imshow(arr[i, j], aspect="auto", origin="lower", vmin=vmin, vmax=vmax, cmap=cmap)
            else:
                im = ax.imshow(arr[i, j], aspect="auto", origin="lower", cmap=cmap, norm=norm)
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
    plt.bar(x - w / 2, target_vals, width=w, label="Target basin mass", color=COLORS["TPE"])
    plt.bar(x + w / 2, gfn_vals, width=w, label="GFN basin mass", color=COLORS["GFN"])
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


def plot_rq2_retrieval_curves(out_path: Path, budget: int, global_best_loss: float, tpe_summary, gfn_summary, random_summary, log_gap_y=True):
    x = np.arange(1, budget + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    eps = 1e-12
    tpe_gap = np.maximum(np.array(pad_curve([v - global_best_loss for v in tpe_summary["best_loss_curve"]], budget)), eps)
    gfn_gap_mean = np.maximum(np.array(gfn_summary["best_loss_curve_mean"]) - global_best_loss, eps)
    gfn_gap_std = np.array(gfn_summary["best_loss_curve_std"])
    rnd_gap_mean = np.maximum(np.array(random_summary["best_loss_curve_mean"]) - global_best_loss, eps)
    rnd_gap_std = np.array(random_summary["best_loss_curve_std"])

    axes[0].plot(x, tpe_gap, label="TPE", linewidth=2, color=COLORS["TPE"])
    axes[0].plot(x, gfn_gap_mean, label="GFN", linewidth=2, color=COLORS["GFN"])
    axes[0].fill_between(
        x,
        np.maximum(gfn_gap_mean - gfn_gap_std, eps),
        gfn_gap_mean + gfn_gap_std,
        alpha=0.2,
        color=COLORS["GFN"],
    )
    axes[0].plot(x, rnd_gap_mean, label="Random", linewidth=2, color=COLORS["Random"])
    axes[0].fill_between(
        x,
        np.maximum(rnd_gap_mean - rnd_gap_std, eps),
        rnd_gap_mean + rnd_gap_std,
        alpha=0.2,
        color=COLORS["Random"],
    )
    if log_gap_y:
        axes[0].set_yscale("log")
    axes[0].set_xlabel("Retrieval budget")
    axes[0].set_ylabel(r"Optimality gap ($L_{\mathrm{best}} - L^{*}$)")
    axes[0].set_title("Retrieval performance (loss gap)")
    axes[0].grid(alpha=0.3)
    axes[0].legend(frameon=False)

    axes[1].plot(x, np.array(pad_curve(tpe_summary["best_reward_curve"], budget)), label="TPE", linewidth=2, color=COLORS["TPE"])
    axes[1].plot(x, np.array(gfn_summary["best_reward_curve_mean"]), label="GFN", linewidth=2, color=COLORS["GFN"])
    axes[1].fill_between(
        x,
        np.array(gfn_summary["best_reward_curve_mean"]) - np.array(gfn_summary["best_reward_curve_std"]),
        np.array(gfn_summary["best_reward_curve_mean"]) + np.array(gfn_summary["best_reward_curve_std"]),
        alpha=0.2,
        color=COLORS["GFN"],
    )
    axes[1].plot(x, np.array(random_summary["best_reward_curve_mean"]), label="Random", linewidth=2, color=COLORS["Random"])
    axes[1].fill_between(
        x,
        np.array(random_summary["best_reward_curve_mean"]) - np.array(random_summary["best_reward_curve_std"]),
        np.array(random_summary["best_reward_curve_mean"]) + np.array(random_summary["best_reward_curve_std"]),
        alpha=0.2,
        color=COLORS["Random"],
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
    plt.bar(x - w, tpe_vals, width=w, label="TPE", color=COLORS["TPE"])
    plt.bar(x, gfn_vals, width=w, yerr=gfn_errs, capsize=3, label="GFN", color=COLORS["GFN"])
    plt.bar(x + w, rnd_vals, width=w, yerr=rnd_errs, capsize=3, label="Random", color=COLORS["Random"])
    plt.xticks(x, [f"Top-{k}" for k in ks])
    plt.ylabel("Unique top-k states found")
    plt.xlabel("Reference set")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_rq2_probability_rank(out_path: Path, p_target: Dict[str, float], p_gfn: Dict[str, float], topk_max: int = 100):
    target_sorted = np.array(sorted(p_target.values(), reverse=True), dtype=float)
    gfn_sorted = np.array(sorted(p_gfn.values(), reverse=True), dtype=float)
    k = np.arange(1, min(topk_max, len(target_sorted)) + 1)

    plt.figure(figsize=(5.4, 4.2))
    plt.plot(k, target_sorted[: len(k)], lw=2, label="Target", color=COLORS["TPE"])
    plt.plot(k, gfn_sorted[: len(k)], lw=2, label="GFN", color=COLORS["GFN"])
    plt.yscale("log")
    plt.xlabel("Rank")
    plt.ylabel("Probability")
    plt.title("Probability-rank comparison")
    plt.grid(alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reward_table", required=True)
    ap.add_argument("--gfn_project", required=True)
    ap.add_argument("--gfn_run_id", required=True)
    ap.add_argument("--bo_project", required=True, help="wandb entity/project for the TPE baseline")
    ap.add_argument("--bo_run_id", required=True, help="wandb run id for the TPE baseline")
    ap.add_argument("--step_fraction", type=float, required=True)
    ap.add_argument("--budget", type=int, required=True, help="Post-training retrieval budget for the enumerable setting")
    ap.add_argument("--artifact_alias", default="final")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n_repeats", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output_dir", default="paper_rq12_outputs")
    ap.add_argument("--top_n_modes", type=int, default=5)
    ap.add_argument("--topk_max", type=int, default=100)
    ap.add_argument("--allow_softmin_target", action="store_true")
    ap.add_argument("--beta", type=float, default=None, help="Used only with --allow_softmin_target")
    ap.add_argument("--loss_norm", choices=["none", "q10q90"], default="q10q90")
    ap.add_argument("--q_low", type=float, default=0.1)
    ap.add_argument("--q_high", type=float, default=0.9)
    ap.add_argument("--linear_gap_y", action="store_true", help="Use linear instead of log scale on the loss-gap panel")
    ap.add_argument("--target_power_gamma", type=float, default=0.5, help="Power-law color normalization for the stored-reward target heatmap (1.0 keeps linear scale)")
    args = ap.parse_args()

    out_dir = Path(args.output_dir) / args.gfn_run_id
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

    ckpt_path = download_gfn_checkpoint(args.gfn_project, args.gfn_run_id, alias=args.artifact_alias)
    tmp_root = ckpt_path.parent.parent if ckpt_path.parent.name == "files" else ckpt_path.parent
    try:
        ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
        model = build_mlp_from_state_dict(extract_forward_state_dict(ckpt))
        env = CropSimEnv(n_cycles=1, step_fraction=args.step_fraction, precomputed=True, device=args.device)
        p_gfn = compute_exact_gfn_distribution(model, env, keys, device=args.device)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    tpe_seq_full = fetch_tpe_trial_sequence(args.bo_project, args.bo_run_id)
    effective_budget = min(args.budget, len(tpe_seq_full))
    if effective_budget < args.budget:
        print(f"[WARN] Requested budget={args.budget}, but TPE history has only {len(tpe_seq_full)} steps. Using {effective_budget}.")
    tpe_seq = tpe_seq_full[:effective_budget]

    sorted_by_loss = sorted(keys, key=lambda k: losses[k])
    topk_sets = {1: set(sorted_by_loss[:1]), 5: set(sorted_by_loss[:5]), 10: set(sorted_by_loss[:10]), 20: set(sorted_by_loss[:20])}
    global_best_loss = losses[sorted_by_loss[0]]

    tpe_summary = summarize_sequence(tpe_seq, losses, rewards, topk_sets, top_mode_keys=top_mode_keys, basin_of=basin_of)

    rng = np.random.default_rng(args.seed)
    random_keys = np.array(keys)
    if effective_budget > len(random_keys):
        raise ValueError(f"Budget {effective_budget} exceeds state space size {len(random_keys)}")

    random_loss_curves = []
    random_reward_curves = []
    random_mode_curves = []
    random_finals_loss = []
    random_finals_reward = []
    random_uniqs = []
    random_mode_hits = []
    random_topk = {k: [] for k in topk_sets}
    random_firsts = {k: [] for k in topk_sets}
    random_hit_rates = {k: [] for k in topk_sets}
    for _ in range(args.n_repeats):
        seq = list(rng.choice(random_keys, size=effective_budget, replace=False))
        s = summarize_sequence(seq, losses, rewards, topk_sets, top_mode_keys=top_mode_keys, basin_of=basin_of)
        random_loss_curves.append(s["best_loss_curve"])
        random_reward_curves.append(s["best_reward_curve"])
        random_mode_curves.append(s["mode_hit_curve"])
        random_finals_loss.append(s["best_final_loss"])
        random_finals_reward.append(s["best_final_reward"])
        random_uniqs.append(s["unique_states"])
        random_mode_hits.append(s["mode_hits_final"])
        for k in topk_sets:
            random_topk[k].append(s["topk_unique"][k])
            random_firsts[k].append(s["first_hit"][k] if s["first_hit"][k] is not None else np.nan)
            random_hit_rates[k].append(0 if s["first_hit"][k] is None else 1)

    random_summary = {
        "best_loss_curve_mean": np.array(random_loss_curves, dtype=float).mean(axis=0).tolist(),
        "best_loss_curve_std": np.array(random_loss_curves, dtype=float).std(axis=0).tolist(),
        "best_reward_curve_mean": np.array(random_reward_curves, dtype=float).mean(axis=0).tolist(),
        "best_reward_curve_std": np.array(random_reward_curves, dtype=float).std(axis=0).tolist(),
        "best_final_loss_mean": float(np.mean(random_finals_loss)),
        "best_final_loss_std": float(np.std(random_finals_loss)),
        "best_final_reward_mean": float(np.mean(random_finals_reward)),
        "best_final_reward_std": float(np.std(random_finals_reward)),
        "unique_states_mean": float(np.mean(random_uniqs)),
        "topk_unique_mean": {k: float(np.mean(v)) for k, v in random_topk.items()},
        "topk_unique_std": {k: float(np.std(v)) for k, v in random_topk.items()},
        "first_hit_mean": {k: _safe_nanmean(v) for k, v in random_firsts.items()},
        "hit_rate": {k: float(np.mean(v)) for k, v in random_hit_rates.items()},
        "mode_hit_curve_mean": np.array(random_mode_curves, dtype=float).mean(axis=0).tolist(),
        "mode_hit_curve_std": np.array(random_mode_curves, dtype=float).std(axis=0).tolist(),
        "mode_hits_final_mean": float(np.mean(random_mode_hits)),
        "mode_hits_final_std": float(np.std(random_mode_hits)),
    }

    gfn_summary = monte_carlo_from_distribution(
        p_gfn,
        effective_budget,
        args.n_repeats,
        losses,
        rewards,
        topk_sets,
        rng,
        top_mode_keys=top_mode_keys,
        basin_of=basin_of,
    )

    arr_target = aggregate_mass(keys, p_target)
    arr_gfn = aggregate_mass(keys, p_gfn)
    target_vmax = float(arr_target.max())
    gfn_vmax = float(arr_gfn.max())
    diff = arr_gfn - arr_target
    diff_abs = float(np.max(np.abs(diff)))

    target_norm = None
    if args.target_power_gamma is not None and abs(args.target_power_gamma - 1.0) > 1e-12:
        target_norm = mcolors.PowerNorm(gamma=args.target_power_gamma, vmin=0.0, vmax=target_vmax)

    plot_faceted_mass(
        arr_target,
        f"Stored reward-defined target distribution ({args.gfn_run_id})",
        out_dir / "rq1_target_distribution_mass_heatmap.png",
        vmin=0.0,
        vmax=target_vmax,
        norm=target_norm,
    )
    plot_faceted_mass(
        arr_gfn,
        f"Exact GFN distribution ({args.gfn_run_id})",
        out_dir / "rq1_gfn_distribution_mass_heatmap.png",
        vmin=0.0,
        vmax=gfn_vmax,
    )
    plot_faceted_mass(diff, f"GFN − target mass difference ({args.gfn_run_id})", out_dir / "rq1_mass_difference_heatmap.png", vmin=-diff_abs, vmax=diff_abs, cmap="coolwarm", cbar_label="Mass difference")
    plot_target_vs_gfn_mode_mass_figure(out_dir / "rq1_target_vs_gfn_top_mode_masses.png", target_mode_rows, p_gfn, top_n_modes=args.top_n_modes)

    gfn_mode_rows, _ = compute_modes_and_basins(p_gfn, neighbors)
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

    rq1_summary = {
        "run_id": args.gfn_run_id,
        "step_fraction": args.step_fraction,
        "effective_budget": effective_budget,
        **target_meta,
        "n_target_modes": len(target_mode_rows),
        "n_gfn_modes": len(gfn_mode_rows),
        "top_target_mode_keys": top_mode_keys,
        "target_top3_basin_mass": float(sum(r["mode_basin_mass"] for r in target_mode_rows[:3])),
        "gfn_mass_on_top3_target_basins": float(sum(sum(p_gfn.get(m, 0.0) for m in r["members"]) for r in target_mode_rows[:3])),
        "gfn_top_target_basin_hits": int(len([r for r in target_mode_rows[:args.top_n_modes] if sum(p_gfn.get(m, 0.0) for m in r["members"]) > 0])),
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
    plot_rq2_probability_rank(out_dir / "rq2_probability_rank.png", p_target, p_gfn, topk_max=args.topk_max)
    dist_metrics = distributional_metrics(p_target, p_gfn, losses)
    with open(out_dir / "rq2_distributional_metrics.json", "w") as f:
        json.dump(dist_metrics, f, indent=2)

    rq2_summary = {
        "effective_budget": effective_budget,
        "tpe": {
            "best_final_loss": tpe_summary["best_final_loss"],
            "best_final_reward": tpe_summary["best_final_reward"],
            "unique_states": tpe_summary["unique_states"],
            "topk_unique": tpe_summary["topk_unique"],
            "first_hit": tpe_summary["first_hit"],
            "mode_hits_final": tpe_summary["mode_hits_final"],
        },
        "gfn": gfn_summary,
        "random": random_summary,
        "distributional_fidelity": dist_metrics,
    }
    with open(out_dir / "rq2_retrieval_summary.json", "w") as f:
        json.dump(rq2_summary, f, indent=2)

    make_summary_table_csv(out_dir / "rq12_results_table.csv", tpe_summary, gfn_summary, random_summary, dist_metrics)

    print(f"Saved RQ1/RQ2 figures to {out_dir}")
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
