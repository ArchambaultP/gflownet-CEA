#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import wandb

from gflownet.envs.greenhouse.sim_env import CropSimEnv
from gflownet.envs.greenhouse.constants import GROUP_ORDER, PERTURBATION_SCHEME


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

# Same ordering as the landscape script.
GROUP_ORDERS = [
    ["none", "increase", "decrease"],
    ["none", "increase", "decrease", "higher_sensitivity", "lower_sensitivity"],
    ["none", "shift_warm", "shift_cold", "widen_optimum", "narrow_optimum"],
    ["none", "increase", "decrease", "higher_sensitivity", "lower_sensitivity"],
    ["none", "more_fruit_growth", "more_veg_growth", "lower_resp_cost", "higher_resp_cost", "higher_sensitivity", "lower_sensitivity"],
]


def normalize_token(tok: str) -> str:
    return tok.strip()


def load_table(path: str):
    with open(path) as f:
        raw = json.load(f)
    states = []
    losses = []
    rewards = []
    for key, val in raw.items():
        toks = tuple(normalize_token(t) for t in key.split("|"))
        if len(toks) != len(GROUP_ORDER):
            raise ValueError(f"Expected {len(GROUP_ORDER)} groups in key {key}")
        states.append(toks)
        losses.append(float(val["loss"]))
        rewards.append(float(val.get("reward", float("nan"))))
    return states, np.array(losses, float), np.array(rewards, float)


def stable_softmin(losses, beta, norm="q10q90", q_low=0.1, q_high=0.9):
    losses = np.asarray(losses, float)
    if norm == "none":
        L = losses.copy()
    elif norm == "q10q90":
        lo = float(np.quantile(losses, q_low))
        hi = float(np.quantile(losses, q_high))
        scale = max(hi - lo, 1e-12)
        L = (losses - lo) / scale
    else:
        raise ValueError(norm)
    logw = -beta * L
    logw -= np.max(logw)
    w = np.exp(logw)
    return w / np.sum(w)


def build_forward_mlp(state_dict):
    linear_weights = []
    for k, v in state_dict.items():
        if k.endswith(".weight") and v.ndim == 2:
            linear_weights.append((k, v))
    linear_weights.sort(key=lambda kv: kv[0])

    layers = []
    for i, (_, w) in enumerate(linear_weights):
        out_dim, in_dim = w.shape
        layers.append(nn.Linear(in_dim, out_dim))
        if i < len(linear_weights) - 1:
            layers.append(nn.ReLU())
    model = nn.Sequential(*layers)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def download_final_ckpt(project, run_id, alias="final"):
    api = wandb.Api()
    artifact_name = f"{project}/ckpt-{run_id}"

    artifact = None
    for try_alias in [alias, "latest"]:
        try:
            artifact = api.artifact(f"{artifact_name}:{try_alias}")
            break
        except wandb.errors.CommError:
            continue
    if artifact is None:
        raise RuntimeError(f"No checkpoint artifact found for {artifact_name} (tried {alias}, latest)")

    tmp_dir = tempfile.mkdtemp(prefix=f"wandb_artifact_{run_id}_")
    artifact_dir = artifact.download(root=tmp_dir)
    ckpt_files = sorted([f for f in os.listdir(artifact_dir) if f.endswith(".ckpt")])
    if not ckpt_files:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise RuntimeError(f"No .ckpt file found in artifact for run {run_id}")
    ckpt_name = "final.ckpt" if "final.ckpt" in ckpt_files else ckpt_files[0]
    return os.path.join(artifact_dir, ckpt_name), tmp_dir


def tokens_to_prefix_state(tokens_prefix, env):
    state = [()]
    for slot_idx, tok in enumerate(tokens_prefix):
        cycle = 1 + slot_idx // env.n_groups
        group_id = slot_idx % env.n_groups
        action_value = env.pert2id[tok]
        state.append((cycle, group_id, action_value))
    return state


def compute_exact_gfn_distribution(model, env, terminal_tokens, device="cpu"):
    model = model.to(device)
    probs = []
    action_space = env.get_action_space()

    with torch.no_grad():
        for toks in terminal_tokens:
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

            # If the env uses an explicit EOS, include it.
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

            probs.append(logp)

    lp = np.array(probs, float)
    lp -= np.max(lp)
    p = np.exp(lp)
    p /= np.sum(p)
    return p


def aggregate_mass(states, p):
    shape = (len(GROUP_ORDERS[0]), len(GROUP_ORDERS[1]), len(GROUP_ORDERS[3]), len(GROUP_ORDERS[2]))
    arr = np.zeros(shape, float)
    idx_maps = [{tok: i for i, tok in enumerate(order)} for order in GROUP_ORDERS]
    for s, prob in zip(states, p):
        i = idx_maps[0][s[0]]
        j = idx_maps[1][s[1]]
        x = idx_maps[2][s[2]]
        y = idx_maps[3][s[3]]
        arr[i, j, y, x] += prob
    return arr


def plot_faceted_mass(arr, title, out_png):
    g1_order, g2_order, g3_order, g4_order, _ = GROUP_ORDERS
    vmin, vmax = 0.0, float(arr.max())
    fig, axes = plt.subplots(len(g1_order), len(g2_order), figsize=(18, 11), squeeze=False)
    im = None
    for i, g1 in enumerate(g1_order):
        for j, g2 in enumerate(g2_order):
            ax = axes[i][j]
            im = ax.imshow(arr[i, j], aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
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
    cbar.set_label("Probability mass")
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def topk_curve(p):
    idx = np.argsort(-p)
    sp = p[idx]
    return np.arange(1, len(p) + 1), sp, np.cumsum(sp)


def build_neighbors(states):
    by_state = {s: i for i, s in enumerate(states)}
    neighbor_idxs = []
    for s in states:
        ns = []
        for g, order in enumerate(GROUP_ORDERS):
            for alt in order:
                if alt != s[g]:
                    t = list(s)
                    t[g] = alt
                    t = tuple(t)
                    j = by_state.get(t)
                    if j is not None:
                        ns.append(j)
        neighbor_idxs.append(ns)
    return by_state, neighbor_idxs


def find_modes(states, p):
    _, neighbor_idxs = build_neighbors(states)
    n = len(states)
    local_max = np.zeros(n, dtype=bool)

    for i in range(n):
        if all(p[i] >= p[j] - 1e-15 for j in neighbor_idxs[i]):
            local_max[i] = True

    sink = np.full(n, -1, int)
    order = np.argsort(-p)
    for i in order:
        if local_max[i]:
            sink[i] = i
            continue
        current = i
        seen = set()
        while True:
            if current in seen:
                best = max(seen, key=lambda k: (p[k], states[k]))
                sink[i] = best
                break
            seen.add(current)
            better = [j for j in neighbor_idxs[current] if p[j] > p[current] + 1e-15]
            if not better:
                sink[i] = current
                local_max[current] = True
                break
            current = max(better, key=lambda j: (p[j], states[j]))
            if sink[current] != -1:
                sink[i] = sink[current]
                break
        for k in list(seen):
            sink[k] = sink[i]

    mode_members = defaultdict(list)
    for i, s in enumerate(sink):
        mode_members[s].append(i)

    rows = []
    for m, members in mode_members.items():
        rows.append({
            "mode_center": "|".join(states[m]),
            "mode_peak_prob": float(p[m]),
            "mode_basin_mass": float(p[members].sum()),
            "basin_size": int(len(members)),
            "center_rank": int(np.sum(p > p[m]) + 1),
        })
    rows.sort(key=lambda r: (-r["mode_basin_mass"], -r["mode_peak_prob"]))
    return rows


def plot_mode_bars(target_modes, gfn_modes, out_png, topn=10):
    trows = target_modes[:topn]
    grows = gfn_modes[:topn]
    labels_t = [r["mode_center"] for r in trows]
    labels_g = [r["mode_center"] for r in grows]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    axes[0].bar(range(len(trows)), [r["mode_basin_mass"] for r in trows])
    axes[0].set_xticks(range(len(trows)))
    axes[0].set_xticklabels(labels_t, rotation=60, ha="right", fontsize=8)
    axes[0].set_ylabel("Basin mass")
    axes[0].set_title("Top target modes")

    axes[1].bar(range(len(grows)), [r["mode_basin_mass"] for r in grows])
    axes[1].set_xticks(range(len(grows)))
    axes[1].set_xticklabels(labels_g, rotation=60, ha="right", fontsize=8)
    axes[1].set_ylabel("Basin mass")
    axes[1].set_title("Top GFN modes")

    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


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


def summarize(p, mode_rows):
    idx = np.argsort(-p)
    return {
        "top1_mass": float(p[idx[:1]].sum()),
        "top3_mass": float(p[idx[:3]].sum()),
        "top10_mass": float(p[idx[:10]].sum()),
        "top20_mass": float(p[idx[:20]].sum()),
        "top3_modes_mass": float(sum(r["mode_basin_mass"] for r in mode_rows[:3])),
        "effective_support": float(1.0 / np.sum(np.square(p))),
        "n_modes": int(len(mode_rows)),
    }


def plot_topk_compare(p_target, p_gfn, out_png, kmax=100):
    k_t, sp_t, cum_t = topk_curve(p_target)
    _, sp_g, cum_g = topk_curve(p_gfn)
    kmax = min(kmax, len(k_t))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].plot(k_t[:kmax], cum_t[:kmax], lw=2, label="target")
    axes[0].plot(k_t[:kmax], cum_g[:kmax], lw=2, label="GFN")
    axes[0].set_xlabel("Top-k states")
    axes[0].set_ylabel("Cumulative mass")
    axes[0].set_title("Cumulative top-k state mass")
    axes[0].grid(alpha=0.3)
    axes[0].legend(frameon=False)

    axes[1].plot(k_t[:kmax], sp_t[:kmax], lw=2, label="target")
    axes[1].plot(k_t[:kmax], sp_g[:kmax], lw=2, label="GFN")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Rank")
    axes[1].set_ylabel("Probability")
    axes[1].set_title("Top-k probability-rank")
    axes[1].grid(alpha=0.3)
    axes[1].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description="Simple wandb-run landscape visualizer: target vs exact GFN distribution for one run."
    )
    ap.add_argument("--reward_table", required=True)
    ap.add_argument("--wandb_project", required=True, help="entity/project")
    ap.add_argument("--wandb_run_id", required=True)
    ap.add_argument("--step_fraction", type=float, required=True)
    ap.add_argument("--out_dir", default="wandb_landscape")
    ap.add_argument("--artifact_alias", default="final")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--beta", type=float, default=None, help="Optional: recompute target from loss with softmin beta instead of using stored reward.")
    ap.add_argument("--loss_norm", choices=["none", "q10q90"], default="q10q90")
    ap.add_argument("--q_low", type=float, default=0.1)
    ap.add_argument("--q_high", type=float, default=0.9)
    args = ap.parse_args()

    out_dir = Path(args.out_dir) / args.wandb_run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    states, losses, stored_rewards = load_table(args.reward_table)
    if args.beta is None and np.all(np.isfinite(stored_rewards)) and stored_rewards.sum() > 0:
        p_target = stored_rewards / stored_rewards.sum()
        target_meta = {"target_source": "stored_reward"}
    else:
        if args.beta is None:
            raise ValueError("No stored reward found in table. Pass --beta to build a softmin target.")
        p_target = stable_softmin(losses, beta=args.beta, norm=args.loss_norm, q_low=args.q_low, q_high=args.q_high)
        target_meta = {
            "target_source": "softmin_from_loss",
            "beta": args.beta,
            "loss_norm": args.loss_norm,
            "q_low": args.q_low,
            "q_high": args.q_high,
        }

    ckpt_path, tmp_dir = download_final_ckpt(args.wandb_project, args.wandb_run_id, alias=args.artifact_alias)
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = build_forward_mlp(ckpt["forward"])

        env = CropSimEnv(
            n_cycles=1,
            step_fraction=args.step_fraction,
            precomputed=True,
            device=args.device,
        )

        p_gfn = compute_exact_gfn_distribution(model, env, states, device=args.device)

        arr_target = aggregate_mass(states, p_target)
        arr_gfn = aggregate_mass(states, p_gfn)

        plot_faceted_mass(arr_target, f"True distribution — target ({args.wandb_run_id})", out_dir / "target_distribution_mass_heatmap.png")
        plot_faceted_mass(arr_gfn, f"True distribution — GFN ({args.wandb_run_id})", out_dir / "gfn_distribution_mass_heatmap.png")
        plot_topk_compare(p_target, p_gfn, out_dir / "topk_compare.png")

        target_modes = find_modes(states, p_target)
        gfn_modes = find_modes(states, p_gfn)
        plot_mode_bars(target_modes, gfn_modes, out_dir / "top_modes_compare.png")
        save_csv(out_dir / "target_modes.csv", target_modes)
        save_csv(out_dir / "gfn_modes.csv", gfn_modes)

        summary = {
            "run_id": args.wandb_run_id,
            "step_fraction": args.step_fraction,
            **target_meta,
            "target": summarize(p_target, target_modes),
            "gfn": summarize(p_gfn, gfn_modes),
        }
        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"Saved figures to {out_dir}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
