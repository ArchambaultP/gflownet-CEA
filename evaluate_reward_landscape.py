#!/usr/bin/env python3
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

GROUP_NAMES = [
    "Canopy",
    "Photosynthesis",
    "Temp. Inhibition",
    "Temp. & Dev.",
    "Biomass",
]

# Order used in the plots and in the discrete-neighborhood mode finder.
GROUP_ORDERS = [
    ["none", "increase", "decrease"],
    ["none", "increase", "decrease", "higher_sensitivity", "lower_sensitivity"],
    ["none", "shift_warm", "shift_cold", "widen_optimum", "narrow_optimum"],
    ["none", "increase", "decrease", "higher_sensitivity", "lower_sensitivity"],
    ["none", "more_fruit_growth", "more_veg_growth", "lower_resp_cost", "higher_resp_cost", "higher_sensitivity", "lower_sensitivity"],
]

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
        if len(toks) != 5:
            raise ValueError(f"Expected 5 groups in key {key}")
        states.append(toks)
        losses.append(float(val["loss"]))
        rewards.append(float(val.get("reward", float("nan"))))
    return states, np.array(losses, float), np.array(rewards, float)

def stable_softmin(losses, beta, norm="q10q90", q_low=0.1, q_high=0.9):
    losses = np.asarray(losses, float)
    if norm == "none":
        L = losses.copy()
        meta = {"norm": "none", "anchor": 0.0, "scale": 1.0, "q_low": None, "q_high": None}
    elif norm == "q10q90":
        lo = float(np.quantile(losses, q_low))
        hi = float(np.quantile(losses, q_high))
        scale = max(hi - lo, 1e-12)
        L = (losses - lo) / scale
        meta = {"norm": "q10q90", "anchor": lo, "scale": scale, "q_low": q_low, "q_high": q_high}
    else:
        raise ValueError(norm)
    logw = -beta * L
    logw -= np.max(logw)
    w = np.exp(logw)
    p = w / np.sum(w)
    return p, meta

def threshold_reward(losses, tau_quantile=0.05, temperature=0.05, tail=1e-3, norm="q10q90", q_low=0.1, q_high=0.9):
    losses = np.asarray(losses, float)
    if norm == "q10q90":
        lo = float(np.quantile(losses, q_low))
        hi = float(np.quantile(losses, q_high))
        scale = max(hi - lo, 1e-12)
        L = (losses - lo) / scale
    elif norm == "none":
        L = losses.copy()
        lo, hi, scale = 0.0, 1.0, 1.0
    else:
        raise ValueError(norm)

    tau = float(np.quantile(L, tau_quantile))
    z = (tau - L) / max(temperature, 1e-12)
    sig = 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))
    w = tail + (1.0 - tail) * sig
    p = w / np.sum(w)
    meta = {
        "norm": norm,
        "anchor": lo,
        "scale": scale,
        "tau_quantile": tau_quantile,
        "tau": tau,
        "temperature": temperature,
        "tail": tail,
    }
    return p, meta

def topk_curve(p):
    idx = np.argsort(-p)
    sp = p[idx]
    return np.arange(1, len(p) + 1), np.cumsum(sp)

def entropy(p):
    p = np.asarray(p, float)
    m = p > 0
    return float(-(p[m] * np.log(p[m])).sum())

def effective_support(p):
    return float(1.0 / np.sum(np.square(p)))

def aggregate_mass(states, p):
    # rows = group1, cols = group2, heatmap x = group3, y = group4, biomass summed
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
    # Reserve a dedicated axis for the colorbar so it cannot overlap the last facets.
    fig.subplots_adjust(left=0.06, right=0.90, bottom=0.08, top=0.90, wspace=0.25, hspace=0.35)
    cax = fig.add_axes([0.92, 0.15, 0.015, 0.68])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Probability mass")
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)

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

    # Discrete mode definition: local maxima on the Hamming-1 neighborhood graph.
    for i in range(n):
        if all(p[i] >= p[j] - 1e-15 for j in neighbor_idxs[i]):
            local_max[i] = True

    # Basin assignment by greedy ascent to the highest-probability improving neighbor.
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
        rows.append(
            {
                "mode_center": "|".join(states[m]),
                "mode_peak_prob": float(p[m]),
                "mode_basin_mass": float(p[members].sum()),
                "basin_size": int(len(members)),
                "center_rank": int(np.sum(p > p[m]) + 1),
            }
        )
    rows.sort(key=lambda r: (-r["mode_basin_mass"], -r["mode_peak_prob"]))
    return rows

def plot_topk(p, title, out_png, kmax=100):
    ranks, cum = topk_curve(p)
    kmax = min(kmax, len(p))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ranks[:kmax], cum[:kmax], lw=2)
    ax.set_xlabel("Top-k states")
    ax.set_ylabel("Cumulative probability mass")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

def plot_mode_bars(mode_rows, title, out_png, topn=10):
    rows = mode_rows[:topn]
    labels = [r["mode_center"] for r in rows]
    masses = [r["mode_basin_mass"] for r in rows]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(range(len(rows)), masses)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Mode basin mass")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

def summarize(p, mode_rows):
    idx = np.argsort(-p)
    return {
        "top1_mass": float(p[idx[:1]].sum()),
        "top3_mass": float(p[idx[:3]].sum()),
        "top10_mass": float(p[idx[:10]].sum()),
        "top1pct_mass": float(p[idx[: max(1, len(p) // 100)]].sum()),
        "entropy": entropy(p),
        "effective_support": effective_support(p),
        "top3_modes_mass": float(sum(r["mode_basin_mass"] for r in mode_rows[:3])),
        "n_modes": int(len(mode_rows)),
    }

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

def main():
    ap = argparse.ArgumentParser(
        description="Evaluate reward distributions and mode structure from an enumerated reward/loss table."
    )
    ap.add_argument("--reward_table", required=True)
    ap.add_argument("--out_dir", default="landscape_eval")
    ap.add_argument(
        "--use_stored_reward",
        action="store_true",
        help="Also evaluate the stored reward field as a distribution.",
    )
    ap.add_argument("--softmin_betas", type=float, nargs="*", default=[])
    ap.add_argument(
        "--target_ratios",
        type=float,
        nargs="*",
        default=[],
        help="Convenience: beta = log(ratio).",
    )
    ap.add_argument("--reward_family", choices=["softmin", "threshold"], default="softmin")
    ap.add_argument("--loss_norm", choices=["none", "q10q90"], default="q10q90")
    ap.add_argument("--q_low", type=float, default=0.1)
    ap.add_argument("--q_high", type=float, default=0.9)
    ap.add_argument("--tau_quantile", type=float, default=0.05, help="For thresholded reward.")
    ap.add_argument("--temperature", type=float, default=0.05, help="For thresholded reward.")
    ap.add_argument("--tail", type=float, default=1e-3, help="For thresholded reward.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    states, losses, stored_rewards = load_table(args.reward_table)
    distributions = []

    if args.use_stored_reward and np.all(np.isfinite(stored_rewards)) and stored_rewards.sum() > 0:
        p = stored_rewards / stored_rewards.sum()
        distributions.append(("stored_reward", p, {"family": "stored"}))

    betas = list(args.softmin_betas) + [math.log(r) for r in args.target_ratios]
    seen = set()
    dedup = []
    for b in betas:
        key = round(float(b), 12)
        if key not in seen:
            seen.add(key)
            dedup.append(float(b))
    betas = dedup

    if args.reward_family == "softmin":
        for beta in betas:
            p, meta = stable_softmin(
                losses,
                beta=beta,
                norm=args.loss_norm,
                q_low=args.q_low,
                q_high=args.q_high,
            )
            distributions.append((f"softmin_beta{beta:.4g}", p, {"family": "softmin", "beta": beta, **meta}))
    else:
        p, meta = threshold_reward(
            losses,
            tau_quantile=args.tau_quantile,
            temperature=args.temperature,
            tail=args.tail,
            norm=args.loss_norm,
            q_low=args.q_low,
            q_high=args.q_high,
        )
        distributions.append((f"threshold_tauq{args.tau_quantile:.3g}_T{args.temperature:.3g}", p, {"family": "threshold", **meta}))

    if not distributions:
        if np.all(np.isfinite(stored_rewards)) and stored_rewards.sum() > 0:
            p = stored_rewards / stored_rewards.sum()
            distributions.append(("stored_reward", p, {"family": "stored"}))
        else:
            beta = math.log(20.0)
            p, meta = stable_softmin(losses, beta=beta, norm=args.loss_norm, q_low=args.q_low, q_high=args.q_high)
            distributions.append((f"softmin_beta{beta:.4g}", p, {"family": "softmin", "beta": beta, **meta}))

    all_summary = []
    for name, p, meta in distributions:
        sub = out_dir / name
        sub.mkdir(parents=True, exist_ok=True)

        arr = aggregate_mass(states, p)
        plot_faceted_mass(arr, f"True distribution — {name}", sub / "true_distribution_mass_heatmap.png")
        plot_topk(p, f"Cumulative top-k state mass — {name}", sub / "topk_state_mass.png")

        modes = find_modes(states, p)
        plot_mode_bars(modes, f"Top mode basin masses — {name}", sub / "top_modes.png")
        save_csv(sub / "modes.csv", modes)

        summary = summarize(p, modes)
        summary.update(meta)
        summary["name"] = name
        with open(sub / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        all_summary.append(summary)

    save_csv(out_dir / "reward_sweep_summary.csv", all_summary)
    print(f"Saved outputs to {out_dir}")

if __name__ == "__main__":
    main()
