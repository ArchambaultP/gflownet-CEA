#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np


def sigmoid(z):
    z = np.asarray(z, dtype=float)
    z = np.clip(z, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def normalize_losses(losses, mode="q10q90", q_low=0.10, q_high=0.90):
    losses = np.asarray(losses, dtype=float)

    if mode == "none":
        return losses.copy(), {
            "loss_norm": "none",
            "anchor": 0.0,
            "scale": 1.0,
            "q_low": None,
            "q_high": None,
        }

    if mode == "q10q90":
        ql = float(np.quantile(losses, q_low))
        qh = float(np.quantile(losses, q_high))
        scale = max(qh - ql, 1e-12)
        normed = (losses - ql) / scale
        return normed, {
            "loss_norm": "q10q90",
            "anchor": ql,
            "scale": scale,
            "q_low": q_low,
            "q_high": q_high,
        }

    raise ValueError(f"Unsupported loss_norm: {mode}")


def build_thresholded_reward(
    losses,
    loss_norm="q10q90",
    q_low=0.10,
    q_high=0.90,
    tau_quantile=0.05,
    temperature=0.05,
    epsilon=1e-3,
):
    losses_norm, norm_meta = normalize_losses(losses, loss_norm, q_low, q_high)

    tau = float(np.quantile(losses_norm, tau_quantile))
    z = (tau - losses_norm) / max(float(temperature), 1e-12)
    rewards = epsilon + (1.0 - epsilon) * sigmoid(z)
    probs = rewards / rewards.sum()

    meta = {
        "reward_family": "thresholded_sigmoid",
        **norm_meta,
        "tau_quantile": float(tau_quantile),
        "tau": tau,
        "temperature": float(temperature),
        "epsilon": float(epsilon),
        "top1_prob": float(np.max(probs)),
        "top10_mass": float(np.sort(probs)[::-1][:10].sum()),
        "top20_mass": float(np.sort(probs)[::-1][:20].sum()),
        "effective_support": float(1.0 / np.sum(probs ** 2)),
    }
    return rewards, probs, losses_norm, meta


def main():
    ap = argparse.ArgumentParser(
        description="Rewrite an enumerated reward table using a smooth thresholded reward."
    )
    ap.add_argument("--input_table", required=True)
    ap.add_argument("--output_table", required=True)
    ap.add_argument("--loss_norm", choices=["none", "q10q90"], default="q10q90")
    ap.add_argument("--q_low", type=float, default=0.10)
    ap.add_argument("--q_high", type=float, default=0.90)
    ap.add_argument("--tau_quantile", type=float, default=0.05,
                    help="Good-set threshold on normalized loss, e.g. 0.05 for top 5 percent lowest losses.")
    ap.add_argument("--temperature", type=float, default=0.05,
                    help="Transition sharpness. Smaller = sharper.")
    ap.add_argument("--epsilon", type=float, default=1e-3,
                    help="Reward floor for bad states.")
    ap.add_argument("--save_metadata", action="store_true")
    args = ap.parse_args()

    with open(args.input_table) as f:
        table = json.load(f)

    keys = list(table.keys())
    losses = np.array([float(table[k]["loss"]) for k in keys], dtype=float)

    rewards, probs, losses_norm, meta = build_thresholded_reward(
        losses=losses,
        loss_norm=args.loss_norm,
        q_low=args.q_low,
        q_high=args.q_high,
        tau_quantile=args.tau_quantile,
        temperature=args.temperature,
        epsilon=args.epsilon,
    )

    out = {}
    for i, k in enumerate(keys):
        row = dict(table[k])
        row["reward"] = float(rewards[i])
        row["reward_probability"] = float(probs[i])
        row["reward_loss"] = float(losses_norm[i])
        out[k] = row

    out_path = Path(args.output_table)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    if args.save_metadata:
        meta_path = out_path.with_name(out_path.stem + "_reward_calibration.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    print(f"Saved rewritten table to {out_path}")
    print("Thresholded reward settings:")
    for k, v in meta.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
