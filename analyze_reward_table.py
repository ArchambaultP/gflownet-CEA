"""
Compute the exact Boltzmann distribution from the precomputed reward table.
Use this to compare against GFlowNet sample distributions.

Usage:
    python analyze_reward_table.py reward_table_sf0.15.json --beta 5.65881
"""

import json
import argparse
import numpy as np


def load_reward_table(path, beta):
    with open(path) as f:
        raw = json.load(f)

    states = []
    losses = []
    for key, entry in raw.items():
        states.append(key)
        losses.append(entry["loss"])

    losses = np.array(losses)
#    rewards = np.exp(-beta * (losses - losses.min()))
#    rewards = np.exp(-beta * losses)
    rewards = (1/losses) ** beta
    Z = rewards.sum()
    probs = rewards / Z

    return states, losses, rewards, probs, Z


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to reward_table JSON")
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--top_k", type=int, default=20)
    args = parser.parse_args()

    states, losses, rewards, probs, Z = load_reward_table(args.path, args.beta)

    # Sort by reward descending
    order = np.argsort(-rewards)

    print(f"Total states:     {len(states)}")
    print(f"Partition Z:      {Z:.6f}")
    print(f"Beta:             {args.beta:.4f}")
    print(f"Loss range:       [{losses.min():.4f}, {losses.max():.4f}]")
    print(f"Reward range:     [{rewards.min():.6f}, {rewards.max():.6f}]")
    print(f"States with R>0.5: {(rewards > 0.5).sum()}")
    print(f"States with R>0.1: {(rewards > 0.1).sum()}")
    print(f"States with R>0.01:{(rewards > 0.01).sum()}")

    # Effective number of modes (exponential of entropy)
    nonzero = probs[probs > 0]
    entropy = -np.sum(nonzero * np.log(nonzero))
    n_eff = np.exp(entropy)
    delta_H = np.log(len(states)) - entropy
    print(f"Entropy:          {entropy:.4f}")
    print(f"Effective modes:  {n_eff:.1f} ({100*n_eff/len(states):.2f}% of state-space)") # this should be between 0.1 - 1% ideally
    print(f"Entropy gap:      {delta_H:.4f}")

    # Top-k states
    print(f"\nTop {args.top_k} states:")
    print(f"{'Rank':<6}{'Loss':<12}{'Reward':<12}{'Prob':<12}{'Actions'}")
    print("-" * 80)
    for rank, idx in enumerate(order[:args.top_k]):
        print(f"{rank+1:<6}{losses[idx]:<12.4f}{rewards[idx]:<12.6f}{probs[idx]:<12.6f}{states[idx]}")

    # Cumulative probability of top-k
    top_k_prob = probs[order[:args.top_k]].sum()
    print(f"\nCumulative prob of top {args.top_k}: {top_k_prob:.4f}")

    # Mode clusters: group by first 3 actions (canopy + photosynthesis + temp_inhib)
    print(f"\nMode clusters (by first 3 group actions):")
    clusters = {}
    for idx in order:
        actions = states[idx].split("|")
        cluster_key = "|".join(actions[:3])
        if cluster_key not in clusters:
            clusters[cluster_key] = {"count": 0, "total_prob": 0.0, "best_loss": losses[idx]}
        clusters[cluster_key]["count"] += 1
        clusters[cluster_key]["total_prob"] += probs[idx]

    sorted_clusters = sorted(clusters.items(), key=lambda x: -x[1]["total_prob"])
    print(f"{'Cluster':<45}{'Count':<8}{'Prob':<12}{'Best Loss'}")
    print("-" * 80)
    for key, info in sorted_clusters[:15]:
        print(f"{key:<45}{info['count']:<8}{info['total_prob']:<12.4f}{info['best_loss']:.4f}")


if __name__ == "__main__":
    main()
