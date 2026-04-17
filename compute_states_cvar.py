#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import numpy as np

from gflownet.envs.greenhouse.constants import (
    BASELINE_PARAMETERS,
    GROUP_ORDER,
    INITIAL_CONDITIONS,
    PARAMETER_BOUNDS,
    PERTURBATION_SCHEME,
)
from fmu.pool.batch_contextual import evaluate_all


DEFAULT_DATA_DIR = "data/greenhouse/secondEdition"
DEFAULT_FMU_PATH = "fmu/FMU/tomato.fmu"
DEFAULT_TEAM_IDS = [
    "Reference",
    "Digilog",
    "IUACAAS",
    "Automatoes",
    "TheAutomators",
    "AICU",
]


def iter_action_sequences(n_cycles: int) -> Iterator[Tuple[str, ...]]:
    spaces = []
    for _ in range(n_cycles):
        for group_name in GROUP_ORDER:
            spaces.append(list(PERTURBATION_SCHEME[group_name].keys()))
    yield from itertools.product(*spaces)


def count_terminal_states(n_cycles: int) -> int:
    total = 1
    for _ in range(n_cycles):
        for group_name in GROUP_ORDER:
            total *= len(PERTURBATION_SCHEME[group_name])
    return total


def make_initial_trainable_params(start_from: str) -> Dict[str, float]:
    params: Dict[str, float] = {}
    for p, (lo, hi) in PARAMETER_BOUNDS.items():
        if start_from == "midpoint":
            params[p] = float((lo + hi) / 2.0)
        elif start_from == "baseline":
            params[p] = float(BASELINE_PARAMETERS.get(p, (lo + hi) / 2.0))
        else:
            raise ValueError(f"Unsupported start_from={start_from!r}")
    return params


def apply_action(
    params: Dict[str, float],
    group_name: str,
    action_name: str,
    step_fraction: float,
) -> None:
    action = PERTURBATION_SCHEME[group_name][action_name]
    for param_name, direction in action.items():
        if direction == 0:
            continue
        lo, hi = PARAMETER_BOUNDS[param_name]
        new_val = params[param_name] + direction * step_fraction * (hi - lo)
        params[param_name] = float(np.clip(new_val, lo, hi))


def params_from_action_sequence(
    action_seq: Tuple[str, ...],
    step_fraction: float,
    decay_factor: float,
    n_cycles: int,
    start_from: str,
) -> Dict[str, float]:
    expected = len(GROUP_ORDER) * n_cycles
    if len(action_seq) != expected:
        raise ValueError(
            f"Action sequence length {len(action_seq)} != expected {expected} "
            f"for n_cycles={n_cycles}"
        )

    params = make_initial_trainable_params(start_from=start_from)

    idx = 0
    for cycle in range(n_cycles):
        cycle_step_fraction = float(step_fraction) * (float(decay_factor) ** cycle)
        for group_name in GROUP_ORDER:
            apply_action(
                params=params,
                group_name=group_name,
                action_name=action_seq[idx],
                step_fraction=cycle_step_fraction,
            )
            idx += 1

    return params


def generate_all_terminal_states(
    step_fraction: float,
    start_from: str,
    decay_factor: float,
    n_cycles: int,
    max_states: int | None = None,
) -> Dict[Tuple[str, ...], Dict[str, float]]:
    states: Dict[Tuple[str, ...], Dict[str, float]] = {}

    for idx, action_seq in enumerate(iter_action_sequences(n_cycles), start=1):
        if max_states is not None and idx > max_states:
            break
        states[action_seq] = params_from_action_sequence(
            action_seq=action_seq,
            step_fraction=step_fraction,
            decay_factor=decay_factor,
            n_cycles=n_cycles,
            start_from=start_from,
        )

    return states


def _normalize_key(key):
    if isinstance(key, tuple):
        return key
    if isinstance(key, str):
        if key.startswith("("):
            return tuple(
                s.strip().strip("'\"")
                for s in key.strip("()").split(",")
                if s.strip()
            )
        if "|" in key:
            return tuple(s.strip() for s in key.split("|") if s.strip())
        if "," in key:
            return tuple(s.strip() for s in key.split(",") if s.strip())
    return (str(key),)


def loss_to_reward(
    loss: float,
    beta: float,
    reward_transform: str = "softmin",
    reward_eps: float = 1e-8,
    reward_clip_min: float = 1e-12,
) -> float:
    loss = float(loss)
    if reward_transform == "softmin":
        reward = float(np.exp(-beta * loss))
    elif reward_transform == "invpower":
        reward = float(np.exp(-beta * np.log(max(loss, reward_eps))))
    else:
        raise ValueError(
            f"Unsupported reward_transform={reward_transform!r}; "
            "expected 'softmin' or 'invpower'"
        )
    return float(max(reward, reward_clip_min))


def compute_context_stats(
    loss_by_context: Dict[Tuple[str, ...], Dict[str, float]],
    team_ids: List[str],
    q_low: float,
    q_high: float,
) -> Dict[str, Dict[str, float]]:
    stats = {}
    for team in team_ids:
        vals = [float(ctx[team]) for ctx in loss_by_context.values() if team in ctx]
        if not vals:
            continue
        arr = np.asarray(vals, dtype=float)
        anchor = float(np.quantile(arr, q_low))
        scale = float(max(np.quantile(arr, q_high) - anchor, 1e-12))
        stats[team] = {"anchor": anchor, "scale": scale}
    return stats


def context_reward_topk_linear(
    loss_by_context_entry: Dict[str, float],
    context_stats: Dict[str, Dict[str, float]],
    team_ids: List[str],
    context_top_k: int,
    context_tau: float,
    reward_epsilon: float,
) -> float:
    scores = []

    for team in team_ids:
        if team not in loss_by_context_entry or team not in context_stats:
            continue
        anchor = float(context_stats[team]["anchor"])
        scale = max(float(context_stats[team]["scale"]), 1e-12)
        z = (float(loss_by_context_entry[team]) - anchor) / scale
        s = max(0.0, 1.0 - (z / max(float(context_tau), 1e-12)))
        scores.append(float(s))

    if not scores:
        return max(float(reward_epsilon), 1e-12)

    scores = sorted(scores, reverse=True)
    k = min(max(int(context_top_k), 1), len(scores))
    agg = float(np.mean(scores[:k]))
    reward = float(reward_epsilon + (1.0 - reward_epsilon) * agg)
    return max(reward, 1e-12)


def export_results(
    terminal_states,
    losses,
    loss_by_context,
    path,
    reward_mode="context_topk_linear",
    beta=10.0,
    reward_transform="softmin",
    reward_eps=1e-8,
    reward_clip_min=1e-12,
    team_ids=None,
    q_low=0.10,
    q_high=0.90,
    context_top_k=2,
    context_tau=0.20,
    reward_epsilon=1e-3,
):
    team_ids = list(team_ids or [])
    normalized_losses = {_normalize_key(k): float(v) for k, v in losses.items()}
    normalized_context = {
        _normalize_key(k): {str(team): float(v) for team, v in ctx.items()}
        for k, ctx in loss_by_context.items()
    }

    context_stats = (
        compute_context_stats(normalized_context, team_ids, q_low=q_low, q_high=q_high)
        if reward_mode == "context_topk_linear"
        else {}
    )

    output = {}
    skipped = 0
    for combo, params in terminal_states.items():
        combo = _normalize_key(combo)
        if combo not in normalized_losses:
            skipped += 1
            continue

        scalar_loss = float(normalized_losses[combo])
        ctx = normalized_context.get(combo, {})

        if reward_mode == "context_topk_linear":
            reward = context_reward_topk_linear(
                ctx,
                context_stats=context_stats,
                team_ids=team_ids,
                context_top_k=context_top_k,
                context_tau=context_tau,
                reward_epsilon=reward_epsilon,
            )
        else:
            reward = loss_to_reward(
                scalar_loss,
                beta=beta,
                reward_transform=reward_transform,
                reward_eps=reward_eps,
                reward_clip_min=reward_clip_min,
            )

        output["|".join(combo)] = {
            "params": {k: float(v) for k, v in params.items()},
            "loss": scalar_loss,
            "loss_by_context": {team: float(v) for team, v in ctx.items()},
            "reward": float(reward),
        }

    with open(path, "w") as f:
        json.dump(output, f, indent=2)

    rewards = [v["reward"] for v in output.values()]
    print(f"  Saved {len(output)} states to {path}")
    if skipped:
        print(f"  Skipped {skipped}/{len(terminal_states)} states (no loss returned)")
    if rewards:
        print(f"  Reward range: [{min(rewards):.6g}, {max(rewards):.6g}]")

    if reward_mode == "context_topk_linear":
        meta = {
            "reward_mode": reward_mode,
            "context_top_k": int(context_top_k),
            "context_tau": float(context_tau),
            "reward_epsilon": float(reward_epsilon),
            "loss_norm": "q10q90",
            "q_low": float(q_low),
            "q_high": float(q_high),
            "team_stats": context_stats,
        }
        meta_path = str(path).replace(".json", "_context_reward_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  Saved context reward metadata to {meta_path}")


def _arr_stats(values: np.ndarray):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "quantiles": {},
        }

    probs = [0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.995, 0.999]
    quant = {str(p): float(np.quantile(values, p)) for p in probs}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "quantiles": quant,
    }


def _histogram(values: np.ndarray):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"bin_edges": [], "counts": []}

    max_v = float(np.max(values))
    base_edges = np.array([0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0])
    edges = base_edges[base_edges < max_v]
    if edges.size == 0 or edges[0] != 0.0:
        edges = np.concatenate([[0.0], edges])
    if max_v <= edges[-1]:
        edges = np.concatenate([edges, [edges[-1] + 1e-9]])
    else:
        edges = np.concatenate([edges, [max_v]])

    counts, used_edges = np.histogram(values, bins=edges)
    return {
        "bin_edges": [float(x) for x in used_edges.tolist()],
        "counts": [int(x) for x in counts.tolist()],
    }


def save_abs_error_distribution(prefix: Path, details: dict, args):
    if not details:
        print("  [WARN] No residual details available; skipped empirical |e| export.")
        return

    abs_e_all = np.asarray(details.get("abs_e_all", []), dtype=np.float64)
    abs_e_dm = np.asarray(details.get("abs_e_dm", []), dtype=np.float64)
    abs_e_n = np.asarray(details.get("abs_e_n", []), dtype=np.float64)
    abs_e_by_team = {
        team: np.asarray(vals, dtype=np.float64)
        for team, vals in details.get("abs_e_by_team", {}).items()
    }

    stats = {
        "metric": "abs_normalized_residual",
        "definition": "|e| where e = (y_hat - y) / max(|y|, floor)",
        "loss_type": args.loss_type,
        "huber_delta": float(args.huber_delta),
        "relative_floor_frac": float(args.relative_floor_frac),
        "relative_floor_abs": float(args.relative_floor_abs),
        "num_team_jobs_completed": int(details.get("num_team_jobs_completed", 0)),
        "num_expected_team_jobs": int(details.get("num_expected_team_jobs", 0)),
        "overall": _arr_stats(abs_e_all),
        "by_channel": {
            "DM_harvest_obs": _arr_stats(abs_e_dm),
            "N_harvest_per_m2": _arr_stats(abs_e_n),
        },
        "by_team": {team: _arr_stats(vals) for team, vals in abs_e_by_team.items()},
        "floors_by_team": details.get("floors_by_team", {}),
        "overall_histogram": _histogram(abs_e_all),
        "delta_candidates": {
            "p50": float(np.quantile(abs_e_all, 0.50)) if abs_e_all.size else None,
            "p75": float(np.quantile(abs_e_all, 0.75)) if abs_e_all.size else None,
            "p90": float(np.quantile(abs_e_all, 0.90)) if abs_e_all.size else None,
            "p95": float(np.quantile(abs_e_all, 0.95)) if abs_e_all.size else None,
        },
    }

    stats_path = prefix.with_name(prefix.name + "_abs_e_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved empirical |e| stats to {stats_path}")

    if args.save_abs_errors_npz:
        arrays = {
            "abs_e_all": abs_e_all,
            "abs_e_dm": abs_e_dm,
            "abs_e_n": abs_e_n,
        }
        for team, vals in abs_e_by_team.items():
            arrays[f"abs_e_team_{team}"] = vals
        npz_path = prefix.with_name(prefix.name + "_abs_e_raw.npz")
        np.savez_compressed(npz_path, **arrays)
        print(f"  Saved raw empirical |e| arrays to {npz_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Enumerate all terminal states and precompute scalar + per-context losses/rewards in parallel."
    )
    parser.add_argument("--step_fractions", type=float, nargs="+", default=[0.15])
    parser.add_argument("--n_cycles", type=int, default=1)
    parser.add_argument("--decay_factor", type=float, default=0.5)
    parser.add_argument("--start_from", choices=["midpoint", "baseline"], default="midpoint")
    parser.add_argument("--output_dir", type=str, default="precomputed_contextual")
    parser.add_argument("--output_template", type=str, default="reward_table_sf{sf}_c{cycles}.json")
    parser.add_argument("--max_states", type=int, default=None)

    parser.add_argument("--reward_mode", choices=["context_topk_linear", "softmin", "invpower"], default="context_topk_linear")
    parser.add_argument("--beta", type=float, default=10.0)
    parser.add_argument("--beta_mode", choices=["fixed", "auto_median"], default="fixed")
    parser.add_argument("--reward_eps", type=float, default=1e-8)
    parser.add_argument("--reward_clip_min", type=float, default=1e-12)
    parser.add_argument("--reward_epsilon", type=float, default=1e-3)

    parser.add_argument("--context_top_k", type=int, default=2)
    parser.add_argument("--context_tau", type=float, default=0.20)
    parser.add_argument("--q_low", type=float, default=0.10)
    parser.add_argument("--q_high", type=float, default=0.90)

    parser.add_argument("--loss_type", choices=["huber_relative", "rse", "absolute_relative"], default="absolute_relative")
    parser.add_argument("--huber_delta", type=float, default=1.0)
    parser.add_argument("--relative_floor_frac", type=float, default=0.05)
    parser.add_argument("--relative_floor_abs", type=float, default=1e-6)

    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--fmu_path", type=str, default=DEFAULT_FMU_PATH)
    parser.add_argument("--team_ids", nargs="+", default=DEFAULT_TEAM_IDS)
    parser.add_argument("--n_workers", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--verbose_batch", action="store_true")
    parser.add_argument("--save_abs_errors_npz", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

    if args.n_workers is None:
        try:
            args.n_workers = min(len(os.sched_getaffinity(0)), 48)
        except Exception:
            args.n_workers = os.cpu_count() or 1

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Precomputing terminal-state losses / rewards")
    print("=" * 60)
    print(f"start_from         : {args.start_from}")
    print(f"n_cycles           : {args.n_cycles}")
    print(f"decay_factor       : {args.decay_factor}")
    print(f"loss_type          : {args.loss_type}")
    print(f"reward_mode        : {args.reward_mode}")
    print(f"n_workers          : {args.n_workers}")
    print(f"total_states(full) : {count_terminal_states(args.n_cycles):,}")
    if args.max_states is not None:
        print(f"max_states         : {args.max_states:,}")
    print()

    for sf in args.step_fractions:
        print(f"\n=== step_fraction = {sf} ===")

        states = generate_all_terminal_states(
            step_fraction=sf,
            start_from=args.start_from,
            decay_factor=args.decay_factor,
            n_cycles=args.n_cycles,
            max_states=args.max_states,
        )
        print(f"  Generated {len(states):,} terminal states")

        losses, eval_details = evaluate_all(
            states=states,
            fmu_path=args.fmu_path,
            team_ids=args.team_ids,
            data_dir=args.data_dir,
            n_workers=args.n_workers,
            timeout=args.timeout,
            verbose=args.verbose_batch,
            loss_type=args.loss_type,
            huber_delta=args.huber_delta,
            relative_floor_frac=args.relative_floor_frac,
            relative_floor_abs=args.relative_floor_abs,
            return_details=True,
        )
        print(f"  {len(losses)}/{len(states)} evaluations completed")

        loss_by_context = eval_details.get("loss_by_context", {}) if eval_details else {}

        all_losses = np.asarray(list(losses.values()), dtype=float)
        if all_losses.size == 0:
            raise RuntimeError("No losses were computed; cannot export reward table.")

        if args.beta_mode == "auto_median":
            median_loss = float(np.median(all_losses))
            beta = float(5.65881 / max(median_loss, 1e-12))
            print(f"  Median loss: {median_loss:.6f}, auto beta: {beta:.6f}")
        else:
            beta = float(args.beta)
            print(f"  Using fixed beta: {beta:.6f}")

        out_name = args.output_template.format(sf=sf, cycles=args.n_cycles)
        out_path = Path(args.output_dir) / out_name
        export_results(
            terminal_states=states,
            losses=losses,
            loss_by_context=loss_by_context,
            path=str(out_path),
            reward_mode=args.reward_mode,
            beta=beta,
            reward_transform=args.reward_mode if args.reward_mode in {"softmin", "invpower"} else "softmin",
            reward_eps=args.reward_eps,
            reward_clip_min=args.reward_clip_min,
            team_ids=args.team_ids,
            q_low=args.q_low,
            q_high=args.q_high,
            context_top_k=args.context_top_k,
            context_tau=args.context_tau,
            reward_epsilon=args.reward_epsilon,
        )
        save_abs_error_distribution(out_path.with_suffix(""), eval_details, args)


if __name__ == "__main__":
    main()
