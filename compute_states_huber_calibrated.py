import argparse
import inspect
import itertools
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from gflownet.envs.greenhouse.constants import (
    BASELINE_PARAMETERS,
    GROUP_ORDER,
    INITIAL_CONDITIONS,
    PARAMETER_BOUNDS,
    PERTURBATION_SCHEME,
)

DRY_RUN = False

if not DRY_RUN:
    DATA_DIR = "data/greenhouse/secondEdition"
    FMU_PATH = "fmu/FMU/tomato.fmu"
    TEAM_IDS = [
        "Reference",
        "Digilog",
        "IUACAAS",
        "Automatoes",
        "TheAutomators",
        "AICU",
    ]


def generate_all_terminal_states(step_fraction: float, start_from: str = "midpoint"):
    modes_per_group = [list(PERTURBATION_SCHEME[group].keys()) for group in GROUP_ORDER]
    all_combos = list(itertools.product(*modes_per_group))

    terminal_states = {}
    for combo in all_combos:
        params = {}
        for p, (lo, hi) in PARAMETER_BOUNDS.items():
            if start_from == "midpoint":
                params[p] = (lo + hi) / 2.0
            else:
                params[p] = BASELINE_PARAMETERS.get(p, (lo + hi) / 2.0)

        for group, mode in zip(GROUP_ORDER, combo):
            directions = PERTURBATION_SCHEME[group][mode]
            for p, direction in directions.items():
                if direction == 0:
                    continue
                lo, hi = PARAMETER_BOUNDS[p]
                params[p] = np.clip(
                    params[p] + direction * step_fraction * (hi - lo),
                    lo,
                    hi,
                )

        terminal_states[combo] = params

    return terminal_states


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
            return tuple(key.split("|"))
        if "," in key:
            return tuple(s.strip() for s in key.split(","))
    return (key,)


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


def fit_loss_normalization(
    losses: Iterable[float],
    mode: str = "none",
    low_q: float = 0.10,
    high_q: float = 0.90,
    eps: float = 1e-8,
) -> dict:
    arr = np.asarray(list(losses), dtype=np.float64)
    if arr.size == 0:
        raise ValueError("No losses provided for normalization/calibration.")

    mode = str(mode).lower()
    stats = {
        "mode": mode,
        "low_q": float(low_q),
        "high_q": float(high_q),
        "eps": float(eps),
        "count": int(arr.size),
        "loss_min": float(np.min(arr)),
        "loss_max": float(np.max(arr)),
        "loss_mean": float(np.mean(arr)),
        "loss_std": float(np.std(arr)),
    }

    if mode == "none":
        stats.update(
            {
                "anchor": 0.0,
                "scale": 1.0,
                "q_low": float(np.quantile(arr, low_q)),
                "q_high": float(np.quantile(arr, high_q)),
            }
        )
        return stats

    if mode != "q10q90":
        raise ValueError(f"Unsupported loss_norm={mode!r}; expected 'none' or 'q10q90'.")

    q_low = float(np.quantile(arr, low_q))
    q_high = float(np.quantile(arr, high_q))
    scale = max(q_high - q_low, float(eps))
    stats.update({"anchor": q_low, "scale": scale, "q_low": q_low, "q_high": q_high})
    return stats


def transform_loss_for_reward(
    loss: float,
    norm_stats: dict,
) -> float:
    loss = float(loss)
    mode = norm_stats.get("mode", "none")
    if mode == "none":
        return loss
    if mode == "q10q90":
        return (loss - float(norm_stats["anchor"])) / max(float(norm_stats["scale"]), float(norm_stats.get("eps", 1e-8)))
    raise ValueError(f"Unsupported loss_norm mode={mode!r}")


def calibrate_beta(
    losses: Iterable[float],
    reward_transform: str,
    beta_mode: str,
    beta: float,
    beta_target_ratio: float,
    norm_stats: dict,
    reward_eps: float = 1e-8,
) -> tuple[float, dict]:
    arr = np.asarray(list(losses), dtype=np.float64)
    if arr.size == 0:
        raise ValueError("No losses available for beta calibration.")

    beta_mode = str(beta_mode).lower()
    reward_transform = str(reward_transform).lower()

    meta = {
        "beta_mode": beta_mode,
        "beta_input": float(beta),
        "beta_target_ratio": float(beta_target_ratio),
        "reward_transform": reward_transform,
    }

    if beta_mode == "fixed":
        meta["beta_reason"] = "user_fixed"
        return float(beta), meta

    if beta_mode == "auto_median":
        L_median = float(np.median(arr))
        resolved_beta = float(5.65881 / max(L_median, 1e-12))
        meta.update({"beta_reason": "legacy_auto_median", "median_loss": L_median})
        return resolved_beta, meta

    if beta_mode != "auto_contrast":
        raise ValueError(
            f"Unsupported beta_mode={beta_mode!r}; expected 'fixed', 'auto_median', or 'auto_contrast'."
        )

    target_ratio = max(float(beta_target_ratio), 1.0 + 1e-12)
    low = float(np.quantile(arr, float(norm_stats["low_q"])))
    high = float(np.quantile(arr, float(norm_stats["high_q"])))
    if reward_transform == "softmin":
        low_t = transform_loss_for_reward(low, norm_stats)
        high_t = transform_loss_for_reward(high, norm_stats)
        gap = max(high_t - low_t, 1e-12)
        resolved_beta = float(np.log(target_ratio) / gap)
        meta.update(
            {
                "beta_reason": "quantile_contrast",
                "loss_low": low,
                "loss_high": high,
                "reward_loss_low": low_t,
                "reward_loss_high": high_t,
                "contrast_gap": gap,
            }
        )
        return resolved_beta, meta

    if reward_transform == "invpower":
        low_safe = max(low, reward_eps)
        high_safe = max(high, reward_eps)
        gap = max(np.log(high_safe) - np.log(low_safe), 1e-12)
        resolved_beta = float(np.log(target_ratio) / gap)
        meta.update(
            {
                "beta_reason": "quantile_contrast_logloss",
                "loss_low": low_safe,
                "loss_high": high_safe,
                "contrast_gap": gap,
            }
        )
        return resolved_beta, meta

    raise ValueError(f"Unsupported reward_transform={reward_transform!r}")
def export_results(
    terminal_states,
    losses,
    beta,
    path,
    reward_transform="softmin",
    reward_eps=1e-8,
    reward_clip_min=1e-12,
    norm_stats=None,
):
    normalized_losses = {_normalize_key(k): v for k, v in losses.items()}
    norm_stats = norm_stats or {"mode": "none", "anchor": 0.0, "scale": 1.0, "eps": 1e-8}

    ts_sample = next(iter(terminal_states.keys()))
    loss_sample_orig = next(iter(losses.keys()))
    loss_sample_norm = next(iter(normalized_losses.keys()))
    print(
        f"  Key debug: terminal_states={ts_sample!r}, "
        f"losses_orig={loss_sample_orig!r}, "
        f"losses_normalized={loss_sample_norm!r}"
    )

    output = {}
    skipped = 0
    for combo, params in terminal_states.items():
        if combo not in normalized_losses:
            skipped += 1
            continue
        L_raw = float(normalized_losses[combo])
        L_reward = float(transform_loss_for_reward(L_raw, norm_stats))
        output["|".join(combo)] = {
            "params": {k: float(v) for k, v in params.items()},
            "loss": L_raw,
            "reward_loss": L_reward,
            "reward": loss_to_reward(
                L_reward,
                beta=beta,
                reward_transform=reward_transform,
                reward_eps=reward_eps,
                reward_clip_min=reward_clip_min,
            ),
        }

    with open(path, "w") as f:
        json.dump(output, f, indent=2)

    rewards = [v["reward"] for v in output.values()]
    reward_losses = np.asarray([v["reward_loss"] for v in output.values()], dtype=np.float64)
    print(f"  Saved {len(output)} states to {path}")
    if skipped:
        print(f"  Skipped {skipped}/{len(terminal_states)} states (no loss returned)")
    print(f"  Reward-loss range: [{reward_losses.min():.6g}, {reward_losses.max():.6g}]")
    print(f"  Reward range: [{min(rewards):.6g}, {max(rewards):.6g}]")
    print(f"  States with reward > 0.5: {sum(1 for r in rewards if r > 0.5)}")



def fake_evaluate_all(terminal_states, seed=42):
    rng = np.random.default_rng(seed)
    losses = {}
    for combo in terminal_states:
        if rng.random() < 0.05:
            continue
        losses[combo] = rng.uniform(0.3, 5.0)
    return losses, {
        "abs_e_all": np.abs(rng.normal(size=1000)),
        "abs_e_dm": np.abs(rng.normal(size=500)),
        "abs_e_n": np.abs(rng.normal(size=500)),
        "abs_e_by_team": {"Reference": np.abs(rng.normal(size=200))},
        "floors_by_team": {},
        "num_team_jobs_completed": 0,
        "num_expected_team_jobs": 0,
    }


def _mean_team_loss(team_losses: Iterable[Iterable[float]], failure_loss: float = 1e6) -> float:
    per_team = []
    for errs in team_losses:
        errs = list(errs)
        if errs:
            per_team.append(float(np.mean(errs)))
    if not per_team:
        return float(failure_loss)
    return float(np.mean(per_team))


def _batch_evaluate_if_supported(states, args):
    try:
        from fmu.pool.batch import evaluate_all
    except Exception:
        return None

    sig = inspect.signature(evaluate_all)
    params = sig.parameters
    required_new_args = {
        "fmu_path",
        "team_ids",
        "data_dir",
        "loss_type",
        "huber_delta",
        "relative_floor_frac",
        "relative_floor_abs",
        "return_details",
    }
    if not required_new_args.issubset(params.keys()):
        print(
            "  [INFO] fmu.pool.batch.evaluate_all does not expose the new Huber + stats arguments; "
            "falling back to direct PersistentFMUPool evaluation (losses only)."
        )
        return None

    call_kwargs = dict(
        states=states,
        fmu_path=args.fmu_path,
        team_ids=args.team_ids,
        data_dir=args.data_dir,
        n_workers=args.n_workers,
        verbose=True,
        timeout=args.timeout,
        loss_type=args.loss_type,
        huber_delta=args.huber_delta,
        relative_floor_frac=args.relative_floor_frac,
        relative_floor_abs=args.relative_floor_abs,
        return_details=True,
    )
    filtered_kwargs = {k: v for k, v in call_kwargs.items() if k in params}
    print("  [INFO] Using updated fmu.pool.batch.evaluate_all with Huber-aware loss + residual stats")
    return evaluate_all(**filtered_kwargs)


def _evaluate_direct(states, args):
    from fmu.pool import PersistentFMUPool

    pool = PersistentFMUPool(
        args.team_ids,
        args.fmu_path,
        args.data_dir,
        step_size=args.step_size,
        max_uses=args.max_uses,
        max_restarts=args.max_restarts,
        loss_type=args.loss_type,
        huber_delta=args.huber_delta,
        relative_floor_frac=args.relative_floor_frac,
        relative_floor_abs=args.relative_floor_abs,
    )

    losses = {}
    items = list(states.items())
    try:
        for idx, (combo, params) in enumerate(items, start=1):
            full_config = {**BASELINE_PARAMETERS, **INITIAL_CONDITIONS, **params}
            team_losses = pool.evaluate(full_config, timeout=args.timeout)
            losses[combo] = _mean_team_loss(team_losses, failure_loss=args.failure_loss)
            if idx == 1 or idx % args.log_every == 0 or idx == len(items):
                print(
                    f"  [{idx:>5}/{len(items)}] {combo} -> loss={losses[combo]:.6f} "
                    f"(successful teams={len(team_losses)}/{len(args.team_ids)})"
                )
    finally:
        pool.shutdown()

    return losses, None


def evaluate_losses(states, args):
    if DRY_RUN:
        return fake_evaluate_all(states)

    if args.prefer_batch:
        out = _batch_evaluate_if_supported(states, args)
        if out is not None:
            return out

    return _evaluate_direct(states, args)


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


def save_reward_calibration(prefix: Path, norm_stats: dict, beta: float, beta_meta: dict, args):
    out = {
        "reward_transform": args.reward_transform,
        "loss_norm": args.loss_norm,
        "loss_norm_low_q": float(args.loss_norm_low_q),
        "loss_norm_high_q": float(args.loss_norm_high_q),
        "loss_norm_eps": float(args.loss_norm_eps),
        "beta": float(beta),
        "beta_mode": args.beta_mode,
        "beta_target_ratio": float(args.beta_target_ratio),
        "normalization": norm_stats,
        "beta_meta": beta_meta,
    }
    path = prefix.with_name(prefix.name + "_reward_calibration.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved reward calibration to {path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Enumerate all terminal states and precompute Huber-aware losses/rewards.")
    parser.add_argument("--step_fractions", type=float, nargs="+", default=[0.10, 0.15, 0.30])
    parser.add_argument("--start_from", choices=["midpoint", "baseline"], default="midpoint")
    parser.add_argument("--output_dir", type=str, default="precomputed")
    parser.add_argument("--output_template", type=str, default="reward_table_sf{sf}.json")

    parser.add_argument("--reward_transform", choices=["softmin", "invpower"], default="softmin")
    parser.add_argument("--beta", type=float, default=10.0)
    parser.add_argument("--beta_mode", choices=["fixed", "auto_median", "auto_contrast"], default="fixed")
    parser.add_argument("--beta_target_ratio", type=float, default=20.0, help="For auto_contrast: desired reward ratio between low- and high-loss quantiles.")
    parser.add_argument("--reward_eps", type=float, default=1e-8)
    parser.add_argument("--reward_clip_min", type=float, default=1e-12)
    parser.add_argument("--loss_norm", choices=["none", "q10q90"], default="q10q90")
    parser.add_argument("--loss_norm_low_q", type=float, default=0.10)
    parser.add_argument("--loss_norm_high_q", type=float, default=0.90)
    parser.add_argument("--loss_norm_eps", type=float, default=1e-8)

    parser.add_argument("--loss_type", choices=["huber_relative", "rse", "absolute_relative"], default="huber_relative")
    parser.add_argument("--huber_delta", type=float, default=1.0)
    parser.add_argument("--relative_floor_frac", type=float, default=0.05)
    parser.add_argument("--relative_floor_abs", type=float, default=1e-6)

    parser.add_argument("--data_dir", type=str, default=DATA_DIR if not DRY_RUN else "data/greenhouse/secondEdition")
    parser.add_argument("--fmu_path", type=str, default=FMU_PATH if not DRY_RUN else "fmu/FMU/tomato.fmu")
    parser.add_argument("--team_ids", nargs="+", default=TEAM_IDS if not DRY_RUN else ["Reference"])
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--step_size", type=float, default=120.0)
    parser.add_argument("--max_uses", type=int, default=1)
    parser.add_argument("--max_restarts", type=int, default=3)
    parser.add_argument("--failure_loss", type=float, default=1e6)

    parser.add_argument("--no_prefer_batch", dest="prefer_batch", action="store_false", help="Skip the batch evaluator and force direct PersistentFMUPool evaluation.")
    parser.set_defaults(prefer_batch=True)
    parser.add_argument("--n_workers", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--save_abs_errors_npz", action="store_true", help="Also save the raw exact |e| arrays as a compressed NPZ sidecar.")
    return parser.parse_args()


def main():
    args = parse_args()

    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

    if args.reward_transform == "invpower" and args.loss_norm != "none":
        raise ValueError("loss_norm must be 'none' when reward_transform='invpower'. Use softmin for normalized losses.")
    if not (0.0 <= args.loss_norm_low_q < args.loss_norm_high_q <= 1.0):
        raise ValueError("Expected 0 <= loss_norm_low_q < loss_norm_high_q <= 1.")

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
    print(f"loss_type          : {args.loss_type}")
    print(f"huber_delta        : {args.huber_delta}")
    print(f"relative_floor_frac: {args.relative_floor_frac}")
    print(f"relative_floor_abs : {args.relative_floor_abs}")
    print(f"reward_transform   : {args.reward_transform}")
    print(f"loss_norm          : {args.loss_norm}")
    print(f"loss_norm_qs       : ({args.loss_norm_low_q}, {args.loss_norm_high_q})")
    print(f"beta_mode          : {args.beta_mode}")
    print(f"beta               : {args.beta}")
    print(f"beta_target_ratio  : {args.beta_target_ratio}")
    print(f"prefer_batch       : {args.prefer_batch}")
    print()

    for sf in args.step_fractions:
        print(f"\n=== step_fraction = {sf} ===")
        states = generate_all_terminal_states(step_fraction=sf, start_from=args.start_from)
        print(f"  Generated {len(states):,} terminal states")

        losses, eval_details = evaluate_losses(states, args)
        print(f"  {len(losses)}/{len(states)} evaluations completed")

        all_losses = np.asarray(list(losses.values()), dtype=float)
        if all_losses.size == 0:
            raise RuntimeError("No losses were computed; cannot export reward table.")

        norm_stats = fit_loss_normalization(
            all_losses,
            mode=args.loss_norm,
            low_q=args.loss_norm_low_q,
            high_q=args.loss_norm_high_q,
            eps=args.loss_norm_eps,
        )
        print(
            f"  Reward-loss normalization: mode={norm_stats['mode']}, "
            f"q_low={norm_stats['q_low']:.6f}, q_high={norm_stats['q_high']:.6f}, "
            f"anchor={norm_stats['anchor']:.6f}, scale={norm_stats['scale']:.6f}"
        )

        beta, beta_meta = calibrate_beta(
            all_losses,
            reward_transform=args.reward_transform,
            beta_mode=args.beta_mode,
            beta=args.beta,
            beta_target_ratio=args.beta_target_ratio,
            norm_stats=norm_stats,
            reward_eps=args.reward_eps,
        )
        if args.beta_mode == "fixed":
            print(f"  Using fixed beta: {beta:.6f}")
        else:
            print(f"  Resolved beta: {beta:.6f} ({beta_meta['beta_reason']})")

        out_name = args.output_template.format(sf=sf)
        out_path = Path(args.output_dir) / out_name
        export_results(
            states,
            losses,
            beta=beta,
            path=str(out_path),
            reward_transform=args.reward_transform,
            reward_eps=args.reward_eps,
            reward_clip_min=args.reward_clip_min,
            norm_stats=norm_stats,
        )
        save_abs_error_distribution(out_path.with_suffix(""), eval_details, args)
        save_reward_calibration(out_path.with_suffix(""), norm_stats, beta, beta_meta, args)


if __name__ == "__main__":
    main()
