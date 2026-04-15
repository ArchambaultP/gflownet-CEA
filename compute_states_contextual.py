
import argparse
import inspect
import itertools
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List

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


def compute_context_stats(loss_by_context: Dict[tuple, Dict[str, float]], team_ids: List[str], q_low: float, q_high: float):
    stats = {}
    for team in team_ids:
        vals = [float(ctx[team]) for ctx in loss_by_context.values() if team in ctx]
        if len(vals) == 0:
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

    if len(scores) == 0:
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

    context_stats = compute_context_stats(normalized_context, team_ids, q_low=q_low, q_high=q_high) if reward_mode == "context_topk_linear" else {}

    output = {}
    skipped = 0
    for combo, params in terminal_states.items():
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


def fake_evaluate_all(terminal_states, team_ids, seed=42):
    rng = np.random.default_rng(seed)
    losses = {}
    loss_by_context = {}
    for combo in terminal_states:
        if rng.random() < 0.05:
            continue
        ctx = {team: float(rng.uniform(0.2, 5.0)) for team in team_ids}
        loss_by_context[combo] = ctx
        losses[combo] = float(np.mean(list(ctx.values())))
    return losses, {
        "loss_by_context": loss_by_context,
        "abs_e_all": np.abs(rng.normal(size=1000)),
        "abs_e_dm": np.abs(rng.normal(size=500)),
        "abs_e_n": np.abs(rng.normal(size=500)),
        "abs_e_by_team": {team_ids[0]: np.abs(rng.normal(size=200))},
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



def _load_module_from_path(module_name: str, path: Path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _batch_evaluate_if_supported(states, args):
    """
    Load a batch evaluator WITHOUT importing the fmu.pool package, because that
    package __init__ may currently be broken by a bad PersistentFMUPool import.
    """
    candidates = []
    if getattr(args, "batch_module_path", None):
        candidates.append(Path(args.batch_module_path))
    candidates.extend([
        Path("batch_contextual.py"),
        Path("fmu/pool/batch_contextual.py"),
        Path("batch.py"),
        Path("fmu/pool/batch.py"),
    ])

    evaluate_all = None
    loaded_from = None
    for idx, path in enumerate(candidates):
        try:
            if path.exists():
                mod = _load_module_from_path(f"context_batch_mod_{idx}", path)
                if hasattr(mod, "evaluate_all"):
                    evaluate_all = mod.evaluate_all
                    loaded_from = path
                    break
        except Exception as e:
            print(f"  [INFO] Failed to load batch evaluator from {path}: {e}")

    if evaluate_all is None:
        print("  [INFO] No loadable batch evaluator found; falling back to direct evaluation.")
        return None

    sig = inspect.signature(evaluate_all)
    params = sig.parameters
    required_args = {
        "fmu_path",
        "team_ids",
        "data_dir",
        "loss_type",
        "huber_delta",
        "relative_floor_frac",
        "relative_floor_abs",
        "return_details",
    }
    if not required_args.issubset(params.keys()):
        print(
            f"  [INFO] Batch evaluator at {loaded_from} does not expose the required arguments; "
            "falling back to direct evaluation."
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
    print(f"  [INFO] Using contextual batch evaluator from {loaded_from}")
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
    loss_by_context = {}
    items = list(states.items())
    try:
        for idx, (combo, params) in enumerate(items, start=1):
            full_config = {**BASELINE_PARAMETERS, **INITIAL_CONDITIONS, **params}
            team_losses = pool.evaluate(full_config, timeout=args.timeout)

            ctx = {}
            for team, errs in zip(args.team_ids, team_losses):
                errs = list(errs)
                if errs:
                    ctx[str(team)] = float(np.mean(errs))

            loss_by_context[combo] = ctx
            losses[combo] = float(np.mean(list(ctx.values()))) if ctx else float(args.failure_loss)

            if idx == 1 or idx % args.log_every == 0 or idx == len(items):
                print(
                    f"  [{idx:>5}/{len(items)}] {combo} -> loss={losses[combo]:.6f} "
                    f"(contexts={len(ctx)}/{len(args.team_ids)})"
                )
    finally:
        pool.shutdown()

    details = {"loss_by_context": loss_by_context}
    return losses, details


def evaluate_losses(states, args):
    if DRY_RUN:
        return fake_evaluate_all(states, args.team_ids)

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


def parse_args():
    parser = argparse.ArgumentParser(description="Enumerate all terminal states and precompute scalar + per-context losses/rewards.")
    parser.add_argument("--step_fractions", type=float, nargs="+", default=[0.10, 0.15, 0.30])
    parser.add_argument("--start_from", choices=["midpoint", "baseline"], default="midpoint")
    parser.add_argument("--output_dir", type=str, default="precomputed_contextual")
    parser.add_argument("--output_template", type=str, default="reward_table_sf{sf}.json")

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
    parser.add_argument("--batch_module_path", type=str, default=None,
                        help="Explicit path to batch_contextual.py or batch.py to avoid importing the broken fmu.pool package.")
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--save_abs_errors_npz", action="store_true", help="Also save the raw exact |e| arrays as a compressed NPZ sidecar.")
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
    print(f"loss_type          : {args.loss_type}")
    print(f"reward_mode        : {args.reward_mode}")
    print(f"context_top_k      : {args.context_top_k}")
    print(f"context_tau        : {args.context_tau}")
    print(f"q_low/q_high       : {args.q_low}/{args.q_high}")
    print(f"prefer_batch       : {args.prefer_batch}")
    print()

    for sf in args.step_fractions:
        print(f"\n=== step_fraction = {sf} ===")
        states = generate_all_terminal_states(step_fraction=sf, start_from=args.start_from)
        print(f"  Generated {len(states):,} terminal states")

        losses, eval_details = evaluate_losses(states, args)
        print(f"  {len(losses)}/{len(states)} evaluations completed")

        loss_by_context = eval_details.get("loss_by_context", {}) if eval_details else {}

        all_losses = np.asarray(list(losses.values()), dtype=float)
        if all_losses.size == 0:
            raise RuntimeError("No losses were computed; cannot export reward table.")

        if args.beta_mode == "auto_median":
            L_median = float(np.median(all_losses))
            beta = float(5.65881 / max(L_median, 1e-12))
            print(f"  Median loss: {L_median:.6f}, auto beta: {beta:.6f}")
        else:
            beta = float(args.beta)
            print(f"  Using fixed beta: {beta:.6f}")

        out_name = args.output_template.format(sf=sf)
        out_path = Path(args.output_dir) / out_name
        export_results(
            states,
            losses,
            loss_by_context,
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
