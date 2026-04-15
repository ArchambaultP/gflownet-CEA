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
            f"Unsupported reward_transform={reward_transform!r}; expected 'softmin' or 'invpower'"
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


def _atomic_json_dump(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    tmp.replace(path)


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

    _atomic_json_dump(output, Path(path))

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
            "n_states_exported": int(len(output)),
        }
        meta_path = str(path).replace(".json", "_context_reward_meta.json")
        _atomic_json_dump(meta, Path(meta_path))
        print(f"  Saved context reward metadata to {meta_path}")


def load_existing_output(path: Path):
    if not path.exists():
        return {}, {}
    with open(path) as f:
        raw = json.load(f)
    losses = {}
    loss_by_context = {}
    for key, entry in raw.items():
        combo = _normalize_key(key)
        if isinstance(entry, dict) and "loss" in entry:
            losses[combo] = float(entry["loss"])
            loss_by_context[combo] = {str(k): float(v) for k, v in entry.get("loss_by_context", {}).items()}
    return losses, loss_by_context


def _batch_evaluate_if_supported(states, args):
    source = None
    evaluate_all = None
    try:
        from fmu.pool.batch_contextual import evaluate_all as _eval
        evaluate_all = _eval
        source = "fmu.pool.batch_contextual"
    except Exception:
        if args.allow_scalar_batch_fallback:
            try:
                from fmu.pool.batch import evaluate_all as _eval
                evaluate_all = _eval
                source = "fmu.pool.batch"
            except Exception:
                return None
        else:
            raise RuntimeError(
                "Could not import fmu.pool.batch_contextual. Refusing to silently fall back to scalar batch.py because context_topk_linear needs loss_by_context."
            )

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
        print("  [INFO] batch evaluator does not expose the required arguments; falling back to direct evaluation.")
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
    print(f"  [INFO] Using batch evaluator: {source}")
    out = evaluate_all(**filtered_kwargs)
    if not isinstance(out, tuple) or len(out) != 2:
        raise RuntimeError(f"{source} did not return (losses, details) with return_details=True")
    losses, details = out
    if args.reward_mode == "context_topk_linear":
        lbc = details.get("loss_by_context", {}) if isinstance(details, dict) else {}
        nonempty = sum(1 for v in lbc.values() if v)
        print(f"  [INFO] loss_by_context nonempty states: {nonempty}/{len(losses)}")
        if nonempty == 0:
            raise RuntimeError(
                f"{source} returned no usable loss_by_context entries. This means you are not actually getting contextual losses."
            )
    return losses, details


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
                    f"  [{idx:>5}/{len(items)}] {combo} -> loss={losses[combo]:.6f} (contexts={len(ctx)}/{len(args.team_ids)})"
                )
    finally:
        pool.shutdown()

    details = {"loss_by_context": loss_by_context}
    return losses, details


def evaluate_losses(states, args):
    if args.prefer_batch:
        out = _batch_evaluate_if_supported(states, args)
        if out is not None:
            return out
    return _evaluate_direct(states, args)


def chunked_items(d: Dict[Tuple[str, ...], dict], chunk_size: int):
    items = list(d.items())
    for i in range(0, len(items), chunk_size):
        yield dict(items[i:i + chunk_size])


def parse_args():
    parser = argparse.ArgumentParser(description="Enumerate all terminal states and precompute scalar + per-context losses/rewards with incremental checkpointing.")
    parser.add_argument("--step_fractions", type=float, nargs="+", default=[0.10, 0.15, 0.30])
    parser.add_argument("--start_from", choices=["midpoint", "baseline"], default="midpoint")
    parser.add_argument("--output_dir", type=str, default="precomputed_contextual")
    parser.add_argument("--output_template", type=str, default="reward_table_sf{sf}.json")

    parser.add_argument("--reward_mode", choices=["context_topk_linear", "softmin", "invpower"], default="context_topk_linear")
    parser.add_argument("--beta", type=float, default=10.0)
    parser.add_argument("--reward_transform", choices=["softmin", "invpower"], default="softmin")
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

    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    parser.add_argument("--fmu_path", type=str, default=FMU_PATH)
    parser.add_argument("--team_ids", nargs="+", default=TEAM_IDS)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--step_size", type=float, default=120.0)
    parser.add_argument("--max_uses", type=int, default=1)
    parser.add_argument("--max_restarts", type=int, default=3)
    parser.add_argument("--failure_loss", type=float, default=1e6)

    parser.add_argument("--no_prefer_batch", dest="prefer_batch", action="store_false")
    parser.set_defaults(prefer_batch=True)
    parser.add_argument("--allow_scalar_batch_fallback", action="store_true", help="Allow falling back to fmu.pool.batch if batch_contextual cannot be imported.")
    parser.add_argument("--n_workers", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=25)

    parser.add_argument("--checkpoint_every_states", type=int, default=100, help="Evaluate and save after each chunk of this many terminal states.")
    parser.add_argument("--no_resume", dest="resume", action="store_false", help="Do not resume from an existing output JSON.")
    parser.set_defaults(resume=True)
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
    print("Precomputing terminal-state losses / rewards (incremental)")
    print("=" * 60)
    print(f"start_from             : {args.start_from}")
    print(f"loss_type              : {args.loss_type}")
    print(f"reward_mode            : {args.reward_mode}")
    print(f"context_top_k          : {args.context_top_k}")
    print(f"context_tau            : {args.context_tau}")
    print(f"q_low/q_high           : {args.q_low}/{args.q_high}")
    print(f"prefer_batch           : {args.prefer_batch}")
    print(f"checkpoint_every_states: {args.checkpoint_every_states}")
    print(f"resume                 : {args.resume}")
    print()

    for sf in args.step_fractions:
        print(f"\n=== step_fraction = {sf} ===")
        states = generate_all_terminal_states(step_fraction=sf, start_from=args.start_from)
        total_states = len(states)
        print(f"  Generated {total_states:,} terminal states")

        out_name = args.output_template.format(sf=sf)
        out_path = Path(args.output_dir) / out_name
        progress_path = out_path.with_name(out_path.stem + "_progress.json")

        cumulative_losses: Dict[tuple, float] = {}
        cumulative_loss_by_context: Dict[tuple, Dict[str, float]] = {}
        if args.resume and out_path.exists():
            cumulative_losses, cumulative_loss_by_context = load_existing_output(out_path)
            print(f"  Resuming from {out_path}: {len(cumulative_losses)}/{total_states} states already saved")

        remaining_states = {combo: params for combo, params in states.items() if combo not in cumulative_losses}
        if len(remaining_states) == 0:
            print("  Nothing left to compute; keeping existing output.")
            continue

        chunk_idx = 0
        for chunk in chunked_items(remaining_states, args.checkpoint_every_states):
            chunk_idx += 1
            chunk_first = next(iter(chunk.keys()))
            print(f"  [chunk {chunk_idx}] evaluating {len(chunk)} states (first={chunk_first})")
            chunk_losses, chunk_details = evaluate_losses(chunk, args)
            chunk_lbc = chunk_details.get("loss_by_context", {}) if isinstance(chunk_details, dict) else {}

            for combo, loss in chunk_losses.items():
                combo_n = _normalize_key(combo)
                cumulative_losses[combo_n] = float(loss)
                cumulative_loss_by_context[combo_n] = {
                    str(k): float(v) for k, v in chunk_lbc.get(combo, chunk_lbc.get(combo_n, {})).items()
                }

            completed_states = {combo: params for combo, params in states.items() if combo in cumulative_losses}
            export_results(
                completed_states,
                cumulative_losses,
                cumulative_loss_by_context,
                path=str(out_path),
                reward_mode=args.reward_mode,
                beta=args.beta,
                reward_transform=args.reward_transform,
                reward_eps=args.reward_eps,
                reward_clip_min=args.reward_clip_min,
                team_ids=args.team_ids,
                q_low=args.q_low,
                q_high=args.q_high,
                context_top_k=args.context_top_k,
                context_tau=args.context_tau,
                reward_epsilon=args.reward_epsilon,
            )

            nonempty_ctx = sum(1 for v in cumulative_loss_by_context.values() if v)
            progress = {
                "step_fraction": sf,
                "completed_states": int(len(cumulative_losses)),
                "total_states": int(total_states),
                "remaining_states": int(total_states - len(cumulative_losses)),
                "nonempty_loss_by_context_states": int(nonempty_ctx),
                "output_path": str(out_path),
            }
            _atomic_json_dump(progress, progress_path)
            print(
                f"  [chunk {chunk_idx}] checkpoint saved: {len(cumulative_losses)}/{total_states} states, "
                f"loss_by_context nonempty={nonempty_ctx}"
            )

        print(f"  Finished step_fraction={sf}. Final output: {out_path}")


if __name__ == "__main__":
    main()
