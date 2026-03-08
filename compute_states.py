import itertools
import json
import os
import numpy as np
from gflownet.envs.greenhouse.constants import BASELINE_PARAMETERS, PARAMETER_BOUNDS, INITIAL_CONDITIONS, GROUP_ORDER, PERTURBATION_SCHEME
# ── Toggle: set to True to run with dummy data (no FMU/torch needed) ──
DRY_RUN = True

if not DRY_RUN:
    from gflownet.envs.greenhouse.constants import (
        BASELINE_PARAMETERS, GROUP_ORDER, PERTURBATION_SCHEME,
        INITIAL_CONDITIONS, PARAMETER_BOUNDS
    )
    from fmu.pool.batch import evaluate_all
    DATA_DIR = "data/greenhouse/secondEdition"



def generate_all_terminal_states(step_fraction, start_from="midpoint"):
    modes_per_group = [
        list(PERTURBATION_SCHEME[group].keys())
        for group in GROUP_ORDER
    ]
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
                    lo, hi
                )

        terminal_states[combo] = params

    return terminal_states


def _normalize_key(key):
    """Convert a losses key to a tuple, regardless of input format."""
    if isinstance(key, tuple):
        return key
    if isinstance(key, str):
        # Handle "('a', 'b', 'c')" format
        if key.startswith("("):
            return tuple(
                s.strip().strip("'\"")
                for s in key.strip("()").split(",")
                if s.strip()
            )
        # Handle "a|b|c" format
        if "|" in key:
            return tuple(key.split("|"))
        # Handle "a,b,c" format
        if "," in key:
            return tuple(s.strip() for s in key.split(","))
    return (key,)


def export_results(terminal_states, losses, beta, path):
    # Normalize all loss keys to tuples for matching
    normalized_losses = {_normalize_key(k): v for k, v in losses.items()}

    # Debug: show key formats
    ts_sample = next(iter(terminal_states.keys()))
    loss_sample_orig = next(iter(losses.keys()))
    loss_sample_norm = next(iter(normalized_losses.keys()))
    print(f"  Key debug: terminal_states={ts_sample!r}, "
          f"losses_orig={loss_sample_orig!r}, "
          f"losses_normalized={loss_sample_norm!r}")

    output = {}
    skipped = 0
    for combo, params in terminal_states.items():
        if combo not in normalized_losses:
            skipped += 1
            continue
        L = normalized_losses[combo]
        output["|".join(combo)] = {
            "params": {k: float(v) for k, v in params.items()},
            "loss": float(L),
            "reward": float(np.exp(-beta * L)),
        }

    with open(path, "w") as f:
        json.dump(output, f, indent=2)

    rewards = [v["reward"] for v in output.values()]
    print(f"  Saved {len(output)} states to {path}")
    if skipped:
        print(f"  Skipped {skipped}/{len(terminal_states)} states (no loss returned)")
    print(f"  Reward range: [{min(rewards):.6f}, {max(rewards):.6f}]")
    print(f"  States with reward > 0.5: {sum(1 for r in rewards if r > 0.5)}")


def fake_evaluate_all(terminal_states, seed=42):
    """Generate fake losses for local testing. Drops ~5% of states to simulate timeouts."""
    rng = np.random.default_rng(seed)
    losses = {}
    for combo in terminal_states:
        if rng.random() < 0.05:
            continue  # simulate a failed evaluation
        losses[combo] = rng.uniform(0.3, 5.0)
    return losses


if __name__ == "__main__":
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

    if not DRY_RUN:
        FMU_PATH = "fmu/FMU/tomato.fmu"
        TEAM_IDS = [
            "Reference",
            "Digilog",
            "IUACAAS",
            "Automatoes",
            "TheAutomators",
            "AICU",
        ]
        cores = min(len(os.sched_getaffinity(0)), 48)

    for sf in [0.10, 0.15, 0.30]:
        print(f"\n=== step_fraction = {sf} ===")
        states = generate_all_terminal_states(step_fraction=sf)
        print(f"  Generated {len(states):,} terminal states")

        if DRY_RUN:
            losses = fake_evaluate_all(states)
        else:
            losses = evaluate_all(
                states, FMU_PATH, TEAM_IDS,
                data_dir=DATA_DIR,
                n_workers=cores,
                verbose=True,
                timeout=600,
            )

        print(f"  {len(losses)}/{len(states)} evaluations completed")

        all_losses = list(losses.values())
        L_median = np.median(all_losses)
        beta = 5.65881 / L_median
        print(f"  Median loss: {L_median:.4f}, beta: {beta:.4f}")

        export_results(states, losses, beta, f"reward_table_sf{sf}.json")