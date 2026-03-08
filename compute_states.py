import itertools
import json
import os
import numpy as np
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


def export_results(terminal_states, losses, beta, path):
    output = {}
    for combo, params in terminal_states.items():
        breakpoint()
        L = losses[combo]
        output[combo] = {
            "params": {k: float(v) for k, v in params.items()},
            "loss": float(L),
            "reward": float(np.exp(-beta * L)),
        }

    with open(path, "w") as f:
        json.dump(output, f, indent=2)

    rewards = [v["reward"] for v in output.values()]
    print(f"  Saved {len(output)} states to {path}")
    print(f"  Reward range: [{min(rewards):.6f}, {max(rewards):.6f}]")
    print(f"  States with reward > 0.5: {sum(1 for r in rewards if r > 0.5)}")


# if __name__ == "__main__":
#     os.environ["OPENBLAS_NUM_THREADS"] = "1"
#     os.environ["MKL_NUM_THREADS"] = "1"
#     os.environ["OMP_NUM_THREADS"] = "1"

#     FMU_PATH = "fmu/FMU/tomato.fmu"
#     TEAM_IDS = [
#         "Reference",
#         "Digilog",
#         "IUACAAS",
#         "Automatoes",
#         "TheAutomators",
#         "AICU",
#     ]

#     cores = min(len(os.sched_getaffinity(0)), 48)

#     for sf in [0.10, 0.15, 0.30]:
#         print(f"\n=== step_fraction = {sf} ===")

#         states = generate_all_terminal_states(step_fraction=sf)

#         print(f"  Generated {len(states):,} terminal states")

#         losses = evaluate_all(
#             states, FMU_PATH, TEAM_IDS,
#             data_dir=DATA_DIR, 
#             n_workers=cores, 
#             verbose=True,
#             timeout=600,
#         )
#         print(f"  {len(losses)}/{len(states)} evaluations completed")

#         all_losses = list(losses.values())
#         L_median = np.median(all_losses)
#         beta = 5.65881 / L_median
#         print(f"  Median loss: {L_median:.4f}, beta: {beta:.4f}")

#         export_results(states, losses, beta, f"reward_table_sf{sf}.json")

if __name__ == "__main__":
    sf = 0.15
    states = generate_all_terminal_states(step_fraction=sf)
    print(f"Generated {len(states)} terminal states")

    # Fake losses: random values in a realistic-ish range
    rng = np.random.default_rng(42)
    losses = {combo: rng.uniform(0.5, 5.0) for combo in states}

    L_median = np.median(list(losses.values()))
    beta = 5.65881 / L_median

    out_path = f"reward_table_sf{sf}.json"
    export_results(states, losses, beta, out_path)

    # Quick verification: read it back
    with open(out_path) as f:
        data = json.load(f)
    print(f"\n  Verified: read back {len(data)} entries from {out_path}")