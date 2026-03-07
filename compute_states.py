import itertools
import json
import numpy as np
from multiprocessing import Pool
from gflownet.envs.greenhouse.constants import (
    BASELINE_PARAMETERS, GROUP_ORDER, PERTURBATION_SCHEME,
    INITIAL_CONDITIONS, PARAMETER_BOUNDS
)
from fmu.tomato_controller import TomatoController
from gflownet.proxy.greenhouse.cropSimulatorProxy import CropSimulatorProxy
# Adjust these imports to match your codebase
import os

DATA_DIR = "data/greenhouse/secondEdition"

def _init_worker(fmu_path, team_ids):
    """Each worker creates its own FMU instances."""
    global _controllers, _team_data, _inputs, _setpoints
    _controllers = {t: TomatoController('fmu/FMU/tomato.fmu', 
                                        start_time=0, # inital simulation time (in seconds). should not change
                                        stop_time=86400.0 * 200, # Final simulation time (in seconds).
                                        step_size=120.0, #numerical solver step size (in seconds)
                                        logger=None)
                    for t in team_ids}
    _team_data = {t: CropSimulatorProxy.get_team_obs_dataset(DATA_DIR, t) for t in team_ids}
    _inputs = {t:CropSimulatorProxy.compute_trace(
                    CropSimulatorProxy.get_team_control_dataset(DATA_DIR, t),
                      delta='30min') for t in team_ids}    
    _setpoints = {t:(_team_data[t].index - _team_data[t].index.min())[1:].total_seconds().tolist() for t in team_ids}


def _evaluate_single(args):
    import time
    """Evaluate one terminal state across all teams."""

    key, params = args

    print(f"Evaluating {key}")
    
    init = {**BASELINE_PARAMETERS, **INITIAL_CONDITIONS, **params}

    errors = []
    try:
        for t in _controllers:
            
            # print(f"last inp: {_inputs[-1]}")
            # time.sleep(20)
            
            sim_out = _controllers[t].simulate(
                _inputs[t], _setpoints[t], init_conds=init
            )

            for idx, (_, output) in enumerate(sim_out):
                y_DM = _team_data[t]["DM_harvest_obs"].iloc[idx]
                y_N = _team_data[t]["N_harvest_per_m2"].iloc[idx]
                y_hat_DM = output["C_harvest"]
                y_hat_N = output["N_harvest"]

                if y_DM > 0:
                    errors.append(((y_hat_DM - y_DM) / y_DM) ** 2)
                if y_N > 0:
                    errors.append(((y_hat_N - y_N) / y_N) ** 2)

        L = np.mean(errors) if errors else 1e6
    except Exception as e:
        print(f"Failed on {key}: {e}")
        L = 1e6

    return key, L


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


def evaluate_all(terminal_states, fmu_path, team_ids, n_workers=48):
    work = [("|".join(combo), params) for combo, params in terminal_states.items()]

    results = {}
    with Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(fmu_path, team_ids),
    ) as pool:
        for i, (key, loss) in enumerate(pool.imap_unordered(_evaluate_single, work)):
            results[key] = loss
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(work)} evaluated")

    return results


def export_results(terminal_states, losses, beta, path):
    output = {}
    for combo, params in terminal_states.items():
        key = "|".join(combo)
        L = losses[key]
        output[key] = {
            "params": {k: float(v) for k, v in params.items()},
            "loss": float(L),
            "reward": float(np.exp(-beta * L)),
        }

    with open(path, "w") as f:
        json.dump(output, f, indent=2)

    # Quick summary
    rewards = [v["reward"] for v in output.values()]
    print(f"  Saved {len(output)} states to {path}")
    print(f"  Reward range: [{min(rewards):.6f}, {max(rewards):.6f}]")
    print(f"  States with reward > 0.5: {sum(1 for r in rewards if r > 0.5)}")


if __name__ == "__main__":
    FMU_PATH = "fmu/FMU/tomato.fmu"
    TEAM_IDS = [
        "Reference",
        "Digilog",
        "IUACAAS",
        "Automatoes",
        "TheAutomators",
        "AICU"
        ]
    
    for sf in [0.10, 0.15, 0.30]:
        print(f"\n=== step_fraction = {sf} ===")

        states = generate_all_terminal_states(step_fraction=sf)
        print(f"  Generated {len(states):,} terminal states")

        cores = len(os.sched_getaffinity(0))
        losses = evaluate_all(states, FMU_PATH, TEAM_IDS, n_workers=cores)

        # Calibrate beta from median loss
        all_losses = list(losses.values())
        L_median = np.median(all_losses)
        beta = 4.6 / L_median
        print(f"  Median loss: {L_median:.4f}, beta: {beta:.4f}")

        export_results(states, losses, beta, f"reward_table_sf{sf}.json")