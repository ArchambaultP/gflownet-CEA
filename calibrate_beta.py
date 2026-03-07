import multiprocessing as mp
import numpy as np
from gflownet.envs.greenhouse.constants import (
    BASELINE_PARAMETERS, INITIAL_CONDITIONS, PARAMETER_BOUNDS
)
from gflownet.proxy.greenhouse.cropSimulatorProxy import CropSimulatorProxy
from fmu.fmu_pool import run_parallel
from fmu.tomato_controller import TomatoController
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_MAIN_FREE"] = "1"  # don't hold onto memory after use

TEAM_IDS = [
    "Reference",
    "Digilog",
    "IUACAAS",
    "Automatoes",
    "TheAutomators",
    "AICU",
]
DATA_DIR = "data/greenhouse/secondEdition"
FMU_PATH = "fmu/FMU/tomato.fmu"
STEP_SIZE = 120.0

def log_callback(instance_environment, instance_name, status, category, message):
    print(f"[FMU] {category}: {message}")

def main():
    midpoint = {p: (lo + hi) / 2.0 for p, (lo, hi) in PARAMETER_BOUNDS.items()}
    init = {**BASELINE_PARAMETERS, **INITIAL_CONDITIONS, **midpoint}

    args_by_team = {}
    team_obs = {}
    for t in TEAM_IDS:
        data = CropSimulatorProxy.get_team_control_dataset(DATA_DIR, t)
        input_trace = CropSimulatorProxy.compute_trace(data, delta="30min")
        obs = CropSimulatorProxy.get_team_obs_dataset(DATA_DIR, t)
        setpoints = (obs.index - obs.index.min())[1:].total_seconds().tolist()
        args_by_team[t] = (input_trace, setpoints, init, STEP_SIZE)
        team_obs[t] = obs

    results = run_parallel(args_by_team, FMU_PATH, timeout=30, verbose=True)

    team_losses = []
    for t, sim_out in results.items():
        obs = team_obs[t]
        team_errors = []
        for idx, (time_val, output) in enumerate(sim_out):
            y_DM = obs["DM_harvest_obs"].iloc[idx]
            y_N = obs["N_harvest_per_m2"].iloc[idx]
            y_hat_DM = output["C_harvest"]
            y_hat_N = output["N_harvest"]

            if y_DM > 0:
                team_errors.append(((y_hat_DM - y_DM) / y_DM) ** 2)
            if y_N > 0:
                team_errors.append(((y_hat_N - y_N) / y_N) ** 2)

        if team_errors:
            team_losses.append(np.mean(team_errors))
            print(f"  {t}: L={np.mean(team_errors):.4f} ({len(team_errors)} observations)")

    if not team_losses:
        print(team_losses)
        print("ERROR: No teams completed successfully")
    else:
        L_midpoint = np.mean(team_losses)
        beta = 4.6 / L_midpoint

        print(f"\nL at midpoint:  {L_midpoint:.4f}")
        print(f"R at midpoint:  {np.exp(-beta * L_midpoint):.4f}")
        print(f"beta:           {beta:.4f}")
        print(f"\nSave this: self.beta = {beta}")

        failed = set(TEAM_IDS) - set(results.keys())
        if failed:
            print(f"\nWARNING: These teams timed out or errored: {failed}")


if __name__ == '__main__':
    # mp.set_start_method("spawn", force=True)
    main()