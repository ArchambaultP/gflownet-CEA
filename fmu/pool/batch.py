"""Batch evaluation of many parameter configs.

Used by compute_states.py to evaluate thousands of terminal states.
Each subprocess handles one (config, team) pair — no reset() needed.
"""
import os
import shutil
import pickle
import tempfile
import subprocess
import sys
import numpy as np


_SINGLE_EVAL_SCRIPT = """
import sys, pickle, os, numpy as np
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

args_file, result_file = sys.argv[1], sys.argv[2]

with open(args_file, 'rb') as f:
    key, params, fmu_path, team, data_dir = pickle.load(f)

from fmu.tomato_controller import TomatoController
from gflownet.envs.greenhouse.constants import BASELINE_PARAMETERS, INITIAL_CONDITIONS
from gflownet.proxy.greenhouse.cropSimulatorProxy import CropSimulatorProxy

team_data = CropSimulatorProxy.get_team_obs_dataset(data_dir, team)
input_trace = CropSimulatorProxy.compute_trace(
    CropSimulatorProxy.get_team_control_dataset(data_dir, team), delta='30min')
setpoints = (team_data.index - team_data.index.min())[1:].total_seconds().tolist()

init = {**BASELINE_PARAMETERS, **INITIAL_CONDITIONS, **params}

controller = TomatoController(
    fmu_path, start_time=0, stop_time=86400.0 * 200,
    step_size=120.0, logger=None)

sim_out = controller.simulate(input_trace, setpoints, init_conds=init)

errors = []
for idx, (_, output) in enumerate(sim_out):
    y_DM = team_data["DM_harvest_obs"].iloc[idx]
    y_N = team_data["N_harvest_per_m2"].iloc[idx]
    y_hat_DM = output["C_harvest"]
    y_hat_N = output["N_harvest"]
    if y_DM > 0:
        errors.append(((y_hat_DM - y_DM) / y_DM) ** 2)
    if y_N > 0:
        errors.append(((y_hat_N - y_N) / y_N) ** 2)

with open(result_file, 'wb') as f:
    pickle.dump((key, team, errors), f)
"""


def evaluate_all(terminal_states, fmu_path, team_ids, data_dir,
                 n_workers=48, timeout=120, verbose=False):
    """Evaluate many parameter configs across all teams.

    Spawns one subprocess per (config, team) pair. Each process
    creates a fresh FMU, runs one simulation, and exits.

    Args:
        terminal_states: {combo_tuple: params_dict}
        fmu_path: path to .fmu file
        team_ids: list of team names
        data_dir: path to greenhouse data
        n_workers: max concurrent subprocesses
        timeout: seconds per subprocess

    Returns:
        {combo_key: mean_loss} for all completed evaluations
    """
    work = [("|".join(combo), params) for combo, params in terminal_states.items()]
    jobs = [(key, params, team) for key, params in work for team in team_ids]

    tmp_dir = tempfile.mkdtemp()
    all_errors = {}
    total_done = 0

    for wave_start in range(0, len(jobs), n_workers):
        wave = jobs[wave_start:wave_start + n_workers]
        procs = []

        for i, (key, params, team) in enumerate(wave):
            idx = wave_start + i
            args_file = os.path.join(tmp_dir, f"args_{idx}.pkl")
            result_file = os.path.join(tmp_dir, f"result_{idx}.pkl")
            with open(args_file, 'wb') as f:
                pickle.dump((key, params, fmu_path, team, data_dir), f)

            p = subprocess.Popen(
                [sys.executable, "-c", _SINGLE_EVAL_SCRIPT,
                 args_file, result_file],
                env={**os.environ,
                     "OPENBLAS_NUM_THREADS": "1",
                     "MKL_NUM_THREADS": "1",
                     "OMP_NUM_THREADS": "1"},
                cwd=os.getcwd(),
            )
            procs.append((p, result_file))

        for p, result_file in procs:
            try:
                p.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
                continue

            if p.returncode != 0:
                continue

            try:
                with open(result_file, 'rb') as f:
                    key, team, errors = pickle.load(f)
                if key not in all_errors:
                    all_errors[key] = []
                all_errors[key].extend(errors)
                total_done += 1
            except Exception:
                pass

        if verbose:
            print(f"  {total_done}/{len(jobs)} team evaluations done")

    results = {}
    for key, errors in all_errors.items():
        results[key] = np.mean(errors) if errors else 1e6

    shutil.rmtree(tmp_dir, ignore_errors=True)
    return results