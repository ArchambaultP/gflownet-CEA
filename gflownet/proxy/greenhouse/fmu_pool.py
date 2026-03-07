"""Parallel FMU execution with hard-kill timeouts."""
import os
import shutil
import pickle
import tempfile
import multiprocessing as mp
import traceback
import subprocess
import json
import sys

def run_parallel(args_by_team, fmu_path, timeout=15, verbose=False, max_workers=3):
    tmp_dir = tempfile.mkdtemp()

    # Pre-copy and pre-serialize
    team_meta = {}
    for i, (t, args) in enumerate(args_by_team.items()):
        local_fmu = os.path.join(tmp_dir, f"tomato_{i}.fmu")
        shutil.copy2(fmu_path, local_fmu)

        args_file = os.path.join(tmp_dir, f"args_{t}.pkl")
        with open(args_file, 'wb') as f:
            pickle.dump(args, f)

        result_file = os.path.join(tmp_dir, f"result_{t}.pkl")
        team_meta[t] = (local_fmu, args_file, result_file)

    all_teams = list(team_meta.items())
    results = {}

    for batch_start in range(0, len(all_teams), max_workers):
        batch = all_teams[batch_start:batch_start + max_workers]
        procs = {}

        for t, (local_fmu, args_file, result_file) in batch:
            p = subprocess.Popen(
                [sys.executable, "-c", _WORKER_SCRIPT, local_fmu, args_file, result_file],
                env={**os.environ,
                     "OPENBLAS_NUM_THREADS": "1",
                     "MKL_NUM_THREADS": "1",
                     "OMP_NUM_THREADS": "1"},
            )
            procs[t] = (p, result_file)
            if verbose:
                print(f"Started FMU for team {t}")

        for t, (p, result_file) in procs.items():
            try:
                p.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
                if verbose:
                    print(f"Hard-killed FMU for team {t}")
                continue

            if p.returncode != 0:
                print(f"FMU crashed for team {t}, exit code: {p.returncode}")
                continue

            try:
                with open(result_file, 'rb') as f:
                    results[t] = pickle.load(f)
                if verbose:
                    print(f"Finished FMU for team {t}")
            except Exception as e:
                print(f"Failed to read result for team {t}: {e}")

    shutil.rmtree(tmp_dir, ignore_errors=True)
    return results


# The child process is a completely fresh Python interpreter — no fork needed
_WORKER_SCRIPT = """
import sys, pickle, os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

fmu_path, args_file, result_file = sys.argv[1], sys.argv[2], sys.argv[3]

with open(args_file, 'rb') as f:
    input_trace, setpoints, init_conds, step_size = pickle.load(f)

from fmu.tomato_controller import TomatoController
controller = TomatoController(
    fmu_path,
    start_time=0.0,
    stop_time=input_trace[-1][0],
    step_size=step_size,
    logger=None,
)
result = controller.simulate(input_trace, setpoints, init_conds=init_conds)

with open(result_file, 'wb') as f:
    pickle.dump(result, f)
"""