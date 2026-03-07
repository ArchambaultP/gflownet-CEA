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

def run_parallel(args_by_team, fmu_path, timeout=15, verbose=False, max_workers=3, work_dir=None):
    if work_dir is None:
        work_dir = os.path.dirname(os.path.abspath(fmu_path))
    tmp_dir = tempfile.mkdtemp(dir=work_dir)

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
                     "OMP_NUM_THREADS": "1",
                     "TMPDIR": tmp_dir},  # FMU extracts here too
            )
            procs[t] = (p, result_file)
            if verbose:
                print(f"Started FMU for team {t}")

        for t, (p, result_file) in procs.items():
            try:
                _, stderr = p.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                p.kill()
                _, stderr = p.communicate()
                if verbose:
                    print(f"Hard-killed FMU for team {t}")
                if stderr:
                    print(f"  stderr: {stderr.decode()}")
                continue

            if stderr and verbose:
                print(f"  [{t}] {stderr.decode()}")

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
import sys, pickle, os, traceback
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

fmu_path, args_file, result_file = sys.argv[1], sys.argv[2], sys.argv[3]

print(f"[worker] Loading args from {args_file}", file=sys.stderr, flush=True)
with open(args_file, 'rb') as f:
    input_trace, setpoints, init_conds, step_size = pickle.load(f)
print(f"[worker] Loaded {len(input_trace)} input points, {len(setpoints)} setpoints", file=sys.stderr, flush=True)

print(f"[worker] Importing TomatoController...", file=sys.stderr, flush=True)
from fmu.tomato_controller import TomatoController
print(f"[worker] Instantiating FMU from {fmu_path}", file=sys.stderr, flush=True)

try:
    controller = TomatoController(
        fmu_path,
        start_time=0.0,
        stop_time=input_trace[-1][0],
        step_size=step_size,
        logger=None,
    )
    print(f"[worker] FMU instantiated, starting simulate...", file=sys.stderr, flush=True)
    result = controller.simulate(input_trace, setpoints, init_conds=init_conds)
    print(f"[worker] Simulate done, {len(result)} outputs", file=sys.stderr, flush=True)

    with open(result_file, 'wb') as f:
        pickle.dump(result, f)
    print(f"[worker] Result saved", file=sys.stderr, flush=True)

except Exception:
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
"""