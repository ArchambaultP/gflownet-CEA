"""Parallel FMU execution with hard-kill timeouts."""
import os
import shutil
import pickle
import tempfile
import struct
import subprocess
import sys
import select

# ─── Worker script for run_parallel (one team per worker) ───

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

from fmu.tomato_controller import TomatoController

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

except Exception:
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
"""


# ─── Worker script for evaluate_all (batch of states per worker) ───

_EVAL_WORKER_SCRIPT = """
import sys, pickle, os, numpy as np
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

args_file, result_file = sys.argv[1], sys.argv[2]

with open(args_file, 'rb') as f:
    work_items, fmu_path, team_ids, data_dir = pickle.load(f)

from fmu.tomato_controller import TomatoController
from gflownet.envs.greenhouse.constants import BASELINE_PARAMETERS, INITIAL_CONDITIONS
from gflownet.proxy.greenhouse.cropSimulatorProxy import CropSimulatorProxy

team_data = {t: CropSimulatorProxy.get_team_obs_dataset(data_dir, t) for t in team_ids}
inputs = {t: CropSimulatorProxy.compute_trace(
    CropSimulatorProxy.get_team_control_dataset(data_dir, t), delta='30min')
    for t in team_ids}
setpoints_map = {t: (team_data[t].index - team_data[t].index.min())[1:].total_seconds().tolist()
    for t in team_ids}

# One controller per team — created once, reused for this batch
controllers = {t: TomatoController(
    fmu_path, start_time=0, stop_time=86400.0 * 200,
    step_size=120.0, logger=None)
    for t in team_ids}

print(f"[worker] Initialized. {len(work_items)} items to evaluate.", file=sys.stderr, flush=True)

results = {}
for i, (key, params) in enumerate(work_items):
    init = {**BASELINE_PARAMETERS, **INITIAL_CONDITIONS, **params}
    errors = []
    try:
        for t in team_ids:
            sim_out = controllers[t].simulate(inputs[t], setpoints_map[t], init_conds=init)
            for idx, (_, output) in enumerate(sim_out):
                y_DM = team_data[t]["DM_harvest_obs"].iloc[idx]
                y_N = team_data[t]["N_harvest_per_m2"].iloc[idx]
                y_hat_DM = output["C_harvest"]
                y_hat_N = output["N_harvest"]
                if y_DM > 0:
                    errors.append(((y_hat_DM - y_DM) / y_DM) ** 2)
                if y_N > 0:
                    errors.append(((y_hat_N - y_N) / y_N) ** 2)
        L = np.mean(errors) if errors else 1e6
    except Exception as e:
        print(f"[worker] Failed on {key}: {e}", file=sys.stderr, flush=True)
        L = 1e6
    results[key] = L

    if (i + 1) % 5 == 0:
        print(f"[worker] {i+1}/{len(work_items)} done", file=sys.stderr, flush=True)

print(f"[worker] Finished all {len(work_items)} items", file=sys.stderr, flush=True)

with open(result_file, 'wb') as f:
    pickle.dump(results, f)
"""

def _recv_timeout(pipe, timeout=15):
    """Receive with timeout. Raises TimeoutError if no data arrives."""
    ready, _, _ = select.select([pipe], [], [], timeout)
    if not ready:
        raise TimeoutError("Worker did not respond in time")
    raw = pipe.read(4)
    if not raw:
        raise EOFError
    size = struct.unpack('<I', raw)[0]
    data = pipe.read(size)
    return pickle.loads(data)

# ─── Public API ───

def run_parallel(args_by_team, fmu_path, timeout=15, verbose=False, max_workers=3, work_dir=None):
    """Run one FMU simulation per team in parallel."""
    if work_dir is None:
        work_dir = os.path.dirname(os.path.abspath(fmu_path))
    tmp_dir = tempfile.mkdtemp(dir=work_dir)

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
                     "TMPDIR": tmp_dir},
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

def _send(pipe, obj):
    """Send a pickled object with length prefix."""
    data = pickle.dumps(obj)
    pipe.write(struct.pack('<I', len(data)))
    pipe.write(data)
    pipe.flush()

def _recv(pipe):
    """Receive a length-prefixed pickled object."""
    raw = pipe.read(4)
    if not raw:
        raise EOFError
    size = struct.unpack('<I', raw)[0]
    data = pipe.read(size)
    return pickle.loads(data)


_PERSISTENT_WORKER = """
import sys, os, pickle, struct, traceback
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

team, fmu_path, data_dir, step_size, max_uses = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), int(sys.argv[5])

from fmu.tomato_controller import TomatoController
from gflownet.envs.greenhouse.constants import BASELINE_PARAMETERS, INITIAL_CONDITIONS
from gflownet.proxy.greenhouse.cropSimulatorProxy import CropSimulatorProxy

team_data = CropSimulatorProxy.get_team_obs_dataset(data_dir, team)
input_trace = CropSimulatorProxy.compute_trace(
    CropSimulatorProxy.get_team_control_dataset(data_dir, team), delta='30min')
setpoints = (team_data.index - team_data.index.min())[1:].total_seconds().tolist()

def send(obj):
    data = pickle.dumps(obj)
    sys.stdout.buffer.write(struct.pack('<I', len(data)))
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()

def recv():
    raw = sys.stdin.buffer.read(4)
    if not raw:
        raise EOFError
    size = struct.unpack('<I', raw)[0]
    return pickle.loads(sys.stdin.buffer.read(size))

def make_controller():
    return TomatoController(
        fmu_path, start_time=0, stop_time=86400.0 * 200,
        step_size=step_size, logger=None)

controller = make_controller()
use_count = 0

# Signal ready
send("READY")

while True:
    try:
        msg = recv()
    except EOFError:
        break

    if msg == "STOP":
        break

    config = msg

    if use_count >= max_uses:
        controller = make_controller()
        use_count = 0

    try:
        sim_out = controller.simulate(input_trace, setpoints, init_conds=config)
        use_count += 1

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

        send(("OK", errors))
    except Exception as e:
        # Reinstantiate after error
        controller = make_controller()
        use_count = 0
        send(("ERROR", str(e)))
"""

class PersistentFMUPool:
    """Pool of long-lived FMU worker subprocesses, one per team."""

    def __init__(self, teams, fmu_path, data_dir, step_size=120.0, max_uses=3):
        self.teams = teams
        self.fmu_path = fmu_path
        self.data_dir = data_dir
        self.step_size = step_size
        self.max_uses = max_uses
        self.workers = {}

        for t in teams:
            self._start_worker(t)

    def _start_worker(self, team):
        p = subprocess.Popen(
            [sys.executable, "-c", _PERSISTENT_WORKER,
             team, self.fmu_path, self.data_dir,
             str(self.step_size), str(self.max_uses)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**os.environ,
                 "OPENBLAS_NUM_THREADS": "1",
                 "MKL_NUM_THREADS": "1",
                 "OMP_NUM_THREADS": "1"},
            cwd=os.getcwd(),
        )
        try:
            ready = _recv(p.stdout)
            assert ready == "READY"
        except Exception:
            stderr = p.stderr.read().decode()
            p.kill()
            p.wait()
            print(f"Worker {team} failed to start: {stderr}")
            raise
        self.workers[team] = p

    def _restart_worker(self, team):
        p = self.workers.get(team)
        if p:
            try:
                p.kill()
                p.wait()
            except Exception:
                pass
        print(f"Restarting worker for team {team}")
        self._start_worker(team)

    def evaluate(self, config, timeout=10):
        """Send config to all teams in parallel, collect errors."""
        dead = []
        for t, p in self.workers.items():
            if p.poll() is not None:
                dead.append(t)
                continue
            try:
                _send(p.stdin, config)
            except (BrokenPipeError, OSError):
                dead.append(t)

        for t in dead:
            try:
                self._restart_worker(t)
                _send(self.workers[t].stdin, config)
            except Exception as e:
                print(f"Failed to restart worker {t}: {e}")

        team_losses = []
        for t, p in self.workers.items():
            try:
                status, payload = _recv_timeout(p.stdout, timeout=timeout)
                if status == "OK" and payload:
                    team_losses.append(payload)
                elif status == "ERROR":
                    print(f"FMU error for team {t}: {payload}")
            except TimeoutError:
                print(f"Worker {t} timed out, restarting")
                try:
                    self._restart_worker(t)
                except Exception as e:
                    print(f"Failed to restart worker {t}: {e}")
            except EOFError:
                stderr = p.stderr.read().decode()
                print(f"Worker {t} died: {stderr}")
                try:
                    self._restart_worker(t)
                except Exception as e:
                    print(f"Failed to restart worker {t}: {e}")

        return team_losses

    def shutdown(self):
        for t, p in self.workers.items():
            try:
                _send(p.stdin, "STOP")
                p.wait(timeout=5)
            except Exception:
                p.kill()
                p.wait()

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

print(f"[{key}|{team}] {len(sim_out)} outputs, {len(setpoints)} setpoints, stop_time={input_trace[-1][0]}", file=sys.stderr, flush=True)
if sim_out:
    _, first_out = sim_out[0]
    print(f"  C_harvest={first_out.get('C_harvest')}, N_harvest={first_out.get('N_harvest')}", file=sys.stderr, flush=True)
else:
    print(f"  WARNING: empty sim_out!", file=sys.stderr, flush=True)

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

print(f"  {len(errors)} error terms, L={np.mean(errors) if errors else 'N/A'}", file=sys.stderr, flush=True)

with open(result_file, 'wb') as f:
    pickle.dump((key, team, errors), f)
"""


def evaluate_all(terminal_states, fmu_path, team_ids, data_dir,
                 n_workers=48, timeout=120, verbose=False):
    """One subprocess per (config, team). No reset() needed."""
    work = [("|".join(combo), params) for combo, params in terminal_states.items()]

    # Build all (key, team) jobs
    jobs = [(key, params, team) for key, params in work for team in team_ids]

    tmp_dir = tempfile.mkdtemp()
    all_errors = {}  # key -> list of errors across teams
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
                [sys.executable, "-c", _SINGLE_EVAL_SCRIPT, args_file, result_file],
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

        if verbose and total_done > 0:
            print(f"  {total_done}/{len(jobs)} team evaluations done")

    # Aggregate: mean error per key -> reward-style loss
    results = {}
    for key, errors in all_errors.items():
        results[key] = np.mean(errors) if errors else 1e6

    shutil.rmtree(tmp_dir, ignore_errors=True)
    return results