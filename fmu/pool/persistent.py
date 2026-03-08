"""Long-lived FMU worker pool for GFlowNet proxy.

Each team gets a persistent subprocess that stays alive across
proxy calls. Workers reinstantiate the FMU every max_uses calls
to avoid reset() corruption.
"""
import os
import subprocess
import sys
from fmu.pool.protocol import send, recv_timeout


_PERSISTENT_WORKER = """
import sys, os, pickle, struct
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

team = sys.argv[1]
fmu_path = sys.argv[2]
data_dir = sys.argv[3]
step_size = float(sys.argv[4])
max_uses = int(sys.argv[5])

from fmu.tomato_controller import TomatoController
from gflownet.envs.greenhouse.constants import BASELINE_PARAMETERS, INITIAL_CONDITIONS
from gflownet.proxy.greenhouse.cropSimulatorProxy import CropSimulatorProxy

team_data = CropSimulatorProxy.get_team_obs_dataset(data_dir, team)
input_trace = CropSimulatorProxy.compute_trace(
    CropSimulatorProxy.get_team_control_dataset(data_dir, team), delta='30min')
setpoints = (team_data.index - team_data.index.min())[1:].total_seconds().tolist()

def _send(obj):
    data = pickle.dumps(obj)
    sys.stdout.buffer.write(struct.pack('<I', len(data)))
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()

def _recv():
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

_send("READY")

while True:
    try:
        msg = _recv()
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

        _send(("OK", errors))
    except Exception as e:
        controller = make_controller()
        use_count = 0
        _send(("ERROR", str(e)))
"""


class PersistentFMUPool:
    """Pool of long-lived FMU worker subprocesses, one per team.

    Args:
        teams: list of team names
        fmu_path: path to .fmu file
        data_dir: path to greenhouse data
        step_size: FMU solver step size
        max_uses: reinstantiate FMU after this many simulate() calls
    """

    def __init__(self, teams, fmu_path, data_dir, step_size=120.0, max_uses=1):
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
                 "OMP_NUM_THREADS": "1",
                 "PYTHONPATH": os.pathsep.join(sys.path),},
            cwd=os.getcwd(),
        )
        try:
            ready = recv_timeout(p.stdout, timeout=30)
            assert ready == "READY"
        except Exception:
            stderr = p.stderr.read().decode()
            p.kill()
            p.wait()
            raise RuntimeError(f"Worker {team} failed to start: {stderr}")
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

    def evaluate(self, config, timeout=15):
        """Send config to all teams, collect per-team errors.
        Returns list of error lists, one per team that succeeded.
        Skips any team whose worker is dead or fails — no retries.
        """
        # Send to all live workers, skip dead ones
        live_teams = []
        for t, p in self.workers.items():
            if p.poll() is not None:
                print(f"Worker {t} already dead, skipping")
                continue
            try:
                send(p.stdin, config)
                live_teams.append(t)
            except (BrokenPipeError, OSError):
                print(f"Worker {t} pipe broken, skipping")

        # Collect results only from teams we successfully sent to
        team_losses = []
        for t in live_teams:
            p = self.workers[t]
            try:
                status, payload = recv_timeout(p.stdout, timeout=timeout)
                if status == "OK" and payload:
                    team_losses.append(payload)
                elif status == "ERROR":
                    print(f"FMU error for team {t}: {payload}")
            except TimeoutError:
                print(f"Worker {t} timed out, killing")
                p.kill()
            except EOFError:
                print(f"Worker {t} died mid-response")

        return team_losses

    def shutdown(self):
        for t, p in self.workers.items():
            try:
                send(p.stdin, "STOP")
                p.wait(timeout=5)
            except Exception:
                p.kill()
                p.wait()