"""Long-lived FMU worker pool for GFlowNet proxy.

Each team gets a persistent subprocess that stays alive across
proxy calls. Workers reinstantiate the FMU every max_uses calls
to avoid reset() corruption.
"""
import os
import subprocess
import sys
import threading
from fmu.pool.protocol import send, recv_timeout


_PERSISTENT_WORKER = r'''
import sys, os, pickle, struct, traceback, signal, select, time

# ── Protect the protocol pipe from FMU C-level stdout writes ──
_PROTO_FD = os.dup(sys.stdout.fileno())   # copy of real stdout
os.dup2(sys.stderr.fileno(), 1)           # fd 1 now → stderr
_proto_out = os.fdopen(_PROTO_FD, 'wb', buffering=0)

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

team = sys.argv[1]
fmu_path = sys.argv[2]
data_dir = sys.argv[3]
step_size = float(sys.argv[4])
max_uses = int(sys.argv[5])
loss_type = sys.argv[6]
huber_delta = float(sys.argv[7])
relative_floor_frac = float(sys.argv[8])
relative_floor_abs = float(sys.argv[9])

# NOTE: We intentionally do NOT import TomatoController here.
# It loads native .so/.dll code with global state that corrupts
# across fork(). Only the child process imports it (post-fork).
from gflownet.envs.greenhouse.constants import BASELINE_PARAMETERS, INITIAL_CONDITIONS
from gflownet.proxy.greenhouse.cropSimulatorProxy import CropSimulatorProxy

# ── Expensive one-time setup (survives child segfaults) ──
team_data = CropSimulatorProxy.get_team_obs_dataset(data_dir, team)
input_trace = CropSimulatorProxy.compute_trace(
    CropSimulatorProxy.get_team_control_dataset(data_dir, team), delta='30min')
setpoints = (team_data.index - team_data.index.min())[1:].total_seconds().tolist()

# Pre-extract plain Python lists for the child (no pandas after fork)
_team_DM = team_data["DM_harvest_obs"].tolist()
_team_N  = team_data["N_harvest_per_m2"].tolist()

# Precompute denominator floors from the observed scale
def _series_floor(values):
    import numpy as np
    arr = np.asarray(values, dtype=float)
    arr = np.abs(arr[np.isfinite(arr)])
    arr = arr[arr > 0]
    if arr.size == 0:
        scale = 1.0
    else:
        scale = float(np.percentile(arr, 90))
    return max(relative_floor_abs, relative_floor_frac * scale)

_DM_floor = _series_floor(_team_DM)
_N_floor = _series_floor(_team_N)

# Pre-pickle heavy objects so the child can deserialize a clean copy
# instead of relying on fork'd pandas internals
_input_trace_bytes = pickle.dumps(input_trace)

def _send(obj):
    data = pickle.dumps(obj)
    _proto_out.write(struct.pack('<I', len(data)))
    _proto_out.write(data)
    _proto_out.flush()

def _recv():
    raw = sys.stdin.buffer.read(4)
    if not raw:
        raise EOFError
    size = struct.unpack('<I', raw)[0]
    return pickle.loads(sys.stdin.buffer.read(size))

def _pipe_read_all(fd, size):
    """Read exactly `size` bytes from a file descriptor."""
    chunks = []
    remaining = size
    while remaining > 0:
        chunk = os.read(fd, remaining)
        if not chunk:
            raise EOFError("child pipe closed prematurely")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b''.join(chunks)

def _as_scalar(v):
    """Extract a float from FMU output (may be list, array, or scalar)."""
    if isinstance(v, (list, tuple)):
        return float(v[-1]) if v else 0.0
    if hasattr(v, '__len__'):
        return float(v[-1]) if len(v) > 0 else 0.0
    return float(v)

def _huber(e):
    ae = abs(float(e))
    if ae <= huber_delta:
        return 0.5 * ae * ae
    return huber_delta * (ae - 0.5 * huber_delta)

def _point_loss(y_hat, y, floor):
    denom = max(abs(float(y)), floor)
    e = (float(y_hat) - float(y)) / denom
    if loss_type == 'huber_relative':
        return _huber(e)
    if loss_type == 'absolute_relative':
        return abs(e)
    return e * e

def run_in_fork(config, timeout=60):
    """Fork a child to run simulate(). Parent survives segfaults.

    The child imports TomatoController fresh (no inherited .so state)
    and deserializes input_trace from bytes (no shared pandas memory).
    """
    r_fd, w_fd = os.pipe()
    pid = os.fork()

    if pid == 0:
        os.close(r_fd)
        try:
            from fmu.tomato_controller import TomatoController

            child_input_trace = pickle.loads(_input_trace_bytes)

            controller = TomatoController(
                fmu_path, start_time=0, stop_time=86400.0 * 200,
                step_size=step_size, logger=None)
            sim_out = controller.simulate(
                child_input_trace, setpoints, init_conds=config)

            errors = []
            for idx, (_, output) in enumerate(sim_out):
                y_DM = _team_DM[idx]
                y_N  = _team_N[idx]
                y_hat_DM = _as_scalar(output["C_harvest"])
                y_hat_N  = _as_scalar(output["N_harvest"])
                if y_DM > 0:
                    errors.append(_point_loss(y_hat_DM, y_DM, _DM_floor))
                if y_N > 0:
                    errors.append(_point_loss(y_hat_N, y_N, _N_floor))

            result = ("OK", errors)
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
            result = ("ERROR", f"{type(e).__name__}: {e}")

        try:
            data = pickle.dumps(result)
            os.write(w_fd, struct.pack('<I', len(data)))
            os.write(w_fd, data)
            controller.close()
        except Exception:
            pass
        os.close(w_fd)
        os._exit(0)

    else:
        os.close(w_fd)
        try:
            deadline = time.monotonic() + timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    os.kill(pid, signal.SIGKILL)
                    os.waitpid(pid, 0)
                    os.close(r_fd)
                    return ("ERROR", "child timed out")
                ready, _, _ = select.select([r_fd], [], [], min(remaining, 1.0))
                if ready:
                    break
                wpid, wstatus = os.waitpid(pid, os.WNOHANG)
                if wpid != 0:
                    os.close(r_fd)
                    if os.WIFSIGNALED(wstatus):
                        sig = os.WTERMSIG(wstatus)
                        try:
                            sig_name = signal.Signals(sig).name
                        except (ValueError, AttributeError):
                            sig_name = f"signal {sig}"
                        return ("ERROR", f"child killed by {sig_name}")
                    return ("ERROR", f"child exited with code {os.WEXITSTATUS(wstatus)}")

            raw_len = _pipe_read_all(r_fd, 4)
            size = struct.unpack('<I', raw_len)[0]
            raw_data = _pipe_read_all(r_fd, size)
            os.close(r_fd)
            os.waitpid(pid, 0)
            return pickle.loads(raw_data)

        except Exception as e:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
            try:
                os.waitpid(pid, 0)
            except ChildProcessError:
                pass
            try:
                os.close(r_fd)
            except OSError:
                pass
            return ("ERROR", f"parent error: {type(e).__name__}: {e}")

_send("READY")

while True:
    try:
        msg = _recv()
    except EOFError:
        break

    if msg == "STOP":
        break

    config = msg
    status, payload = run_in_fork(config, timeout=60)
    _send((status, payload))
'''


def _drain_stderr(pipe, team, lines_buf):
    """Background thread to continuously read stderr from a worker."""
    try:
        for line in iter(pipe.readline, b''):
            decoded = line.decode("utf-8", errors="replace").rstrip()
            if decoded:
                lines_buf.append(decoded)
    except Exception:
        pass


class PersistentFMUPool:
    """Pool of long-lived FMU worker subprocesses, one per team."""

    def __init__(
        self,
        teams,
        fmu_path,
        data_dir,
        step_size=120.0,
        max_uses=1,
        max_restarts=3,
        loss_type='huber_relative',
        huber_delta=1.0,
        relative_floor_frac=0.05,
        relative_floor_abs=1e-6,
    ):
        self.teams = teams
        self.fmu_path = fmu_path
        self.data_dir = data_dir
        self.step_size = step_size
        self.max_uses = max_uses
        self.max_restarts = max_restarts
        self.loss_type = str(loss_type)
        self.huber_delta = float(huber_delta)
        self.relative_floor_frac = float(relative_floor_frac)
        self.relative_floor_abs = float(relative_floor_abs)
        self.workers = {}
        self._stderr_bufs = {}
        self._stderr_threads = {}
        self._restart_counts = {t: 0 for t in teams}

        for t in teams:
            self._start_worker(t)

    def _start_worker(self, team):
        self._stderr_bufs[team] = []
        p = subprocess.Popen(
            [
                sys.executable,
                "-c",
                _PERSISTENT_WORKER,
                team,
                self.fmu_path,
                self.data_dir,
                str(self.step_size),
                str(self.max_uses),
                self.loss_type,
                str(self.huber_delta),
                str(self.relative_floor_frac),
                str(self.relative_floor_abs),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={
                **os.environ,
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "PYTHONPATH": os.pathsep.join(sys.path),
            },
            cwd=os.getcwd(),
        )
        buf = self._stderr_bufs[team]
        t = threading.Thread(target=_drain_stderr, args=(p.stderr, team, buf), daemon=True)
        t.start()
        self._stderr_threads[team] = t

        try:
            ready = recv_timeout(p.stdout, timeout=30)
            assert ready == "READY"
        except Exception:
            t.join(timeout=2)
            stderr_text = "\n".join(buf[-50:])
            p.kill()
            p.wait()
            raise RuntimeError(
                f"Worker {team} failed to start.\n--- stderr ---\n{stderr_text}"
            )
        self.workers[team] = p
        self._restart_counts[team] = 0
        print(f"Worker {team} started (pid={p.pid})")

    def _restart_worker(self, team):
        p = self.workers.get(team)
        if p:
            try:
                p.kill()
                p.wait(timeout=5)
            except Exception:
                pass

        self._restart_counts[team] = self._restart_counts.get(team, 0) + 1
        if self._restart_counts[team] > self.max_restarts:
            stderr_lines = self._stderr_bufs.get(team, [])
            print(
                f"Worker {team} exceeded {self.max_restarts} restarts, giving up. Last stderr:\n"
                + "\n".join(stderr_lines[-30:])
            )
            return False

        print(
            f"Restarting worker {team} "
            f"(attempt {self._restart_counts[team]}/{self.max_restarts})"
        )
        try:
            self._start_worker(team)
            return True
        except RuntimeError as e:
            print(f"Failed to restart worker {team}: {e}")
            return False

    def _get_worker_stderr(self, team):
        return "\n".join(self._stderr_bufs.get(team, [])[-20:])

    def evaluate(self, config, timeout=90):
        for t in self.teams:
            p = self.workers.get(t)
            if p is None or p.poll() is not None:
                rc = p.returncode if p else None
                if rc is not None and rc < 0:
                    import signal as _sig
                    try:
                        sig_name = _sig.Signals(-rc).name
                    except (ValueError, AttributeError):
                        sig_name = f"signal {-rc}"
                    print(f"Worker {t} killed by {sig_name} (rc={rc})")
                elif rc is not None:
                    print(f"Worker {t} exited with code {rc}")
                stderr_text = self._get_worker_stderr(t)
                if stderr_text:
                    print(f"Worker {t} stderr:\n{stderr_text}")
                self._restart_worker(t)

        live_teams = []
        for t in self.teams:
            p = self.workers.get(t)
            if p is None or p.poll() is not None:
                continue
            try:
                send(p.stdin, config)
                live_teams.append(t)
            except (BrokenPipeError, OSError):
                print(f"Worker {t} pipe broken on send, will restart next call")

        team_losses = []
        for t in live_teams:
            p = self.workers[t]
            try:
                status, payload = recv_timeout(p.stdout, timeout=timeout)
                if status == "OK" and payload:
                    team_losses.append(payload)
                    print(f"Worker {t} computed loss: {payload}")
                    self._restart_counts[t] = 0
                elif status == "ERROR":
                    print(f"FMU error for team {t}: {payload}")
            except TimeoutError:
                print(f"Worker {t} timed out ({timeout}s), killing")
                p.kill()
                p.wait()
            except EOFError:
                rc = p.returncode if p.poll() is not None else "still running?"
                stderr_text = self._get_worker_stderr(t)
                print(f"Worker {t} died mid-response (rc={rc}). stderr:\n{stderr_text}")

        return team_losses

    def shutdown(self):
        for t, p in self.workers.items():
            try:
                send(p.stdin, "STOP")
                p.wait(timeout=5)
            except Exception:
                try:
                    p.kill()
                    p.wait()
                except Exception:
                    pass
