"""Long-lived FMU worker pool for GFlowNet proxy.

This version is rebuilt from the older persistent worker design:
- one long-lived Python worker subprocess per team
- each worker survives across proxy calls
- each actual FMU simulation runs in a fresh forked child so the parent
  survives native crashes / reset corruption
- returns per-point losses per team, matching the current proxy contract

Notes
-----
- `max_uses` is kept for API compatibility with older callers, but the
  fork-per-simulation design already gives each evaluation a fresh FMU
  controller instance.
- Loss computation matches the current batch evaluator semantics:
  stabilized relative residuals using per-channel floors, with support for
  `huber_relative`, `absolute_relative`, and squared relative error (`rse`).
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
from typing import Dict, List

from fmu.pool.protocol import recv_timeout, send


_PERSISTENT_WORKER = r'''
import os
import pickle
import select
import signal
import struct
import sys
import time
import traceback

# Protect the protocol channel from C-level stdout writes coming from the FMU.
_PROTO_FD = os.dup(sys.stdout.fileno())
os.dup2(sys.stderr.fileno(), 1)
_proto_out = os.fdopen(_PROTO_FD, "wb", buffering=0)

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

team = sys.argv[1]
fmu_path = sys.argv[2]
data_dir = sys.argv[3]
step_size = float(sys.argv[4])
max_uses = int(sys.argv[5])  # kept for compatibility; not used directly
loss_type = sys.argv[6]
huber_delta = float(sys.argv[7])
relative_floor_frac = float(sys.argv[8])
relative_floor_abs = float(sys.argv[9])

from gflownet.envs.greenhouse.constants import BASELINE_PARAMETERS, INITIAL_CONDITIONS
from gflownet.proxy.greenhouse.cropSimulatorProxy import CropSimulatorProxy


def _send(obj):
    data = pickle.dumps(obj)
    _proto_out.write(struct.pack("<I", len(data)))
    _proto_out.write(data)
    _proto_out.flush()


def _recv():
    raw = sys.stdin.buffer.read(4)
    if not raw:
        raise EOFError
    size = struct.unpack("<I", raw)[0]
    payload = sys.stdin.buffer.read(size)
    return pickle.loads(payload)


def _pipe_read_all(fd, size):
    chunks = []
    remaining = size
    while remaining > 0:
        chunk = os.read(fd, remaining)
        if not chunk:
            raise EOFError("child pipe closed prematurely")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _as_scalar(v):
    if isinstance(v, (list, tuple)):
        return float(v[-1]) if v else 0.0
    if hasattr(v, "__len__"):
        return float(v[-1]) if len(v) > 0 else 0.0
    return float(v)


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


def _huber(e):
    ae = abs(float(e))
    if ae <= huber_delta:
        return 0.5 * ae * ae
    return huber_delta * (ae - 0.5 * huber_delta)


def _point_loss(y_hat, y, floor):
    denom = max(abs(float(y)), floor)
    e = (float(y_hat) - float(y)) / denom
    if loss_type == "huber_relative":
        return _huber(e)
    if loss_type == "absolute_relative":
        return abs(e)
    # default: squared relative error (rse)
    return e * e


# One-time team setup in the persistent parent worker.
team_data = CropSimulatorProxy.get_team_obs_dataset(data_dir, team)
ctrl_data = CropSimulatorProxy.get_team_control_dataset(data_dir, team)
climate_start = ctrl_data.index.min()
input_trace = CropSimulatorProxy.compute_trace(ctrl_data, delta="30min")
setpoints = (team_data.index - climate_start).total_seconds().tolist()

_team_DM = team_data["DM_harvest_obs"].tolist()
_team_N = team_data["N_harvest_per_m2"].tolist()

_DM_floor = _series_floor(_team_DM)
_N_floor = _series_floor(_team_N)

# Serialize heavy objects once so each child can deserialize a clean copy.
_input_trace_bytes = pickle.dumps(input_trace)


def run_in_fork(config, timeout=60):
    """Run one FMU simulation in a forked child process.

    The parent worker survives segfaults and native-library corruption.
    """
    r_fd, w_fd = os.pipe()
    pid = os.fork()

    if pid == 0:
        os.close(r_fd)
        controller = None
        try:
            from fmu.tomato_controller import TomatoController

            child_input_trace = pickle.loads(_input_trace_bytes)
            controller = TomatoController(
                fmu_path,
                start_time=0,
                stop_time=86400.0 * 200,
                step_size=step_size,
                logger=None,
            )

            sim_out = controller.simulate(
                child_input_trace,
                setpoints,
                init_conds=config,
            )

            point_losses = []
            for idx, (_, output) in enumerate(sim_out):
                y_DM = _team_DM[idx]
                y_N = _team_N[idx]
                y_hat_DM = _as_scalar(output["C_harvest"])
                y_hat_N = _as_scalar(output["N_harvest"])

                if y_DM > 0:
                    point_losses.append(_point_loss(y_hat_DM, y_DM, _DM_floor))
                if y_N > 0:
                    point_losses.append(_point_loss(y_hat_N, y_N, _N_floor))

            result = ("OK", point_losses)
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
            result = ("ERROR", f"{type(e).__name__}: {e}")

        try:
            data = pickle.dumps(result)
            os.write(w_fd, struct.pack("<I", len(data)))
            os.write(w_fd, data)
        except Exception:
            pass

        try:
            if controller is not None and hasattr(controller, "close"):
                controller.close()
        except Exception:
            pass

        try:
            os.close(w_fd)
        except OSError:
            pass
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
                        except Exception:
                            sig_name = f"signal {sig}"
                        return ("ERROR", f"child killed by {sig_name}")
                    return ("ERROR", f"child exited with code {os.WEXITSTATUS(wstatus)}")

            raw_len = _pipe_read_all(r_fd, 4)
            size = struct.unpack("<I", raw_len)[0]
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

    status, payload = run_in_fork(msg, timeout=90)
    _send((status, payload))
'''


def _drain_stderr(pipe, lines_buf: List[str]) -> None:
    try:
        for line in iter(pipe.readline, b""):
            decoded = line.decode("utf-8", errors="replace").rstrip()
            if decoded:
                lines_buf.append(decoded)
    except Exception:
        pass


class PersistentFMUPool:
    """Pool of persistent team workers.

    Public contract
    ---------------
    evaluate(config, timeout=90) -> list[list[float]]
        Returns one list of point losses per successful team.
    """

    def __init__(
        self,
        teams,
        fmu_path,
        data_dir,
        step_size=120.0,
        max_uses=1,
        max_restarts=3,
        loss_type="huber_relative",
        huber_delta=1.0,
        relative_floor_frac=0.05,
        relative_floor_abs=1e-6,
    ):
        self.teams = list(teams)
        self.fmu_path = str(fmu_path)
        self.data_dir = str(data_dir)
        self.step_size = float(step_size)
        self.max_uses = int(max_uses)
        self.max_restarts = int(max_restarts)
        self.loss_type = str(loss_type)
        self.huber_delta = float(huber_delta)
        self.relative_floor_frac = float(relative_floor_frac)
        self.relative_floor_abs = float(relative_floor_abs)

        self.workers: Dict[str, subprocess.Popen] = {}
        self._stderr_bufs: Dict[str, List[str]] = {}
        self._stderr_threads: Dict[str, threading.Thread] = {}
        self._restart_counts: Dict[str, int] = {t: 0 for t in self.teams}

        for team in self.teams:
            self._start_worker(team)

    def _start_worker(self, team: str) -> None:
        stderr_buf: List[str] = []
        self._stderr_bufs[team] = stderr_buf

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

        t = threading.Thread(target=_drain_stderr, args=(p.stderr, stderr_buf), daemon=True)
        t.start()
        self._stderr_threads[team] = t

        try:
            ready = recv_timeout(p.stdout, timeout=30)
            if ready != "READY":
                raise RuntimeError(f"unexpected worker handshake: {ready!r}")
        except Exception:
            t.join(timeout=2)
            stderr_text = "\n".join(stderr_buf[-50:])
            p.kill()
            p.wait()
            raise RuntimeError(
                f"Worker {team} failed to start.\n--- stderr ---\n{stderr_text}"
            )

        self.workers[team] = p
        self._restart_counts[team] = 0
        print(f"Worker {team} started (pid={p.pid})")

    def _worker_stderr(self, team: str) -> str:
        return "\n".join(self._stderr_bufs.get(team, [])[-30:])

    def _restart_worker(self, team: str) -> bool:
        p = self.workers.get(team)
        if p is not None:
            try:
                p.kill()
                p.wait(timeout=5)
            except Exception:
                pass

        self._restart_counts[team] = self._restart_counts.get(team, 0) + 1
        if self._restart_counts[team] > self.max_restarts:
            print(
                f"Worker {team} exceeded {self.max_restarts} restart attempts.\n"
                f"Last stderr:\n{self._worker_stderr(team)}"
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

    def _ensure_live_workers(self) -> None:
        for team in self.teams:
            p = self.workers.get(team)
            if p is None or p.poll() is not None:
                rc = p.returncode if p is not None else None
                if rc is not None and rc < 0:
                    try:
                        sig_name = signal.Signals(-rc).name
                    except Exception:
                        sig_name = f"signal {-rc}"
                    print(f"Worker {team} killed by {sig_name} (rc={rc})")
                elif rc is not None:
                    print(f"Worker {team} exited with code {rc}")

                stderr_text = self._worker_stderr(team)
                if stderr_text:
                    print(f"Worker {team} stderr:\n{stderr_text}")

                self._restart_worker(team)

    def evaluate(self, config, timeout=90):
        """Evaluate one full parameter configuration across all teams.

        Returns
        -------
        list[list[float]]
            Per-team point loss lists for successful teams.
        """
        self._ensure_live_workers()

        live_teams = []
        for team in self.teams:
            p = self.workers.get(team)
            if p is None or p.poll() is not None:
                continue
            try:
                send(p.stdin, config)
                live_teams.append(team)
            except (BrokenPipeError, OSError):
                print(f"Worker {team} pipe broken on send")

        team_losses = []
        for team in live_teams:
            p = self.workers[team]
            try:
                status, payload = recv_timeout(p.stdout, timeout=timeout)
                if status == "OK" and payload:
                    team_losses.append(payload)
                    self._restart_counts[team] = 0
                elif status == "ERROR":
                    print(f"FMU error for team {team}: {payload}")
            except TimeoutError:
                print(f"Worker {team} timed out after {timeout}s; killing")
                try:
                    p.kill()
                    p.wait()
                except Exception:
                    pass
            except EOFError:
                rc = p.returncode if p.poll() is not None else None
                print(
                    f"Worker {team} died mid-response (rc={rc}).\n"
                    f"stderr:\n{self._worker_stderr(team)}"
                )

        return team_losses

    def shutdown(self) -> None:
        for team, p in self.workers.items():
            try:
                send(p.stdin, "STOP")
                p.wait(timeout=5)
            except Exception:
                try:
                    p.kill()
                    p.wait()
                except Exception:
                    pass
