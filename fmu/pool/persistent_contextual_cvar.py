
from __future__ import annotations

import os
import pickle
import select
import signal
import struct
import subprocess
import sys
from typing import Dict, List, Optional

# ----------------------------
# Simple binary protocol
# ----------------------------

def _send(pipe, obj) -> None:
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    pipe.write(struct.pack("<I", len(data)))
    pipe.write(data)
    pipe.flush()


def _recv_timeout(pipe, timeout: float):
    fd = pipe.fileno()
    ready, _, _ = select.select([fd], [], [], timeout)
    if not ready:
        raise TimeoutError(f"Timed out waiting for worker after {timeout}s")
    raw_len = pipe.read(4)
    if not raw_len:
        raise EOFError("Worker pipe closed")
    size = struct.unpack("<I", raw_len)[0]
    payload = pipe.read(size)
    if not payload:
        raise EOFError("Worker pipe closed mid-message")
    return pickle.loads(payload)


# ----------------------------
# Long-lived worker code
# ----------------------------

_WORKER = r"""
from __future__ import annotations

import os
import pickle
import struct
import sys
import traceback
from typing import Dict, List

import numpy as np

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

team = sys.argv[1]
fmu_path = sys.argv[2]
data_dir = sys.argv[3]
step_size = float(sys.argv[4])
loss_type = sys.argv[5]
huber_delta = float(sys.argv[6])
relative_floor_frac = float(sys.argv[7])
relative_floor_abs = float(sys.argv[8])
setpoint_mode = sys.argv[9]

from fmu.tomato_controller import TomatoController
from gflownet.envs.greenhouse.constants import BASELINE_PARAMETERS, INITIAL_CONDITIONS
from data.greenhouse.secondEdition.extract import (
    load_climate_data,
    load_prod_data,
    load_tomato_data,
    load_parameter_data,
)

def _send(obj):
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    sys.stdout.buffer.write(struct.pack("<I", len(data)))
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()

def _recv():
    raw = sys.stdin.buffer.read(4)
    if not raw:
        raise EOFError
    size = struct.unpack("<I", raw)[0]
    payload = sys.stdin.buffer.read(size)
    return pickle.loads(payload)

def _as_scalar(v):
    if isinstance(v, (list, tuple)):
        return float(v[-1]) if v else 0.0
    if hasattr(v, "__len__") and not isinstance(v, (str, bytes, dict)):
        return float(v[-1]) if len(v) > 0 else 0.0
    return float(v)

def _series_floor(values):
    arr = np.asarray(values, dtype=float)
    arr = np.abs(arr[np.isfinite(arr)])
    arr = arr[arr > 0]
    if arr.size == 0:
        scale = 1.0
    else:
        scale = float(np.percentile(arr, 90))
    return max(relative_floor_abs, relative_floor_frac * scale)

def _huber(e: float) -> float:
    ae = abs(float(e))
    if ae <= huber_delta:
        return 0.5 * ae * ae
    return huber_delta * (ae - 0.5 * huber_delta)

def _point_loss(y_hat: float, y: float, floor: float) -> float:
    denom = max(abs(float(y)), floor)
    e = (float(y_hat) - float(y)) / denom
    if loss_type == "huber_relative":
        return _huber(e)
    if loss_type == "absolute_relative":
        return abs(e)
    return e * e

def _load_team_control_dataset(data_dir: str, team: str):
    fp_climate = f"{data_dir}/{team}/GreenhouseClimate.csv"
    climate_df = load_climate_data(fp_climate)
    return climate_df[["CO2air", "PAR", "Tair"]]

def _load_team_obs_dataset(data_dir: str, team: str):
    fp_production = f"{data_dir}/{team}/Production.csv"
    fp_tomato = f"{data_dir}/{team}/TomQuality.csv"
    fp_parameter = f"{data_dir}/{team}/CropParameters.csv"

    prod_df = load_prod_data(fp_production)
    import pandas as pd
    prod_df = pd.DataFrame({
        "N": prod_df["nClassA"] + prod_df["nClassB"],
        "N_Sum": (prod_df["nClassA"] + prod_df["nClassB"]).cumsum(),
        "Yield": prod_df["gClassA"] + prod_df["gClassB"],
        "Yield_Sum": (prod_df["gClassA"] + prod_df["gClassB"]).cumsum(),
        "DAP": prod_df["DAP"],
    })

    tomato_df = load_tomato_data(fp_tomato)
    param_df = load_parameter_data(fp_parameter)

    df = pd.merge(prod_df, param_df, on="Time", how="outer")
    df = pd.merge(df, tomato_df, on="Time", how="inner")
    df = df.ffill()

    df["N_harvest_per_m2"] = ((df["N"] / 10) * df["stem_density"]).cumsum()
    df["yield_fw_g_m2"] = (df["Yield"] / 10) * df["stem_density"]
    df["dry_weight_g_m2"] = df["yield_fw_g_m2"] * (df["dryMatterPercent"] / 100)
    df["dry_weight_mg_CH2O_m2"] = df["dry_weight_g_m2"] * 1000
    df["DM_harvest_obs"] = df["dry_weight_mg_CH2O_m2"].cumsum()

    return df[["DM_harvest_obs", "N_harvest_per_m2"]]

def _compute_trace(sim_df, delta="30min"):
    sim_df = sim_df.copy()
    sim_df["Tair24"] = sim_df["Tair"].groupby(sim_df.index.date).transform("mean").round(2)
    sim_df.index = sim_df.index.round(delta)
    sim_df = sim_df.groupby(level=0).mean()
    sim_df.index = (sim_df.index - sim_df.index.min()).total_seconds()
    return [
        (float(t), {
            "CO2_Air": float(row.CO2air),
            "PAR_gh": float(row.PAR),
            "TCan": float(row.Tair),
            "TCan24": float(row.Tair24),
        })
        for t, row in sim_df.iterrows()
    ]

team_data = _load_team_obs_dataset(data_dir, team)
ctrl_data = _load_team_control_dataset(data_dir, team)
input_trace = _compute_trace(ctrl_data, delta="30min")

if setpoint_mode == "obs_start":
    setpoints = (team_data.index - team_data.index.min())[1:].total_seconds().tolist()
else:
    climate_start = ctrl_data.index.min()
    setpoints = (team_data.index - climate_start).total_seconds().tolist()

team_DM = team_data["DM_harvest_obs"].tolist()
team_N = team_data["N_harvest_per_m2"].tolist()
DM_floor = _series_floor(team_DM)
N_floor = _series_floor(team_N)

_send(("READY", {"team": team}))

while True:
    try:
        msg = _recv()
    except EOFError:
        break

    if msg == "STOP":
        break

    try:
        full_config = {**BASELINE_PARAMETERS, **INITIAL_CONDITIONS, **msg}
        controller = TomatoController(
            fmu_path,
            start_time=0.0,
            stop_time=86400.0 * 200.0,
            step_size=step_size,
            logger=None,
        )
        try:
            sim_out = controller.simulate(
                input_trace,
                setpoints,
                init_conds=full_config,
            )
        finally:
            try:
                controller.close()
            except Exception:
                pass

        point_losses: List[float] = []
        for idx, (_, output) in enumerate(sim_out):
            if idx >= len(team_DM) or idx >= len(team_N):
                break

            y_DM = team_DM[idx]
            y_N = team_N[idx]
            y_hat_DM = _as_scalar(output["C_harvest"])
            y_hat_N = _as_scalar(output["N_harvest"])

            if y_DM > 0:
                point_losses.append(_point_loss(y_hat_DM, y_DM, DM_floor))
            if y_N > 0:
                point_losses.append(_point_loss(y_hat_N, y_N, N_floor))

        _send(("OK", point_losses))
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        _send(("ERROR", f"{type(e).__name__}: {e}"))
"""


class PersistentFMUPool:
    """
    Persistent one-worker-per-team pool.

    For contextual/CVaR rewards, the proxy should call evaluate_contextual()
    and aggregate returned per-team point-loss lists into per-team means and
    then into its contextual reward.
    """

    def __init__(
        self,
        teams,
        fmu_path,
        data_dir,
        step_size: float = 120.0,
        max_restarts: int = 3,
        loss_type: str = "absolute_relative",
        huber_delta: float = 1.0,
        relative_floor_frac: float = 0.05,
        relative_floor_abs: float = 1e-6,
        setpoint_mode: str = "climate_start",
        verbose: bool = False,
    ):
        self.teams = list(teams)
        self.fmu_path = str(fmu_path)
        self.data_dir = str(data_dir)
        self.step_size = float(step_size)
        self.max_restarts = int(max_restarts)
        self.loss_type = str(loss_type)
        self.huber_delta = float(huber_delta)
        self.relative_floor_frac = float(relative_floor_frac)
        self.relative_floor_abs = float(relative_floor_abs)
        self.setpoint_mode = str(setpoint_mode)
        self.verbose = bool(verbose)

        self.workers: Dict[str, subprocess.Popen] = {}
        self.restart_counts: Dict[str, int] = {team: 0 for team in self.teams}

        for team in self.teams:
            self._start_worker(team)

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    def _start_worker(self, team: str) -> None:
        p = subprocess.Popen(
            [
                sys.executable,
                "-c",
                _WORKER,
                team,
                self.fmu_path,
                self.data_dir,
                str(self.step_size),
                self.loss_type,
                str(self.huber_delta),
                str(self.relative_floor_frac),
                str(self.relative_floor_abs),
                self.setpoint_mode,
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
        status, payload = _recv_timeout(p.stdout, timeout=30)
        if status != "READY":
            try:
                err = p.stderr.read().decode("utf-8", errors="replace")
            except Exception:
                err = ""
            p.kill()
            p.wait()
            raise RuntimeError(f"Worker {team} failed to start: {payload}\n{err}")
        self.workers[team] = p
        self.restart_counts[team] = 0
        self._log(f"Worker {team} started (pid={p.pid})")

    def _restart_worker(self, team: str) -> bool:
        p = self.workers.get(team)
        if p is not None:
            try:
                p.kill()
                p.wait(timeout=5)
            except Exception:
                pass
        self.restart_counts[team] += 1
        if self.restart_counts[team] > self.max_restarts:
            return False
        self._start_worker(team)
        return True

    def _ensure_live_workers(self) -> None:
        for team in self.teams:
            p = self.workers.get(team)
            if p is None or p.poll() is not None:
                self._restart_worker(team)

    def _dispatch(self, config: Dict[str, float]) -> List[str]:
        self._ensure_live_workers()
        live = []
        for team in self.teams:
            p = self.workers.get(team)
            if p is None or p.poll() is not None:
                continue
            try:
                _send(p.stdin, config)
                live.append(team)
            except Exception:
                pass
        return live

    def _collect(self, live_teams: List[str], timeout: float) -> Dict[str, List[float]]:
        out: Dict[str, List[float]] = {}
        for team in live_teams:
            p = self.workers[team]
            try:
                status, payload = _recv_timeout(p.stdout, timeout=timeout)
                if status == "OK":
                    out[team] = payload
                    self.restart_counts[team] = 0
                else:
                    self._log(f"Worker {team} error: {payload}")
            except TimeoutError:
                self._log(f"Worker {team} timed out after {timeout}s")
                try:
                    p.kill()
                    p.wait(timeout=5)
                except Exception:
                    pass
            except Exception as e:
                self._log(f"Worker {team} failed during receive: {type(e).__name__}: {e}")
        return out

    def evaluate_contextual(self, config: Dict[str, float], timeout: float = 120.0) -> Dict[str, List[float]]:
        live = self._dispatch(config)
        return self._collect(live, timeout=timeout)

    def evaluate(self, config: Dict[str, float], timeout: float = 120.0) -> List[List[float]]:
        return list(self.evaluate_contextual(config, timeout=timeout).values())

    def shutdown(self) -> None:
        for team, p in list(self.workers.items()):
            try:
                _send(p.stdin, "STOP")
                p.wait(timeout=5)
            except Exception:
                try:
                    p.kill()
                    p.wait(timeout=5)
                except Exception:
                    pass
