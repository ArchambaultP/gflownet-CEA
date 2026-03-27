"""Batch evaluation of many parameter configs with Huber-aware losses.

Used by compute_states to evaluate thousands of terminal states.
Each subprocess handles one (config, team) pair â€” no reset() needed.

Main additions over the original batch.py:
- supports huber_relative / rse / absolute_relative
- uses stabilized relative residuals: e = (y_hat - y) / max(|y|, floor)
- can optionally return exact empirical |e| data for delta selection
- aggregates losses as mean of per-team means, matching the live proxy path
"""

import os
import pickle
import shutil
import subprocess
import sys
import tempfile
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


_SINGLE_EVAL_SCRIPT = r"""
import os
import pickle
import sys
import numpy as np

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

args_file, result_file = sys.argv[1], sys.argv[2]

with open(args_file, 'rb') as f:
    (
        key,
        params,
        fmu_path,
        team,
        data_dir,
        loss_type,
        huber_delta,
        relative_floor_frac,
        relative_floor_abs,
    ) = pickle.load(f)

from fmu.tomato_controller import TomatoController
from gflownet.envs.greenhouse.constants import BASELINE_PARAMETERS, INITIAL_CONDITIONS
from gflownet.proxy.greenhouse.cropSimulatorProxy import CropSimulatorProxy


def _as_scalar(v):
    if isinstance(v, (list, tuple)):
        return float(v[-1]) if v else 0.0
    if hasattr(v, '__len__'):
        return float(v[-1]) if len(v) > 0 else 0.0
    return float(v)


def _series_floor(values, relative_floor_frac, relative_floor_abs):
    arr = np.asarray(values, dtype=float)
    arr = np.abs(arr[np.isfinite(arr)])
    arr = arr[arr > 0]
    if arr.size == 0:
        scale = 1.0
    else:
        scale = float(np.percentile(arr, 90))
    return max(float(relative_floor_abs), float(relative_floor_frac) * scale)


def _huber(e, delta):
    ae = abs(float(e))
    if ae <= delta:
        return 0.5 * ae * ae
    return float(delta) * (ae - 0.5 * float(delta))


def _point_loss(e, loss_type, huber_delta):
    if loss_type == 'huber_relative':
        return _huber(e, huber_delta)
    if loss_type == 'absolute_relative':
        return abs(float(e))
    return float(e) * float(e)


team_data = CropSimulatorProxy.get_team_obs_dataset(data_dir, team)
input_trace = CropSimulatorProxy.compute_trace(
    CropSimulatorProxy.get_team_control_dataset(data_dir, team), delta='30min')
setpoints = (team_data.index - team_data.index.min())[1:].total_seconds().tolist()

init = {**BASELINE_PARAMETERS, **INITIAL_CONDITIONS, **params}
controller = TomatoController(
    fmu_path, start_time=0, stop_time=86400.0 * 200,
    step_size=120.0, logger=None)

sim_out = controller.simulate(input_trace, setpoints, init_conds=init)

DM_floor = _series_floor(team_data['DM_harvest_obs'].values, relative_floor_frac, relative_floor_abs)
N_floor = _series_floor(team_data['N_harvest_per_m2'].values, relative_floor_frac, relative_floor_abs)

point_losses = []
abs_e_all = []
abs_e_dm = []
abs_e_n = []
for idx, (_, output) in enumerate(sim_out):
    y_DM = float(team_data['DM_harvest_obs'].iloc[idx])
    y_N = float(team_data['N_harvest_per_m2'].iloc[idx])
    y_hat_DM = _as_scalar(output['C_harvest'])
    y_hat_N = _as_scalar(output['N_harvest'])

    if y_DM > 0:
        e_dm = (float(y_hat_DM) - y_DM) / max(abs(y_DM), DM_floor)
        point_losses.append(_point_loss(e_dm, loss_type, huber_delta))
        ae = abs(float(e_dm))
        abs_e_dm.append(ae)
        abs_e_all.append(ae)

    if y_N > 0:
        e_n = (float(y_hat_N) - y_N) / max(abs(y_N), N_floor)
        point_losses.append(_point_loss(e_n, loss_type, huber_delta))
        ae = abs(float(e_n))
        abs_e_n.append(ae)
        abs_e_all.append(ae)

with open(result_file, 'wb') as f:
    pickle.dump(
        {
            'key': key,
            'team': team,
            'point_losses': point_losses,
            'abs_e_all': abs_e_all,
            'abs_e_dm': abs_e_dm,
            'abs_e_n': abs_e_n,
            'dm_floor': float(DM_floor),
            'n_floor': float(N_floor),
        },
        f,
    )
"""


def _normalize_combo_key(key):
    if isinstance(key, tuple):
        return key
    if isinstance(key, str) and "|" in key:
        return tuple(key.split("|"))
    return key


def _safe_mean(values: Iterable[float], default: float = 1e6) -> float:
    values = list(values)
    if not values:
        return float(default)
    return float(np.mean(values))


def evaluate_all(
    states=None,
    terminal_states=None,
    fmu_path=None,
    team_ids=None,
    data_dir=None,
    n_workers=48,
    timeout=120,
    verbose=False,
    loss_type="huber_relative",
    huber_delta=1.0,
    relative_floor_frac=0.05,
    relative_floor_abs=1e-6,
    return_details=False,
):
    """Evaluate many parameter configs across all teams.

    Args:
        states / terminal_states: {combo_tuple: params_dict}
        return_details: if True, also returns exact empirical |e| data

    Returns:
        losses if return_details=False
        (losses, details) if return_details=True
    """
    if states is None:
        states = terminal_states
    if states is None:
        raise ValueError("One of `states` or `terminal_states` must be provided.")

    work = [("|".join(combo), params) for combo, params in states.items()]
    jobs = [(key, params, team) for key, params in work for team in team_ids]

    temp_root = temp_root or os.environ.get('BATCH_EVAL_TMPDIR')
    if temp_root:
        os.makedirs(temp_root, exist_ok=True)

    total_done = 0

    team_point_losses: Dict[str, Dict[str, List[float]]] = {}
    abs_e_all: List[float] = []
    abs_e_dm: List[float] = []
    abs_e_n: List[float] = []
    abs_e_by_team: Dict[str, List[float]] = {team: [] for team in team_ids}
    floors_by_team: Dict[str, Dict[str, float]] = {}

    with tempfile.TemporaryDirectory(prefix="batch_eval_", dir=temp_root) as tmp_dir:
        for wave_start in range(0, len(jobs), n_workers):
            wave = jobs[wave_start:wave_start + n_workers]
            procs = []
            wave_files = []

            for i, (key, params, team) in enumerate(wave):
                idx = wave_start + i
                args_file = os.path.join(tmp_dir, f"args_{idx}.pkl")
                result_file = os.path.join(tmp_dir, f"result_{idx}.pkl")
                wave_files.extend([args_file, result_file])
                with open(args_file, 'wb') as f:
                    pickle.dump(
                        (
                            key,
                            params,
                            fmu_path,
                            team,
                            data_dir,
                            loss_type,
                            huber_delta,
                            relative_floor_frac,
                            relative_floor_abs,
                        ),
                        f,
                    )

                p = subprocess.Popen(
                    [sys.executable, "-c", _SINGLE_EVAL_SCRIPT, args_file, result_file],
                    env={
                        **os.environ,
                        "OPENBLAS_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "1",
                        "OMP_NUM_THREADS": "1",
                        "PYTHONPATH": os.pathsep.join(sys.path),
                        **({"TMPDIR": temp_root} if temp_root else {}),
                    },
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
                        payload = pickle.load(f)
                    key = payload['key']
                    team = payload['team']
                    point_losses = list(payload.get('point_losses', []))
                    team_point_losses.setdefault(key, {})[team] = point_losses
                    abs_e_all.extend(payload.get('abs_e_all', []))
                    abs_e_dm.extend(payload.get('abs_e_dm', []))
                    abs_e_n.extend(payload.get('abs_e_n', []))
                    abs_e_by_team.setdefault(team, []).extend(payload.get('abs_e_all', []))
                    floors_by_team[team] = {
                        'DM_floor': float(payload.get('dm_floor', np.nan)),
                        'N_floor': float(payload.get('n_floor', np.nan)),
                    }
                    total_done += 1
                except Exception:
                    pass

            for fp in wave_files:
                try:
                    if os.path.exists(fp):
                        os.remove(fp)
                except OSError:
                    pass

            if verbose:
                print(f"  {total_done}/{len(jobs)} team evaluations done")

        losses = {}
        for key, team_map in team_point_losses.items():
            per_team_means = [_safe_mean(vals) for vals in team_map.values() if vals]
            losses[_normalize_combo_key(key)] = _safe_mean(per_team_means)

        if not return_details:
            return losses

        details = {
            'abs_e_all': np.asarray(abs_e_all, dtype=np.float64),
            'abs_e_dm': np.asarray(abs_e_dm, dtype=np.float64),
            'abs_e_n': np.asarray(abs_e_n, dtype=np.float64),
            'abs_e_by_team': {
                team: np.asarray(vals, dtype=np.float64) for team, vals in abs_e_by_team.items()
            },
            'floors_by_team': floors_by_team,
            'num_team_jobs_completed': int(total_done),
            'num_expected_team_jobs': int(len(jobs)),
        }
        return losses, details
