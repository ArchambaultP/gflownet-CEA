
import hashlib
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from gflownet.proxy.base import Proxy
from gflownet.envs.greenhouse.constants import (
    BASELINE_PARAMETERS,
    INITIAL_CONDITIONS,
)
from data.greenhouse.secondEdition.extract import (
    load_climate_data,
    load_prod_data,
    load_tomato_data,
    load_parameter_data,
)


SCALAR_REWARD_MODES = {
    "softmin",
    "thresholded_sigmoid",
}

CONTEXT_REWARD_MODES = {
    "context_kth_exp_simple",
    "context_cvar_exp_simple",
    "context_cvar_blend_exp_simple",
}


def _maybe_float(x, name: str):
    if x is None:
        return None
    if isinstance(x, (int, float, np.floating)):
        return float(x)
    if isinstance(x, str):
        s = x.strip()
        if s == "" or s.lower() in {"none", "null"}:
            return None
        try:
            return float(s)
        except ValueError as e:
            raise ValueError(f"{name} must be numeric or null, got {x!r}.") from e
    raise TypeError(f"{name} must be numeric or null, got type {type(x).__name__}")


def _sigmoid(z: float) -> float:
    z = float(np.clip(z, -60.0, 60.0))
    return float(1.0 / (1.0 + np.exp(-z)))


class CropSimulatorProxy(Proxy):
    """
    Contextual crop simulator proxy using PersistentFMUPool for fast live evaluation.

    Key speed differences vs. the slow contextual proxy:
    - reuses a single PersistentFMUPool instead of spawning subprocesses per miss
    - batches cache misses inside __call__ and deduplicates repeated misses
    - recalibrates contextual reward statistics periodically instead of on every miss

    Cache-key handling:
    - if the proxy item is a string, use it directly
    - if the proxy item is (state_key, values), use state_key
    - if the proxy item is {"state_key": ..., "values": ...}, use state_key
    - otherwise fall back to a rounded-value hash

    Expected reward_cache format:
    {
      "state_key_or_hash": {
        "loss": <float>,
        "loss_by_context": {"Reference": ..., ...}
      }
    }
    """

    def __init__(
        self,
        reward_cache_path: Optional[str] = None,
        reward_mode: str = "context_cvar_blend_exp_simple",
        beta: Optional[float] = None,
        cache_save_every: int = 100,
        recalibrate_every: int = 64,
        loss_norm: str = "q10q90",
        q_low: float = 0.10,
        q_high: float = 0.90,
        reward_anchor: Optional[float] = None,
        reward_scale: Optional[float] = None,
        reward_tau: Optional[float] = None,
        tau_quantile: float = 0.05,
        threshold_temperature: float = 0.05,
        reward_epsilon: float = 1e-3,
        context_top_k: int = 3,
        context_tail_count: int = 2,
        context_cvar_lambda: float = 0.25,
        context_team_ids: Optional[List[str]] = None,
        context_fallback_to_scalar: bool = True,
        scalar_fallback_reward_mode: str = "softmin",
        # live evaluation / pool settings
        data_dir: str = "data/greenhouse/secondEdition",
        fmu_path: str = "fmu/FMU/tomato.fmu",
        step_size: float = 120.0,
        pool_max_uses: int = 1,
        pool_max_restarts: int = 3,
        live_loss_type: str = "absolute_relative",
        live_huber_delta: float = 1.0,
        live_relative_floor_frac: float = 0.05,
        live_relative_floor_abs: float = 1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.teams = context_team_ids or [
            "Reference",
            "Digilog",
            "IUACAAS",
            "Automatoes",
            "TheAutomators",
            "AICU",
        ]
        self.data_dir = str(data_dir)
        self.fmu_path = str(fmu_path)
        self.step_size = float(step_size)
        self.parameter_names = sorted(BASELINE_PARAMETERS.keys())

        self.reward_mode = str(reward_mode)
        self.scalar_fallback_reward_mode = str(scalar_fallback_reward_mode)

        if self.reward_mode not in (SCALAR_REWARD_MODES | CONTEXT_REWARD_MODES):
            raise ValueError(f"Unsupported reward_mode: {self.reward_mode}")
        if self.scalar_fallback_reward_mode not in SCALAR_REWARD_MODES:
            raise ValueError(
                f"Unsupported scalar_fallback_reward_mode: {self.scalar_fallback_reward_mode}"
            )

        beta_val = _maybe_float(beta, "beta")
        if beta_val is None:
            beta_val = 3.0 if self.reward_mode == "softmin" else 1.0
        self.beta = float(beta_val)

        self.loss_norm = str(loss_norm)
        self.q_low = float(q_low)
        self.q_high = float(q_high)

        self.reward_anchor = _maybe_float(reward_anchor, "reward_anchor")
        self.reward_scale = _maybe_float(reward_scale, "reward_scale")
        self.reward_tau = _maybe_float(reward_tau, "reward_tau")
        self.tau_quantile = float(tau_quantile)
        self.threshold_temperature = float(threshold_temperature)
        self.reward_epsilon = float(reward_epsilon)

        self.context_top_k = int(context_top_k)
        self.context_tail_count = int(context_tail_count)
        self.context_cvar_lambda = float(context_cvar_lambda)
        if not (0.0 <= self.context_cvar_lambda <= 1.0):
            raise ValueError("context_cvar_lambda must be in [0, 1].")
        self.context_fallback_to_scalar = bool(context_fallback_to_scalar)

        self.live_loss_type = str(live_loss_type)
        self.live_huber_delta = float(live_huber_delta)
        self.live_relative_floor_frac = float(live_relative_floor_frac)
        self.live_relative_floor_abs = float(live_relative_floor_abs)

        self.pool_max_uses = int(pool_max_uses)
        self.pool_max_restarts = int(pool_max_restarts)
        self.pool = None

        self.reward_cache: Dict[str, Dict[str, object]] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_new = 0
        self.cache_save_every = int(cache_save_every)
        self.recalibrate_every = max(int(recalibrate_every), 1)
        self.new_since_recalibration = 0
        self.reward_cache_path = reward_cache_path

        self.context_rank_arrays: Dict[str, np.ndarray] = {}
        self.context_quantile_stats: Dict[str, Tuple[float, float]] = {}

        if reward_cache_path and os.path.exists(reward_cache_path):
            self._load_cache(reward_cache_path)
            self.precomputed = len(self.reward_cache) > 0
            print(f"Loaded {len(self.reward_cache)} cached evaluations from {reward_cache_path}")
        else:
            self.precomputed = False
            if reward_cache_path:
                print(f"No cache found at {reward_cache_path} — will build lazily")
            else:
                print("No reward cache path — will use live FMU evaluation (no persistent cache)")

        self._maybe_init_reward_calibration_from_cache(force=True)

    def _contextual_reward_mode(self) -> bool:
        return self.reward_mode in CONTEXT_REWARD_MODES

    def _hash_from_values(self, values) -> str:
        if hasattr(values, "tolist"):
            vals = values.tolist()
        elif isinstance(values, (list, tuple)):
            vals = list(values)
        else:
            vals = [float(values)]
        rounded = tuple(round(float(v), 8) for v in vals)
        return hashlib.sha256(str(rounded).encode()).hexdigest()[:16]

    def _extract_cache_key_and_config(self, states_proxy_item):
        # string: can only serve as semantic key unless cached already
        if isinstance(states_proxy_item, str):
            return states_proxy_item, None

        # tuple/list form: (state_key, values_or_config)
        if isinstance(states_proxy_item, (tuple, list)) and len(states_proxy_item) == 2 and isinstance(states_proxy_item[0], str):
            state_key = str(states_proxy_item[0])
            payload = states_proxy_item[1]
            config = self._payload_to_config(payload)
            return state_key, config

        # dict form
        if isinstance(states_proxy_item, dict):
            if "state_key" in states_proxy_item:
                state_key = str(states_proxy_item["state_key"])
                if "config" in states_proxy_item:
                    return state_key, {str(k): float(v) for k, v in states_proxy_item["config"].items()}
                if "values" in states_proxy_item:
                    return state_key, self._payload_to_config(states_proxy_item["values"])
                return state_key, None
            if "config" in states_proxy_item:
                cfg = {str(k): float(v) for k, v in states_proxy_item["config"].items()}
                return self._hash_from_values([cfg[k] for k in self.parameter_names]), cfg

        # object with semantic state key
        if hasattr(states_proxy_item, "state_key"):
            state_key = str(states_proxy_item.state_key)
            if hasattr(states_proxy_item, "config"):
                return state_key, {str(k): float(v) for k, v in states_proxy_item.config.items()}
            if hasattr(states_proxy_item, "values"):
                return state_key, self._payload_to_config(states_proxy_item.values)
            return state_key, None

        # fallback: hashed numeric vector
        config = self._payload_to_config(states_proxy_item)
        return self._hash_from_values([config[k] for k in self.parameter_names]), config

    def _payload_to_config(self, payload) -> Optional[Dict[str, float]]:
        if payload is None:
            return None
        if isinstance(payload, dict):
            return {str(k): float(v) for k, v in payload.items()}

        if hasattr(payload, "tolist"):
            values = payload.tolist()
        else:
            values = list(payload)

        if len(values) < len(self.parameter_names):
            raise ValueError(
                f"Proxy payload length {len(values)} is smaller than number of parameters {len(self.parameter_names)}"
            )

        config = {}
        for i, name in enumerate(self.parameter_names):
            config[name] = float(values[i])
        return config

    def _load_cache(self, path):
        with open(path) as f:
            raw = json.load(f)

        for key, entry in raw.items():
            if isinstance(entry, dict):
                loss = float(entry["loss"]) if "loss" in entry else float(entry.get("scalar_loss", 1e6))
                loss_by_context = {
                    str(team): float(val)
                    for team, val in entry.get("loss_by_context", {}).items()
                }
                self.reward_cache[str(key)] = {
                    "loss": loss,
                    "loss_by_context": loss_by_context,
                }
            else:
                self.reward_cache[str(key)] = {
                    "loss": float(entry),
                    "loss_by_context": {},
                }

    def _save_cache(self):
        if not self.reward_cache_path:
            return

        cache_out = {}
        for key, entry in self.reward_cache.items():
            cache_out[str(key)] = {
                "loss": float(entry["loss"]),
                "loss_by_context": {
                    str(team): float(v)
                    for team, v in entry.get("loss_by_context", {}).items()
                },
            }

        os.makedirs(os.path.dirname(self.reward_cache_path) or ".", exist_ok=True)
        with open(self.reward_cache_path, "w") as f:
            json.dump(cache_out, f, indent=2)

    def _context_cache_available(self) -> bool:
        return any(entry.get("loss_by_context") for entry in self.reward_cache.values())

    def _build_context_rank_stats(self) -> Dict[str, np.ndarray]:
        by_team: Dict[str, List[float]] = {}
        for entry in self.reward_cache.values():
            for team, loss in entry.get("loss_by_context", {}).items():
                by_team.setdefault(str(team), []).append(float(loss))

        out: Dict[str, np.ndarray] = {}
        for team, losses in by_team.items():
            arr = np.asarray(sorted(losses), dtype=float)
            if arr.size > 0:
                out[team] = arr
        return out

    def _build_context_quantile_stats(self) -> Dict[str, Tuple[float, float]]:
        by_team: Dict[str, List[float]] = {}
        for entry in self.reward_cache.values():
            for team, loss in entry.get("loss_by_context", {}).items():
                by_team.setdefault(str(team), []).append(float(loss))

        out: Dict[str, Tuple[float, float]] = {}
        for team, losses in by_team.items():
            arr = np.asarray(losses, dtype=float)
            if arr.size == 0:
                continue
            lo = float(np.quantile(arr, self.q_low))
            hi = float(np.quantile(arr, self.q_high))
            out[team] = (lo, max(hi - lo, 1e-12))
        return out

    def _maybe_init_reward_calibration_from_cache(self, force: bool = False):
        if len(self.reward_cache) == 0:
            return

        losses = np.array(
            [float(entry["loss"]) for entry in self.reward_cache.values()],
            dtype=float,
        )

        if force or self.reward_anchor is None or self.reward_scale is None:
            if self.loss_norm == "none":
                self.reward_anchor = 0.0
                self.reward_scale = 1.0
            elif self.loss_norm == "q10q90":
                self.reward_anchor = float(np.quantile(losses, self.q_low))
                qh = float(np.quantile(losses, self.q_high))
                self.reward_scale = max(qh - self.reward_anchor, 1e-12)
            else:
                raise ValueError(f"Unsupported loss_norm: {self.loss_norm}")

        if self.reward_mode == "thresholded_sigmoid" and (force or self.reward_tau is None):
            loss_normed = (
                (losses - float(self.reward_anchor)) / max(float(self.reward_scale), 1e-12)
                if self.loss_norm != "none"
                else losses.copy()
            )
            self.reward_tau = float(np.quantile(loss_normed, self.tau_quantile))

        if self._contextual_reward_mode() and self._context_cache_available():
            self.context_rank_arrays = self._build_context_rank_stats()
            self.context_quantile_stats = self._build_context_quantile_stats()

        if self.reward_mode == "context_kth_exp_simple":
            print(
                "[reward calibration] "
                f"mode={self.reward_mode}, k={self.context_top_k}, beta={self.beta}, "
                f"contexts={sorted(self.context_rank_arrays.keys())}"
            )
        elif self.reward_mode == "context_cvar_exp_simple":
            print(
                "[reward calibration] "
                f"mode={self.reward_mode}, tail_count={self.context_tail_count}, beta={self.beta}, "
                f"contexts={sorted(self.context_quantile_stats.keys())}, "
                f"q_low={self.q_low}, q_high={self.q_high}"
            )
        elif self.reward_mode == "context_cvar_blend_exp_simple":
            print(
                "[reward calibration] "
                f"mode={self.reward_mode}, tail_count={self.context_tail_count}, "
                f"lambda={self.context_cvar_lambda}, beta={self.beta}, "
                f"contexts={sorted(self.context_quantile_stats.keys())}, "
                f"q_low={self.q_low}, q_high={self.q_high}"
            )
        elif self.reward_mode == "thresholded_sigmoid":
            print(
                "[reward calibration] "
                f"mode={self.reward_mode}, anchor={self.reward_anchor:.6g}, "
                f"scale={self.reward_scale:.6g}, tau={self.reward_tau:.6g}, "
                f"T={self.threshold_temperature}, eps={self.reward_epsilon}, beta={self.beta}"
            )
        elif self.reward_mode == "softmin":
            print(
                "[reward calibration] "
                f"mode={self.reward_mode}, anchor={self.reward_anchor:.6g}, "
                f"scale={self.reward_scale:.6g}, beta={self.beta}"
            )

    def _normalize_loss(self, loss: float) -> float:
        if self.loss_norm == "none":
            return float(loss)

        if self.reward_anchor is None or self.reward_scale is None:
            raise RuntimeError(
                "Reward normalization statistics are not initialized. "
                "Provide reward_anchor/reward_scale or load a cache with enough losses."
            )

        return (float(loss) - float(self.reward_anchor)) / max(float(self.reward_scale), 1e-12)

    @staticmethod
    def _percentile_rank(sorted_arr: np.ndarray, value: float) -> float:
        n = int(sorted_arr.size)
        if n <= 1:
            return 0.0
        idx = int(np.searchsorted(sorted_arr, value, side="left"))
        return float(idx / (n - 1))

    def _transformed_context_values_rank(self, loss_by_context: Dict[str, float]) -> List[float]:
        vals: List[float] = []
        for team in self.teams:
            if team not in loss_by_context:
                continue
            if team not in self.context_rank_arrays:
                continue
            vals.append(self._percentile_rank(self.context_rank_arrays[team], float(loss_by_context[team])))
        return vals

    def _transformed_context_values_cvar(self, loss_by_context: Dict[str, float]) -> List[float]:
        vals: List[float] = []
        for team in self.teams:
            if team not in loss_by_context:
                continue
            if team not in self.context_quantile_stats:
                continue
            lo, scale = self.context_quantile_stats[team]
            vals.append((float(loss_by_context[team]) - lo) / max(scale, 1e-12))
        return vals

    def _scalar_loss_to_reward(self, loss: float, mode: Optional[str] = None) -> float:
        mode = self.scalar_fallback_reward_mode if mode is None else str(mode)
        loss_normed = self._normalize_loss(loss)

        if mode == "softmin":
            reward = float(np.exp(-self.beta * loss_normed))
        elif mode == "thresholded_sigmoid":
            if self.reward_tau is None:
                raise RuntimeError(
                    "reward_tau is not initialized for thresholded reward. "
                    "Provide it explicitly or load a cache so it can be inferred."
                )
            z = (float(self.reward_tau) - loss_normed) / max(float(self.threshold_temperature), 1e-12)
            s = _sigmoid(z)
            reward = float(self.reward_epsilon + (1.0 - self.reward_epsilon) * (s ** self.beta))
        else:
            raise ValueError(f"Unsupported scalar reward mode: {mode}")

        return max(reward, 1e-12)

    def _context_loss_to_reward(self, loss_by_context: Dict[str, float]) -> Optional[float]:
        if self.reward_mode == "context_kth_exp_simple":
            vals = self._transformed_context_values_rank(loss_by_context)
            if not vals:
                if self.context_fallback_to_scalar:
                    return None
                return max(self.reward_epsilon, 1e-12)
            vals_sorted = sorted(float(v) for v in vals)
            k = min(max(int(self.context_top_k), 1), len(vals_sorted))
            kth_rank = float(vals_sorted[k - 1])
            return max(float(np.exp(-self.beta * kth_rank)), 1e-12)

        if self.reward_mode in {"context_cvar_exp_simple", "context_cvar_blend_exp_simple"}:
            vals = self._transformed_context_values_cvar(loss_by_context)
            if not vals:
                if self.context_fallback_to_scalar:
                    return None
                return max(self.reward_epsilon, 1e-12)
            vals_sorted = sorted(float(v) for v in vals)
            m = min(max(int(self.context_tail_count), 1), len(vals_sorted))
            tail_mean = float(np.mean(vals_sorted[-m:]))
            if self.reward_mode == "context_cvar_exp_simple":
                agg = tail_mean
            else:
                mean_all = float(np.mean(vals_sorted))
                agg = (1.0 - self.context_cvar_lambda) * mean_all + self.context_cvar_lambda * tail_mean
            return max(float(np.exp(-self.beta * agg)), 1e-12)

        raise ValueError(f"Unsupported contextual reward mode: {self.reward_mode}")

    def _loss_to_reward(self, loss: float, loss_by_context: Optional[Dict[str, float]] = None) -> float:
        if self._contextual_reward_mode():
            reward = self._context_loss_to_reward(loss_by_context or {})
            if reward is not None:
                return reward
            return self._scalar_loss_to_reward(loss)
        return self._scalar_loss_to_reward(loss, mode=self.reward_mode)

    def _init_pool(self):
        from fmu.pool import PersistentFMUPool

        if self.pool is None:
            self.pool = PersistentFMUPool(
                self.teams,
                self.fmu_path,
                self.data_dir,
                step_size=self.step_size,
                max_uses=self.pool_max_uses,
                max_restarts=self.pool_max_restarts,
                loss_type=self.live_loss_type,
                huber_delta=self.live_huber_delta,
                relative_floor_frac=self.live_relative_floor_frac,
                relative_floor_abs=self.live_relative_floor_abs,
            )

    def _evaluate_live_with_pool(self, config: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        self._init_pool()

        full_config = {**BASELINE_PARAMETERS, **INITIAL_CONDITIONS, **config}
        team_losses = self.pool.evaluate(full_config)
        if not team_losses:
            return 1e6, {}

        ctx = {}
        per_team = []
        for team, errs in zip(self.teams, team_losses):
            errs = list(errs)
            if errs:
                mean_err = float(np.mean(errs))
                ctx[str(team)] = mean_err
                per_team.append(mean_err)

        scalar = float(np.mean(per_team)) if per_team else 1e6
        return scalar, ctx

    def _should_recalibrate(self) -> bool:
        if self.reward_anchor is None or self.reward_scale is None:
            return True
        if self.reward_mode == "thresholded_sigmoid" and self.reward_tau is None:
            return True
        if self._contextual_reward_mode():
            if not self._context_cache_available():
                return False
            return self.new_since_recalibration >= self.recalibrate_every
        return False

    @torch.no_grad()
    def __call__(self, states_proxy):
        items = list(states_proxy)
        keys_in_order: List[str] = []
        pending: Dict[str, Optional[Dict[str, float]]] = {}

        # First pass: identify hits vs unique misses
        for item in items:
            cache_key, config = self._extract_cache_key_and_config(item)
            keys_in_order.append(cache_key)
            if cache_key in self.reward_cache:
                self.cache_hits += 1
            else:
                self.cache_misses += 1
                if cache_key not in pending:
                    pending[cache_key] = config

        # Evaluate each unique miss once, reusing the persistent pool
        if pending:
            for cache_key, config in pending.items():
                if config is None:
                    loss, ctx = 1e6, {}
                else:
                    loss, ctx = self._evaluate_live_with_pool(config)

                self.reward_cache[cache_key] = {
                    "loss": float(loss),
                    "loss_by_context": {str(team): float(v) for team, v in (ctx or {}).items()},
                }
                self.cache_new += 1
                self.new_since_recalibration += 1

            if self._should_recalibrate():
                self._maybe_init_reward_calibration_from_cache(force=True)
                self.new_since_recalibration = 0

            if self.reward_cache_path and self.cache_new >= self.cache_save_every:
                self._save_cache()
                print(
                    f"  [CACHE] Saved {len(self.reward_cache)} entries "
                    f"({self.cache_hits} hits, {self.cache_misses} misses)"
                )
                self.cache_new = 0

        # Second pass: compute rewards in original order
        out = []
        for cache_key in keys_in_order:
            entry = self.reward_cache[cache_key]
            reward = self._loss_to_reward(
                float(entry["loss"]),
                dict(entry.get("loss_by_context", {})),
            )
            out.append(reward)

        return torch.tensor(out, dtype=self.float, device=self.device)

    def save_final_cache(self):
        if self.cache_new > 0:
            self._save_cache()
            print(
                f"  [CACHE] Final save: {len(self.reward_cache)} entries "
                f"({self.cache_hits} hits, {self.cache_misses} misses)"
            )
            self.cache_new = 0

        if self.pool is not None:
            try:
                self.pool.shutdown()
            except Exception:
                pass

    @staticmethod
    def compare_to_baseline(c):
        CropSimulatorProxy.compare_configs(c, BASELINE_PARAMETERS)

    @staticmethod
    def compare_configs(c1, c2):
        for key in sorted(set(c1) | set(c2)):
            val1 = float(c1.get(key, np.nan))
            val2 = float(c2.get(key, np.nan))
            print(f"{key}: param={val1} | baseline={val2} | diff={np.abs(val1 - val2)}")

    @staticmethod
    def get_team_control_dataset(data_dir, team):
        fp_climate = f"{data_dir}/{team}/GreenhouseClimate.csv"
        climate_df = load_climate_data(fp_climate)
        return climate_df[["CO2air", "PAR", "Tair"]]

    @staticmethod
    def get_team_obs_dataset(data_dir, team):
        fp_production = f"{data_dir}/{team}/Production.csv"
        fp_tomato = f"{data_dir}/{team}/TomQuality.csv"
        fp_parameter = f"{data_dir}/{team}/CropParameters.csv"

        prod_df = load_prod_data(fp_production)
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

    @staticmethod
    def compute_trace(sim_df, delta="5min"):
        sim_df = sim_df.copy()
        sim_df["Tair24"] = (
            sim_df["Tair"]
            .groupby(sim_df.index.date)
            .transform("mean")
            .round(2)
        )
        sim_df.index = sim_df.index.round(delta)
        sim_df = sim_df.groupby(level=0).mean()
        sim_df.index = (sim_df.index - sim_df.index.min()).total_seconds()
        trace = [
            (t, {
                "CO2_Air": row.CO2air,
                "PAR_gh": row.PAR,
                "TCan": row.Tair,
                "TCan24": row.Tair24,
            })
            for t, row in sim_df.iterrows()
        ]
        return trace
