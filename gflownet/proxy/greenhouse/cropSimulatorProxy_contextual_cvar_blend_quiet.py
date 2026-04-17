
from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from gflownet.proxy.base import Proxy
from gflownet.envs.greenhouse.constants import BASELINE_PARAMETERS


SCALAR_REWARD_MODES = {"softmin", "thresholded_sigmoid"}
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


def _to_int(x, name: str) -> int:
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return int(x)
    if isinstance(x, str):
        return int(x.strip())
    raise TypeError(f"{name} must be int-like, got {type(x).__name__}")


def _sigmoid(z):
    z = np.clip(z, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


class CropSimulatorProxy(Proxy):
    """
    Quiet contextual proxy.
    - no debug prints
    - persistent FMU pool live evaluation
    - returns scalar rewards for the GFlowNet proxy interface
    """

    def __init__(
        self,
        reward_cache_path: Optional[str] = None,
        reward_mode: str = "context_cvar_blend_exp_simple",
        beta: Optional[float] = None,
        cache_save_every: int = 100,
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
        live_loss_type: str = "absolute_relative",
        live_huber_delta: float = 1.0,
        live_relative_floor_frac: float = 0.05,
        live_relative_floor_abs: float = 1e-6,
        pool_eval_timeout: int = 120,
        pool_max_restarts: int = 3,
        setpoint_mode: str = "climate_start",
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
        self.data_dir = "data/greenhouse/secondEdition"
        self.fmu_path = "fmu/FMU/tomato.fmu"
        self.step_size = 120.0
        self.parameter_names = sorted(BASELINE_PARAMETERS.keys())

        self.reward_mode = str(reward_mode)
        self.scalar_fallback_reward_mode = str(scalar_fallback_reward_mode)
        if self.reward_mode not in (SCALAR_REWARD_MODES | CONTEXT_REWARD_MODES):
            raise ValueError(f"Unsupported reward_mode: {self.reward_mode}")
        if self.scalar_fallback_reward_mode not in SCALAR_REWARD_MODES:
            raise ValueError(f"Unsupported scalar_fallback_reward_mode: {self.scalar_fallback_reward_mode}")

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

        self.context_top_k = _to_int(context_top_k, "context_top_k")
        self.context_tail_count = _to_int(context_tail_count, "context_tail_count")
        self.context_cvar_lambda = float(context_cvar_lambda)
        self.context_fallback_to_scalar = bool(context_fallback_to_scalar)

        self.live_loss_type = str(live_loss_type)
        self.live_huber_delta = float(live_huber_delta)
        self.live_relative_floor_frac = float(live_relative_floor_frac)
        self.live_relative_floor_abs = float(live_relative_floor_abs)
        self.pool_eval_timeout = _to_int(pool_eval_timeout, "pool_eval_timeout")
        self.pool_max_restarts = _to_int(pool_max_restarts, "pool_max_restarts")
        self.setpoint_mode = str(setpoint_mode)

        self.reward_cache: Dict[str, Dict[str, object]] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_new = 0
        self.cache_save_every = _to_int(cache_save_every, "cache_save_every")
        self.reward_cache_path = reward_cache_path

        self.context_rank_arrays: Dict[str, np.ndarray] = {}
        self.context_quantile_stats: Dict[str, Tuple[float, float]] = {}
        self.pool = None

        if reward_cache_path and os.path.exists(reward_cache_path):
            self._load_cache(reward_cache_path)

        self._maybe_init_reward_calibration_from_cache(force=True)

    def _contextual_reward_mode(self) -> bool:
        return self.reward_mode in CONTEXT_REWARD_MODES

    def _make_fallback_hash_key(self, values):
        if hasattr(values, "tolist"):
            vals = values.tolist()
        elif isinstance(values, (list, tuple)):
            vals = list(values)
        else:
            vals = [float(values)]
        rounded = tuple(round(float(v), 8) for v in vals)
        return hashlib.sha256(str(rounded).encode()).hexdigest()[:16]

    def _unpack_proxy_item(self, item):
        if isinstance(item, str):
            return item, None
        if isinstance(item, (tuple, list)) and len(item) == 2 and isinstance(item[0], str):
            return str(item[0]), item[1]
        if isinstance(item, dict) and "state_key" in item:
            return str(item["state_key"]), item.get("values", item.get("config", None))
        if hasattr(item, "state_key"):
            return str(getattr(item, "state_key")), getattr(item, "values", None)
        return self._make_fallback_hash_key(item), item

    def _load_cache(self, path):
        with open(path) as f:
            raw = json.load(f)
        for key, entry in raw.items():
            if isinstance(entry, dict):
                self.reward_cache[str(key)] = {
                    "loss": float(entry.get("loss", entry.get("scalar_loss", 1e6))),
                    "loss_by_context": {
                        str(team): float(v)
                        for team, v in entry.get("loss_by_context", {}).items()
                    },
                }
            else:
                self.reward_cache[str(key)] = {"loss": float(entry), "loss_by_context": {}}

    def _save_cache(self):
        if not self.reward_cache_path:
            return
        out = {}
        for key, entry in self.reward_cache.items():
            out[key] = {
                "loss": float(entry["loss"]),
                "loss_by_context": {str(team): float(v) for team, v in entry.get("loss_by_context", {}).items()},
            }
        os.makedirs(os.path.dirname(self.reward_cache_path) or ".", exist_ok=True)
        tmp = self.reward_cache_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(out, f, indent=2)
        os.replace(tmp, self.reward_cache_path)

    def _build_context_rank_stats(self):
        by_team: Dict[str, List[float]] = {}
        for entry in self.reward_cache.values():
            for team, loss in entry.get("loss_by_context", {}).items():
                by_team.setdefault(str(team), []).append(float(loss))
        return {team: np.asarray(sorted(vals), dtype=float) for team, vals in by_team.items() if vals}

    def _build_context_quantile_stats(self):
        by_team: Dict[str, List[float]] = {}
        for entry in self.reward_cache.values():
            for team, loss in entry.get("loss_by_context", {}).items():
                by_team.setdefault(str(team), []).append(float(loss))
        out: Dict[str, Tuple[float, float]] = {}
        for team, vals in by_team.items():
            arr = np.asarray(vals, dtype=float)
            if arr.size:
                lo = float(np.quantile(arr, self.q_low))
                hi = float(np.quantile(arr, self.q_high))
                out[team] = (lo, max(hi - lo, 1e-12))
        return out

    def _maybe_init_reward_calibration_from_cache(self, force: bool = False):
        if not self.reward_cache:
            return
        losses = np.asarray([float(v["loss"]) for v in self.reward_cache.values()], dtype=float)
        if force or self.reward_anchor is None or self.reward_scale is None:
            if self.loss_norm == "none":
                self.reward_anchor = 0.0
                self.reward_scale = 1.0
            else:
                self.reward_anchor = float(np.quantile(losses, self.q_low))
                qh = float(np.quantile(losses, self.q_high))
                self.reward_scale = max(qh - self.reward_anchor, 1e-12)
        if self.reward_mode == "thresholded_sigmoid" and (force or self.reward_tau is None):
            normed = (losses - self.reward_anchor) / max(self.reward_scale, 1e-12)
            self.reward_tau = float(np.quantile(normed, self.tau_quantile))
        if self._contextual_reward_mode():
            self.context_rank_arrays = self._build_context_rank_stats()
            self.context_quantile_stats = self._build_context_quantile_stats()

    def _normalize_scalar_losses(self, losses: np.ndarray) -> np.ndarray:
        if self.loss_norm == "none":
            return losses.astype(float)
        return (losses.astype(float) - float(self.reward_anchor)) / max(float(self.reward_scale), 1e-12)

    def _scalar_rewards_vectorised(self, losses: np.ndarray) -> np.ndarray:
        normed = self._normalize_scalar_losses(losses)
        mode = self.reward_mode if self.reward_mode in SCALAR_REWARD_MODES else self.scalar_fallback_reward_mode
        if mode == "softmin":
            rewards = np.exp(-self.beta * normed)
        elif mode == "thresholded_sigmoid":
            z = (float(self.reward_tau) - normed) / max(float(self.threshold_temperature), 1e-12)
            s = _sigmoid(z)
            rewards = self.reward_epsilon + (1.0 - self.reward_epsilon) * np.power(s, self.beta)
        else:
            raise ValueError(f"Unsupported scalar mode {mode}")
        return np.maximum(rewards, 1e-12)

    def _compute_rewards_vectorised(self, cache_keys: List[str]) -> np.ndarray:
        losses = np.asarray([float(self.reward_cache[k]["loss"]) for k in cache_keys], dtype=float)
        if not self._contextual_reward_mode() or not self.context_quantile_stats:
            return self._scalar_rewards_vectorised(losses)

        rewards = []
        for key in cache_keys:
            ctx = self.reward_cache[key].get("loss_by_context", {})
            if self.reward_mode == "context_kth_exp_simple":
                vals = []
                for team in self.teams:
                    if team in ctx and team in self.context_rank_arrays:
                        arr = self.context_rank_arrays[team]
                        idx = int(np.searchsorted(arr, float(ctx[team]), side="left"))
                        vals.append(idx / max(arr.size - 1, 1))
                if not vals:
                    rewards.append(float(self._scalar_rewards_vectorised(np.asarray([self.reward_cache[key]["loss"]]))[0]))
                else:
                    vals = sorted(vals)
                    k = min(max(self.context_top_k, 1), len(vals))
                    rewards.append(float(np.exp(-self.beta * vals[k - 1])))
                continue

            vals = []
            for team in self.teams:
                if team in ctx and team in self.context_quantile_stats:
                    lo, scale = self.context_quantile_stats[team]
                    vals.append((float(ctx[team]) - lo) / max(scale, 1e-12))
            if not vals:
                if self.context_fallback_to_scalar:
                    rewards.append(float(self._scalar_rewards_vectorised(np.asarray([self.reward_cache[key]["loss"]]))[0]))
                else:
                    rewards.append(max(self.reward_epsilon, 1e-12))
                continue
            vals = sorted(vals)
            m = min(max(self.context_tail_count, 1), len(vals))
            tail_mean = float(np.mean(vals[-m:]))
            if self.reward_mode == "context_cvar_exp_simple":
                agg = tail_mean
            else:
                mean_all = float(np.mean(vals))
                agg = (1.0 - self.context_cvar_lambda) * mean_all + self.context_cvar_lambda * tail_mean
            rewards.append(float(np.exp(-self.beta * agg)))

        return np.maximum(np.asarray(rewards, dtype=float), 1e-12)

    def _ensure_pool(self):
        if self.pool is not None:
            return
        from fmu.pool.persistent_contextual_cvar import PersistentFMUPool
        self.pool = PersistentFMUPool(
            self.teams,
            self.fmu_path,
            self.data_dir,
            step_size=self.step_size,
            max_restarts=self.pool_max_restarts,
            loss_type=self.live_loss_type,
            huber_delta=self.live_huber_delta,
            relative_floor_frac=self.live_relative_floor_frac,
            relative_floor_abs=self.live_relative_floor_abs,
            setpoint_mode=self.setpoint_mode,
            verbose=False,
        )

    def _evaluate_live_batch(self, pending_configs: Dict[str, Optional[Dict[str, float]]]) -> Dict[str, Dict[str, object]]:
        results: Dict[str, Dict[str, object]] = {}
        if not pending_configs:
            return results

        for key, cfg in pending_configs.items():
            if cfg is None:
                results[key] = {"loss": 1e6, "loss_by_context": {}}

        numeric_states = {key: cfg for key, cfg in pending_configs.items() if cfg is not None}
        if not numeric_states:
            return results

        self._ensure_pool()

        for key, config in numeric_states.items():
            team_results = self.pool.evaluate_contextual(config, timeout=self.pool_eval_timeout)
            ctx = {}
            means = []
            for team, point_losses in team_results.items():
                if point_losses:
                    m = float(np.mean(point_losses))
                    ctx[str(team)] = m
                    means.append(m)
            scalar = float(np.mean(means)) if means else 1e6
            results[key] = {"loss": scalar, "loss_by_context": ctx}

        return results

    @torch.no_grad()
    def __call__(self, states_proxy):
        items = list(states_proxy)
        cache_keys: List[str] = []
        pending_configs: Dict[str, Optional[Dict[str, float]]] = {}

        for item in items:
            key, values_like = self._unpack_proxy_item(item)
            cache_keys.append(key)

            if key in self.reward_cache:
                self.cache_hits += 1
                continue
            if key in pending_configs:
                continue

            self.cache_misses += 1
            if values_like is None:
                pending_configs[key] = None
            else:
                values = values_like.tolist() if hasattr(values_like, "tolist") else list(values_like)
                pending_configs[key] = {name: float(values[i]) for i, name in enumerate(self.parameter_names)}

        if pending_configs:
            batch_results = self._evaluate_live_batch(pending_configs)
            for key, entry in batch_results.items():
                self.reward_cache[key] = {
                    "loss": float(entry["loss"]),
                    "loss_by_context": {str(team): float(v) for team, v in entry.get("loss_by_context", {}).items()},
                }
            self.cache_new += len(batch_results)
            if self.reward_cache_path and self.cache_new >= self.cache_save_every:
                self._save_cache()
                self.cache_new = 0
            if self._contextual_reward_mode() and batch_results:
                self._maybe_init_reward_calibration_from_cache(force=True)

        rewards = self._compute_rewards_vectorised(cache_keys)
        return torch.tensor(rewards, dtype=self.float, device=self.device)

    def save_final_cache(self):
        if self.cache_new > 0:
            self._save_cache()
            self.cache_new = 0
        if self.pool is not None:
            try:
                self.pool.shutdown()
            except Exception:
                pass
