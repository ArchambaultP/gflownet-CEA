
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from gflownet.proxy.base import Proxy
from gflownet.envs.greenhouse.constants import BASELINE_PARAMETERS, INITIAL_CONDITIONS


SCALAR_REWARD_MODES = {"softmin", "thresholded_sigmoid"}
CONTEXT_REWARD_MODES = {
    "context_kth_exp_simple",
    "context_cvar_exp_simple",
    "context_cvar_blend_exp_simple",
}


def _dbg(msg: str) -> None:
    try:
        os.write(2, (msg.rstrip() + "\n").encode())
    except Exception:
        pass


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


def _sigmoid(z):
    z = np.clip(z, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


class CropSimulatorProxy(Proxy):
    """
    Rewritten contextual crop simulator proxy.

    Main goals:
    - always return a Tensor, never None
    - explicit stderr debug at each stage
    - support a hybrid live evaluator:
        * pool for small batches
        * batch_contextual for larger pending miss batches
    - periodic calibration / cache save only
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
        # live eval controls
        live_eval_backend: str = "auto",   # auto | pool | batch
        batch_eval_min_pending: int = 8,
        batch_eval_n_workers: int = 24,
        batch_eval_timeout: int = 600,
        pool_eval_timeout: int = 120,
        live_loss_type: str = "absolute_relative",
        live_huber_delta: float = 1.0,
        live_relative_floor_frac: float = 0.05,
        live_relative_floor_abs: float = 1e-6,
        pool_max_uses: int = 1,
        recalibrate_every: int = 64,
        verbose_debug: bool = True,
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

        self.context_top_k = int(context_top_k)
        self.context_tail_count = int(context_tail_count)
        self.context_cvar_lambda = float(context_cvar_lambda)
        self.context_fallback_to_scalar = bool(context_fallback_to_scalar)

        self.live_eval_backend = str(live_eval_backend)
        self.batch_eval_min_pending = int(max(1, batch_eval_min_pending))
        self.batch_eval_n_workers = int(max(1, batch_eval_n_workers))
        self.batch_eval_timeout = int(max(1, batch_eval_timeout))
        self.pool_eval_timeout = int(max(1, pool_eval_timeout))
        self.live_loss_type = str(live_loss_type)
        self.live_huber_delta = float(live_huber_delta)
        self.live_relative_floor_frac = float(live_relative_floor_frac)
        self.live_relative_floor_abs = float(live_relative_floor_abs)
        self.pool_max_uses = int(max(1, pool_max_uses))
        self.recalibrate_every = int(max(1, recalibrate_every))
        self.verbose_debug = bool(verbose_debug)

        self.reward_cache: Dict[str, Dict[str, object]] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_new = 0
        self.cache_new_since_recalibration = 0
        self.cache_save_every = int(cache_save_every)
        self.reward_cache_path = reward_cache_path

        self.context_rank_arrays: Dict[str, np.ndarray] = {}
        self.context_quantile_stats: Dict[str, Tuple[float, float]] = {}

        self.pool = None

        _dbg(f"[PROXY IMPORT/INIT] pid={os.getpid()} backend={self.live_eval_backend}")

        if reward_cache_path and os.path.exists(reward_cache_path):
            self._load_cache(reward_cache_path)
            _dbg(f"[CACHE LOAD] entries={len(self.reward_cache)} path={reward_cache_path}")
        else:
            _dbg(f"[CACHE LOAD] none path={reward_cache_path}")

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
            return item, None, False
        if isinstance(item, (tuple, list)) and len(item) == 2 and isinstance(item[0], str):
            return str(item[0]), item[1], False
        if isinstance(item, dict) and "state_key" in item:
            return str(item["state_key"]), item.get("values", item.get("config", None)), False
        if hasattr(item, "state_key"):
            return str(getattr(item, "state_key")), getattr(item, "values", None), False
        return self._make_fallback_hash_key(item), item, True

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

    def _context_cache_available(self) -> bool:
        return any(entry.get("loss_by_context") for entry in self.reward_cache.values())

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

        if self._contextual_reward_mode() and self._context_cache_available():
            self.context_rank_arrays = self._build_context_rank_stats()
            self.context_quantile_stats = self._build_context_quantile_stats()

        _dbg(
            f"[CALIBRATE] force={force} cache={len(self.reward_cache)} "
            f"anchor={self.reward_anchor} scale={self.reward_scale} tau={self.reward_tau}"
        )

    def _should_recalibrate(self) -> bool:
        if self.reward_anchor is None or self.reward_scale is None:
            return True
        if self.reward_mode == "thresholded_sigmoid" and self.reward_tau is None:
            return True
        if self._contextual_reward_mode():
            return self.cache_new_since_recalibration >= self.recalibrate_every
        return False

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

        if self.reward_mode == "context_kth_exp_simple":
            # simpler loop, robust
            rewards = []
            for key in cache_keys:
                ctx = self.reward_cache[key].get("loss_by_context", {})
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
            return np.maximum(np.asarray(rewards, dtype=float), 1e-12)

        rewards = []
        for key in cache_keys:
            ctx = self.reward_cache[key].get("loss_by_context", {})
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
        _dbg("[POOL CREATE START]")
        from fmu.pool.persistent import PersistentFMUPool
        self.pool = PersistentFMUPool(
            self.teams,
            self.fmu_path,
            self.data_dir,
            step_size=self.step_size,
            max_uses=self.pool_max_uses,
            loss_type=self.live_loss_type,
            huber_delta=self.live_huber_delta,
            relative_floor_frac=self.live_relative_floor_frac,
            relative_floor_abs=self.live_relative_floor_abs,
        )
        _dbg("[POOL CREATE DONE]")

    def _evaluate_one_with_pool(self, config: Dict[str, float]) -> Dict[str, object]:
        self._ensure_pool()
        full_config = {**BASELINE_PARAMETERS, **INITIAL_CONDITIONS, **config}
        if self._contextual_reward_mode():
            team_results = self.pool.evaluate_contextual(full_config, timeout=self.pool_eval_timeout)
            ctx = {}
            means = []
            for team, point_losses in team_results.items():
                if point_losses:
                    m = float(np.mean(point_losses))
                    ctx[str(team)] = m
                    means.append(m)
            scalar = float(np.mean(means)) if means else 1e6
            return {"loss": scalar, "loss_by_context": ctx}
        else:
            team_losses = self.pool.evaluate(full_config, timeout=self.pool_eval_timeout)
            means = [float(np.mean(x)) for x in team_losses if x]
            scalar = float(np.mean(means)) if means else 1e6
            return {"loss": scalar, "loss_by_context": {}}

    def _evaluate_pending_with_batch(self, numeric_states: Dict[str, Dict[str, float]]) -> Optional[Dict[str, Dict[str, object]]]:
        try:
            from fmu.pool.batch_contextual import evaluate_all
        except Exception as e:
            _dbg(f"[BATCH EVAL IMPORT FAIL] {type(e).__name__}: {e}")
            return None

        states = {(key,): cfg for key, cfg in numeric_states.items()}
        _dbg(f"[BATCH EVAL START] n_states={len(states)} workers={self.batch_eval_n_workers}")
        t0 = time.perf_counter()
        try:
            losses, details = evaluate_all(
                states=states,
                fmu_path=self.fmu_path,
                team_ids=self.teams,
                data_dir=self.data_dir,
                n_workers=self.batch_eval_n_workers,
                timeout=self.batch_eval_timeout,
                verbose=False,
                loss_type=self.live_loss_type,
                huber_delta=self.live_huber_delta,
                relative_floor_frac=self.live_relative_floor_frac,
                relative_floor_abs=self.live_relative_floor_abs,
                return_details=True,
            )
        except Exception as e:
            _dbg(f"[BATCH EVAL FAIL] {type(e).__name__}: {e}")
            return None

        out: Dict[str, Dict[str, object]] = {}
        lbc = details.get("loss_by_context", {}) if isinstance(details, dict) else {}
        for tup_key, loss in losses.items():
            key = tup_key[0] if isinstance(tup_key, tuple) else str(tup_key)
            ctx = lbc.get(tup_key, lbc.get(key, {}))
            out[key] = {
                "loss": float(loss),
                "loss_by_context": {str(team): float(v) for team, v in ctx.items()},
            }
        _dbg(f"[BATCH EVAL DONE] elapsed={time.perf_counter()-t0:.3f}s got={len(out)}")
        return out

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

        use_batch = False
        if self.live_eval_backend == "batch":
            use_batch = True
        elif self.live_eval_backend == "auto" and len(numeric_states) >= self.batch_eval_min_pending:
            use_batch = True

        batch_results = None
        if use_batch and self._contextual_reward_mode():
            batch_results = self._evaluate_pending_with_batch(numeric_states)

        if batch_results is not None:
            results.update(batch_results)
            missing = [k for k in numeric_states if k not in results]
        else:
            missing = list(numeric_states.keys())

        if missing:
            _dbg(f"[POOL FALLBACK START] n_states={len(missing)}")
            for key in missing:
                _dbg(f"[POOL EVAL START] key={key}")
                t0 = time.perf_counter()
                try:
                    results[key] = self._evaluate_one_with_pool(numeric_states[key])
                except Exception as e:
                    _dbg(f"[POOL EVAL FAIL] key={key} {type(e).__name__}: {e}")
                    results[key] = {"loss": 1e6, "loss_by_context": {}}
                _dbg(f"[POOL EVAL DONE] key={key} elapsed={time.perf_counter()-t0:.3f}s")
            _dbg("[POOL FALLBACK DONE]")

        return results

    @torch.no_grad()
    def __call__(self, states_proxy):
        _dbg(f"[PROXY CALL ENTER] pid={os.getpid()} type={type(states_proxy).__name__}")
        t0 = time.perf_counter()
        items = list(states_proxy)
        _dbg(f"[PROXY CALL AFTER LIST] n_items={len(items)} elapsed={time.perf_counter()-t0:.3f}s")

        cache_keys: List[str] = []
        pending_configs: Dict[str, Optional[Dict[str, float]]] = {}
        fallback_hash_count = 0

        for item in items:
            key, values_like, used_hash = self._unpack_proxy_item(item)
            cache_keys.append(key)
            if used_hash:
                fallback_hash_count += 1

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

        if fallback_hash_count:
            _dbg(f"[CACHE KEY WARNING] fallback_hash_count={fallback_hash_count}")

        _dbg(f"[MISS BLOCK START] pending={len(pending_configs)} hits={self.cache_hits} misses={self.cache_misses}")
        if pending_configs:
            t_eval = time.perf_counter()
            batch_results = self._evaluate_live_batch(pending_configs)
            _dbg(f"[MISS BLOCK END] got_results={len(batch_results)} elapsed={time.perf_counter()-t_eval:.3f}s")

            for key, entry in batch_results.items():
                self.reward_cache[key] = {
                    "loss": float(entry["loss"]),
                    "loss_by_context": {str(team): float(v) for team, v in entry.get("loss_by_context", {}).items()},
                }
            self.cache_new += len(batch_results)
            self.cache_new_since_recalibration += len(batch_results)

            if self._should_recalibrate():
                self._maybe_init_reward_calibration_from_cache(force=True)
                self.cache_new_since_recalibration = 0

            if self.reward_cache_path and self.cache_new >= self.cache_save_every:
                self._save_cache()
                _dbg(f"[CACHE SAVE] entries={len(self.reward_cache)}")
                self.cache_new = 0

        t1 = time.perf_counter()
        _dbg(f"[BEFORE REWARD] elapsed={t1-t0:.3f}s")
        rewards = self._compute_rewards_vectorised(cache_keys)
        t2 = time.perf_counter()
        _dbg(f"[AFTER REWARD] elapsed={t2-t1:.3f}s batch_size={len(cache_keys)}")
        out = torch.tensor(rewards, dtype=self.float, device=self.device)
        _dbg(f"[PROXY RETURN] shape={tuple(out.shape)}")
        return out

    def save_final_cache(self):
        if self.cache_new > 0:
            self._save_cache()
            _dbg(f"[CACHE FINAL SAVE] entries={len(self.reward_cache)}")
            self.cache_new = 0
        if self.pool is not None:
            try:
                self.pool.shutdown()
            except Exception:
                pass
