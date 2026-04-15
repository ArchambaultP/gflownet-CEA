import hashlib
import json
import os
from typing import Dict, List, Optional

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
    "context_topk_linear",
    "context_topk_exp",
    "context_topk_powerlaw",
    "context_kbest_exp",
    "context_kbest_powerlaw",
    "context_softctx_exp",
    "context_softctx_powerlaw",
    "context_kth_exp",
    "context_kth_exp_power",
    "context_kth_sigmoid",
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
            raise ValueError(
                f"{name} must be numeric or null, got {x!r}. "
                f"If you copied a placeholder like '<anchor from calibration>', "
                f"replace it with a real number or remove the field."
            ) from e
    raise TypeError(f"{name} must be numeric or null, got type {type(x).__name__}")


def _sigmoid(z: float) -> float:
    z = float(np.clip(z, -60.0, 60.0))
    return float(1.0 / (1.0 + np.exp(-z)))


class CropSimulatorProxy(Proxy):
    """
    Crop simulator proxy with scalar and context-aware reward support.

    Scalar reward modes
    -------------------
    - softmin:
        reward = exp(-beta * loss_norm)

    - thresholded_sigmoid:
        s = sigmoid((tau - loss_norm) / T)
        reward = eps + (1 - eps) * s**beta

    Context reward modes
    --------------------
    These mirror rebuild_contextual.py:

    - context_topk_linear
    - context_topk_exp
    - context_topk_powerlaw
    - context_kbest_exp
    - context_kbest_powerlaw
    - context_softctx_exp
    - context_softctx_powerlaw
    - context_kth_exp
    - context_kth_exp_power
    - context_kth_sigmoid

    Context transform
    -----------------
    - normalized_loss
    - percentile_rank
    """

    def __init__(
        self,
        reward_cache_path: Optional[str] = None,
        reward_mode: str = "thresholded_sigmoid",
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
        powerlaw_eps: float = 0.05,
        context_transform: str = "percentile_rank",
        context_top_k: int = 2,
        context_tau: float = 0.20,
        softctx_tau: float = 0.20,
        rank_power: float = 1.0,
        sigmoid_tau: float = 0.10,
        sigmoid_temperature: float = 0.02,
        clip_upper: bool = False,
        context_team_ids: Optional[List[str]] = None,
        context_fallback_to_scalar: bool = True,
        scalar_fallback_reward_mode: str = "softmin",
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

        if self.reward_mode not in SCALAR_REWARD_MODES | CONTEXT_REWARD_MODES:
            raise ValueError(f"Unsupported reward_mode: {self.reward_mode}")

        if self.scalar_fallback_reward_mode not in SCALAR_REWARD_MODES:
            raise ValueError(
                f"Unsupported scalar_fallback_reward_mode: {self.scalar_fallback_reward_mode}"
            )

        self.loss_norm = str(loss_norm)
        self.q_low = float(q_low)
        self.q_high = float(q_high)

        self.reward_anchor = _maybe_float(reward_anchor, "reward_anchor")
        self.reward_scale = _maybe_float(reward_scale, "reward_scale")
        self.reward_tau = _maybe_float(reward_tau, "reward_tau")
        self.tau_quantile = float(tau_quantile)
        self.threshold_temperature = float(threshold_temperature)
        self.reward_epsilon = float(reward_epsilon)
        self.powerlaw_eps = float(powerlaw_eps)

        self.context_transform = str(context_transform)
        if self.context_transform not in {"normalized_loss", "percentile_rank"}:
            raise ValueError(f"Unsupported context_transform: {self.context_transform}")

        self.context_top_k = int(context_top_k)
        self.context_tau = float(context_tau)
        self.softctx_tau = float(softctx_tau)
        self.rank_power = float(rank_power)
        self.sigmoid_tau = float(sigmoid_tau)
        self.sigmoid_temperature = float(sigmoid_temperature)
        self.clip_upper = bool(clip_upper)
        self.context_fallback_to_scalar = bool(context_fallback_to_scalar)

        beta_val = _maybe_float(beta, "beta")
        if beta_val is None:
            beta_val = 3.0 if self.reward_mode == "softmin" else 1.0
        self.beta = float(beta_val)

        self.reward_cache: Dict[str, Dict[str, object]] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_new = 0
        self.cache_save_every = int(cache_save_every)
        self.reward_cache_path = reward_cache_path

        self.context_anchor_by_team: Dict[str, float] = {}
        self.context_scale_by_team: Dict[str, float] = {}
        self.context_rank_arrays: Dict[str, np.ndarray] = {}

        if reward_cache_path and os.path.exists(reward_cache_path):
            self._load_cache(reward_cache_path)
            self.precomputed = len(self.reward_cache) > 0
            print(f"Loaded {len(self.reward_cache)} cached evaluations from {reward_cache_path}")
        else:
            self.precomputed = False
            if reward_cache_path:
                print(f"No cache found at {reward_cache_path} — will build lazily")
            else:
                print("No reward cache path — will use live FMU evaluation (no caching)")

        self._maybe_init_reward_calibration_from_cache(force=True)

    def _contextual_reward_mode(self) -> bool:
        return self.reward_mode in CONTEXT_REWARD_MODES

    def _make_cache_key(self, states_proxy_item):
        if isinstance(states_proxy_item, str):
            return states_proxy_item

        if hasattr(states_proxy_item, "tolist"):
            vals = states_proxy_item.tolist()
        elif isinstance(states_proxy_item, (list, tuple)):
            vals = list(states_proxy_item)
        else:
            vals = [float(states_proxy_item)]

        rounded = tuple(round(float(v), 8) for v in vals)
        return hashlib.sha256(str(rounded).encode()).hexdigest()[:16]

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
                self.reward_cache[key] = {
                    "loss": loss,
                    "loss_by_context": loss_by_context,
                }
            else:
                self.reward_cache[key] = {
                    "loss": float(entry),
                    "loss_by_context": {},
                }

    def _save_cache(self):
        if not self.reward_cache_path:
            return

        cache_out = {}
        for key, entry in self.reward_cache.items():
            cache_out[key] = {
                "loss": float(entry["loss"]),
                "loss_by_context": {
                    team: float(v) for team, v in entry.get("loss_by_context", {}).items()
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
            loss_normed = np.asarray(
                [(float(x) - float(self.reward_anchor)) / max(float(self.reward_scale), 1e-12) for x in losses],
                dtype=float,
            ) if self.loss_norm != "none" else losses.copy()
            self.reward_tau = float(np.quantile(loss_normed, self.tau_quantile))

        if self._context_cache_available():
            by_team: Dict[str, List[float]] = {}
            for entry in self.reward_cache.values():
                for team, loss in entry.get("loss_by_context", {}).items():
                    by_team.setdefault(str(team), []).append(float(loss))

            if self.context_transform == "normalized_loss":
                self.context_rank_arrays = {}
                self.context_anchor_by_team = {}
                self.context_scale_by_team = {}
                for team, team_losses in by_team.items():
                    arr = np.asarray(team_losses, dtype=float)
                    if self.loss_norm == "none":
                        anchor, scale = 0.0, 1.0
                    elif self.loss_norm == "q10q90":
                        anchor = float(np.quantile(arr, self.q_low))
                        qh = float(np.quantile(arr, self.q_high))
                        scale = max(qh - anchor, 1e-12)
                    else:
                        raise ValueError(f"Unsupported loss_norm: {self.loss_norm}")

                    self.context_anchor_by_team[team] = float(anchor)
                    self.context_scale_by_team[team] = float(scale)

            elif self.context_transform == "percentile_rank":
                self.context_anchor_by_team = {}
                self.context_scale_by_team = {}
                self.context_rank_arrays = self._build_context_rank_stats()

            else:
                raise ValueError(f"Unsupported context_transform: {self.context_transform}")

        if self._contextual_reward_mode():
            print(
                "[reward calibration] "
                f"mode={self.reward_mode}, transform={self.context_transform}, "
                f"k={self.context_top_k}, beta={self.beta}, rank_power={self.rank_power}, "
                f"context_tau={self.context_tau}, softctx_tau={self.softctx_tau}, "
                f"sigmoid_tau={self.sigmoid_tau}, sigmoid_T={self.sigmoid_temperature}, "
                f"contexts={sorted(set(self.context_anchor_by_team) | set(self.context_rank_arrays))}"
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

    def _normalize_context_loss(self, team: str, loss: float) -> float:
        if self.loss_norm == "none":
            return float(loss)

        if team not in self.context_anchor_by_team or team not in self.context_scale_by_team:
            raise RuntimeError(
                f"Context normalization stats not initialized for team={team!r}. "
                "Precompute states with loss_by_context or use percentile_rank."
            )
        return (float(loss) - self.context_anchor_by_team[team]) / max(
            self.context_scale_by_team[team], 1e-12
        )

    @staticmethod
    def _percentile_rank(sorted_arr: np.ndarray, value: float) -> float:
        n = int(sorted_arr.size)
        if n <= 1:
            return 0.0
        idx = int(np.searchsorted(sorted_arr, value, side="left"))
        return float(idx / (n - 1))

    def _transformed_context_values(self, loss_by_context: Dict[str, float]) -> List[float]:
        vals: List[float] = []

        for team in self.teams:
            if team not in loss_by_context:
                continue

            loss = float(loss_by_context[team])

            if self.context_transform == "normalized_loss":
                z = self._normalize_context_loss(team, loss)
                vals.append(float(z))
            elif self.context_transform == "percentile_rank":
                if team not in self.context_rank_arrays:
                    continue
                r = self._percentile_rank(self.context_rank_arrays[team], loss)
                vals.append(float(r))
            else:
                raise ValueError(f"Unsupported context_transform: {self.context_transform}")

        return vals

    def _clipped_linear_scores(self, vals: List[float]) -> List[float]:
        out: List[float] = []
        tau = max(float(self.context_tau), 1e-12)
        for v in vals:
            s = 1.0 - float(v) / tau
            s = max(0.0, s)
            if self.clip_upper:
                s = min(1.0, s)
            out.append(float(s))
        return out

    def _aggregate_contextual_loss(self, vals: List[float]) -> Optional[float]:
        if not vals:
            return None

        if self.reward_mode in {"context_topk_linear", "context_topk_exp", "context_topk_powerlaw"}:
            scores = self._clipped_linear_scores(vals)
            if not scores:
                return None
            scores = sorted(scores, reverse=True)
            k = min(max(int(self.context_top_k), 1), len(scores))
            mean_top = float(np.mean(scores[:k]))
            return float(max(0.0, 1.0 - mean_top))

        if self.reward_mode in {"context_kbest_exp", "context_kbest_powerlaw"}:
            vals_sorted = sorted(float(v) for v in vals)
            k = min(max(int(self.context_top_k), 1), len(vals_sorted))
            return float(np.mean(vals_sorted[:k]))

        if self.reward_mode in {"context_softctx_exp", "context_softctx_powerlaw"}:
            tau = max(float(self.softctx_tau), 1e-12)
            arr = np.asarray(vals, dtype=float)
            m = float(np.min(arr))
            agg = m - tau * np.log(np.mean(np.exp(-(arr - m) / tau)))
            return float(agg)

        if self.reward_mode in {"context_kth_exp", "context_kth_exp_power", "context_kth_sigmoid"}:
            vals_sorted = sorted(float(v) for v in vals)
            k = min(max(int(self.context_top_k), 1), len(vals_sorted))
            return float(vals_sorted[k - 1])

        raise ValueError(f"Unsupported contextual reward_mode: {self.reward_mode}")

    def _reward_from_agg_loss(self, agg_loss: float) -> float:
        beta = float(self.beta)
        rank_power = float(self.rank_power)

        if self.reward_mode in {"context_topk_exp", "context_kbest_exp", "context_softctx_exp", "context_kth_exp"}:
            r = float(np.exp(-beta * float(agg_loss)))
        elif self.reward_mode == "context_kth_exp_power":
            r = float(np.exp(-beta * (max(float(agg_loss), 0.0) ** rank_power)))
        elif self.reward_mode in {"context_topk_powerlaw", "context_kbest_powerlaw", "context_softctx_powerlaw"}:
            r = float((float(agg_loss) + float(self.powerlaw_eps)) ** (-beta))
        elif self.reward_mode == "context_topk_linear":
            r = float(
                self.reward_epsilon
                + (1.0 - self.reward_epsilon) * max(0.0, 1.0 - float(agg_loss))
            )
        elif self.reward_mode == "context_kth_sigmoid":
            z = (float(self.sigmoid_tau) - float(agg_loss)) / max(
                float(self.sigmoid_temperature), 1e-12
            )
            s = _sigmoid(z)
            r = float(self.reward_epsilon + (1.0 - self.reward_epsilon) * (s ** beta))
        else:
            raise ValueError(f"Unsupported contextual reward_mode: {self.reward_mode}")

        return max(r, 1e-12)

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
            z = (float(self.reward_tau) - loss_normed) / max(
                float(self.threshold_temperature), 1e-12
            )
            s = float(_sigmoid(z))
            reward = float(
                self.reward_epsilon + (1.0 - self.reward_epsilon) * (s ** self.beta)
            )
        else:
            raise ValueError(f"Unsupported scalar reward mode: {mode}")

        return max(reward, 1e-12)

    def _context_loss_to_reward(self, loss_by_context: Dict[str, float]) -> Optional[float]:
        vals = self._transformed_context_values(loss_by_context)
        agg_loss = self._aggregate_contextual_loss(vals)

        if agg_loss is None:
            if self.context_fallback_to_scalar:
                return None
            return max(self.reward_epsilon, 1e-12)

        return self._reward_from_agg_loss(agg_loss)

    def _loss_to_reward(
        self,
        loss: float,
        loss_by_context: Optional[Dict[str, float]] = None,
    ) -> float:
        if self._contextual_reward_mode():
            reward = self._context_loss_to_reward(loss_by_context or {})
            if reward is not None:
                return reward
            return self._scalar_loss_to_reward(loss)

        return self._scalar_loss_to_reward(loss, mode=self.reward_mode)

    def _evaluate_live(self, config):
        """
        Live evaluation path.

        For any contextual reward mode, use the batch-contextual evaluator on a
        single state so we can recover named per-context losses robustly.
        """
        if self._contextual_reward_mode():
            try:
                from fmu.pool.batch_contextual import evaluate_all
            except Exception as e:
                if not self.context_fallback_to_scalar:
                    raise
                print(
                    "[WARN] batch_contextual unavailable for live miss, "
                    f"falling back to scalar pool eval: {e}"
                )
            else:
                states = {("live",): {**config}}
                losses, details = evaluate_all(
                    states=states,
                    fmu_path=self.fmu_path,
                    team_ids=self.teams,
                    data_dir=self.data_dir,
                    n_workers=min(len(self.teams), 8),
                    timeout=600,
                    verbose=False,
                    loss_type="absolute_relative",
                    huber_delta=1.0,
                    relative_floor_frac=0.05,
                    relative_floor_abs=1e-6,
                    return_details=True,
                )
                loss = float(losses.get(("live",), 1e6))
                ctx = details.get("loss_by_context", {}).get(("live",), {})
                return loss, {str(team): float(v) for team, v in ctx.items()}

        from fmu.pool import PersistentFMUPool

        if not hasattr(self, "pool"):
            self.pool = PersistentFMUPool(
                self.teams,
                self.fmu_path,
                self.data_dir,
                step_size=self.step_size,
                max_uses=1,
            )

        full_config = {**BASELINE_PARAMETERS, **INITIAL_CONDITIONS, **config}

        team_losses = self.pool.evaluate(full_config)
        if not team_losses:
            return 1e6, {}

        per_team = [float(np.mean(errs)) for errs in team_losses if len(errs)]
        scalar = float(np.mean(per_team)) if per_team else 1e6

        ctx = {}
        for team, errs in zip(self.teams, team_losses):
            errs = list(errs)
            if errs:
                ctx[str(team)] = float(np.mean(errs))
        return scalar, ctx

    @torch.no_grad()
    def __call__(self, states_proxy):
        out = []

        for batch in states_proxy:
            cache_key = self._make_cache_key(batch)

            if cache_key in self.reward_cache:
                entry = self.reward_cache[cache_key]
                loss = float(entry["loss"])
                loss_by_context = dict(entry.get("loss_by_context", {}))
                self.cache_hits += 1
            else:
                self.cache_misses += 1

                if isinstance(batch, str):
                    loss = 1e6
                    loss_by_context = {}
                else:
                    if hasattr(batch, "tolist"):
                        values = batch.tolist()
                    else:
                        values = list(batch)

                    config = {}
                    for i, name in enumerate(self.parameter_names):
                        config[name] = float(values[i])

                    loss, loss_by_context = self._evaluate_live(config)

                self.reward_cache[cache_key] = {
                    "loss": float(loss),
                    "loss_by_context": {
                        team: float(v) for team, v in (loss_by_context or {}).items()
                    },
                }
                self.cache_new += 1

                need_recalibration = (
                    self.reward_anchor is None
                    or self.reward_scale is None
                    or (self.reward_mode == "thresholded_sigmoid" and self.reward_tau is None)
                    or (
                        self._contextual_reward_mode()
                        and self.context_transform == "normalized_loss"
                        and len(self.context_anchor_by_team) == 0
                    )
                    or (
                        self._contextual_reward_mode()
                        and self.context_transform == "percentile_rank"
                    )
                )
                if need_recalibration:
                    self._maybe_init_reward_calibration_from_cache(force=True)

                if self.reward_cache_path and self.cache_new >= self.cache_save_every:
                    self._save_cache()
                    print(
                        f"  [CACHE] Saved {len(self.reward_cache)} entries "
                        f"({self.cache_hits} hits, {self.cache_misses} misses)"
                    )
                    self.cache_new = 0

            reward = self._loss_to_reward(loss, loss_by_context)
            out.append(reward)

        return torch.tensor(out, dtype=self.float, device=self.device)

    def save_final_cache(self):
        if self.cache_new > 0:
            self._save_cache()
            print(
                f"  [CACHE] Final save: {len(self.reward_cache)} entries "
                f"({self.cache_hits} hits, {self.cache_misses} misses)"
            )
        if hasattr(self, "pool"):
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
        prod_df = pd.DataFrame(
            {
                "N": prod_df["nClassA"] + prod_df["nClassB"],
                "N_Sum": (prod_df["nClassA"] + prod_df["nClassB"]).cumsum(),
                "Yield": prod_df["gClassA"] + prod_df["gClassB"],
                "Yield_Sum": (prod_df["gClassA"] + prod_df["gClassB"]).cumsum(),
                "DAP": prod_df["DAP"],
            }
        )

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
            (
                t,
                {
                    "CO2_Air": row.CO2air,
                    "PAR_gh": row.PAR,
                    "TCan": row.Tair,
                    "TCan24": row.Tair24,
                },
            )
            for t, row in sim_df.iterrows()
        ]
        return trace
