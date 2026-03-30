import hashlib
import json
import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
from gflownet.proxy.base import Proxy
from gflownet.envs.greenhouse.constants import (
    BASELINE_PARAMETERS, INITIAL_CONDITIONS,
)
from data.greenhouse.secondEdition.extract import (
    load_climate_data, load_prod_data, load_tomato_data, load_parameter_data,
)


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


class CropSimulatorProxy(Proxy):
    """
    Crop simulator proxy with thresholded reward support.

    Reward modes
    ------------
    - softmin:
        reward = exp(-beta * loss_norm)

    - thresholded_sigmoid:
        s = sigmoid((tau - loss_norm) / T)
        reward = eps + (1 - eps) * s**beta

      Here beta acts as a power-law sharpening parameter on top of the
      thresholded reward. beta=1 leaves the plain thresholded reward unchanged.
      beta>1 emphasizes the very best states inside the good region.
      beta<1 broadens the reward inside the good region.
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
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.teams = [
            "Reference", "Digilog", "IUACAAS",
            "Automatoes", "TheAutomators", "AICU",
        ]
        self.data_dir = "data/greenhouse/secondEdition"
        self.fmu_path = "fmu/FMU/tomato.fmu"
        self.step_size = 120.0
        self.parameter_names = sorted(BASELINE_PARAMETERS.keys())

        self.reward_mode = str(reward_mode)

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

        self.reward_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_new = 0
        self.cache_save_every = int(cache_save_every)
        self.reward_cache_path = reward_cache_path

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

        self._maybe_init_reward_calibration_from_cache()

        if not self.precomputed:
            self._init_fmu()

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
            if isinstance(entry, dict) and "loss" in entry:
                self.reward_cache[key] = float(entry["loss"])
            else:
                self.reward_cache[key] = float(entry)

    def _save_cache(self):
        if not self.reward_cache_path:
            return

        cache_out = {}
        for key, loss in self.reward_cache.items():
            cache_out[key] = {"loss": float(loss)}

        os.makedirs(os.path.dirname(self.reward_cache_path) or ".", exist_ok=True)
        with open(self.reward_cache_path, "w") as f:
            json.dump(cache_out, f, indent=2)

    @staticmethod
    def _sigmoid(z):
        z = np.clip(z, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-z))

    def _maybe_init_reward_calibration_from_cache(self):
        if len(self.reward_cache) == 0:
            return

        losses = np.array(list(self.reward_cache.values()), dtype=float)

        if self.loss_norm == "none":
            if self.reward_anchor is None:
                self.reward_anchor = 0.0
            if self.reward_scale is None:
                self.reward_scale = 1.0
            loss_normed = losses.copy()

        elif self.loss_norm == "q10q90":
            if self.reward_anchor is None:
                self.reward_anchor = float(np.quantile(losses, self.q_low))
            if self.reward_scale is None:
                qh = float(np.quantile(losses, self.q_high))
                self.reward_scale = max(qh - self.reward_anchor, 1e-12)
            loss_normed = (losses - self.reward_anchor) / max(self.reward_scale, 1e-12)

        else:
            raise ValueError(f"Unsupported loss_norm: {self.loss_norm}")

        if self.reward_mode == "thresholded_sigmoid" and self.reward_tau is None:
            self.reward_tau = float(np.quantile(loss_normed, self.tau_quantile))

        if self.reward_mode == "thresholded_sigmoid":
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

    def _loss_to_reward(self, loss: float) -> float:
        loss_normed = self._normalize_loss(loss)

        if self.reward_mode == "softmin":
            reward = float(np.exp(-self.beta * loss_normed))

        elif self.reward_mode == "thresholded_sigmoid":
            if self.reward_tau is None:
                raise RuntimeError(
                    "reward_tau is not initialized for thresholded reward. "
                    "Provide it explicitly or load a cache so it can be inferred."
                )
            z = (float(self.reward_tau) - loss_normed) / max(float(self.threshold_temperature), 1e-12)
            s = float(self._sigmoid(z))
            reward = float(self.reward_epsilon + (1.0 - self.reward_epsilon) * (s ** self.beta))

        else:
            raise ValueError(f"Unsupported reward_mode: {self.reward_mode}")

        return max(reward, 1e-12)

    def _init_fmu(self):
        from fmu.pool import PersistentFMUPool

        self.team_obs_data = {}
        self.team_input = {}

        for t in self.teams:
            control_df = self.get_team_control_dataset(self.data_dir, t)
            self.team_input[t] = self.compute_trace(control_df, delta="30min")
            self.team_obs_data[t] = self.get_team_obs_dataset(self.data_dir, t)

        self.pool = PersistentFMUPool(
            self.teams, self.fmu_path, self.data_dir,
            step_size=self.step_size, max_uses=1,
        )

    def _evaluate_live(self, config):
        if not hasattr(self, "pool"):
            self._init_fmu()

        full_config = {**BASELINE_PARAMETERS, **INITIAL_CONDITIONS, **config}

        team_losses = self.pool.evaluate(full_config)
        if not team_losses:
            return 1e6

        per_team = [np.mean(errs) for errs in team_losses]
        return float(np.mean(per_team))

    @torch.no_grad()
    def __call__(self, states_proxy):
        out = []

        for batch in states_proxy:
            cache_key = self._make_cache_key(batch)

            if cache_key in self.reward_cache:
                loss = self.reward_cache[cache_key]
                self.cache_hits += 1
            else:
                self.cache_misses += 1

                if isinstance(batch, str):
                    loss = 1e6
                else:
                    if hasattr(batch, "tolist"):
                        values = batch.tolist()
                    else:
                        values = list(batch)

                    config = {}
                    for i, name in enumerate(self.parameter_names):
                        config[name] = float(values[i])

                    loss = self._evaluate_live(config)

                self.reward_cache[cache_key] = float(loss)
                self.cache_new += 1

                if self.reward_anchor is None or self.reward_scale is None or (
                    self.reward_mode == "thresholded_sigmoid" and self.reward_tau is None
                ):
                    self._maybe_init_reward_calibration_from_cache()

                if self.reward_cache_path and self.cache_new >= self.cache_save_every:
                    self._save_cache()
                    print(
                        f"  [CACHE] Saved {len(self.reward_cache)} entries "
                        f"({self.cache_hits} hits, {self.cache_misses} misses)"
                    )
                    self.cache_new = 0

            reward = self._loss_to_reward(loss)
            out.append(reward)

        return torch.tensor(out, dtype=self.float, device=self.device)

    def save_final_cache(self):
        if self.cache_new > 0:
            self._save_cache()
            print(
                f"  [CACHE] Final save: {len(self.reward_cache)} entries "
                f"({self.cache_hits} hits, {self.cache_misses} misses)"
            )

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
