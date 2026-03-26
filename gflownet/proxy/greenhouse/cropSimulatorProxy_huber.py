
import hashlib
import json
import os

import numpy as np
import pandas as pd
import torch
from fmu.tomato_controller import TomatoController
from gflownet.proxy.base import Proxy
from gflownet.envs.greenhouse.constants import (
    BASELINE_PARAMETERS, PARAMETER_BOUNDS, INITIAL_CONDITIONS,
)
from data.greenhouse.secondEdition.extract import (
    load_climate_data, load_prod_data, load_tomato_data, load_parameter_data,
)


class CropSimulatorProxy(Proxy):

    def __init__(
        self,
        reward_cache_path=None,
        beta=None,
        cache_save_every=100,
        reward_transform="softmin",
        reward_eps=1e-8,
        reward_clip_min=1e-12,
        loss_type="huber_relative",
        huber_delta=1.0,
        relative_floor_frac=0.05,
        relative_floor_abs=1e-6,
        **kwargs,
    ):
        """
        Parameters
        ----------
        reward_cache_path : str or None
            Path to a JSON file for the reward cache. If the file exists, it is
            loaded. New evaluations are appended and periodically saved back.
        beta : float or None
            Inverse-temperature for the reward transform.
        cache_save_every : int
            Save the cache to disk every N new evaluations. Default: 100.
        reward_transform : {"softmin", "invpower"}
            softmin  -> reward = exp(-beta * loss)
            invpower -> reward = (1 / max(loss, eps)) ** beta
        reward_eps : float
            Small floor used for stable reward computation.
        reward_clip_min : float
            Minimum positive reward after transformation.
        loss_type : {"huber_relative", "rse", "absolute_relative"}
            Pointwise error to compute inside each team trajectory.
        huber_delta : float
            Transition point of the Huber loss, expressed in normalized
            relative-error units.
        relative_floor_frac : float
            Denominator floor as a fraction of the per-series scale
            (90th percentile of |y|).
        relative_floor_abs : float
            Absolute denominator floor, in case a series is near-zero.
        """
        super().__init__(**kwargs)

        self.teams = [
            "Reference", "Digilog", "IUACAAS",
            "Automatoes", "TheAutomators", "AICU",
        ]
        self.data_dir = "data/greenhouse/secondEdition"
        self.fmu_path = "fmu/FMU/tomato.fmu"
        self.step_size = 120.0
        self.parameter_names = sorted(BASELINE_PARAMETERS.keys())
        self.beta = float(beta if beta is not None else 10.0)
        self.reward_transform = str(reward_transform).lower()
        self.reward_eps = float(reward_eps)
        self.reward_clip_min = float(reward_clip_min)
        self.loss_type = str(loss_type).lower()
        self.huber_delta = float(huber_delta)
        self.relative_floor_frac = float(relative_floor_frac)
        self.relative_floor_abs = float(relative_floor_abs)

        if self.reward_transform not in {"softmin", "invpower"}:
            raise ValueError(
                f"Unsupported reward_transform={self.reward_transform!r}; "
                "expected 'softmin' or 'invpower'."
            )
        if self.loss_type not in {"huber_relative", "rse", "absolute_relative"}:
            raise ValueError(
                f"Unsupported loss_type={self.loss_type!r}; expected "
                "'huber_relative', 'rse', or 'absolute_relative'."
            )

        # Cache configuration
        self.reward_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_new = 0  # count of new entries since last save
        self.cache_save_every = cache_save_every
        self.reward_cache_path = reward_cache_path

        # Load existing cache if available
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

        # Only initialize FMU infrastructure if we might need it
        if not self.precomputed:
            self._init_fmu()

    def _make_cache_key(self, states_proxy_item):
        """
        Create a deterministic cache key from a proxy state.

        For action-string keys (1-cycle precomputed): the key is the string itself.
        For parameter vectors (2-cycle live): hash the rounded values to avoid
        floating-point noise causing cache misses.
        """
        if isinstance(states_proxy_item, str):
            return states_proxy_item

        # Numeric vector — round to avoid float noise, then hash
        if hasattr(states_proxy_item, 'tolist'):
            vals = states_proxy_item.tolist()
        elif isinstance(states_proxy_item, (list, tuple)):
            vals = list(states_proxy_item)
        else:
            vals = [float(states_proxy_item)]

        # Round to 8 decimal places to handle float imprecision
        rounded = tuple(round(v, 8) for v in vals)
        return hashlib.sha256(str(rounded).encode()).hexdigest()[:16]

    def _load_cache(self, path):
        with open(path) as f:
            raw = json.load(f)

        for key, entry in raw.items():
            if isinstance(entry, dict) and "loss" in entry:
                self.reward_cache[key] = entry["loss"]
            else:
                # Support simple key: loss format too
                self.reward_cache[key] = entry

    def _save_cache(self):
        """Save the current cache to disk."""
        if not self.reward_cache_path:
            return

        cache_out = {}
        for key, loss in self.reward_cache.items():
            if isinstance(loss, (int, float)):
                cache_out[key] = {"loss": float(loss)}
            else:
                cache_out[key] = loss

        os.makedirs(os.path.dirname(self.reward_cache_path) or ".", exist_ok=True)
        with open(self.reward_cache_path, "w") as f:
            json.dump(cache_out, f)

    def _init_fmu(self):
        from fmu.pool import PersistentFMUPool

        self.team_obs_data = {}
        self.team_input = {}

        for t in self.teams:
            control_df = self.get_team_control_dataset(self.data_dir, t)
            self.team_input[t] = self.compute_trace(control_df, delta="30min")
            self.team_obs_data[t] = self.get_team_obs_dataset(self.data_dir, t)

        self.pool = PersistentFMUPool(
            self.teams,
            self.fmu_path,
            self.data_dir,
            step_size=self.step_size,
            max_uses=1,
            loss_type=self.loss_type,
            huber_delta=self.huber_delta,
            relative_floor_frac=self.relative_floor_frac,
            relative_floor_abs=self.relative_floor_abs,
        )

    def _loss_to_log_reward(self, loss):
        loss = float(loss)
        if not np.isfinite(loss):
            loss = 1e6
        safe_loss = max(loss, self.reward_eps)
        if self.reward_transform == "softmin":
            return -self.beta * safe_loss
        if self.reward_transform == "invpower":
            return -self.beta * np.log(safe_loss)
        raise RuntimeError(f"Unexpected reward_transform={self.reward_transform!r}")

    def _loss_to_reward(self, loss):
        log_reward = self._loss_to_log_reward(loss)
        log_reward = np.clip(log_reward, np.log(self.reward_clip_min), 50.0)
        reward = float(np.exp(log_reward))
        return max(reward, self.reward_clip_min)

    def _evaluate_live(self, config):
        if not hasattr(self, "pool"):
            self._init_fmu()

        full_config = {**BASELINE_PARAMETERS, **INITIAL_CONDITIONS, **config}

        team_losses = self.pool.evaluate(full_config)
        if not team_losses:
            return 1e6

        per_team = [np.mean(errs) for errs in team_losses if len(errs) > 0]
        if not per_team:
            return 1e6
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
                # Build parameter config from GFlowNet output
                self.cache_misses += 1
                config = {}
                if isinstance(batch, str):
                    # Action-string format — shouldn't happen for uncached states
                    # but handle gracefully
                    loss = 1e6
                else:
                    if hasattr(batch, 'tolist'):
                        values = batch.tolist()
                    else:
                        values = list(batch)
                    for i, name in enumerate(self.parameter_names):
                        config[name] = float(values[i])
                    loss = self._evaluate_live(config)

                # Store in cache
                self.reward_cache[cache_key] = float(loss)
                self.cache_new += 1

                # Periodically save cache to disk
                if (self.reward_cache_path and self.cache_new >= self.cache_save_every):
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
        """Call at the end of training to flush any remaining cached entries."""
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
