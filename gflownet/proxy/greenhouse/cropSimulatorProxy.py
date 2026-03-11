import hashlib
import json
import os
import time

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

    def __init__(self, reward_cache_path=None, beta=None, **kwargs):
        super().__init__(**kwargs)

        self.teams = [
            "Reference", "Digilog", "IUACAAS",
            "Automatoes", "TheAutomators", "AICU",
        ]
        self.data_dir = "data/greenhouse/secondEdition"
        self.fmu_path = "fmu/FMU/tomato.fmu"
        self.step_size = 120.0
        self.parameter_names = sorted(PARAMETER_BOUNDS.keys())
        self.beta = beta if beta is not None else 95

        # Load precomputed cache if available
        self.reward_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

        if reward_cache_path and os.path.exists(reward_cache_path):
            self._load_cache(reward_cache_path)
            self.precomputed = True
            print(f"Loaded {len(self.reward_cache)} cached evaluations from {reward_cache_path}")
        else:
            self.precomputed = False
            print("No reward cache found — will use live FMU evaluation")

        # Only initialize FMU infrastructure if we might need it
        if not self.reward_cache:
            self._init_fmu()

    def _load_cache(self, path):
        with open(path) as f:
            raw = json.load(f)

        for key, entry in raw.items():
            params = entry["params"]
            loss = entry["loss"]
            # Re-key by parameter hash so proxy can look up from values

            cache_key = key
            self.reward_cache[cache_key] = loss

        # Also store action-keyed version for debugging
        self.action_cache = {k: v["loss"] for k, v in raw.items()}

    def _hash_params(self, params):
        rounded = tuple(
            round(float(params[k]), 4)
            for k in self.parameter_names
            if k in params
        )
        return hashlib.md5(str(rounded).encode()).hexdigest()

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
        
        # def log_callback(instance_environment, instance_name, status, category, message):
        #     print(f"[FMU] {category}: {message}")

        # tomato = TomatoController('fmu/FMU/tomato.fmu', 
        #                             start_time=0, # inital simulation time (in seconds). should not change
        #                             stop_time=86400.0 * 200, # Final simulation time (in seconds).
        #                             step_size=120.0, #numerical solver step size (in seconds)
        #                             logger=log_callback)
        
        # test_inp = self.team_input['Reference']
        # test_cond = full_config
        # # test_setpoint = [86400*150]

        # test_out = tomato.simulate(test_inp, test_setpoint, init_conds=test_cond)
        # # inputs = [(0, {"CO2_Air":400.0, "PAR_gh":500.0, "TCan":20.0, "TCan24":20.0})]
        # # setpoints = [86400.0 * 30]

        # # out = tomato.simulate(inputs, setpoints, None)
        # # breakpoint()
        
        team_losses = self.pool.evaluate(full_config)
        breakpoint()
        if not team_losses:
            return 1e6

        per_team = [np.mean(errs) for errs in team_losses]
        return np.mean(per_team)

    @torch.no_grad()
    def __call__(self, states_proxy):
        out = []

        for batch in states_proxy:
            if self.precomputed:
                if batch in self.reward_cache:
                    loss = self.reward_cache[batch]
                    self.cache_hits += 1
            else:
                # Build parameter config from GFlowNet output
                self.cache_misses +=1
                config = {}
                for i, name in enumerate(self.parameter_names):
                    config[name] = float(batch[i])
                loss = self._evaluate_live(config)


            # we switch reward from exponential to power law
            # beta is named that to simplify code
            # it should be called alpha
            reward = (1/loss) ** self.beta # -beta * loss
            reward = np.clip(reward, min=1e-12)
            out.append(reward)

        return torch.tensor(out, dtype=self.float, device=self.device)

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