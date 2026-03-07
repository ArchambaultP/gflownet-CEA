import pickle
import os
import torch
from gflownet.proxy.base import Proxy
from fmu.tomato_controller import TomatoController
from gflownet.envs.greenhouse.constants import BASELINE_PARAMETERS, PARAMETER_BOUNDS, INITIAL_CONDITIONS
from data.greenhouse.secondEdition.extract import load_climate_data, load_prod_data, load_tomato_data, load_parameter_data
from botorch.models.transforms.input import InputStandardize
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.kernels import MaternKernel, ScaleKernel
import numpy as np
import pandas as pd
import time
import signal
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeoutError
import shutil
import tempfile
import multiprocessing as mp
# mp.set_start_method("spawn", force=True)
from fmu.pool import PersistentFMUPool

# mp.set_start_method("spawn", force=True)

class CropSimulatorProxy(Proxy):

    def __init__(self, n_samples=1000, **kwargs):
        super().__init__(**kwargs)
        self.teams = [
        "Reference",
        "Digilog",
        "IUACAAS",
        "Automatoes",
        "TheAutomators",
        "AICU"
        ]
        
        self.team_obs_data = {}
        self.team_input = {}

        self.data_dir = "data/greenhouse/secondEdition"
        self.fmu_path = "fmu/FMU/tomato.fmu"

        #Single β to calibrate. Run the FMU once at your starting parameters (midpoint of bounds), compute L at those parameters, call it L_start. Set self.beta = 4.6 / L_start. That's it. This gives R ≈ 0.01 at starting params and R = 1.0 for a perfect fit.
        self.beta = 5.65881 #calibrate with calibrate_beta.py
        self.step_size = 120.0
        self.parameter_names = sorted(BASELINE_PARAMETERS.keys())

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
    
    @torch.no_grad()
    def __call__(self, states_proxy):
        config = {}
        for batch in states_proxy:
            for i, name in enumerate(self.parameter_names):
                config[name] = batch[i]

        config = {**BASELINE_PARAMETERS, **INITIAL_CONDITIONS, **config}

        team_losses = self.pool.evaluate(config)

        if not team_losses:
            reward = 0.0
        else:
            per_team = [np.mean(errs) for errs in team_losses]
            L = np.mean(per_team)
            reward = np.exp(-self.beta * L)

        return torch.tensor(reward, dtype=self.float, device=self.device)
    
    # @torch.no_grad()
    # def __call__(self, states_proxy):
    #     config = {}
    #     for batch in states_proxy:
    #         for i, name in enumerate(self.parameter_names):
    #             config[name] = batch[i]

    #     # Build args per team
    #     args_by_team = {}
    #     for t in self.teams:
    #         obs_data = self.team_obs_data[t]
    #         setpoints = (obs_data.index - obs_data.index.min())[1:].total_seconds().tolist()
    #         args_by_team[t] = (self.team_input[t], setpoints, config, self.step_size)

    #     # Run all teams in parallel
    #     results = run_parallel(args_by_team, self.fmu_path, timeout=10, verbose=True)

    #     # Score
    #     team_losses = []
    #     for t, sim_out in results.items():
    #         obs_data = self.team_obs_data[t]
    #         team_errors = []
    #         for idx, (time_val, output) in enumerate(sim_out):
    #             y_DM = obs_data["DM_harvest_obs"].iloc[idx]
    #             y_N = obs_data["N_harvest_per_m2"].iloc[idx]
    #             y_hat_DM = output["C_harvest"]
    #             y_hat_N = output["N_harvest"]

    #             if y_DM > 0:
    #                 team_errors.append(((y_hat_DM - y_DM) / y_DM) ** 2)
    #             if y_N > 0:
    #                 team_errors.append(((y_hat_N - y_N) / y_N) ** 2)

    #         if team_errors:
    #             team_losses.append(np.mean(team_errors))

    #     if not team_losses:
    #         reward = 0.0
    #     else:
    #         L = np.mean(team_losses)
    #         reward = np.exp(-self.beta * L)

    #     return torch.tensor(reward, dtype=self.float, device=self.device)
    
    @staticmethod
    def compare_to_baseline(c):
        CropSimulatorProxy.compare_configs(c, BASELINE_PARAMETERS)
    
    @staticmethod
    def compare_configs(c1, c2):
        for key in sorted(set(c1) | set(c2)):
            val1 = float(c1.get(key, np.nan))
            val2 = float(c2.get(key, np.nan))
            print(f"{key}: param={val1} | baseline={val2} | diff={np.abs(val1-val2)}")
    
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
            "DAP":prod_df["DAP"],
        })
        
        tomato_df = load_tomato_data(fp_tomato)
        param_df = load_parameter_data(fp_parameter)

        # df = pd.merge(climate_df, prod_df, on="Time", how="outer")
        df = pd.merge(prod_df, param_df, on="Time", how="outer")
        df = pd.merge(df, tomato_df, on="Time", how="inner")
        df = df.ffill()

        df["N_harvest_per_m2"] = ((df["N"]/10) * df["stem_density"]).cumsum()

        # Raw data: grams total across 10 sample stems
        df["yield_fw_g_m2"] = (df["Yield"] / 10) * df["stem_density"]
        # Fresh weight → dry matter (g/m²)
        df["dry_weight_g_m2"] = df["yield_fw_g_m2"] * (df["dryMatterPercent"] / 100)
        # g{DM}/m² → mg{CH2O}/m²
        df["dry_weight_mg_CH2O_m2"] = df["dry_weight_g_m2"] * 1000
        # Cumulative for comparison against model DM_Harvest
        df["DM_harvest_obs"] = df["dry_weight_mg_CH2O_m2"].cumsum()

        return df[["DM_harvest_obs", "N_harvest_per_m2"]]

    @staticmethod
    def compute_trace(sim_df, delta="5min"):
        sim_df["Tair24"] = sim_df["Tair"].groupby(sim_df.index.date).transform("mean").round(2)
        sim_df.index = sim_df.index.round(delta)
        sim_df = sim_df.groupby(level=0).mean() #remove potential duplicates
        sim_df.index = (sim_df.index - sim_df.index.min()).total_seconds()
        trace = [(t,
                    {'CO2_Air': row.CO2air, 
                    'PAR_gh': row.PAR, 
                    'TCan': row.Tair, 
                    'TCan24': row.Tair24,
                    }) 
                for t, row in sim_df.iterrows()]

        return trace