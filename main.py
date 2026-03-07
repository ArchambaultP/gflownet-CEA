# import torch
# from dataclasses import dataclass
# from tqdm import tqdm
from fmu.tomato_controller import TomatoController

# from typing import List
# from gflownet.envs.greenhouse.crop_env import CropEnv
# from gflownet.proxy.greenhouse.secondEdition import GreenHouseChallenge2ndEdition
from gflownet.proxy.greenhouse.cropSimulatorProxy import CropSimulatorProxy
from gflownet.envs.greenhouse.constants import (
    BASELINE_PARAMETERS, INITIAL_CONDITIONS, PARAMETER_BOUNDS
)

from fmu.fmu_pool import run_parallel
# import pandas as pd
# import datetime
# from data.greenhouse.secondEdition.extract import load_data

# from gflownet.envs.greenhouse.sim_env import CropSimEnv, BASELINE_PARAMETERS

# from botorch.models import SingleTaskGP
# from botorch.fit import fit_gpytorch_mll
# from botorch.models.transforms.input import InputStandardize
# from botorch.models.transforms.outcome import Standardize
# from gpytorch.mlls import ExactMarginalLogLikelihood
# from gpytorch.likelihoods import GaussianLikelihood
# from gpytorch.constraints import GreaterThan
# from gpytorch.kernels import MaternKernel, ScaleKernel

# from data.greenhouse.secondEdition.extract import load_prod_data, load_climate_data, extract_2nd_edition_climate_data, extract_2nd_edition_production_data
# import matplotlib.pyplot as plt
# import pickle
# import numpy as np

TEAM_IDS = [
    "Reference",
    "Digilog",
    "IUACAAS",
    "Automatoes",
    "TheAutomators",
    "AICU",
]

DATA_DIR = "data/greenhouse/secondEdition"
FMU_PATH = "fmu/FMU/tomato.fmu"
STEP_SIZE = 120.0

def log_callback(instance_environment, instance_name, status, category, message):
    print(f"[FMU] {category}: {message}")

def main():
    t = "Reference"
    data = CropSimulatorProxy.get_team_control_dataset(DATA_DIR, t)
    input_trace = CropSimulatorProxy.compute_trace(data, delta="30min")
    obs = CropSimulatorProxy.get_team_obs_dataset(DATA_DIR, t)
    setpoints = (obs.index - obs.index.min())[1:].total_seconds().tolist()

    print(f"stop_time: {input_trace[-1][0]}")
    print(f"setpoints: {setpoints}")
    print(f"n_inputs: {len(input_trace)}")

    tomato = TomatoController(
        'fmu/FMU/tomato.fmu',
        start_time=0,
        stop_time=input_trace[-1][0],
        step_size=120.0,
        logger=log_callback,
    )
    out = tomato.simulate(input_trace, setpoints, None)
    print(f"Direct result: {len(out)} outputs")

# def main():


#     tomato = TomatoController('fmu/FMU/tomato.fmu', 
#                                         start_time=0, # inital simulation time (in seconds). should not change
#                                         stop_time=86400.0 * 200, # Final simulation time (in seconds).
#                                         step_size=120.0, #numerical solver step size (in seconds)
#                                         logger=log_callback)

#     inputs = [(0, {"CO2_Air":400.0, "PAR_gh":500.0, "TCan":20.0, "TCan24":20.0})]
#     setpoints = [86400.0 * 30]

#     out = tomato.simulate(inputs, setpoints, None)

#     args_by_team = {}
#     team_obs = {}
#     init = BASELINE_PARAMETERS | INITIAL_CONDITIONS
#     for t in TEAM_IDS:
#         data = CropSimulatorProxy.get_team_control_dataset(DATA_DIR, t)
#         input_trace = CropSimulatorProxy.compute_trace(data, delta="30min")
#         obs = CropSimulatorProxy.get_team_obs_dataset(DATA_DIR, t)
#         setpoints = (obs.index - obs.index.min())[1:].total_seconds().tolist()
#         args_by_team[t] = (input_trace, setpoints, init, STEP_SIZE)
#         team_obs[t] = obs

#     results = run_parallel(args_by_team, FMU_PATH, timeout=30, verbose=True, max_workers=1, work_dir=None)


    # out = proxy([test])
    # env = CropSimEnv(init_profile=None, fmu_path = 'fmu/FMU/tomato.fmu', growth_step=1, growth_period=2,device="CUDA")
    
    

    # params = [ "LAI_max", "SLA", "rho_can", "rho_floor", "n_plants", "K1", "K2", "J_max_leaf", "Jpot_activation", "Jpot_deactivation", "Jpot_entropy", "Jpot_ref_temp", "alpha", "deg_curv_elec_transport", "Tcan_CO2_comp_point", "net_ass_rate", "G_max", "CO2_air_stomata", "bias_g_Tcan24", "slope_g_Tcan24", "molar_gas_constant", "mass_CH20", "c_fruit_growth", "c_leaf_growth", "c_stem_growth", "c_fruit_maintenance", "c_leaf_maintenance", "c_stem_maintenance", "Q_10_maintenance", "rg_fruit", "rg_leaf", "rg_stem", "c_rgr", "TS_start", "TS_end", "c_dev1", "c_dev2", "n_fruit_phases", "r_fruit_Set", "k_sw_max_Cbuff", "k_sw_min_Cbuff", "c_max_buf_fruit_1", "c_max_buf_fruit_2", "s_MCairbuf_Cbuf", "s_MCbuforg_Cbuf", "s_harvest" ]
    # param_dict = {}

    # print("{")
    # for s in params:
    #     id = tomato.param_vars[s]
    #     val = tomato.fmu.getFloat64([id])

    #     print(f"\"{s}\":{round(val[0],5)},")
    # print("}")

    # proxy = CropSimulatorProxy()
    # data_dir = "data/greenhouse/secondEdition"
    # team = "AICU"
    # climate_file = f"Reference/GreenhouseClimate.csv"
    # production_file = f"Reference/Production.csv"

    # out = proxy([[np.float64(0.65), np.float64(1700.0), np.float64(225.0), np.float64(43400.0), np.float64(195000.0), np.float64(710.0), 298.15, np.float64(0.7999999999999999), np.float64(0.7999999999999999), np.float64(3.75), np.float64(2.2), np.float64(2.75e-05), np.float64(800.0), np.float64(0.0), np.float64(2.26), np.float64(0.375), np.float64(-1.25), np.float64(-8.5e-09), np.float64(1.8999999999999998e-08), np.float64(0.29000000000000004), np.float64(1.5e-07), np.float64(0.29000000000000004), np.float64(4.2499999999999995e-07), np.float64(-1.25e-07), np.float64(1.14e-06), np.float64(3000000.0), np.float64(0.32), np.float64(1.85e-07), np.float64(0.644), np.float64(23400.0), np.float64(33.5), np.float64(23.5), np.float64(1490.0), np.float64(10.5), np.float64(15.0), 0.03, 8.314, 50.0, np.float64(4.25), np.float64(4.5), np.float64(0.185), np.float64(0.44999999999999996), np.float64(0.09), np.float64(0.06999999999999999), np.float64(0.096), np.float64(0.5), np.float64(-0.000275), np.float64(-0.000275), np.float64(-0.000275), np.float64(0.65), np.float64(1.25), np.float64(-0.95), np.float64(-1.25), np.float64(0.158)]])


    breakpoint()
    print("done")

# from gfn.gflownet import TBGFlowNet
# from gfn.gym import HyperGrid  # We use the hyper grid environment
# from gfn.preprocessors import KHotPreprocessor
# from gfn.modules import DiscretePolicyEstimator
# from gfn.samplers import Sampler
# from gfn.utils.modules import MLP  # is a simple multi-layer perceptron (MLP)

# def main():
#     # 1 - We define the environment.
#     env = HyperGrid(ndim=4, height=8, R0=0.01)  # Grid of size 8x8x8x8
#     preprocessor = KHotPreprocessor(ndim=env.ndim, height=env.height)

#     # 2 - We define the needed modules (neural networks).
#     module_PF = MLP(
#         input_dim=preprocessor.output_dim,
#         output_dim=env.n_actions
#     )  # Neural network for the forward policy, with as many outputs as there are actions

#     module_PB = MLP(
#         input_dim=preprocessor.output_dim,
#         output_dim=env.n_actions - 1,
#         trunk=module_PF.trunk  # We share all the parameters of P_F and P_B, except for the last layer
#     )

#     # 3 - We define the estimators.
#     pf_estimator = DiscretePolicyEstimator(module_PF, env.n_actions, is_backward=False, preprocessor=preprocessor)
#     pb_estimator = DiscretePolicyEstimator(module_PB, env.n_actions, is_backward=True, preprocessor=preprocessor)

#     # 4 - We define the GFlowNet.
#     gfn = TBGFlowNet(logZ=0., pf=pf_estimator, pb=pb_estimator)  # We initialize logZ to 0

#     # 5 - We define the sampler and the optimizer.
#     sampler = Sampler(estimator=pf_estimator)  # We use an on-policy sampler, based on the forward policy

#     # Different policy parameters can have their own LR.
#     # Log Z gets dedicated learning rate (typically higher).
#     optimizer = torch.optim.Adam(gfn.pf_pb_parameters(), lr=1e-3)
#     optimizer.add_param_group({"params": gfn.logz_parameters(), "lr": 1e-1})

#     # 6 - We train the GFlowNet for 1000 iterations, with 16 trajectories per iteration
#     for i in (pbar := tqdm(range(1000))):
#         trajectories = sampler.sample_trajectories(env=env, n=16, save_logprobs=True)  # The save_logprobs=True makes on-policy training faster
#         optimizer.zero_grad()
#         loss = gfn.loss(env, trajectories)
#         loss.backward()
#         optimizer.step()
#         if i % 25 == 0:
#             pbar.set_postfix({"loss": loss.item()})

if __name__ == "__main__":
    main()
