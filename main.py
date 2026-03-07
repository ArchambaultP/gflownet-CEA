import torch
from dataclasses import dataclass
from tqdm import tqdm
# from models.plant import GrowthController
# from models.tomato_controller import TomatoController
from fmu.tomato_controller import TomatoController

from typing import List
from gflownet.envs.greenhouse.crop_env import CropEnv
from gflownet.proxy.greenhouse.secondEdition import GreenHouseChallenge2ndEdition
from gflownet.proxy.greenhouse.cropSimulatorProxy import CropSimulatorProxy

import pandas as pd
import datetime
from data.greenhouse.secondEdition.extract import load_data

from gflownet.envs.greenhouse.sim_env import CropSimEnv, BASELINE_PARAMETERS

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
# from botorch.utils import standardize
from botorch.models.transforms.input import InputStandardize
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import MaternKernel, ScaleKernel

from data.greenhouse.secondEdition.extract import load_prod_data, load_climate_data, extract_2nd_edition_climate_data, extract_2nd_edition_production_data
import matplotlib.pyplot as plt
import pickle
import numpy as np


def log_callback(instance_environment, instance_name, status, category, message):
    print(f"[FMU] {category}: {message}")

def main():
    # cropenv = CropEnv()
    proxy = GreenHouseChallenge2ndEdition()

    tomato = TomatoController('fmu/FMU/tomato_limited.fmu', 
                                        start_time=0, # inital simulation time (in seconds). should not change
                                        stop_time=86400.0 * 25, # Final simulation time (in seconds).
                                        step_size=120.0, #numerical solver step size (in seconds)
                                        logger=log_callback)
    


    test = BASELINE_PARAMETERS

    # out = proxy([test])
    breakpoint()
    # env = CropSimEnv(init_profile=None, fmu_path = 'fmu/FMU/tomato.fmu', growth_step=1, growth_period=2,device="CUDA")
    
    

    # params = [ "LAI_max", "SLA", "rho_can", "rho_floor", "n_plants", "K1", "K2", "J_max_leaf", "Jpot_activation", "Jpot_deactivation", "Jpot_entropy", "Jpot_ref_temp", "alpha", "deg_curv_elec_transport", "Tcan_CO2_comp_point", "net_ass_rate", "G_max", "CO2_air_stomata", "bias_g_Tcan24", "slope_g_Tcan24", "molar_gas_constant", "mass_CH20", "c_fruit_growth", "c_leaf_growth", "c_stem_growth", "c_fruit_maintenance", "c_leaf_maintenance", "c_stem_maintenance", "Q_10_maintenance", "rg_fruit", "rg_leaf", "rg_stem", "c_rgr", "TS_start", "TS_end", "c_dev1", "c_dev2", "n_fruit_phases", "r_fruit_Set", "k_sw_max_Cbuff", "k_sw_min_Cbuff", "c_max_buf_fruit_1", "c_max_buf_fruit_2", "s_MCairbuf_Cbuf", "s_MCbuforg_Cbuf", "s_harvest" ]
    # param_dict = {}

    # print("{")
    # for s in params:
    #     id = tomato.param_vars[s]
    #     val = tomato.fmu.getFloat64([id])

    #     print(f"\"{s}\":{round(val[0],5)},")
    # print("}")

    proxy = CropSimulatorProxy()
    data_dir = "data/greenhouse/secondEdition"
    team = "AICU"
    climate_file = f"Reference/GreenhouseClimate.csv"
    production_file = f"Reference/Production.csv"

    out = proxy([[np.float64(0.65), np.float64(1700.0), np.float64(225.0), np.float64(43400.0), np.float64(195000.0), np.float64(710.0), 298.15, np.float64(0.7999999999999999), np.float64(0.7999999999999999), np.float64(3.75), np.float64(2.2), np.float64(2.75e-05), np.float64(800.0), np.float64(0.0), np.float64(2.26), np.float64(0.375), np.float64(-1.25), np.float64(-8.5e-09), np.float64(1.8999999999999998e-08), np.float64(0.29000000000000004), np.float64(1.5e-07), np.float64(0.29000000000000004), np.float64(4.2499999999999995e-07), np.float64(-1.25e-07), np.float64(1.14e-06), np.float64(3000000.0), np.float64(0.32), np.float64(1.85e-07), np.float64(0.644), np.float64(23400.0), np.float64(33.5), np.float64(23.5), np.float64(1490.0), np.float64(10.5), np.float64(15.0), 0.03, 8.314, 50.0, np.float64(4.25), np.float64(4.5), np.float64(0.185), np.float64(0.44999999999999996), np.float64(0.09), np.float64(0.06999999999999999), np.float64(0.096), np.float64(0.5), np.float64(-0.000275), np.float64(-0.000275), np.float64(-0.000275), np.float64(0.65), np.float64(1.25), np.float64(-0.95), np.float64(-1.25), np.float64(0.158)]]
)


    # test = proxy.simulate("AICU", BASELINE_PARAMETERS)
    # breakpoint() 

    # env.get_action_space()

    breakpoint()
    print("done")
    
    # train_X, train_Y, val_X, val_Y, test_X, test_Y = load_data()

    # in_standard = InputStandardize(d=train_X.shape[-1]).to(train_X)
    # out_standard = Standardize(m=train_Y.shape[-1]).to(train_Y)
    # model = SingleTaskGP(train_X,
    #                      train_Y,
    #                      input_transform=in_standard,
    #                      outcome_transform=out_standard,
    #                      covar_module=ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=train_X.shape[-1])))
    # mll = ExactMarginalLogLikelihood(model.likelihood, model)
    # mll = fit_gpytorch_mll(mll)

    # fp = "gflownet/proxy/greenhouse/secondEdition.pkl"
    # with open(fp, 'wb') as file:
    #     pickle.dump(model, file)

    # n_samples = 1000
    # with torch.no_grad():
    #     posterior = model.posterior(val_X)
    #     samples = posterior.rsample(sample_shape=torch.Size([n_samples]))
    # samples = samples.squeeze(-1).cpu().numpy()

    # std = torch.sqrt(posterior.variance.cpu())
    # lower_bound = (posterior.mean - std).squeeze()
    # upper_bound = (posterior.mean + std).squeeze()

    # plt.figure(figsize=(8, 6))
    # plt.plot(val_X[:, 0].cpu(), posterior.mean.cpu(), '--', color="blue", alpha=0.6, marker="+", label="Posterior mean")
    # plt.fill_between(
    #     val_X[:, 0].cpu(),  # X-axis values
    #     lower_bound,                # Lower boundary of the band
    #     upper_bound,                # Upper boundary of the band
    #     color="blue",
    #     alpha=0.2,                  # Make the band transparent
    #     label="Standard deviation"
    # ) 

    # mean_prediction = samples.mean(axis=0)
    # std_prediction = samples.std(axis=0)
    # confidence_multiplier = 1.96 
    # lower_bound = mean_prediction - confidence_multiplier * std_prediction
    # upper_bound = mean_prediction + confidence_multiplier * std_prediction

    
    # # for i in range(n_samples):
    #     # plt.plot(val_X[:, 0].cpu(), samples[i], alpha=0.3)
    # # plt.plot(val_X[:,0].cpu(), mean_prediction, label="Predicted Yield")

    # plt.scatter(train_X[:, 0].cpu(), train_Y.cpu(), color="grey", label="Training Data", alpha=0.5)
    # plt.scatter(val_X[:, 0].cpu(), val_Y.cpu(), color="orange", label="Validation Data")
    # plt.scatter(val_X[:, 0].cpu(), mean_prediction, color="red", marker="x", label="Predictions")
    # plt.fill_between(
    #     val_X[:, 0].cpu(),  # X-axis values
    #     lower_bound,                # Lower boundary of the band
    #     upper_bound,                # Upper boundary of the band
    #     color="red",
    #     alpha=0.2,                  # Make the band transparent
    #     label="95% Confidence Band"
    # )

    # plt.legend()
    # plt.title("Posterior Samples at Training Inputs")
    # plt.show()



def test(model, train_X, train_Y, val_X):

    # 1) Basic model / data checks
    print("train_X:", train_X.shape, train_X.dtype, train_X.device)
    print("train_Y:", train_Y.shape, train_Y.dtype, train_Y.device)
    print("val_X:  ", val_X.shape, val_X.dtype, val_X.device)

    # 2) Transforms
    print("input_transform:", getattr(model, "input_transform", None))
    print("outcome_transform:", getattr(model, "outcome_transform", None))

    # 3) Posterior at training points (should match y closely if fitted)
    with torch.no_grad():
        post_train = model.posterior(train_X)
    print("train posterior mean (first 8):", post_train.mean[:8].squeeze(-1).cpu().numpy())
    print("train posterior var  (first 8):", post_train.variance[:8].squeeze(-1).cpu().numpy())
    print("train_Y (first 8):", train_Y[:8].squeeze(-1).cpu().numpy())

    # 4) Posterior at validation (what you already plotted)
    with torch.no_grad():
        post_val = model.posterior(val_X)
    print("val posterior mean (first 8):", post_val.mean[:8].squeeze(-1).cpu().numpy())
    print("val posterior var  (first 8):", post_val.variance[:8].squeeze(-1).cpu().numpy())

    # 5) Check learned hyperparameters (lengthscale, outputscale, noise, mean)
    try:
        ls = model.covar_module.base_kernel.lengthscale.detach().cpu().numpy()
        os = model.covar_module.outputscale.detach().cpu().numpy()
        print("lengthscale:", ls)
        print("outputscale:", os)
    except Exception as e:
        print("Couldn't read lengthscale/outputscale:", e)

    try:
        mean_const = model.mean_module.constant.detach().cpu().numpy()
        print("mean_module.constant:", mean_const)
    except Exception:
        print("No simple constant mean or couldn't read it.")

    print("likelihood noise:", model.likelihood.noise.detach().cpu().numpy())

    # 6) Compute k(x*, X_train). Are cross-covariances tiny?
    # (uses the learned lengthscale/outputscale above; fallback if None)
    def rbf_like(x1, x2, lengthscale, outputscale):
        L = lengthscale.reshape(1, -1) if getattr(lengthscale, "ndim", None) else lengthscale
        d2 = (((x1.unsqueeze(1) - x2.unsqueeze(0)) / L)**2).sum(-1)
        return outputscale * torch.exp(-0.5 * d2)

    if 'ls' in locals():
        k_xX = rbf_like(val_X.cpu(), train_X.cpu(), torch.tensor(ls), torch.tensor(os))
        print("k(x*,X) stats: min", k_xX.min().item(), "max", k_xX.max().item(), "mean", k_xX.mean().item())
    else:
        print("skip k(x*,X) check (lengthscale unknown)")

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
