"""
Optuna baseline for GFlowNet comparison.
Searches the same discrete perturbation space as the GFlowNet:
  - 5 parameter groups, each with a set of named perturbation actions.
  - Each action applies a coordinated +1/0/-1 pattern across the group's parameters.
  - Actual parameter update: np.clip(val + direction * step_fraction * (hi - lo), lo, hi)
  - At each cycle, one action is chosen per group.

Plug in your own controller, proxy, and parameter bounds where indicated.
"""
from gflownet.envs.greenhouse.constants import GROUPS, BASELINE_PARAMETERS, GROUP_ORDER, PERTURBATION_SCHEME, PARAMETER_BOUNDS, INITIAL_CONDITIONS
from gflownet.proxy.greenhouse.cropSimulatorProxy import CropSimulatorProxy
import numpy as np
import optuna
import numpy as np
from copy import deepcopy
import wandb
from datetime import datetime


# Number of cycles (1 or 2)
N_CYCLES = 1

# Step fraction
STEP_FRACTION = 0.1  
# Reward sharpness 
BETA = 4.0 

proxy = CropSimulatorProxy()

N_EVALUATIONS = 100 # Replace with actual training budget

# ============================================================
# 3. Apply perturbation (same logic as your GFlowNet)
# ============================================================

def apply_perturbation(current_params: dict, group_name: str, action_name: str, step_fraction: float):
    """
    Apply a named perturbation action to the parameters of a group.
    Mirrors the GFlowNet's update: np.clip(val + direction * step_fraction * (hi - lo), lo, hi)
    Step fraction is annealed per cycle (halved each cycle).
    """
    action = PERTURBATION_SCHEME[group_name][action_name]
    for param_name, direction in action.items():
        if direction == 0:
            continue
        lo, hi = PARAMETER_BOUNDS[param_name]
        val = current_params[param_name]
        new_val = np.clip(val + direction * step_fraction * (hi - lo), lo, hi)
        current_params[param_name] = new_val


def build_config(config, normalize=False):
    parameters = [0.0] * len(BASELINE_PARAMETERS.keys())
    for i, k in enumerate(sorted(BASELINE_PARAMETERS)):
        parameters[i] = config.get(k, INITIAL_CONDITIONS.get(k, BASELINE_PARAMETERS[k]))
    
    if normalize:
        for i,k in enumerate(sorted(BASELINE_PARAMETERS)):
            lo, hi = PARAMETER_BOUNDS.get(k, (0, 0))
            if lo == hi:
                parameters[i] = 0.5 #no bounds -> parameter stays fixed
            else:
                parameters[i] = (parameters[i] - lo) / (hi-lo)
    return parameters
    
def simulate_and_evaluate(param_values: dict) -> float:
    """
    Replace with your actual simulation pipeline.
    Takes parameter values, runs 6 trajectories through the simulator,
    computes the mean relative squared error:
        mean of ((y_hat - y) / y) ** 2
    across observation setpoints and trajectories.

    Returns:
        loss: mean relative squared error
    """
    proxy_config = build_config(param_values)
    loss = proxy([proxy_config])
    return loss


def compute_reward(loss: float, beta: float) -> float:
    """Power-law reward: (1 / loss) ** beta"""
    return (1.0 / (loss + 1e-10)) ** beta


# ============================================================
# 5. Optuna objective
# ============================================================

def objective(trial: optuna.Trial) -> float:
    """
    Each trial proposes one action per group per cycle,
    then evaluates the resulting parameter configuration.
    """

    current_params = {}
    for k in BASELINE_PARAMETERS.keys():
        current_params[k] = INITIAL_CONDITIONS.get(k, BASELINE_PARAMETERS[k])

    # For each cycle, choose one action per group (step fraction halves each cycle)
    for cycle in range(N_CYCLES):
        cycle_step_fraction = STEP_FRACTION / (2 ** cycle)
        for group_name in GROUP_ORDER:
            available_actions = list(PERTURBATION_SCHEME[group_name].keys())
            action = trial.suggest_categorical(
                f"{group_name}_cycle{cycle}",
                available_actions,
            )
            apply_perturbation(current_params, group_name, action, cycle_step_fraction)

    # Extract just the values for simulation

    # Evaluate
    loss = simulate_and_evaluate(current_params)
    reward = compute_reward(loss, BETA)


    wandb.log({
        "loss": loss,
        "reward": reward,
        "trial_step": trial.number,
        **current_params,
    })

    return reward


# ============================================================
# 6. Run
# ============================================================

if __name__ == "__main__":
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    wandb.init(mode="online", name=run_name, project="optuna-crop-calibration")
    wandb.define_metric("*", step_metric="trial_step")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=0),
    )

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, 
                   n_trials=N_EVALUATIONS,
                   )

    # --------------------------------------------------------
    # 7. Collect results
    # --------------------------------------------------------

    print(f"Best reward: {study.best_value:.6f}")
    print(f"Best actions: {study.best_params}")

    all_rewards = [t.value for t in study.trials if t.value is not None]

    print(f"\nTotal trials:  {len(all_rewards)}")
    print(f"Mean reward:   {np.mean(all_rewards):.6f}")
    print(f"Median reward: {np.median(all_rewards):.6f}")
    print(f"Top-10 mean:   {np.mean(sorted(all_rewards)[-10:]):.6f}")

    # --------------------------------------------------------
    # 8. Extract action sequences for diversity analysis
    # --------------------------------------------------------
    # Each trial's params is a dict like:
    #   {"leaf_and_canopy_geometry_cycle0": "increase",
    #    "photosynthetic_potential_cycle0": "higher_sensitivity", ...}
    #
    # To compute diversity: convert action names to integer indices,
    # form a vector per trial, then compute pairwise Hamming distances.

    def trial_to_action_vector(trial_params):
        """Convert a trial's action choices to an integer vector."""
        vector = []
        for cycle in range(N_CYCLES):
            for group_name in GROUP_ORDER:
                key = f"{group_name}_cycle{cycle}"
                action_name = trial_params[key]
                action_idx = list(PERTURBATION_SCHEME[group_name].keys()).index(action_name)
                vector.append(action_idx)
        return vector

    action_vectors = [
        trial_to_action_vector(t.params)
        for t in study.trials if t.value is not None
    ]

    # Count unique configurations
    unique_configs = set(tuple(v) for v in action_vectors)
    print(f"Unique configs: {len(unique_configs)} / {len(action_vectors)}")