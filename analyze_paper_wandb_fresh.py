#!/usr/bin/env python3
"""
GFlowNet Crop Calibration — Paper Figure Generation (1-Cycle)
==============================================================

Produces figures for three research questions:
  Q1: Mode discovery — Can the GFN discover interesting modes?
  Q2: Performance   — Can the GFN approximate the true reward-proportional distribution?
  Q3: Hyperparameter sensitivity — How do (step_fraction × lr × seed) affect convergence?

Design:
  - 3 step_fractions × 3 learning rates × 3 seeds = 27 runs total
  - For Q1/Q2, all 9 runs per step_fraction are used (mean ± std)
  - For Q3, wandb training curves + final metrics across the full grid

Usage:
    python analyze_gfn_paper.py \
        --reward_tables precomputed/reward_table_sf0.15.json precomputed/reward_table_sf0.3.json precomputed/reward_table_sf0.4.json \
        --step_fractions 0.15 0.3 0.4 \
        --beta 50 \
        --wandb_project parcham-udem/gfn-crop-calibration \
        --output_dir figures

  Every run fetches wandb metadata fresh and downloads checkpoint artifacts
  fresh from wandb. No local cache, checkpoint_map.json, or experiment
  logdir fallback is used.

Requirements:
    pip install torch numpy matplotlib scipy seaborn wandb
"""

import json
import argparse
import itertools
import os
import shutil
import tempfile
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import jensenshannon
import seaborn as sns

# ═══════════════════════════════════════════════════════════════════
# PUBLICATION STYLE
# ═══════════════════════════════════════════════════════════════════

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLORS = {
    "gt": "#2c3e50",
    "gfn": "#e74c3c",
    "gfn_fill": "#f5b7b1",
    "accent": "#2980b9",
}

# ═══════════════════════════════════════════════════════════════════
# CONSTANTS (self-contained from gflownet/envs/greenhouse/constants.py)
# ═══════════════════════════════════════════════════════════════════

GROUP_ORDER = [
    "leaf_and_canopy_geometry",
    "photosynthetic_potential",
    "temperature_inhibition",
    "temperature_and_development",
    "biomass_growth_and_maintenance",
]

GROUP_LABELS = ["Canopy", "Photosynthesis", "Temp. Inhibition", "Temp. & Dev.", "Biomass"]

PERTURBATION_SCHEME = {
    "leaf_and_canopy_geometry": {
        "none": {"LAI_max": 0, "SLA": 0, "n_plants": 0},
        "increase": {"LAI_max": +1, "SLA": +1, "n_plants": +1},
        "decrease": {"LAI_max": -1, "SLA": -1, "n_plants": -1},
    },
    "photosynthetic_potential": {
        "none": {"J_max_leaf": 0, "Jpot_activation": 0, "Jpot_deactivation": 0, "Jpot_entropy": 0, "alpha": 0, "deg_curv_elec_transport": 0, "Tcan_CO2_comp_point": 0, "CO2_air_stomata": 0, "net_ass_rate": 0},
        "increase": {"J_max_leaf": +1, "alpha": +1, "CO2_air_stomata": +1, "net_ass_rate": -1, "Tcan_CO2_comp_point": -1, "Jpot_activation": 0, "Jpot_deactivation": 0, "Jpot_entropy": 0, "deg_curv_elec_transport": 0},
        "decrease": {"J_max_leaf": -1, "alpha": -1, "CO2_air_stomata": -1, "net_ass_rate": +1, "Tcan_CO2_comp_point": +1, "Jpot_activation": 0, "Jpot_deactivation": 0, "Jpot_entropy": 0, "deg_curv_elec_transport": 0},
        "higher_sensitivity": {"Jpot_activation": +1, "Jpot_deactivation": -1, "deg_curv_elec_transport": -1, "Tcan_CO2_comp_point": +1, "J_max_leaf": 0, "Jpot_entropy": 0, "alpha": 0, "CO2_air_stomata": 0, "net_ass_rate": 0},
        "lower_sensitivity": {"Jpot_activation": -1, "Jpot_deactivation": +1, "deg_curv_elec_transport": +1, "Tcan_CO2_comp_point": -1, "J_max_leaf": 0, "Jpot_entropy": 0, "alpha": 0, "CO2_air_stomata": 0, "net_ass_rate": 0},
    },
    "temperature_inhibition": {
        "none": {"k_sw_min_Tcan": 0, "s_min_Tcan": 0, "k_sw_max_Tcan": 0, "s_max_Tcan": 0, "k_sw_min_Tcan24": 0, "s_min_Tcan24": 0, "k_sw_max_Tcan24": 0, "s_max_Tcan24": 0},
        "shift_warm": {"k_sw_min_Tcan": +1, "s_min_Tcan": 0, "k_sw_max_Tcan": +1, "s_max_Tcan": 0, "k_sw_min_Tcan24": +1, "s_min_Tcan24": 0, "k_sw_max_Tcan24": +1, "s_max_Tcan24": 0},
        "shift_cold": {"k_sw_min_Tcan": -1, "s_min_Tcan": 0, "k_sw_max_Tcan": -1, "s_max_Tcan": 0, "k_sw_min_Tcan24": -1, "s_min_Tcan24": 0, "k_sw_max_Tcan24": -1, "s_max_Tcan24": 0},
        "widen_optimum": {"k_sw_min_Tcan": -1, "s_min_Tcan": +1, "k_sw_max_Tcan": +1, "s_max_Tcan": -1, "k_sw_min_Tcan24": -1, "s_min_Tcan24": +1, "k_sw_max_Tcan24": +1, "s_max_Tcan24": -1},
        "narrow_optimum": {"k_sw_min_Tcan": +1, "s_min_Tcan": -1, "k_sw_max_Tcan": -1, "s_max_Tcan": +1, "k_sw_min_Tcan24": +1, "s_min_Tcan24": -1, "k_sw_max_Tcan24": -1, "s_max_Tcan24": +1},
    },
    "temperature_and_development": {
        "none": {"bias_g_Tcan24": 0, "slope_g_Tcan24": 0, "TS_start": 0, "TS_end": 0, "c_dev1": 0, "c_dev2": 0, "r_fruit_Set": 0},
        "increase": {"bias_g_Tcan24": +1, "slope_g_Tcan24": +1, "TS_end": -1, "c_dev1": +1, "c_dev2": +1, "r_fruit_Set": -1, "TS_start": 0},
        "decrease": {"bias_g_Tcan24": -1, "slope_g_Tcan24": -1, "TS_end": +1, "c_dev1": -1, "c_dev2": -1, "r_fruit_Set": +1, "TS_start": 0},
        "higher_sensitivity": {"slope_g_Tcan24": +1, "c_dev2": +1, "bias_g_Tcan24": 0, "TS_start": 0, "TS_end": 0, "c_dev1": 0, "r_fruit_Set": 0},
        "lower_sensitivity": {"slope_g_Tcan24": -1, "c_dev2": -1, "bias_g_Tcan24": 0, "TS_start": 0, "TS_end": 0, "c_dev1": 0, "r_fruit_Set": 0},
    },
    "biomass_growth_and_maintenance": {
        "none": {"G_max": 0, "c_fruit_growth": 0, "c_leaf_growth": 0, "c_stem_growth": 0, "c_fruit_maintenance": 0, "c_leaf_maintenance": 0, "c_stem_maintenance": 0, "Q_10_maintenance": 0, "rg_fruit": 0, "rg_leaf": 0, "rg_stem": 0, "c_rgr": 0},
        "more_fruit_growth": {"G_max": -1, "rg_fruit": +1, "rg_leaf": -1, "rg_stem": -1, "c_fruit_growth": 0, "c_leaf_growth": 0, "c_stem_growth": 0, "c_fruit_maintenance": 0, "c_leaf_maintenance": 0, "c_stem_maintenance": 0, "Q_10_maintenance": 0, "c_rgr": 0},
        "more_veg_growth": {"G_max": +1, "rg_fruit": -1, "rg_leaf": +1, "rg_stem": +1, "c_fruit_growth": 0, "c_leaf_growth": 0, "c_stem_growth": 0, "c_fruit_maintenance": 0, "c_leaf_maintenance": 0, "c_stem_maintenance": 0, "Q_10_maintenance": 0, "c_rgr": 0},
        "lower_resp_cost": {"G_max": 0, "rg_fruit": 0, "rg_leaf": 0, "rg_stem": 0, "c_fruit_growth": -1, "c_leaf_growth": -1, "c_stem_growth": -1, "c_fruit_maintenance": -1, "c_leaf_maintenance": -1, "c_stem_maintenance": -1, "Q_10_maintenance": 0, "c_rgr": 0},
        "higher_resp_cost": {"G_max": 0, "rg_fruit": 0, "rg_leaf": 0, "rg_stem": 0, "c_fruit_growth": +1, "c_leaf_growth": +1, "c_stem_growth": +1, "c_fruit_maintenance": +1, "c_leaf_maintenance": +1, "c_stem_maintenance": +1, "Q_10_maintenance": 0, "c_rgr": 0},
        "higher_sensitivity": {"Q_10_maintenance": +1, "c_rgr": +1, "G_max": 0, "rg_fruit": 0, "rg_leaf": 0, "rg_stem": 0, "c_fruit_growth": 0, "c_leaf_growth": 0, "c_stem_growth": 0, "c_fruit_maintenance": 0, "c_leaf_maintenance": 0, "c_stem_maintenance": 0},
        "lower_sensitivity": {"Q_10_maintenance": -1, "c_rgr": -1, "G_max": 0, "rg_fruit": 0, "rg_leaf": 0, "rg_stem": 0, "c_fruit_growth": 0, "c_leaf_growth": 0, "c_stem_growth": 0, "c_fruit_maintenance": 0, "c_leaf_maintenance": 0, "c_stem_maintenance": 0},
    },
}

PARAMETER_BOUNDS = {
    "LAI_max": (2.0, 5.5), "SLA": (1.5e-5, 4.0e-5), "n_plants": (1.2, 2.2),
    "J_max_leaf": (100.0, 350.0), "Jpot_activation": (280000.0, 500000.0),
    "Jpot_deactivation": (150000.0, 300000.0), "Jpot_entropy": (640.0, 780.0),
    "alpha": (0.25, 0.50), "deg_curv_elec_transport": (0.50, 0.98),
    "Tcan_CO2_comp_point": (1.0, 2.8), "CO2_air_stomata": (0.50, 0.80),
    "net_ass_rate": (3.0, 6.0),
    "k_sw_min_Tcan": (5.0, 16.0), "k_sw_max_Tcan": (28.0, 39.0),
    "k_sw_min_Tcan24": (10.0, 20.0), "k_sw_max_Tcan24": (19.0, 28.0),
    "s_min_Tcan": (-1.5, -0.4), "s_max_Tcan": (0.3, 1.0),
    "s_min_Tcan24": (-2.0, -0.5), "s_max_Tcan24": (0.5, 2.0),
    "bias_g_Tcan24": (-2.0, -0.5), "slope_g_Tcan24": (0.06, 0.20),
    "TS_start": (-20.0, 20.0), "TS_end": (400.0, 1200.0),
    "c_dev1": (-1.5e-8, -2.0e-9), "c_dev2": (5.0e-9, 2.5e-8),
    "r_fruit_Set": (0.02, 0.35), "G_max": (400.0, 3000.0),
    "c_fruit_growth": (0.18, 0.40), "c_leaf_growth": (0.18, 0.40),
    "c_stem_growth": (0.20, 0.44), "c_fruit_maintenance": (5.0e-8, 2.5e-7),
    "c_leaf_maintenance": (1.5e-7, 7.0e-7), "c_stem_maintenance": (7.0e-8, 3.0e-7),
    "Q_10_maintenance": (1.4, 3.0), "rg_fruit": (0.20, 0.70),
    "rg_leaf": (0.03, 0.15), "rg_stem": (0.02, 0.12), "c_rgr": (1.0e6, 5.0e6),
}

BASELINE_PARAMETERS = {
    "LAI_max": 3.0, "SLA": 3e-05, "n_plants": 2.5,
    "J_max_leaf": 210.0, "Jpot_activation": 370000.0,
    "Jpot_deactivation": 220000.0, "Jpot_entropy": 710.0,
    "alpha": 0.385, "deg_curv_elec_transport": 0.7,
    "Tcan_CO2_comp_point": 1.7, "CO2_air_stomata": 0.67,
    "net_ass_rate": 1.0,
    "k_sw_min_Tcan": 10.0, "s_min_Tcan": -0.869,
    "k_sw_max_Tcan": 34.0, "s_max_Tcan": 0.5793,
    "k_sw_min_Tcan24": 14.0, "s_min_Tcan24": -1.1587,
    "k_sw_max_Tcan24": 24.5, "s_max_Tcan24": 1.13904,
    "bias_g_Tcan24": 0.06, "slope_g_Tcan24": 0.047,
    "TS_start": 0.0, "TS_end": 1035.0,
    "c_dev1": -7.64e-9, "c_dev2": 1.16e-8,
    "r_fruit_Set": 0.1, "G_max": 10000.0,
    "c_fruit_growth": 0.27, "c_leaf_growth": 0.28,
    "c_stem_growth": 0.3, "c_fruit_maintenance": 1.16e-7,
    "c_leaf_maintenance": 3.47e-7, "c_stem_maintenance": 1.47e-7,
    "Q_10_maintenance": 2.0, "rg_fruit": 0.328,
    "rg_leaf": 0.095, "rg_stem": 0.074, "c_rgr": 2850000.0,
    # Fixed (not explored)
    "rho_can": 0.07, "rho_floor": 0.5, "K1": 0.7, "K2": 0.7,
    "Jpot_ref_temp": 298.15, "molar_gas_constant": 8.314,
    "mass_CH20": 0.03, "n_fruit_phases": 50.0,
    "k_sw_max_Cbuff": 20000.0, "k_sw_min_Cbuff": 1000.0,
    "c_max_buf_fruit_1": -1.71e-7, "c_max_buf_fruit_2": 7.31e-7,
    "s_MCairbuf_Cbuf": 0.0005, "s_MCbuforg_Cbuf": -0.005,
    "s_harvest": -5e-05,
}

GROUPS = {g: list(PERTURBATION_SCHEME[g][list(PERTURBATION_SCHEME[g].keys())[0]].keys()) for g in GROUP_ORDER}
SORTED_PARAM_KEYS = sorted(BASELINE_PARAMETERS.keys())
N_PARAMS = len(SORTED_PARAM_KEYS)
N_GROUPS = len(GROUP_ORDER)
ACTIONS_PER_GROUP = [list(PERTURBATION_SCHEME[g].keys()) for g in GROUP_ORDER]

# Exact pert2id from trained CropSimEnv (hash-order dependent — do NOT change)
PERT2ID = {
    "increase": 0, "lower_sensitivity": 1, "shift_warm": 2,
    "higher_resp_cost": 3, "higher_sensitivity": 4, "more_veg_growth": 5,
    "shift_cold": 6, "none": 7, "widen_optimum": 8, "lower_resp_cost": 9,
    "more_fruit_growth": 10, "decrease": 11, "narrow_optimum": 12,
}
ID2PERT = {i: p for p, i in PERT2ID.items()}
N_ACTIONS = len(PERT2ID)


# ═══════════════════════════════════════════════════════════════════
# ENVIRONMENT RECONSTRUCTION
# ═══════════════════════════════════════════════════════════════════

def apply_perturbation(step_fraction, group_name, perturb_name, values=None):
    if values is None:
        values = {}
    scheme = PERTURBATION_SCHEME[group_name][perturb_name]
    for p in GROUPS[group_name]:
        lo, hi = PARAMETER_BOUNDS[p]
        val = values.get(p, (hi + lo) / 2)
        val = np.clip(val + scheme[p] * step_fraction * (hi - lo), lo, hi)
        values[p] = val
    return values


def build_config_from_actions(action_names, step_fraction):
    combined = {}
    for group_idx, pert_name in enumerate(action_names):
        combined = apply_perturbation(step_fraction, GROUP_ORDER[group_idx], pert_name, combined)
    return combined


def normalize_params(config):
    params = [0.0] * N_PARAMS
    for i, k in enumerate(SORTED_PARAM_KEYS):
        val = config.get(k, BASELINE_PARAMETERS[k])
        lo, hi = PARAMETER_BOUNDS.get(k, (0, 0))
        params[i] = (val - lo) / (hi - lo) if lo != hi else 0.5
    return params


def enumerate_all_states():
    return list(itertools.product(*ACTIONS_PER_GROUP))


def state_to_key(action_names):
    return "|".join(action_names)


def get_valid_action_ids(group_idx):
    group_name = GROUP_ORDER[group_idx]
    return {PERT2ID[name] for name in PERTURBATION_SCHEME[group_name].keys()}


# ═══════════════════════════════════════════════════════════════════
# FORWARD POLICY — EXACT ENUMERATION
# ═══════════════════════════════════════════════════════════════════

def build_forward_mlp(state_dict, input_dim=61, hidden_dim=128, output_dim=13):
    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


def build_policy_input(step_num, step_fraction, pert_ids_so_far, config_so_far):
    n_ops = N_GROUPS  # 1 cycle
    vec = [-1.0] * (2 + n_ops + N_PARAMS)
    vec[0] = float(step_num)
    vec[1] = float(step_fraction)
    for i, pid in enumerate(pert_ids_so_far):
        vec[2 + i] = float(pid)
    vec[2 + n_ops:] = normalize_params(config_so_far)
    return vec


@torch.no_grad()
def compute_exact_gfn_distribution(model, step_fraction, device="cpu"):
    """
    Enumerate all 2625 trajectories through the forward policy.
    Returns dict: state_key → probability (sums to 1).
    """
    model = model.to(device)
    all_states = enumerate_all_states()
    log_probs = {}

    # Detect output dim from model
    out_dim = list(model.parameters())[-1].shape[0]

    for state_actions in all_states:
        log_p = 0.0
        config = {}
        pert_ids = []

        for group_idx, pert_name in enumerate(state_actions):
            step_num = group_idx + 1
            vec = build_policy_input(step_num, step_fraction, pert_ids, config)
            x = torch.tensor([vec], dtype=torch.float32, device=device)
            logits = model(x)[0]

            # Mask invalid actions (mask covers full output dim, including EOS)
            valid_ids = get_valid_action_ids(group_idx)
            mask = torch.tensor([pid in valid_ids for pid in range(out_dim)],
                                dtype=torch.bool, device=device)
            masked_logits = logits.clone()
            masked_logits[~mask] = float("-inf")
            log_softmax = torch.log_softmax(masked_logits, dim=0)

            action_id = PERT2ID[pert_name]
            log_p += log_softmax[action_id].item()
            pert_ids.append(action_id)
            config = apply_perturbation(step_fraction, GROUP_ORDER[group_idx], pert_name, config)

        log_probs[state_to_key(state_actions)] = log_p

    # Normalize (should already sum to ~1, but numerical safety)
    keys = list(log_probs.keys())
    lp = np.array([log_probs[k] for k in keys])
    max_lp = lp.max()
    probs = np.exp(lp - max_lp)
    probs /= probs.sum()
    return {k: float(p) for k, p in zip(keys, probs)}


@torch.no_grad()
def sample_gfn_distribution(model, step_fraction, n_samples, device="cpu"):
    """
    Sample n_samples trajectories from the forward policy and estimate the
    distribution by counting terminal state visits.
    Returns dict: state_key → probability (sums to 1).
    """
    model = model.to(device)
    counts = defaultdict(int)

    # Detect output dim from model
    out_dim = list(model.parameters())[-1].shape[0]

    for _ in range(n_samples):
        config = {}
        pert_ids = []
        action_names = []

        for group_idx in range(N_GROUPS):
            step_num = group_idx + 1
            vec = build_policy_input(step_num, step_fraction, pert_ids, config)
            x = torch.tensor([vec], dtype=torch.float32, device=device)
            logits = model(x)[0]

            # Mask invalid actions (mask covers full output dim, including EOS)
            valid_ids = get_valid_action_ids(group_idx)
            mask = torch.tensor([pid in valid_ids for pid in range(out_dim)],
                                dtype=torch.bool, device=device)
            masked_logits = logits.clone()
            masked_logits[~mask] = float("-inf")

            # Sample from the policy
            probs = torch.softmax(masked_logits, dim=0)
            action_id = torch.multinomial(probs, 1).item()
            pert_name = ID2PERT[action_id]

            pert_ids.append(action_id)
            action_names.append(pert_name)
            config = apply_perturbation(
                step_fraction, GROUP_ORDER[group_idx], pert_name, config
            )

        key = state_to_key(tuple(action_names))
        counts[key] += 1

    total = sum(counts.values())
    return {k: count / total for k, count in counts.items()}


# ═══════════════════════════════════════════════════════════════════
# GROUND TRUTH
# ═══════════════════════════════════════════════════════════════════

def load_reward_table(path, beta):
    with open(path) as f:
        raw = json.load(f)
    states = list(raw.keys())
    losses = np.array([raw[k]["loss"] for k in states])
    rewards = (1.0 / losses) ** beta
    probs = rewards / rewards.sum()
    return {k: float(p) for k, p in zip(states, probs)}, {k: raw[k]["loss"] for k in states}


# ═══════════════════════════════════════════════════════════════════
# WANDB
# ═══════════════════════════════════════════════════════════════════

WANDB_METRICS = [
    "Jensen Shannon Div.", "KL Div.", "L1 error", "Loss",
    "Corr. (test probs., rewards)", "logZ",
]


def build_wandb_filters(
    seeds=None,
    step_fractions=None,
    betas=None,
    learning_rates=None,
    lr_z_mults=None,
    random_action_probs=None,
    states=None,
):
    """
    Build a MongoDB-style filter dict for wandb api.runs().

    Each parameter can be a single value or a list of values.
    Only non-None parameters are included in the filter.

    By default, states=None means no state filter (includes running,
    finished, crashed, etc.). Pass states=["finished"] to restrict.

    Note: wandb stores nested config values under '.value' keys, so
    the filter paths must include them (e.g. config.gflownet.value.seed).
    """
    filters = {}

    def _set(key, values):
        if len(values) == 1:
            filters[key] = values[0]
        else:
            filters[key] = {"$in": values}

    if seeds is not None:
        seeds = [seeds] if not isinstance(seeds, list) else seeds
        _set("config.gflownet.value.seed", seeds)

    if step_fractions is not None:
        step_fractions = [step_fractions] if not isinstance(step_fractions, list) else step_fractions
        _set("config.env.value.step_fraction", step_fractions)

    if betas is not None:
        betas = [betas] if not isinstance(betas, list) else betas
        _set("config.proxy.value.beta", betas)

    if learning_rates is not None:
        learning_rates = [learning_rates] if not isinstance(learning_rates, list) else learning_rates
        _set("config.gflownet.value.optimizer.lr", learning_rates)

    if lr_z_mults is not None:
        lr_z_mults = [lr_z_mults] if not isinstance(lr_z_mults, list) else lr_z_mults
        _set("config.gflownet.value.optimizer.lr_z_mult", lr_z_mults)

    if random_action_probs is not None:
        random_action_probs = [random_action_probs] if not isinstance(random_action_probs, list) else random_action_probs
        _set("config.gflownet.value.random_action_prob", random_action_probs)

    if states is not None:
        states = [states] if not isinstance(states, list) else states
        _set("state", states)

    return filters


def fetch_wandb_runs(
    project,
    seeds=None,
    step_fractions=None,
    betas=None,
    learning_rates=None,
    lr_z_mults=None,
    random_action_probs=None,
    states=None,
    run_ids=None,
    after_run_id=None,
    history_samples=500,
):
    import wandb
    api = wandb.Api()

    # If after_run_id is set, look up its creation time to use as a cutoff
    after_time = None
    if after_run_id:
        try:
            ref_run = api.run(f"{project}/{after_run_id}")
            after_time = ref_run.created_at
            print(f"[wandb] Filtering runs created after {after_run_id} ({after_time})")
        except wandb.errors.CommError:
            print(f"[WARN] Reference run {after_run_id} not found — ignoring --wandb_after")

    # If specific run IDs are given, fetch them directly (skip filters)
    if run_ids:
        print(f"[wandb] Fetching {len(run_ids)} specific runs by ID")
        runs = []
        for rid in run_ids:
            try:
                runs.append(api.run(f"{project}/{rid}"))
            except wandb.errors.CommError:
                print(f"  [SKIP] {rid}: run not found")
        print(f"[wandb] Found {len(runs)} runs")
    else:
        filters = build_wandb_filters(
            seeds=seeds,
            step_fractions=step_fractions,
            betas=betas,
            learning_rates=learning_rates,
            lr_z_mults=lr_z_mults,
            random_action_probs=random_action_probs,
            states=states,
        )

        # Add time filter if after_run_id was specified
        if after_time:
            filters["created_at"] = {"$gte": after_time}

        print(f"[wandb] Querying {project} with filters: {filters or '(none)'}")
        runs = api.runs(project, filters=filters)
        print(f"[wandb] Found {len(runs)} matching runs")

    out = []
    for run in runs:
        cfg = run.config

        if not cfg:
            # Config is empty (common with Hydra multirun bug).
            # Still include the run — artifact metadata may have what we need,
            # and the checkpoint can still be evaluated.
            print(f"  [WARN] {run.id}: empty config, using artifact metadata if available (state={run.state})")

            # Try to get metadata from the run's checkpoint artifact
            meta = {}
            try:
                import wandb as _wb
                _api = _wb.Api()
                artifact = _api.artifact(f"{project}/ckpt-{run.id}:latest")
                meta = artifact.metadata or {}
            except Exception:
                pass

            sf = meta.get("step_fraction", "unknown")
            lr = meta.get("lr")
            seed = meta.get("seed")
            beta = meta.get("beta")
            lr_z_mult = meta.get("lr_z_mult")
            random_action_prob = meta.get("random_action_prob")
            logdir = ""
        else:
            # step_fraction
            sf = cfg.get("env", {}).get("step_fraction", None)
            if sf is None or sf == "unknown":
                sf = cfg.get("step_fraction", None)
            if sf is None or sf == "unknown":
                # Last resort: parse from reward_cache_path
                cache_path = cfg.get("proxy", {}).get("reward_cache_path", "")
                if isinstance(cache_path, str) and "sf" in cache_path:
                    try:
                        sf = float(cache_path.split("sf")[1].split(".json")[0])
                    except (ValueError, IndexError):
                        sf = "unknown"
                else:
                    sf = "unknown"

            lr = cfg.get("gflownet", {}).get("optimizer", {}).get("lr", None)
            seed = cfg.get("gflownet", {}).get("seed", cfg.get("seed", None))
            beta = cfg.get("proxy", {}).get("beta", None)
            lr_z_mult = cfg.get("gflownet", {}).get("optimizer", {}).get("lr_z_mult", None)
            random_action_prob = cfg.get("gflownet", {}).get(
                "random_action_prob", cfg.get("random_action_prob", None)
            )
            logdir = cfg.get("logger", {}).get("logdir", {}).get("path", "")

        hist = run.history(samples=history_samples)
        history = {}
        for m in WANDB_METRICS:
            if m in hist.columns:
                vals = hist[["step", m]].dropna()
                history[m] = {"step": vals["step"].tolist(), "value": vals[m].tolist()}

        out.append({
            "id": run.id, "name": run.name, "state": run.state,
            "step_fraction": sf, "lr": lr, "seed": seed, "beta": beta,
            "lr_z_mult": lr_z_mult, "random_action_prob": random_action_prob,
            "logdir": logdir, "history": history,
        })
        print(f"  {run.id}: sf={sf}, lr={lr}, seed={seed}, "
              f"lr_z_mult={lr_z_mult}, ε={random_action_prob}, state={run.state}")

    return out


def _download_wandb_artifact_ckpt(project, run_id, alias="final"):
    """
    Download the checkpoint artifact for one wandb run into a fresh temporary
    directory and return ``(ckpt_path, tmp_dir)``. The caller is responsible
    for removing ``tmp_dir`` when finished.
    """
    import wandb

    api = wandb.Api()
    artifact_name = f"{project}/ckpt-{run_id}"

    artifact = None
    tried = []
    for try_alias in [alias, "latest"]:
        tried.append(try_alias)
        try:
            artifact = api.artifact(f"{artifact_name}:{try_alias}")
            break
        except wandb.errors.CommError:
            continue

    if artifact is None:
        print(f"  [SKIP] {run_id}: no artifact found (tried {tried})")
        return None, None

    tmp_dir = tempfile.mkdtemp(prefix=f"wandb_artifact_{run_id}_")
    try:
        artifact_dir = artifact.download(root=tmp_dir)
        ckpt_files = [f for f in os.listdir(artifact_dir) if f.endswith(".ckpt")]
        if not ckpt_files:
            print(f"  [SKIP] {run_id}: no .ckpt file in artifact")
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return None, None

        ckpt_file = "final.ckpt" if "final.ckpt" in ckpt_files else sorted(ckpt_files)[0]
        ckpt_path = os.path.join(artifact_dir, ckpt_file)
        print(f"  [OK] {run_id}: downloaded fresh artifact to {ckpt_path}")
        return ckpt_path, tmp_dir
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


def download_wandb_artifacts(runs_data, project, alias="final"):
    """
    Download checkpoint artifacts from wandb for each run and build a map of
    fresh temporary checkpoint paths.
    """
    ckpt_map = defaultdict(list)

    for run in runs_data:
        run_id = run["id"]
        sf = run["step_fraction"]
        if sf is None or sf == "unknown":
            print(f"  [SKIP] {run_id}: unknown step_fraction")
            continue

        ckpt_path, tmp_dir = _download_wandb_artifact_ckpt(
            project=project,
            run_id=run_id,
            alias=alias,
        )
        if ckpt_path is None:
            continue

        ckpt_map[str(sf)].append({
            "path": ckpt_path,
            "tmp_dir": tmp_dir,
            "lr": run.get("lr"),
            "seed": run.get("seed"),
            "wandb_id": run_id,
        })

    return dict(ckpt_map)



def compute_metrics(p_gfn, p_star, all_keys):
    p = np.array([p_gfn.get(k, 0.0) for k in all_keys])
    q = np.array([p_star.get(k, 0.0) for k in all_keys])
    p /= p.sum(); q /= q.sum()

    jsd = jensenshannon(p, q) ** 2
    kl = np.sum(q * np.log((q + 1e-30) / (p + 1e-30)))
    l1 = np.sum(np.abs(p - q))
    corr_p, _ = pearsonr(p, q)
    corr_s, _ = spearmanr(p, q)

    ent_p = -np.sum(p[p > 0] * np.log(p[p > 0]))
    ent_q = -np.sum(q[q > 0] * np.log(q[q > 0]))

    return {
        "jsd": jsd, "kl": kl, "l1": l1,
        "pearson_r": corr_p, "spearman_rho": corr_s,
        "n_eff_gfn": np.exp(ent_p), "n_eff_star": np.exp(ent_q),
        "entropy_gfn": ent_p, "entropy_star": ent_q,
    }


# ═══════════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════════

def _short(name):
    m = {
        "none": "none", "increase": "↑", "decrease": "↓",
        "higher_sensitivity": "sens+", "lower_sensitivity": "sens−",
        "shift_warm": "warm", "shift_cold": "cold",
        "widen_optimum": "widen", "narrow_optimum": "narrow",
        "more_fruit_growth": "fruit+", "more_veg_growth": "veg+",
        "lower_resp_cost": "resp−", "higher_resp_cost": "resp+",
    }
    return m.get(name, name)


def _marginalize_g2g4(prob_dict):
    """Marginalize to (Temp. Inhibition × Biomass) — legacy overview used in Q2."""
    g2 = ACTIONS_PER_GROUP[2]
    g4 = ACTIONS_PER_GROUP[4]
    grid = np.zeros((len(g2), len(g4)))
    for sa in enumerate_all_states():
        key = state_to_key(sa)
        grid[g2.index(sa[2]), g4.index(sa[4])] += prob_dict.get(key, 0.0)
    return grid


def _state_prob_tensor(prob_dict):
    """
    Convert a flat state → probability mapping into a dense tensor with shape
    (group1, group2, group3, group4, group5).
    """
    shape = tuple(len(actions) for actions in ACTIONS_PER_GROUP)
    tensor = np.zeros(shape, dtype=float)
    index_maps = [{name: i for i, name in enumerate(actions)} for actions in ACTIONS_PER_GROUP]

    for state_actions in enumerate_all_states():
        key = state_to_key(state_actions)
        idx = tuple(index_maps[g][state_actions[g]] for g in range(N_GROUPS))
        tensor[idx] = prob_dict.get(key, 0.0)

    return tensor


def _mode_heatmap_cube(prob_dict):
    """
    Build the 5D mode-discovery heatmap requested for Q1.

    Faceting:
      - rows    = group 1 (Canopy)
      - cols    = group 2 (Photosynthesis)
      - x-axis  = group 3 (Temp. Inhibition)
      - y-axis  = group 4 (Temp. & Dev.)
      - average = group 5 (Biomass)

    Returns
    -------
    cube : np.ndarray
        Shape = (n_group1, n_group2, n_group4, n_group3), so each facet can be
        displayed directly with imshow(cube[i, j], origin='lower').
    """
    tensor = _state_prob_tensor(prob_dict)
    cube = tensor.mean(axis=4)           # average over group 5 (Biomass)
    cube = np.transpose(cube, (0, 1, 3, 2))  # -> (group1, group2, group4, group3)
    return cube



def _plot_mode_heatmap_grid(cube, title, output_dir, filename, cmap='YlOrRd',
                            vmin=None, vmax=None, cbar_label='Avg. probability'):
    """Plot the faceted 5-group heatmap for Q1 mode discovery."""
    row_actions = ACTIONS_PER_GROUP[0]
    col_actions = ACTIONS_PER_GROUP[1]
    x_actions = ACTIONS_PER_GROUP[2]
    y_actions = ACTIONS_PER_GROUP[3]

    row_labels = [_short(a) for a in row_actions]
    col_labels = [_short(a) for a in col_actions]
    x_labels = [_short(a) for a in x_actions]
    y_labels = [_short(a) for a in y_actions]

    n_rows = len(row_actions)
    n_cols = len(col_actions)
    fig, axes = plt.subplots(
        n_rows,
        n_cols + 1,
        figsize=(1.6 * n_cols + 1.25, 1.65 * n_rows + 1.1),
        gridspec_kw={'width_ratios': [1] * n_cols + [0.08], 'wspace': 0.12, 'hspace': 0.18},
    )

    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    im = None
    for i in range(n_rows):
        for j in range(n_cols):
            ax = axes[i, j]
            im = ax.imshow(
                cube[i, j],
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                aspect='auto',
                origin='lower',
            )
            ax.set_xticks(range(len(x_actions)))
            ax.set_yticks(range(len(y_actions)))

            if i == n_rows - 1:
                ax.set_xticklabels(x_labels, rotation=45, ha='right')
            else:
                ax.set_xticklabels([])

            if j == 0:
                ax.set_yticklabels(y_labels)
                ax.set_ylabel(f"{GROUP_LABELS[0]} = {row_labels[i]}\n{GROUP_LABELS[3]}")
            else:
                ax.set_yticklabels([])

            if i == 0:
                ax.set_title(f"{GROUP_LABELS[1]} = {col_labels[j]}", fontsize=8)

            if i == n_rows - 1:
                ax.set_xlabel(GROUP_LABELS[2])

    cax = axes[:, -1].ravel()[0]
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label(cbar_label, fontsize=8)

    # Hide the extra colorbar column axes created by subplots, except the first.
    for extra_cax in axes[:, -1].ravel()[1:]:
        extra_cax.set_visible(False)

    fig.suptitle(
        title + "\nrows = Canopy, cols = Photosynthesis, x = Temp. Inhibition, y = Temp. & Dev., Biomass averaged",
        fontsize=10.5,
        y=1.01,
    )
    fig.subplots_adjust(left=0.08, right=0.94, bottom=0.11, top=0.88, wspace=0.12, hspace=0.18)
    _save(fig, output_dir, filename)



def _cluster_by_first_n(prob_dict, n):
    clusters = defaultdict(float)
    for key, prob in prob_dict.items():
        parts = key.split('|')
        clusters['|'.join(parts[:n])] += prob
    return dict(clusters)



def _topk_mode_curves(p_star, p_gfn, k_max):
    """Curves useful for quantifying mode discovery against the target top-k set."""
    keys = sorted(p_star.keys(), key=lambda k: -p_star[k])
    k_max = min(k_max, len(keys))
    target_order = keys[:k_max]
    gfn_order = sorted(keys, key=lambda k: -p_gfn.get(k, 0.0))
    gfn_rank = {k: i + 1 for i, k in enumerate(gfn_order)}

    p_star_top = np.array([p_star[k] for k in target_order])
    p_gfn_top = np.array([p_gfn.get(k, 0.0) for k in target_order])
    cum_star = np.cumsum(p_star_top)
    cum_gfn = np.cumsum(p_gfn_top)

    overlap = []
    weighted_recall = []
    target_prefix = []
    target_set = set()
    gfn_prefix_set = set()
    gfn_order_k = gfn_order[:k_max]

    for k in range(1, k_max + 1):
        target_set.add(target_order[k - 1])
        gfn_prefix_set.add(gfn_order_k[k - 1])
        inter = target_set & gfn_prefix_set
        overlap.append(len(inter) / k)
        denom = sum(p_star[s] for s in target_set)
        weighted_recall.append(sum(p_star[s] for s in inter) / denom if denom > 0 else 0.0)
        target_prefix.append(target_set.copy())

    optimum_rank = gfn_rank[target_order[0]]
    return {
        'k': np.arange(1, k_max + 1),
        'p_star_top': p_star_top,
        'p_gfn_top': p_gfn_top,
        'cum_star': cum_star,
        'cum_gfn': cum_gfn,
        'overlap': np.array(overlap),
        'weighted_recall': np.array(weighted_recall),
        'optimum_rank': optimum_rank,
    }



def figure_q1(sf, p_star, gfn_runs, output_dir, top_k=50):
    """
    Q1: Mode discovery.

    Produces:
      1) Faceted target heatmap over the 5-group decomposition.
      2) Faceted mean GFN heatmap over the same decomposition.
      3) Faceted across-run std heatmap.
      4) Top-k mode-discovery chart.
    """
    star_cube = _mode_heatmap_cube(p_star)
    gfn_cubes = np.stack([_mode_heatmap_cube(r['p_gfn']) for r in gfn_runs], axis=0)
    gfn_cube_mean = gfn_cubes.mean(axis=0)
    gfn_cube_std = gfn_cubes.std(axis=0)

    shared_vmin = min(star_cube.min(), gfn_cube_mean.min())
    shared_vmax = max(star_cube.max(), gfn_cube_mean.max())

    _plot_mode_heatmap_grid(
        star_cube,
        title=rf"Q1 mode heatmap — target $P^*$ (step fraction = {sf})",
        output_dir=output_dir,
        filename=f"q1_heatmap_target_sf{sf}",
        cmap='YlOrRd',
        vmin=shared_vmin,
        vmax=shared_vmax,
        cbar_label='Avg. probability',
    )
    _plot_mode_heatmap_grid(
        gfn_cube_mean,
        title=rf"Q1 mode heatmap — mean $P_{{\mathrm{{GFN}}}}$ (step fraction = {sf}, n = {len(gfn_runs)})",
        output_dir=output_dir,
        filename=f"q1_heatmap_gfn_mean_sf{sf}",
        cmap='YlOrRd',
        vmin=shared_vmin,
        vmax=shared_vmax,
        cbar_label='Avg. probability',
    )
    _plot_mode_heatmap_grid(
        gfn_cube_std,
        title=rf"Q1 mode heatmap — across-run $\sigma(P_{{\mathrm{{GFN}}}})$ (step fraction = {sf})",
        output_dir=output_dir,
        filename=f"q1_heatmap_gfn_std_sf{sf}",
        cmap='Blues',
        vmin=0.0,
        vmax=float(gfn_cube_std.max()) if np.isfinite(gfn_cube_std.max()) else None,
        cbar_label='Across-run std',
    )

    # Top-k / rank-based mode-discovery figure
    k_max = min(top_k, len(p_star))
    per_run = [_topk_mode_curves(p_star, r['p_gfn'], k_max) for r in gfn_runs]
    k_vals = per_run[0]['k']

    p_star_top = per_run[0]['p_star_top']
    p_gfn_top = np.stack([d['p_gfn_top'] for d in per_run], axis=0)
    cum_star = per_run[0]['cum_star']
    cum_gfn = np.stack([d['cum_gfn'] for d in per_run], axis=0)
    overlap = np.stack([d['overlap'] for d in per_run], axis=0)
    weighted_recall = np.stack([d['weighted_recall'] for d in per_run], axis=0)
    optimum_ranks = np.array([d['optimum_rank'] for d in per_run])

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.2))

    # Panel A: top-k probability-rank distribution
    ax = axes[0]
    ax.plot(k_vals, p_star_top, '-', color=COLORS['gt'], lw=1.5, label=r'$P^*$')
    ax.plot(k_vals, p_gfn_top.mean(axis=0), '-', color=COLORS['gfn'], lw=1.5,
            label=r'$\overline{P}_{\mathrm{GFN}}$')
    ax.fill_between(
        k_vals,
        np.percentile(p_gfn_top, 10, axis=0),
        np.percentile(p_gfn_top, 90, axis=0),
        color=COLORS['gfn_fill'],
        alpha=0.4,
        label='10–90th pctl.',
    )
    ax.set_yscale('log')
    ax.set_xlabel(r'Target rank $k$')
    ax.set_ylabel('Probability')
    ax.set_title(f'Top-{k_max} probability–rank', fontsize=10)
    ax.legend(frameon=False, fontsize=7)

    # Panel B: cumulative mass assigned to the target top-k states
    ax = axes[1]
    ax.plot(k_vals, cum_star, '-', color=COLORS['gt'], lw=1.5, label='target cumulative mass')
    ax.plot(k_vals, cum_gfn.mean(axis=0), '-', color=COLORS['gfn'], lw=1.5,
            label='GFN cumulative mass')
    ax.fill_between(
        k_vals,
        np.percentile(cum_gfn, 10, axis=0),
        np.percentile(cum_gfn, 90, axis=0),
        color=COLORS['gfn_fill'],
        alpha=0.4,
    )
    ax.set_xlabel(r'Target rank cutoff $k$')
    ax.set_ylabel('Cumulative probability mass')
    ax.set_title('Mass captured on target modes', fontsize=10)
    ax.legend(frameon=False, fontsize=7)

    # Panel C: top-k overlap / weighted recall
    ax = axes[2]
    ax.plot(k_vals, overlap.mean(axis=0), '-', color=COLORS['accent'], lw=1.5,
            label='Top-k overlap')
    ax.fill_between(
        k_vals,
        np.percentile(overlap, 10, axis=0),
        np.percentile(overlap, 90, axis=0),
        color=COLORS['accent'],
        alpha=0.18,
    )
    ax.plot(k_vals, weighted_recall.mean(axis=0), '--', color=COLORS['gfn'], lw=1.5,
            label='Weighted recall')
    ax.fill_between(
        k_vals,
        np.percentile(weighted_recall, 10, axis=0),
        np.percentile(weighted_recall, 90, axis=0),
        color=COLORS['gfn_fill'],
        alpha=0.3,
    )
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel(r'Top-k cutoff')
    ax.set_ylabel('Recovery score')
    ax.set_title('Mode-set recovery', fontsize=10)
    txt = (
        f"optimum rank (mean ± sd) = {optimum_ranks.mean():.1f} ± {optimum_ranks.std():.1f}\n"
        f"median optimum rank = {np.median(optimum_ranks):.0f}\n"
        f"best / worst = {optimum_ranks.min():.0f} / {optimum_ranks.max():.0f}"
    )
    ax.text(
        0.04,
        0.04,
        txt,
        transform=ax.transAxes,
        fontsize=7,
        va='bottom',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.9, edgecolor='lightgray'),
    )
    ax.legend(frameon=False, fontsize=7, loc='lower right')

    fig.suptitle(f"Q1 mode discovery diagnostics — step fraction = {sf}", fontsize=11, y=1.02)
    plt.tight_layout()
    _save(fig, output_dir, f"q1_topk_sf{sf}")

# ─────────────────────────────────────────────────────────────────
# Q2: Performance
# ─────────────────────────────────────────────────────────────────

def figure_q2(sf, p_star, gfn_runs, output_dir):
    """
    Panel A: Log-log scatter (all 9 runs overlaid with transparency)
    Panel B: Rank plot (P* vs mean P_GFN ± ribbon)
    Panel C: Metrics table
    """
    all_keys = sorted(p_star.keys())
    q_arr = np.array([p_star[k] for k in all_keys])

    # Collect per-run probability arrays
    p_matrix = np.array([[r["p_gfn"].get(k, 0.0) for k in all_keys] for r in gfn_runs])
    p_mean = p_matrix.mean(axis=0)
    p_std = p_matrix.std(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))

    # Panel A: Log-log scatter
    ax = axes[0]
    for i, run in enumerate(gfn_runs):
        p_i = p_matrix[i]
        mask = (p_i > 0) & (q_arr > 0)
        ax.scatter(q_arr[mask], p_i[mask], s=4, alpha=0.15, c=COLORS["gfn"],
                   edgecolors="none", rasterized=True, label="" if i > 0 else None)

    # Mean
    mask = (p_mean > 0) & (q_arr > 0)
    ax.scatter(q_arr[mask], p_mean[mask], s=6, alpha=0.5, c=COLORS["accent"],
               edgecolors="none", rasterized=True, zorder=5)

    lim_lo = min(q_arr[q_arr > 0].min(), p_mean[p_mean > 0].min()) * 0.3
    lim_hi = max(q_arr.max(), p_mean.max()) * 3
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "--", color="gray", lw=0.8, zorder=0)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$P^*(x)$")
    ax.set_ylabel(r"$P_{\mathrm{GFN}}(x)$")
    ax.set_title("State-level agreement", fontsize=10)

    # Metrics annotation (using mean distribution)
    metrics_mean = compute_metrics(dict(zip(all_keys, p_mean)), p_star, all_keys)
    txt = (f"JSD = {metrics_mean['jsd']:.4f}\n"
           f"KL  = {metrics_mean['kl']:.4f}\n"
           f"L1  = {metrics_mean['l1']:.4f}\n"
           f"ρ   = {metrics_mean['pearson_r']:.4f}\n"
           f"$N_{{eff}}^{{GFN}}$ = {metrics_mean['n_eff_gfn']:.0f}\n"
           f"$N_{{eff}}^*$     = {metrics_mean['n_eff_star']:.0f}")
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, fontsize=7,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="lightgray"))

    # Panel B: Rank plot with ribbon
    ax = axes[1]
    q_rank_order = np.argsort(-q_arr)
    q_sorted = q_arr[q_rank_order]

    p_sorted_per_run = p_matrix[:, q_rank_order]  # rank by P* order
    p_mean_sorted = p_sorted_per_run.mean(axis=0)
    p_lo = np.percentile(p_sorted_per_run, 10, axis=0)
    p_hi = np.percentile(p_sorted_per_run, 90, axis=0)

    ranks = np.arange(1, len(q_sorted) + 1)
    ax.plot(ranks, q_sorted, "-", color=COLORS["gt"], lw=1.3, label=r"$P^*$")
    ax.plot(ranks, p_mean_sorted, "-", color=COLORS["gfn"], lw=1.3, label=r"$\overline{P}_{\mathrm{GFN}}$")
    ax.fill_between(ranks, p_lo, p_hi, color=COLORS["gfn_fill"], alpha=0.4, label="10–90th pctl.")

    ax.set_yscale("log")
    ax.set_xlabel("State rank (by $P^*$)")
    ax.set_ylabel("Probability")
    ax.set_title("Rank-ordered probabilities", fontsize=10)
    ax.legend(frameon=False, fontsize=7)

    fig.suptitle(f"Distribution Accuracy — sf = {sf}", fontsize=11, y=1.02)
    plt.tight_layout()
    _save(fig, output_dir, f"q2_performance_sf{sf}")


def figure_q2_combined(all_results, output_dir):
    """
    Combined heatmap: one row per step_fraction, columns = P* | mean P_GFN.
    """
    sfs = sorted(all_results.keys())
    n_sf = len(sfs)
    fig, axes = plt.subplots(n_sf, 3, figsize=(7.0, 2.4 * n_sf),
                              gridspec_kw={"width_ratios": [1, 1, 0.05]})
    if n_sf == 1:
        axes = axes.reshape(1, -1)

    g2 = ACTIONS_PER_GROUP[2]
    g4 = ACTIONS_PER_GROUP[4]
    g2_labels = [_short(a) for a in g2]
    g4_labels = [_short(a) for a in g4]

    for row, sf in enumerate(sfs):
        r = all_results[sf]
        gs = _marginalize_g2g4(r["p_star"])
        grids_gfn = np.stack([_marginalize_g2g4(run["p_gfn"]) for run in r["gfn_runs"]])
        gg = grids_gfn.mean(axis=0)
        vmin = min(gs.min(), gg.min())
        vmax = max(gs.max(), gg.max())

        for col, (grid, title) in enumerate([(gs, r"$P^*$"), (gg, r"$\overline{P}_{\mathrm{GFN}}$")]):
            ax = axes[row, col]
            im = ax.imshow(grid, cmap="YlOrRd", vmin=vmin, vmax=vmax, aspect="auto", origin="lower")
            ax.set_xticks(range(len(g4)))
            ax.set_xticklabels(g4_labels if row == n_sf - 1 else [], rotation=45, ha="right")
            ax.set_yticks(range(len(g2)))
            ax.set_yticklabels(g2_labels if col == 0 else [])
            if row == 0:
                ax.set_title(title, fontsize=10, fontweight="bold")

        axes[row, 0].set_ylabel(f"sf = {sf}\n{GROUP_LABELS[2]}", fontsize=8)
        cb = plt.colorbar(im, cax=axes[row, 2])
        cb.set_label("Prob.", fontsize=7)

    fig.suptitle("Mode Heatmaps Across Step Fractions", fontsize=11, y=1.01)
    plt.tight_layout()
    _save(fig, output_dir, "q2_combined_heatmap")


def figure_q2_metrics_table(all_results, output_dir):
    """Metrics summary table as a figure (for supplementary or at-a-glance)."""
    sfs = sorted(all_results.keys())
    rows = []
    for sf in sfs:
        r = all_results[sf]
        all_keys = sorted(r["p_star"].keys())
        for run in r["gfn_runs"]:
            m = compute_metrics(run["p_gfn"], r["p_star"], all_keys)
            m["sf"] = sf
            m["lr"] = run.get("lr", "?")
            m["seed"] = run.get("seed", "?")
            rows.append(m)

    # Build table data
    col_labels = ["sf", "lr", "seed", "JSD", "KL", "L1", "ρ", "N_eff(GFN)", "N_eff(*)"]
    cell_text = []
    for r in rows:
        cell_text.append([
            f"{r['sf']}", f"{r['lr']}", f"{r['seed']}",
            f"{r['jsd']:.5f}", f"{r['kl']:.3f}", f"{r['l1']:.3f}",
            f"{r['pearson_r']:.4f}", f"{r['n_eff_gfn']:.0f}", f"{r['n_eff_star']:.0f}",
        ])

    fig, ax = plt.subplots(figsize=(8, 0.35 * len(rows) + 1.0))
    ax.axis("off")
    table = ax.table(cellText=cell_text, colLabels=col_labels, loc="center",
                     cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.auto_set_column_width(range(len(col_labels)))

    # Header style
    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor("#2c3e50")
        table[(0, j)].set_text_props(color="white", fontweight="bold")

    fig.suptitle("Per-Run Metrics Summary", fontsize=11)
    plt.tight_layout()
    _save(fig, output_dir, "q2_metrics_table")


# ─────────────────────────────────────────────────────────────────
# Q3: Hyperparameter Sensitivity
# ─────────────────────────────────────────────────────────────────

def figure_q3_convergence(runs_data, output_dir):
    """JSD convergence curves: one subplot per step_fraction, colored by lr."""
    sf_groups = defaultdict(lambda: defaultdict(list))
    for run in runs_data:
        sf = run["step_fraction"]
        lr = run.get("lr", "?")
        if "Jensen Shannon Div." in run.get("history", {}):
            sf_groups[sf][lr].append(run)

    sorted_sfs = sorted([sf for sf in sf_groups if sf != "unknown"])
    if not sorted_sfs:
        print("[WARN] No JSD data in wandb runs. Skipping Q3 convergence.")
        return

    n_sf = len(sorted_sfs)
    fig, axes = plt.subplots(1, n_sf, figsize=(3.0 * n_sf, 3.2), sharey=True)
    if n_sf == 1:
        axes = [axes]

    lr_cmap = plt.cm.Set1
    all_lrs = sorted({run.get("lr") for run in runs_data if run.get("lr") is not None})
    lr_colors = {lr: lr_cmap(i / max(len(all_lrs) - 1, 1)) for i, lr in enumerate(all_lrs)}

    for ax, sf in zip(axes, sorted_sfs):
        for lr in sorted(sf_groups[sf].keys()):
            color = lr_colors.get(lr, "gray")
            for run in sf_groups[sf][lr]:
                h = run["history"]["Jensen Shannon Div."]
                ax.plot(h["step"], h["value"], color=color, alpha=0.4, lw=0.8)
            # Mean per lr
            steps, vals = _interpolate_runs(sf_groups[sf][lr], "Jensen Shannon Div.")
            if steps is not None:
                ax.plot(steps, vals, color=color, lw=2.0, label=f"lr={lr}")

        ax.set_xlabel("Training step")
        ax.set_title(f"sf = {sf}", fontsize=10)
        ax.set_yscale("log")
        ax.legend(frameon=False, fontsize=6)

    axes[0].set_ylabel("Jensen-Shannon Divergence")
    fig.suptitle("Convergence by Step Fraction and Learning Rate", fontsize=11, y=1.02)
    plt.tight_layout()
    _save(fig, output_dir, "q3_convergence")


def figure_q3_sensitivity_heatmap(runs_data, output_dir):
    """
    Heatmap: rows = step_fraction, cols = learning_rate.
    Cell color = mean final JSD across seeds. Annotated with mean ± std.
    """
    final_metrics = defaultdict(lambda: defaultdict(list))
    for run in runs_data:
        sf = run["step_fraction"]
        lr = run.get("lr", "?")
        h = run.get("history", {}).get("Jensen Shannon Div.", {})
        if h.get("value"):
            final_metrics[sf][lr].append(h["value"][-1])

    sorted_sfs = sorted([sf for sf in final_metrics if sf != "unknown"])
    sorted_lrs = sorted({lr for sf_lrs in final_metrics.values() for lr in sf_lrs})

    if not sorted_sfs or not sorted_lrs:
        print("[WARN] Insufficient data for Q3 sensitivity heatmap.")
        return

    grid = np.full((len(sorted_sfs), len(sorted_lrs)), np.nan)
    annot = [['' for _ in sorted_lrs] for _ in sorted_sfs]

    for i, sf in enumerate(sorted_sfs):
        for j, lr in enumerate(sorted_lrs):
            vals = final_metrics[sf].get(lr, [])
            if vals:
                grid[i, j] = np.mean(vals)
                annot[i][j] = f"{np.mean(vals):.4f}\n±{np.std(vals):.4f}"

    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    sns.heatmap(grid, ax=ax, annot=annot, fmt="",
                xticklabels=[f"{lr}" for lr in sorted_lrs],
                yticklabels=[f"{sf}" for sf in sorted_sfs],
                cmap="RdYlGn_r", linewidths=0.5, linecolor="white",
                cbar_kws={"label": "Final JSD"})
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Step fraction")
    ax.set_title("Final JSD — Hyperparameter Sensitivity", fontsize=10)
    plt.tight_layout()
    _save(fig, output_dir, "q3_sensitivity_heatmap")


def figure_q3_loss_logz(runs_data, output_dir):
    """TB loss and logZ convergence, grouped by step_fraction."""
    sf_groups = defaultdict(list)
    for run in runs_data:
        sf_groups[run["step_fraction"]].append(run)

    sorted_sfs = sorted([sf for sf in sf_groups if sf != "unknown"])
    cmap = plt.cm.viridis

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
    for panel_idx, metric_key in enumerate(["Loss", "logZ"]):
        ax = axes[panel_idx]
        for i, sf in enumerate(sorted_sfs):
            color = cmap(i / max(len(sorted_sfs) - 1, 1))
            for run in sf_groups[sf]:
                if metric_key in run.get("history", {}):
                    h = run["history"][metric_key]
                    ax.plot(h["step"], h["value"], color=color, alpha=0.2, lw=0.6)
            steps, vals = _interpolate_runs(sf_groups[sf], metric_key)
            if steps is not None:
                ax.plot(steps, vals, color=color, lw=2.0, label=f"sf={sf}")

        ax.set_xlabel("Training step")
        ax.set_ylabel(metric_key)
        ax.set_title(f"{metric_key} convergence", fontsize=10)
        ax.legend(frameon=False, fontsize=7)
        if metric_key == "Loss":
            ax.set_yscale("log")

    plt.tight_layout()
    _save(fig, output_dir, "q3_loss_logz")


# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════

def _interpolate_runs(runs, metric_key, n_points=200):
    valid = [r for r in runs if metric_key in r.get("history", {})]
    if not valid:
        return None, None
    step_lists = [r["history"][metric_key]["step"] for r in valid]
    val_lists = [r["history"][metric_key]["value"] for r in valid]
    lo = max(s[0] for s in step_lists if s)
    hi = min(s[-1] for s in step_lists if s)
    if lo >= hi:
        return None, None
    common = np.linspace(lo, hi, n_points)
    interp = [np.interp(common, s, v) for s, v in zip(step_lists, val_lists) if len(s) >= 2]
    return (common, np.mean(interp, axis=0)) if interp else (None, None)


def _save(fig, output_dir, name):
    for ext in ["pdf", "png"]:
        path = os.path.join(output_dir, f"{name}.{ext}")
        fig.savefig(path)
    print(f"  [SAVED] {name}.pdf / .png")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="GFlowNet Paper Figures (1-Cycle)")
    parser.add_argument("--reward_tables", nargs="+", required=True)
    parser.add_argument("--step_fractions", nargs="+", type=float, required=True)
    parser.add_argument("--beta", type=float, default=50.0)
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--wandb_seeds", nargs="+", type=int, default=None,
                        help="Filter wandb runs by GFLOWNET.SEED values")
    parser.add_argument("--wandb_betas", nargs="+", type=float, default=None,
                        help="Filter wandb runs by PROXY.BETA values")
    parser.add_argument("--wandb_lrs", nargs="+", type=float, default=None,
                        help="Filter wandb runs by GFLOWNET.OPTIMIZER.LEARNING_RATE values")
    parser.add_argument("--wandb_states", nargs="+", default=None,
                        help="Filter wandb runs by state (e.g. finished running crashed). Default: all states")
    parser.add_argument("--wandb_lr_z_mults", nargs="+", type=float, default=None,
                        help="Filter wandb runs by GFLOWNET.OPTIMIZER.LR_Z_MULT values")
    parser.add_argument("--wandb_random_action_probs", nargs="+", type=float, default=None,
                        help="Filter wandb runs by GFLOWNET.RANDOM_ACTION_PROB values")
    parser.add_argument("--wandb_run_ids", nargs="+", default=None,
                        help="Fetch specific wandb runs by ID (e.g. f048bllh 7ssnab2n). "
                             "Overrides all other filters when specified.")
    parser.add_argument("--wandb_after", default=None, metavar="RUN_ID",
                        help="Only include runs created after this run ID. "
                             "Useful to skip old runs with broken configs.")
    parser.add_argument("--output_dir", default="figures")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip_checkpoints", action="store_true")
    parser.add_argument("--artifact_alias", default="final",
                        help="Artifact alias to download (default: final)")
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Number of trajectories to sample for evaluation. "
                             "If not set, uses exact enumeration of all 2625 states.")
    parser.add_argument("--q1_top_k", type=int, default=50,
                        help="Number of target-ranked states to show in the Q1 top-k mode discovery figure.")
    parser.add_argument("--wandb_history_samples", type=int, default=500,
                        help="Number of history samples to fetch per wandb run (default: 500)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    assert len(args.reward_tables) == len(args.step_fractions)

    # ── Early wandb fetch (always fresh, no cache) ──
    runs_data = None
    if args.wandb_project:
        print("=" * 60)
        print("STEP 0: Fetching wandb data")
        print("=" * 60)
        try:
            runs_data = fetch_wandb_runs(
                args.wandb_project,
                seeds=args.wandb_seeds,
                step_fractions=args.step_fractions,
                betas=args.wandb_betas,
                learning_rates=args.wandb_lrs,
                lr_z_mults=args.wandb_lr_z_mults,
                random_action_probs=args.wandb_random_action_probs,
                states=args.wandb_states,
                run_ids=args.wandb_run_ids,
                after_run_id=args.wandb_after,
                history_samples=args.wandb_history_samples,
            )
            print(f"  Retrieved {len(runs_data)} run records from wandb (fresh query)")
        except Exception as e:
            print(f"  [ERROR] wandb: {e}")
            runs_data = None
        print()

    # ── Checkpoint source policy ──
    if not args.skip_checkpoints and not args.wandb_project:
        raise ValueError(
            "--wandb_project is required unless --skip_checkpoints is set, "
            "because checkpoints are always loaded fresh from wandb artifacts."
        )

    # ── Step 1: Ground truth ──
    print("=" * 60)
    print("STEP 1: Loading ground truth distributions")
    print("=" * 60)
    all_results = {}
    for rt_path, sf in zip(args.reward_tables, args.step_fractions):
        p_star, losses = load_reward_table(rt_path, args.beta)
        all_results[sf] = {"p_star": p_star, "losses": losses, "gfn_runs": []}
        print(f"  sf={sf}: {len(p_star)} states, top P*={max(p_star.values()):.6f}")
    print()

    # ── Step 2: GFN distributions from checkpoints ──
    eval_method = "sample" if args.n_samples else "exact"
    print("=" * 60)
    print(f"STEP 2: Computing GFN distributions ({eval_method}"
          f"{f', n={args.n_samples}' if args.n_samples else ', all 2625 states'})")
    print("=" * 60)

    if not args.skip_checkpoints:
        if not runs_data:
            print("  [WARN] No wandb run metadata is available. Skipping checkpoint loading.")
            print("         Provide --wandb_project and matching filters or --wandb_run_ids.")
        else:
            print("  Resolving checkpoints from wandb artifacts (fresh download, no cache)")
            ckpt_map = download_wandb_artifacts(
                runs_data, args.wandb_project,
                alias=args.artifact_alias,
            )

            if not ckpt_map:
                print("  [WARN] No checkpoint artifacts were found on wandb.")
            else:
                for sf_str, entries in ckpt_map.items():
                    try:
                        sf = float(sf_str)
                    except (ValueError, TypeError):
                        print(f"  [SKIP] sf={sf_str} (not a valid number)")
                        continue
                    if sf not in all_results:
                        print(f"  [SKIP] sf={sf} not in --step_fractions")
                        continue

                    for entry in entries:
                        ckpt_path = entry["path"]
                        tmp_dir = entry.get("tmp_dir")
                        lr = entry.get("lr")
                        seed = entry.get("seed")
                        wandb_id = entry.get("wandb_id", "")

                        try:
                            print(f"  Loading {ckpt_path} (sf={sf}, lr={lr}, seed={seed}, wandb_id={wandb_id})")
                            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                            in_dim = ckpt["forward"]["0.weight"].shape[1]
                            hid_dim = ckpt["forward"]["0.weight"].shape[0]
                            out_dim = ckpt["forward"]["4.weight"].shape[0]
                            assert out_dim >= N_ACTIONS, (
                                f"Output dim {out_dim} < {N_ACTIONS} (expected {N_ACTIONS} "
                                f"or {N_ACTIONS + 1} with EOS)"
                            )
                            model = build_forward_mlp(ckpt["forward"], in_dim, hid_dim, out_dim)

                            if args.n_samples:
                                print(f"    Sampling {args.n_samples} trajectories...")
                                p_gfn = sample_gfn_distribution(
                                    model, sf, args.n_samples, args.device
                                )
                            else:
                                print(f"    Enumerating 2625 states...")
                                p_gfn = compute_exact_gfn_distribution(
                                    model, sf, args.device
                                )
                        finally:
                            if tmp_dir:
                                shutil.rmtree(tmp_dir, ignore_errors=True)

                        all_results[sf]["gfn_runs"].append({
                            "p_gfn": p_gfn, "lr": lr, "seed": seed,
                            "wandb_id": wandb_id,
                        })
    else:
        print("  Skipping checkpoint evaluation because --skip_checkpoints was set.")
    print()

    # ── Step 3: Metrics ──
    print("=" * 60)
    print("STEP 3: Per-run metrics")
    print("=" * 60)
    for sf in args.step_fractions:
        r = all_results[sf]
        all_keys = sorted(r["p_star"].keys())
        for run in r["gfn_runs"]:
            m = compute_metrics(run["p_gfn"], r["p_star"], all_keys)
            run["metrics"] = m
            print(f"  sf={sf} lr={run['lr']} seed={run['seed']}: "
                  f"JSD={m['jsd']:.5f} KL={m['kl']:.3f} ρ={m['pearson_r']:.4f}")
    print()

    # ── Step 4: Q1 + Q2 figures ──
    print("=" * 60)
    print("STEP 4: Generating Q1 & Q2 figures")
    print("=" * 60)
    for sf in args.step_fractions:
        r = all_results[sf]
        if not r["gfn_runs"]:
            print(f"  [SKIP] sf={sf}: no GFN distributions available")
            continue
        print(f"  Q1 — sf={sf} ({len(r['gfn_runs'])} runs)")
        figure_q1(sf, r["p_star"], r["gfn_runs"], args.output_dir, top_k=args.q1_top_k)
        print(f"  Q2 — sf={sf}")
        figure_q2(sf, r["p_star"], r["gfn_runs"], args.output_dir)

    valid = {sf: r for sf, r in all_results.items() if r["gfn_runs"]}
    if valid:
        print("  Q2 — combined heatmap")
        figure_q2_combined(valid, args.output_dir)
        print("  Q2 — metrics table")
        figure_q2_metrics_table(valid, args.output_dir)
    print()

    # ── Step 5: Q3 figures (reuse wandb data from Step 0) ──
    if runs_data:
        print("=" * 60)
        print("STEP 5: Q3 figures")
        print("=" * 60)
        figure_q3_convergence(runs_data, args.output_dir)
        figure_q3_sensitivity_heatmap(runs_data, args.output_dir)
        figure_q3_loss_logz(runs_data, args.output_dir)

    # ── Summary ──
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Build lookup from wandb_id to run metadata
    runs_lookup = {}
    if runs_data:
        for r in runs_data:
            runs_lookup[r["id"]] = r

    print(f"{'run_id':<12}{'sf':<6}{'lr':<8}{'seed':<6}{'ε':<6}{'JSD':<10}{'KL':<8}{'L1':<8}{'ρ':<8}{'N_eff(GFN)':<12}{'N_eff(*)'}")
    print("-" * 94)
    for sf in sorted(args.step_fractions):
        for run in all_results[sf]["gfn_runs"]:
            m = run.get("metrics", {})
            if m:
                wid = run.get("wandb_id", "?")
                eps = runs_lookup.get(wid, {}).get("random_action_prob", "?")
                print(f"{wid:<12}{sf:<6}{run['lr']!s:<8}{run['seed']!s:<6}"
                      f"{eps!s:<6}"
                      f"{m['jsd']:<10.5f}{m['kl']:<8.3f}{m['l1']:<8.3f}"
                      f"{m['pearson_r']:<8.4f}{m['n_eff_gfn']:<12.0f}{m['n_eff_star']:.0f}")

    print(f"\nFigures → {args.output_dir}/")


if __name__ == "__main__":
    main()
