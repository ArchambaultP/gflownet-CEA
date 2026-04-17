from types import GenericAlias

# ─────────────────────────────────────────────────────────────────
# Group ordering: follows causal carbon flow through the model
# ─────────────────────────────────────────────────────────────────

GROUP_ORDER = [
    "leaf_and_canopy_geometry",
    "photosynthetic_potential",
    "temperature_inhibition",
    "temperature_and_development",
    "biomass_growth_and_maintenance",
]

# ─────────────────────────────────────────────────────────────────
# Group membership
# ─────────────────────────────────────────────────────────────────

GROUPS = {
    "leaf_and_canopy_geometry": [
        "LAI_max",
        "SLA",
        "n_plants",
    ],

    "photosynthetic_potential": [
        "J_max_leaf",
        "Jpot_activation",
        "Jpot_deactivation",
        "Jpot_entropy",
        "alpha",
        "deg_curv_elec_transport",
        "Tcan_CO2_comp_point",
        "CO2_air_stomata",
        "net_ass_rate",
    ],

    "temperature_inhibition": [
        "k_sw_min_Tcan",
        "s_min_Tcan",
        "k_sw_max_Tcan",
        "s_max_Tcan",
        "k_sw_min_Tcan24",
        "s_min_Tcan24",
        "k_sw_max_Tcan24",
        "s_max_Tcan24",
    ],

    "temperature_and_development": [
        "bias_g_Tcan24",
        "slope_g_Tcan24",
        "TS_start",
        "TS_end",
        "c_dev1",
        "c_dev2",
        "r_fruit_Set",
    ],

    "biomass_growth_and_maintenance": [
        "G_max",
        "c_fruit_growth",
        "c_leaf_growth",
        "c_stem_growth",
        "c_fruit_maintenance",
        "c_leaf_maintenance",
        "c_stem_maintenance",
        "Q_10_maintenance",
        "rg_fruit",
        "rg_leaf",
        "rg_stem",
        "c_rgr",
    ],
}

# ─────────────────────────────────────────────────────────────────
# Unique action names per group
# ─────────────────────────────────────────────────────────────────

CANONICAL_ACTIONS_BY_GROUP = {
    "leaf_and_canopy_geometry": [
        "none",
        "increase",
        "decrease",
    ],
    "photosynthetic_potential": [
        "none",
        "increase",
        "decrease",
        "higher_sensitivity",
        "lower_sensitivity",
    ],
    "temperature_inhibition": [
        "none",
        "shift_warm",
        "shift_cold",
        "widen_optimum",
        "narrow_optimum",
    ],
    "temperature_and_development": [
        "none",
        "increase",
        "decrease",
        "higher_sensitivity",
        "lower_sensitivity",
    ],
    "biomass_growth_and_maintenance": [
        "none",
        "more_fruit_growth",
        "more_veg_growth",
        "lower_resp_cost",
        "higher_resp_cost",
        "higher_sensitivity",
        "lower_sensitivity",
    ],
}

GROUP_ACTION_PREFIX = {
    "leaf_and_canopy_geometry": "canopy",
    "photosynthetic_potential": "photo",
    "temperature_inhibition": "temp_inhib",
    "temperature_and_development": "temp_dev",
    "biomass_growth_and_maintenance": "biomass",
}

UNIQUE_ACTION_NAME_BY_GROUP_AND_CANONICAL = {
    group: {canonical: f"{GROUP_ACTION_PREFIX[group]}__{canonical}" for canonical in actions}
    for group, actions in CANONICAL_ACTIONS_BY_GROUP.items()
}

CANONICAL_ACTION_NAME_BY_UNIQUE = {
    unique: canonical
    for group, mapping in UNIQUE_ACTION_NAME_BY_GROUP_AND_CANONICAL.items()
    for canonical, unique in mapping.items()
}

DISPLAY_ACTION_LABELS = {
    unique: canonical
    for unique, canonical in CANONICAL_ACTION_NAME_BY_UNIQUE.items()
}


def unique_action_name(group_name: str, canonical_action_name: str) -> str:
    return UNIQUE_ACTION_NAME_BY_GROUP_AND_CANONICAL[group_name][canonical_action_name]


def canonical_action_name(action_name: str) -> str:
    return CANONICAL_ACTION_NAME_BY_UNIQUE.get(action_name, action_name)


def old_state_key_to_new(state_key: str) -> str:
    toks = [t.strip() for t in state_key.split("|")]
    if len(toks) != len(GROUP_ORDER):
        raise ValueError(f"Expected {len(GROUP_ORDER)} actions, got {len(toks)} for key={state_key!r}")
    new_toks = [
        UNIQUE_ACTION_NAME_BY_GROUP_AND_CANONICAL[group][tok]
        for group, tok in zip(GROUP_ORDER, toks)
    ]
    return "|".join(new_toks)


def new_state_key_to_old(state_key: str) -> str:
    toks = [t.strip() for t in state_key.split("|")]
    if len(toks) != len(GROUP_ORDER):
        raise ValueError(f"Expected {len(GROUP_ORDER)} actions, got {len(toks)} for key={state_key!r}")
    old_toks = [CANONICAL_ACTION_NAME_BY_UNIQUE.get(tok, tok) for tok in toks]
    return "|".join(old_toks)

# ─────────────────────────────────────────────────────────────────
# Perturbation scheme: same directions, unique action names
# ─────────────────────────────────────────────────────────────────

PERTURBATION_SCHEME = {

    "leaf_and_canopy_geometry": {
        "canopy__none": {
            "LAI_max": 0, "SLA": 0, "n_plants": 0,
        },
        "canopy__increase": {
            "LAI_max": +1, "SLA": +1, "n_plants": +1,
        },
        "canopy__decrease": {
            "LAI_max": -1, "SLA": -1, "n_plants": -1,
        },
    },

    "photosynthetic_potential": {
        "photo__none": {
            "J_max_leaf": 0, "Jpot_activation": 0, "Jpot_deactivation": 0,
            "Jpot_entropy": 0, "alpha": 0, "deg_curv_elec_transport": 0,
            "Tcan_CO2_comp_point": 0, "CO2_air_stomata": 0, "net_ass_rate": 0,
        },
        "photo__increase": {
            "J_max_leaf": +1, "alpha": +1, "CO2_air_stomata": +1,
            "net_ass_rate": -1, "Tcan_CO2_comp_point": -1,
            "Jpot_activation": 0, "Jpot_deactivation": 0,
            "Jpot_entropy": 0, "deg_curv_elec_transport": 0,
        },
        "photo__decrease": {
            "J_max_leaf": -1, "alpha": -1, "CO2_air_stomata": -1,
            "net_ass_rate": +1, "Tcan_CO2_comp_point": +1,
            "Jpot_activation": 0, "Jpot_deactivation": 0,
            "Jpot_entropy": 0, "deg_curv_elec_transport": 0,
        },
        "photo__higher_sensitivity": {
            "Jpot_activation": +1, "Jpot_deactivation": -1,
            "deg_curv_elec_transport": -1, "Tcan_CO2_comp_point": +1,
            "J_max_leaf": 0, "Jpot_entropy": 0,
            "alpha": 0, "CO2_air_stomata": 0, "net_ass_rate": 0,
        },
        "photo__lower_sensitivity": {
            "Jpot_activation": -1, "Jpot_deactivation": +1,
            "deg_curv_elec_transport": +1, "Tcan_CO2_comp_point": -1,
            "J_max_leaf": 0, "Jpot_entropy": 0,
            "alpha": 0, "CO2_air_stomata": 0, "net_ass_rate": 0,
        },
    },

    "temperature_inhibition": {
        "temp_inhib__none": {
            "k_sw_min_Tcan": 0, "s_min_Tcan": 0,
            "k_sw_max_Tcan": 0, "s_max_Tcan": 0,
            "k_sw_min_Tcan24": 0, "s_min_Tcan24": 0,
            "k_sw_max_Tcan24": 0, "s_max_Tcan24": 0,
        },
        "temp_inhib__shift_warm": {
            "k_sw_min_Tcan": +1, "s_min_Tcan": 0,
            "k_sw_max_Tcan": +1, "s_max_Tcan": 0,
            "k_sw_min_Tcan24": +1, "s_min_Tcan24": 0,
            "k_sw_max_Tcan24": +1, "s_max_Tcan24": 0,
        },
        "temp_inhib__shift_cold": {
            "k_sw_min_Tcan": -1, "s_min_Tcan": 0,
            "k_sw_max_Tcan": -1, "s_max_Tcan": 0,
            "k_sw_min_Tcan24": -1, "s_min_Tcan24": 0,
            "k_sw_max_Tcan24": -1, "s_max_Tcan24": 0,
        },
        "temp_inhib__widen_optimum": {
            "k_sw_min_Tcan": -1, "s_min_Tcan": +1,
            "k_sw_max_Tcan": +1, "s_max_Tcan": -1,
            "k_sw_min_Tcan24": -1, "s_min_Tcan24": +1,
            "k_sw_max_Tcan24": +1, "s_max_Tcan24": -1,
        },
        "temp_inhib__narrow_optimum": {
            "k_sw_min_Tcan": +1, "s_min_Tcan": -1,
            "k_sw_max_Tcan": -1, "s_max_Tcan": +1,
            "k_sw_min_Tcan24": +1, "s_min_Tcan24": -1,
            "k_sw_max_Tcan24": -1, "s_max_Tcan24": +1,
        },
    },

    "temperature_and_development": {
        "temp_dev__none": {
            "bias_g_Tcan24": 0, "slope_g_Tcan24": 0,
            "TS_start": 0, "TS_end": 0,
            "c_dev1": 0, "c_dev2": 0, "r_fruit_Set": 0,
        },
        "temp_dev__increase": {
            "bias_g_Tcan24": +1, "slope_g_Tcan24": +1,
            "TS_end": -1, "c_dev1": +1, "c_dev2": +1,
            "r_fruit_Set": -1, "TS_start": 0,
        },
        "temp_dev__decrease": {
            "bias_g_Tcan24": -1, "slope_g_Tcan24": -1,
            "TS_end": +1, "c_dev1": -1, "c_dev2": -1,
            "r_fruit_Set": +1, "TS_start": 0,
        },
        "temp_dev__higher_sensitivity": {
            "slope_g_Tcan24": +1, "c_dev2": +1,
            "bias_g_Tcan24": 0, "TS_start": 0, "TS_end": 0,
            "c_dev1": 0, "r_fruit_Set": 0,
        },
        "temp_dev__lower_sensitivity": {
            "slope_g_Tcan24": -1, "c_dev2": -1,
            "bias_g_Tcan24": 0, "TS_start": 0, "TS_end": 0,
            "c_dev1": 0, "r_fruit_Set": 0,
        },
    },

    "biomass_growth_and_maintenance": {
        "biomass__none": {
            "G_max": 0,
            "c_fruit_growth": 0, "c_leaf_growth": 0, "c_stem_growth": 0,
            "c_fruit_maintenance": 0, "c_leaf_maintenance": 0, "c_stem_maintenance": 0,
            "Q_10_maintenance": 0, "rg_fruit": 0, "rg_leaf": 0,
            "rg_stem": 0, "c_rgr": 0,
        },
        "biomass__more_fruit_growth": {
            "G_max": -1, "rg_fruit": +1, "rg_leaf": -1, "rg_stem": -1,
            "c_fruit_growth": 0, "c_leaf_growth": 0, "c_stem_growth": 0,
            "c_fruit_maintenance": 0, "c_leaf_maintenance": 0, "c_stem_maintenance": 0,
            "Q_10_maintenance": 0, "c_rgr": 0,
        },
        "biomass__more_veg_growth": {
            "G_max": +1, "rg_fruit": -1, "rg_leaf": +1, "rg_stem": +1,
            "c_fruit_growth": 0, "c_leaf_growth": 0, "c_stem_growth": 0,
            "c_fruit_maintenance": 0, "c_leaf_maintenance": 0, "c_stem_maintenance": 0,
            "Q_10_maintenance": 0, "c_rgr": 0,
        },
        "biomass__lower_resp_cost": {
            "G_max": 0, "rg_fruit": 0, "rg_leaf": 0, "rg_stem": 0,
            "c_fruit_growth": -1, "c_leaf_growth": -1, "c_stem_growth": -1,
            "c_fruit_maintenance": -1, "c_leaf_maintenance": -1, "c_stem_maintenance": -1,
            "Q_10_maintenance": 0, "c_rgr": 0,
        },
        "biomass__higher_resp_cost": {
            "G_max": 0, "rg_fruit": 0, "rg_leaf": 0, "rg_stem": 0,
            "c_fruit_growth": +1, "c_leaf_growth": +1, "c_stem_growth": +1,
            "c_fruit_maintenance": +1, "c_leaf_maintenance": +1, "c_stem_maintenance": +1,
            "Q_10_maintenance": 0, "c_rgr": 0,
        },
        "biomass__higher_sensitivity": {
            "Q_10_maintenance": +1, "c_rgr": +1,
            "G_max": 0, "rg_fruit": 0, "rg_leaf": 0, "rg_stem": 0,
            "c_fruit_growth": 0, "c_leaf_growth": 0, "c_stem_growth": 0,
            "c_fruit_maintenance": 0, "c_leaf_maintenance": 0, "c_stem_maintenance": 0,
        },
        "biomass__lower_sensitivity": {
            "Q_10_maintenance": -1, "c_rgr": -1,
            "G_max": 0, "rg_fruit": 0, "rg_leaf": 0, "rg_stem": 0,
            "c_fruit_growth": 0, "c_leaf_growth": 0, "c_stem_growth": 0,
            "c_fruit_maintenance": 0, "c_leaf_maintenance": 0, "c_stem_maintenance": 0,
        },
    },
}

# ─────────────────────────────────────────────────────────────────
# Parameter bounds — adjusted so Vanthoor defaults are inside
# ─────────────────────────────────────────────────────────────────

PARAMETER_BOUNDS = {
    "LAI_max":       (2.0,     5.5),
    "SLA":           (1.5e-5,  4.0e-5),
    "n_plants":      (1.2,     2.5),

    "J_max_leaf":              (100.0,     350.0),
    "Jpot_activation":         (280_000.0, 500_000.0),
    "Jpot_deactivation":       (150_000.0, 300_000.0),
    "Jpot_entropy":            (640.0,     780.0),
    "alpha":                   (0.25,      0.50),
    "deg_curv_elec_transport": (0.50,      0.98),
    "Tcan_CO2_comp_point":     (1.0,       2.8),
    "CO2_air_stomata":         (0.50,      0.80),
    "net_ass_rate":            (1.0,       6.0),

    "k_sw_min_Tcan":   (5.0,   16.0),
    "k_sw_max_Tcan":   (28.0,  39.0),
    "k_sw_min_Tcan24": (10.0,  20.0),
    "k_sw_max_Tcan24": (19.0,  28.0),
    "s_min_Tcan":      (-1.5,  -0.4),
    "s_max_Tcan":      (0.3,   1.0),
    "s_min_Tcan24":    (-2.0,  -0.5),
    "s_max_Tcan24":    (0.5,   2.0),

    "bias_g_Tcan24":    (-2.0,      0.06),
    "slope_g_Tcan24":   (0.047,     0.20),
    "TS_start":         (-20.0,    20.0),
    "TS_end":           (400.0,    1200.0),
    "c_dev1":           (-1.5e-8,  -2.0e-9),
    "c_dev2":           (5.0e-9,   2.5e-8),
    "r_fruit_Set":      (0.02,     0.35),

    "G_max":               (400.0,   10_000.0),
    "c_fruit_growth":      (0.18,     0.40),
    "c_leaf_growth":       (0.18,     0.40),
    "c_stem_growth":       (0.20,     0.44),
    "c_fruit_maintenance": (5.0e-8,   2.5e-7),
    "c_leaf_maintenance":  (1.5e-7,   7.0e-7),
    "c_stem_maintenance":  (7.0e-8,   3.0e-7),
    "Q_10_maintenance":    (1.4,      3.0),
    "rg_fruit":            (0.20,     0.70),
    "rg_leaf":             (0.03,     0.15),
    "rg_stem":             (0.02,     0.12),
    "c_rgr":               (1.0e6,    5.0e6),
}

CONSTRAINTS = {
    "temperature_inhibition": {
        "min_gap_inst": 12.0,
        "min_gap_24h":   6.0,
    },
}

INITIAL_CONDITIONS = {
    "init_Cbuff": 1000.0,
    "init_Cleaf": 15000.0,
    "init_Cstem": 8000.0,
    "init_Cfruits": [0.0] * 50,
    "init_Nfruits": [0.0] * 50,
    "init_TScan": 0.0,
}

BASELINE_PARAMETERS = {
    "LAI_max": 3.0,
    "SLA": 3e-05,
    "n_plants": 2.5,
    "J_max_leaf": 210.0,
    "Jpot_activation": 370_000.0,
    "Jpot_deactivation": 220_000.0,
    "Jpot_entropy": 710.0,
    "alpha": 0.385,
    "deg_curv_elec_transport": 0.7,
    "Tcan_CO2_comp_point": 1.7,
    "CO2_air_stomata": 0.67,
    "net_ass_rate": 1.0,
    "k_sw_min_Tcan": 10.0,
    "s_min_Tcan": -0.869,
    "k_sw_max_Tcan": 34.0,
    "s_max_Tcan": 0.5793,
    "k_sw_min_Tcan24": 14.0,
    "s_min_Tcan24": -1.1587,
    "k_sw_max_Tcan24": 24.5,
    "s_max_Tcan24": 1.13904,
    "bias_g_Tcan24": 0.06,
    "slope_g_Tcan24": 0.047,
    "TS_start": 0.0,
    "TS_end": 1035.0,
    "c_dev1": -7.64e-9,
    "c_dev2": 1.16e-8,
    "r_fruit_Set": 0.1,
    "G_max": 10_000.0,
    "c_fruit_growth": 0.27,
    "c_leaf_growth": 0.28,
    "c_stem_growth": 0.3,
    "c_fruit_maintenance": 1.16e-7,
    "c_leaf_maintenance": 3.47e-7,
    "c_stem_maintenance": 1.47e-7,
    "Q_10_maintenance": 2.0,
    "rg_fruit": 0.328,
    "rg_leaf": 0.095,
    "rg_stem": 0.074,
    "c_rgr": 2_850_000.0,

    "rho_can": 0.07,
    "rho_floor": 0.5,
    "K1": 0.7,
    "K2": 0.7,
    "Jpot_ref_temp": 298.15,
    "molar_gas_constant": 8.314,
    "mass_CH20": 0.03,
    "n_fruit_phases": 50.0,
    "k_sw_max_Cbuff": 20_000.0,
    "k_sw_min_Cbuff": 1_000.0,
    "c_max_buf_fruit_1": -1.71e-7,
    "c_max_buf_fruit_2": 7.31e-7,
    "s_MCairbuf_Cbuf": 0.0005,
    "s_MCbuforg_Cbuf": -0.005,
    "s_harvest": -5e-05,
}

OUTPUT_TYPES = {
    "C_stems": float,
    "C_leaves": float,
    "LeafAreaIndex": float,
    "RelativeGrowthRate": float,
    "N_fruits": list[50],
    "C_fruits": list[50],
    "Sum_Cfruits": float,
    "Sum_Nfruits": float,
    "C_buffer": float,
    "N_harvest": float,
    "C_harvest": float,
}


def parse_output_type(k):
    t = OUTPUT_TYPES[k]
    if isinstance(t, GenericAlias) and t.__origin__ is list:
        return ("list", t.__args__[0])
    else:
        return ("scalar", t)
