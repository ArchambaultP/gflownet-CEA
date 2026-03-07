from types import GenericAlias

# ─────────────────────────────────────────────────────────────────
# Group ordering: follows causal carbon flow through the model
# ─────────────────────────────────────────────────────────────────

GROUP_ORDER = [
    "leaf_and_canopy_geometry",        # Step 1: how much PAR is intercepted
    "photosynthetic_potential",        # Step 2: how efficiently PAR → CH2O
    "temperature_inhibition",          # Step 3: hard envelope on growth
    "temperature_and_development",     # Step 4: phenology within envelope
    "biomass_growth_and_maintenance",  # Step 5: carbon partitioning & costs
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
# Perturbation scheme: direction vectors per mode per group
# ─────────────────────────────────────────────────────────────────

PERTURBATION_SCHEME = {

    "leaf_and_canopy_geometry": {
        "none": {
            "LAI_max": 0, "SLA": 0, "n_plants": 0,
        },
        "increase": {
            "LAI_max": +1, "SLA": +1, "n_plants": +1,
        },
        "decrease": {
            "LAI_max": -1, "SLA": -1, "n_plants": -1,
        },
    },

    "photosynthetic_potential": {
        "none": {
            "J_max_leaf": 0, "Jpot_activation": 0, "Jpot_deactivation": 0,
            "Jpot_entropy": 0, "alpha": 0, "deg_curv_elec_transport": 0,
            "Tcan_CO2_comp_point": 0, "CO2_air_stomata": 0, "net_ass_rate": 0,
        },
        "increase": {
            "J_max_leaf": +1, "alpha": +1, "CO2_air_stomata": +1,
            "net_ass_rate": -1, "Tcan_CO2_comp_point": -1,
            "Jpot_activation": 0, "Jpot_deactivation": 0,
            "Jpot_entropy": 0, "deg_curv_elec_transport": 0,
        },
        "decrease": {
            "J_max_leaf": -1, "alpha": -1, "CO2_air_stomata": -1,
            "net_ass_rate": +1, "Tcan_CO2_comp_point": +1,
            "Jpot_activation": 0, "Jpot_deactivation": 0,
            "Jpot_entropy": 0, "deg_curv_elec_transport": 0,
        },
        "higher_sensitivity": {
            "Jpot_activation": +1, "Jpot_deactivation": -1,
            "deg_curv_elec_transport": -1, "Tcan_CO2_comp_point": +1,
            "J_max_leaf": 0, "Jpot_entropy": 0,
            "alpha": 0, "CO2_air_stomata": 0, "net_ass_rate": 0,
        },
        "lower_sensitivity": {
            "Jpot_activation": -1, "Jpot_deactivation": +1,
            "deg_curv_elec_transport": +1, "Tcan_CO2_comp_point": -1,
            "J_max_leaf": 0, "Jpot_entropy": 0,
            "alpha": 0, "CO2_air_stomata": 0, "net_ass_rate": 0,
        },
    },

    "temperature_inhibition": {
        "none": {
            "k_sw_min_Tcan": 0, "s_min_Tcan": 0,
            "k_sw_max_Tcan": 0, "s_max_Tcan": 0,
            "k_sw_min_Tcan24": 0, "s_min_Tcan24": 0,
            "k_sw_max_Tcan24": 0, "s_max_Tcan24": 0,
        },
        "shift_warm": {
            "k_sw_min_Tcan": +1, "s_min_Tcan": 0,
            "k_sw_max_Tcan": +1, "s_max_Tcan": 0,
            "k_sw_min_Tcan24": +1, "s_min_Tcan24": 0,
            "k_sw_max_Tcan24": +1, "s_max_Tcan24": 0,
        },
        "shift_cold": {
            "k_sw_min_Tcan": -1, "s_min_Tcan": 0,
            "k_sw_max_Tcan": -1, "s_max_Tcan": 0,
            "k_sw_min_Tcan24": -1, "s_min_Tcan24": 0,
            "k_sw_max_Tcan24": -1, "s_max_Tcan24": 0,
        },
        "widen_optimum": {
            "k_sw_min_Tcan": -1, "s_min_Tcan": +1,
            "k_sw_max_Tcan": +1, "s_max_Tcan": -1,
            "k_sw_min_Tcan24": -1, "s_min_Tcan24": +1,
            "k_sw_max_Tcan24": +1, "s_max_Tcan24": -1,
        },
        "narrow_optimum": {
            "k_sw_min_Tcan": +1, "s_min_Tcan": -1,
            "k_sw_max_Tcan": -1, "s_max_Tcan": +1,
            "k_sw_min_Tcan24": +1, "s_min_Tcan24": -1,
            "k_sw_max_Tcan24": -1, "s_max_Tcan24": +1,
        },
    },

    "temperature_and_development": {
        "none": {
            "bias_g_Tcan24": 0, "slope_g_Tcan24": 0,
            "TS_start": 0, "TS_end": 0,
            "c_dev1": 0, "c_dev2": 0, "r_fruit_Set": 0,
        },
        "increase": {
            "bias_g_Tcan24": +1, "slope_g_Tcan24": +1,
            "TS_end": -1, "c_dev1": +1, "c_dev2": +1,
            "r_fruit_Set": -1, "TS_start": 0,
        },
        "decrease": {
            "bias_g_Tcan24": -1, "slope_g_Tcan24": -1,
            "TS_end": +1, "c_dev1": -1, "c_dev2": -1,
            "r_fruit_Set": +1, "TS_start": 0,
        },
        "higher_sensitivity": {
            "slope_g_Tcan24": +1, "c_dev2": +1,
            "bias_g_Tcan24": 0, "TS_start": 0, "TS_end": 0,
            "c_dev1": 0, "r_fruit_Set": 0,
        },
        "lower_sensitivity": {
            "slope_g_Tcan24": -1, "c_dev2": -1,
            "bias_g_Tcan24": 0, "TS_start": 0, "TS_end": 0,
            "c_dev1": 0, "r_fruit_Set": 0,
        },
    },

    "biomass_growth_and_maintenance": {
        "none": {
            "G_max": 0,
            "c_fruit_growth": 0, "c_leaf_growth": 0, "c_stem_growth": 0,
            "c_fruit_maintenance": 0, "c_leaf_maintenance": 0, "c_stem_maintenance": 0,
            "Q_10_maintenance": 0, "rg_fruit": 0, "rg_leaf": 0,
            "rg_stem": 0, "c_rgr": 0,
        },
        "more_fruit_growth": {
            "G_max": -1, "rg_fruit": +1, "rg_leaf": -1, "rg_stem": -1,
            "c_fruit_growth": 0, "c_leaf_growth": 0, "c_stem_growth": 0,
            "c_fruit_maintenance": 0, "c_leaf_maintenance": 0, "c_stem_maintenance": 0,
            "Q_10_maintenance": 0, "c_rgr": 0,
        },
        "more_veg_growth": {
            "G_max": +1, "rg_fruit": -1, "rg_leaf": +1, "rg_stem": +1,
            "c_fruit_growth": 0, "c_leaf_growth": 0, "c_stem_growth": 0,
            "c_fruit_maintenance": 0, "c_leaf_maintenance": 0, "c_stem_maintenance": 0,
            "Q_10_maintenance": 0, "c_rgr": 0,
        },
        "lower_resp_cost": {
            "G_max": 0, "rg_fruit": 0, "rg_leaf": 0, "rg_stem": 0,
            "c_fruit_growth": -1, "c_leaf_growth": -1, "c_stem_growth": -1,
            "c_fruit_maintenance": -1, "c_leaf_maintenance": -1, "c_stem_maintenance": -1,
            "Q_10_maintenance": 0, "c_rgr": 0,
        },
        "higher_resp_cost": {
            "G_max": 0, "rg_fruit": 0, "rg_leaf": 0, "rg_stem": 0,
            "c_fruit_growth": +1, "c_leaf_growth": +1, "c_stem_growth": +1,
            "c_fruit_maintenance": +1, "c_leaf_maintenance": +1, "c_stem_maintenance": +1,
            "Q_10_maintenance": 0, "c_rgr": 0,
        },
        "higher_sensitivity": {
            "Q_10_maintenance": +1, "c_rgr": +1,
            "G_max": 0, "rg_fruit": 0, "rg_leaf": 0, "rg_stem": 0,
            "c_fruit_growth": 0, "c_leaf_growth": 0, "c_stem_growth": 0,
            "c_fruit_maintenance": 0, "c_leaf_maintenance": 0, "c_stem_maintenance": 0,
        },
        "lower_sensitivity": {
            "Q_10_maintenance": -1, "c_rgr": -1,
            "G_max": 0, "rg_fruit": 0, "rg_leaf": 0, "rg_stem": 0,
            "c_fruit_growth": 0, "c_leaf_growth": 0, "c_stem_growth": 0,
            "c_fruit_maintenance": 0, "c_leaf_maintenance": 0, "c_stem_maintenance": 0,
        },
    },
}

# ─────────────────────────────────────────────────────────────────
# Parameter bounds — cherry-adapted
# ─────────────────────────────────────────────────────────────────

PARAMETER_BOUNDS = {
    # Group 1: Canopy architecture
    "LAI_max":       (2.0,     5.5),      # low-wire pruned → high-wire indeterminate
    "SLA":           (1.5e-5,  4.0e-5),   # thick sun leaf → thin shade leaf
    "n_plants":      (1.2,     2.2),      # stem density from challenge data

    # Group 2: Photosynthetic biochemistry
    "J_max_leaf":              (100.0,     350.0),     # shade → high-N sun leaf
    "Jpot_activation":         (280_000.0, 500_000.0), # TODO: verify units — Vanthoor is 37k J/mol
    "Jpot_deactivation":       (150_000.0, 300_000.0), # high-T decline; Farquhar 1980
    "Jpot_entropy":            (640.0,     780.0),     # coupled to deactivation
    "alpha":                   (0.25,      0.50),      # quantum yield
    "deg_curv_elec_transport": (0.50,      0.98),      # J-vs-PAR curvature
    "Tcan_CO2_comp_point":     (1.0,       2.8),       # CO2 comp point slope vs T
    "CO2_air_stomata":         (0.50,      0.80),      # stomatal conductance ratio
    "net_ass_rate":            (3.0,       6.0),       # electrons per CO2

    # Group 3: Temperature inhibition
    "k_sw_min_Tcan":   (5.0,   16.0),     # rising sigmoid midpoint for hTcan
    "k_sw_max_Tcan":   (28.0,  39.0),     # falling sigmoid midpoint for hTcan
    "k_sw_min_Tcan24": (10.0,  20.0),     # rising sigmoid midpoint for hTcan24
    "k_sw_max_Tcan24": (19.0,  28.0),     # falling sigmoid midpoint for hTcan24
    "s_min_Tcan":      (-1.5,  -0.4),     # slope; nominal -0.869
    "s_max_Tcan":      (0.3,   1.0),      # slope; nominal 0.579
    "s_min_Tcan24":    (-2.0,  -0.5),     # slope; nominal -1.159
    "s_max_Tcan24":    (0.5,   2.0),      # slope; nominal 1.139

    # Group 4: Temperature & development
    "bias_g_Tcan24":    (-2.0,     -0.5),  # growth-T intercept
    "slope_g_Tcan24":   (0.06,     0.20),  # growth-T slope
    "TS_start":         (-20.0,    20.0),   # thermal sum start offset
    "TS_end":           (400.0,    1200.0), # cherry develops faster
    "c_dev1":           (-1.5e-8,  -2.0e-9),# wider for cherry
    "c_dev2":           (5.0e-9,   2.5e-8), # wider for cherry
    "r_fruit_Set":      (0.02,     0.35),   # low threshold for cherry

    # Group 5: Biomass growth & maintenance
    "G_max":               (400.0,    3_000.0),  # cherry: 0.5–2g DM per fruit
    "c_fruit_growth":      (0.18,     0.40),     # growth resp; biochemistry conserved
    "c_leaf_growth":       (0.18,     0.40),
    "c_stem_growth":       (0.20,     0.44),
    "c_fruit_maintenance": (5.0e-8,   2.5e-7),   # Heuvelink 1996
    "c_leaf_maintenance":  (1.5e-7,   7.0e-7),
    "c_stem_maintenance":  (7.0e-8,   3.0e-7),
    "Q_10_maintenance":    (1.4,      3.0),
    "rg_fruit":            (0.20,     0.70),     # high for cherry (many small sinks)
    "rg_leaf":             (0.03,     0.15),     # compact cherry plants
    "rg_stem":             (0.02,     0.12),     # compact cherry plants
    "c_rgr":               (1.0e6,    5.0e6),
}

# ─────────────────────────────────────────────────────────────────
# Post-perturbation safety constraints
# ─────────────────────────────────────────────────────────────────

CONSTRAINTS = {
    "temperature_inhibition": {
        "min_gap_inst": 12.0,   # k_sw_max_Tcan - k_sw_min_Tcan >= 12°C
        "min_gap_24h":   6.0,   # k_sw_max_Tcan24 - k_sw_min_Tcan24 >= 6°C
    },
}

# ─────────────────────────────────────────────────────────────────
# Initial conditions (fixed, not explored by GFlowNet)
# ─────────────────────────────────────────────────────────────────

INITIAL_CONDITIONS = {
    "init_Cbuff": 1000.0,
    "init_Cleaf": 15000.0,
    "init_Cstem": 8000.0,
    "init_Cfruits": [0.0] * 50,
    "init_Nfruits": [0.0] * 50,
    "init_TScan": 0.0,
}

# ─────────────────────────────────────────────────────────────────
# Baseline parameters (Vanthoor defaults + fixed params)
# Parameters not in PARAMETER_BOUNDS are passed through unchanged.
# ─────────────────────────────────────────────────────────────────

BASELINE_PARAMETERS = {
    # Explored parameters (defaults from Vanthoor)
    "LAI_max": 3.0,
    "SLA": 3e-05,
    "n_plants": 2.5,
    "J_max_leaf": 210.0,
    "Jpot_activation": 370_000.0,      # TODO: verify units vs Vanthoor's 37k J/mol
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

    # Fixed parameters (not explored, passed through to FMU)
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

# ─────────────────────────────────────────────────────────────────
# FMU output types
# ─────────────────────────────────────────────────────────────────

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