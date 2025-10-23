"""
Configuration file for the Chemical Experiment Optimization application.

This file defines:
- Variable specifications (names, types, bounds, levels, descriptions).
- Column ordering for UI and model consistency.
- Paths and constants for data handling and Bayesian Optimization.
- Helper structures for categorical variable processing.
- Reference point for hypervolume calculations in BoTorch.
"""
import os
import torch

# --- Define Domain Variables & Columns ---
variables_spec = [
    {'name': 'base', 'type': 'categorical', 'levels': ['KOH', 'KOtBu'], 'description': 'Organic Base'},
    {'name': 'base_eq', 'type': 'continuous', 'bounds': [0.0, 2.0], 'description': 'Base Eq'},
    {'name': 'lag', 'type': 'categorical', 'levels': ['DMSO', 'CH3CN'], 'description': 'Solvent'},
    {'name': 'eta', 'type': 'continuous', 'bounds': [0.0, 1.0], 'description': 'Eta'},
    {'name': 'time', 'type': 'continuous', 'bounds': [10.0, 300.0], 'description': 'Time (sec)'},
    {'name': 'freq', 'type': 'continuous', 'bounds': [3.0, 30.0], 'description': 'Frequency (Hz)'},
    
    # --- Outcomes (user-provided results of an experiment) ---
    {'name': 'yield_', 'type': 'outcome', 'bounds': [0.0, 1.0], 'description': 'Yield'},
    {'name': 'pmi', 'type': 'outcome', 'bounds': [2.18, 150.0], 'description': 'PMI Score'},

    # --- Objective (calculated, for BO) ---
    {'name': 'euclidean_dist', 'type': 'objective', 'bounds': [0.0, 180.0], 'maximize': False, 'description': 'Euclidean Distance (from Yield=100, PMI=2.18)'}
]

# Ordered list of input column names for UI and model consistency
ORDERED_INPUT_COLS_FOR_UI = [
    spec['name'] for spec in variables_spec if spec['type'] in ['categorical', 'continuous']
]

# Objective column names
objective_columns = [spec['name'] for spec in variables_spec if spec['type'] == 'objective']
# Columns the user provides as results
ORDERED_OBJECTIVE_COLS_FOR_UI = ['yield_', 'pmi'] 

# Path to the data file
DATA_FILE_PATH = 'data/ilidi_data.csv'

# Minimum samples required to switch from random sampling to BoTorch
MIN_SAMPLES_FOR_BO = 8 #max(10, 2 * processed_input_dim) 

# All columns expected in the data file
columns = ORDERED_INPUT_COLS_FOR_UI + [spec['name'] for spec in variables_spec if spec['type'] == 'outcome'] + objective_columns

MAX_SUGGESTIONS_DISPLAY = 20

# --- PMI Calculation Constants ---
PMI_CONFIG = {
    'reagents': {
        'chalcone': {'mass_mg': 41.7},
        'TMSOI': {'mass_mg': 44.0}
    },
    'bases': {
        'KOH': {'mw': 56.11},
        'KOtBu': {'mw': 112.21}
    },
    'solvents': {
        'CH3CN': {'density': 0.787}, 
        'DMSO': {'density': 1.1}
    },
    'product': {
        'theoretical_mass_mg': 44.45
    },
    'mmol_scale': 0.2
}

cat_var_details = {}
for var_spec in variables_spec:
    if var_spec['type'] == 'categorical':
        num_levels = len(var_spec['levels'])
        cat_var_details[var_spec['name']] = {
            'levels': var_spec['levels'],
            'num_levels': num_levels,
            'idx_to_level': {i: level for i, level in enumerate(var_spec['levels'])},
            'level_to_idx': {level: i for i, level in enumerate(var_spec['levels'])}
        }

os.makedirs('data', exist_ok=True)