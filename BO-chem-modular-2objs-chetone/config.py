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

# --- Domain Variables & Columns (English) ---
variables_spec = [
    {'name': 's_salt',  'type': 'categorical', 'levels': ['TMSI', 'TMSOI'], 'description': 'S-Salt (1 eq)'},
    {'name': 'base',    'type': 'categorical', 'levels': ['NaOH', 'KOH', 'KOtBu'], 'description': 'Base'},
    {'name': 'base_eq', 'type': 'continuous',  'bounds': [1.0, 3.0], 'description': 'Base Equivalents'},
    {'name': 'solvent', 'type': 'categorical', 'levels': ['CH₃CN', 'DMSO', 'THF'], 'description': 'LAG'},
    {'name': 'eta',     'type': 'continuous',  'bounds': [0.0, 1.0], 'description': 'η'},
    {'name': 'freq',    'type': 'continuous',  'bounds': [3.0, 30.0], 'description': 'Frequency (Hz)'},
    {'name': 'time',    'type': 'continuous',  'bounds': [5.0, 60.0], 'description': 'Time (min)'},
    {'name': 'yield_',  'type': 'objective',   'bounds': [0.0, 1.0], 'maximize': True,  'description': 'Yield'},
    {'name': 'pmi',     'type': 'objective',   'bounds': [0.0, 150.0], 'maximize': False, 'description': 'PMI Score'}
]

ORDERED_INPUT_COLS_FOR_UI = [v['name'] for v in variables_spec if v['type'] in ('categorical','continuous')]
objective_columns = [v['name'] for v in variables_spec if v['type'] == 'objective']
ORDERED_OBJECTIVE_COLS_FOR_UI = ['yield_', 'pmi']

DATA_FILE_PATH = 'data/ilidi_data.csv'
MIN_SAMPLES_FOR_BO = 10
columns = ORDERED_INPUT_COLS_FOR_UI + objective_columns
MAX_SUGGESTIONS_DISPLAY = 20

# PMI Constants (Ketone + S-Salt -> Epoxide)
PMI_CONFIG = {
    'reagents': {
        'ketone': {'mw': 182.21, 'mass_mg': 36.4},
        's_salt': {
            'TMSI':  {'mw': 204.07},
            'TMSOI': {'mw': 220.07}
        }
    },
    'bases': {
        'NaOH':  {'mw': 39.997},
        'KOH':   {'mw': 56.11},
        'KOtBu': {'mw': 112.21}
    },
    'solvents': {
        'CH₃CN': {'density': 0.786},
        'DMSO':  {'density': 1.1},
        'THF':   {'density': 0.899}
    },
    'product': {
        'theoretical_mass_mg': 39.2
    },
    'mmol_scale': 0.2
}

cat_var_details = {}
for spec in variables_spec:
    if spec['type'] == 'categorical':
        levels = spec['levels']
        cat_var_details[spec['name']] = {
            'levels': levels,
            'num_levels': len(levels),
            'idx_to_level': {i: l for i, l in enumerate(levels)},
            'level_to_idx': {l: i for i, l in enumerate(levels)}
        }

REF_POINT = torch.tensor([-0.01, -150.1], dtype=torch.double)

os.makedirs('data', exist_ok=True)