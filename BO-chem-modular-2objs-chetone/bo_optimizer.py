"""
Handles Bayesian Optimization logic using BoTorch for suggesting new experiments.

This module includes functions to:
- Generate single or batch experiment suggestions.
- Fallback to random sampling if BoTorch conditions are not met or if errors occur.
- Preprocess data for BoTorch (normalization, one-hot encoding).
- Postprocess BoTorch outputs into a user-friendly format.
"""
import torch
import pandas as pd
import numpy as np
import random
import traceback
from scipy.stats.qmc import LatinHypercube
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
from botorch.optim import optimize_acqf
from botorch.models.transforms.outcome import Standardize
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.hypervolume import Hypervolume

from config import (
    variables_spec, objective_columns, REF_POINT,
    cat_var_details, 
    MIN_SAMPLES_FOR_BO, ORDERED_INPUT_COLS_FOR_UI
)
from data_utils import load_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"Running on GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")


def _lhs_sample_suggestions(num_suggestions=1):
    """Generates a specified number of experiment suggestions using Latin Hypercube Sampling."""
    samples = []
    
    # Separate variable specs by type
    continuous_vars_spec = [
        spec for spec in variables_spec 
        if spec['name'] in ORDERED_INPUT_COLS_FOR_UI and spec['type'] == 'continuous'
    ]
    categorical_vars_spec = [
        spec for spec in variables_spec 
        if spec['name'] in ORDERED_INPUT_COLS_FOR_UI and spec['type'] == 'categorical'
    ]

    # LHS for continuous variables
    lhs_samples_cont_scaled = None
    if continuous_vars_spec:
        d_continuous = len(continuous_vars_spec)
        # Add a random seed for reproducibility within a single call, but variety across calls
        sampler_cont = LatinHypercube(d=d_continuous, seed=random.randint(0, 100000))
        lhs_samples_cont_norm = sampler_cont.random(n=num_suggestions) # Shape: (num_suggestions, d_continuous)
        
        lhs_samples_cont_scaled = np.empty_like(lhs_samples_cont_norm)
        for i, var_spec in enumerate(continuous_vars_spec):
            bounds = var_spec['bounds']
            lhs_samples_cont_scaled[:, i] = bounds[0] + lhs_samples_cont_norm[:, i] * (bounds[1] - bounds[0])
    lhs_samples_cat_chosen_levels = {}
    for cat_spec in categorical_vars_spec:
        levels = cat_spec['levels']
        num_levels = len(levels)
        sampler_cat = LatinHypercube(d=1, seed=random.randint(0, 100000))
        lhs_norm_vals_cat = sampler_cat.random(n=num_suggestions).flatten() 
        chosen_indices = np.floor(lhs_norm_vals_cat * num_levels).astype(int)
        lhs_samples_cat_chosen_levels[cat_spec['name']] = [levels[idx] for idx in chosen_indices]

    for i in range(num_suggestions):
        sample = {}
        
        if continuous_vars_spec and lhs_samples_cont_scaled is not None:
            for j, var_spec in enumerate(continuous_vars_spec):
                sample[var_spec['name']] = lhs_samples_cont_scaled[i, j]
        
        for cat_spec in categorical_vars_spec:
            sample[cat_spec['name']] = lhs_samples_cat_chosen_levels[cat_spec['name']][i]
            
        samples.append(sample)
        
    return pd.DataFrame(samples)

def get_suggestions_botorch(df, num_suggestions=1):
    """
    Generates experiment suggestions using BoTorch with multi-objective optimization.
    Falls back to Latin Hypercube Sampling (LHS) if data is insufficient or errors occur.
    """
    if df.shape[0] < MIN_SAMPLES_FOR_BO:
        print(f"BoTorch fallback: Not enough initial samples ({df.shape[0]} < {MIN_SAMPLES_FOR_BO}). Using LHS.")
        return _lhs_sample_suggestions(num_suggestions)

    try:
        df_bo = df.copy()
        for obj_col in objective_columns:
            df_bo[obj_col] = pd.to_numeric(df_bo[obj_col], errors='coerce')
        
        df_bo.dropna(subset=objective_columns, inplace=True)
        
        if df_bo.shape[0] < MIN_SAMPLES_FOR_BO:
            print(f"BoTorch fallback: Not enough valid samples after cleaning ({df_bo.shape[0]} < {MIN_SAMPLES_FOR_BO}). Using LHS.")
            return _lhs_sample_suggestions(num_suggestions)

        train_Y_list = []
        for i, obj_col_name in enumerate(objective_columns):
            obj_spec = next(spec for spec in variables_spec if spec['name'] == obj_col_name)
            obj_values = torch.tensor(df_bo[obj_col_name].values, dtype=torch.float64)
            if not obj_spec['maximize']:
                obj_values = -obj_values
            train_Y_list.append(obj_values.unsqueeze(-1))
        
        train_Y_raw = torch.cat(train_Y_list, dim=-1).to(device)
        
        transformed_X_parts = []
        tensor_col_map = [] 
        current_tensor_idx = 0
        combined_bounds_list_normalized = []

        for var_spec_item in variables_spec:
            col_name = var_spec_item['name']
            if col_name not in ORDERED_INPUT_COLS_FOR_UI:
                continue

            if var_spec_item['type'] == 'categorical':
                details = cat_var_details[col_name]
                num_levels = details['num_levels']
                cat_data = pd.Categorical(df_bo[col_name], categories=details['levels'])
                one_hot_encoded = pd.get_dummies(cat_data, columns=[col_name], prefix=col_name).reindex(
                    columns=[f"{col_name}_{level}" for level in details['levels']], fill_value=0
                )
                transformed_X_parts.append(torch.tensor(one_hot_encoded.values, dtype=torch.float64))
                tensor_col_map.append({
                    'name': col_name, 'type': 'categorical',
                    'start_idx': current_tensor_idx, 'end_idx': current_tensor_idx + num_levels,
                    'details': details, 'one_hot_levels': [f"{col_name}_{level}" for level in details['levels']]
                })
                combined_bounds_list_normalized.extend([[0.0, 1.0]] * num_levels)
                current_tensor_idx += num_levels
            elif var_spec_item['type'] == 'continuous':
                bounds = var_spec_item['bounds']
                norm_values = (df_bo[col_name].astype(float).values - bounds[0]) / (bounds[1] - bounds[0])
                norm_values = np.clip(norm_values, 0.0, 1.0)
                transformed_X_parts.append(torch.tensor(norm_values, dtype=torch.float64).unsqueeze(-1))
                tensor_col_map.append({
                    'name': col_name, 'type': 'continuous',
                    'start_idx': current_tensor_idx, 'end_idx': current_tensor_idx + 1,
                    'original_bounds': bounds 
                })
                combined_bounds_list_normalized.extend([[0.0, 1.0]])
                current_tensor_idx += 1
        
        train_X_normalized = torch.cat(transformed_X_parts, dim=1).to(device)

        botorch_domain_bounds_normalized = torch.tensor(combined_bounds_list_normalized, dtype=torch.float64).transpose(0, 1).to(device)
        if botorch_domain_bounds_normalized.numel() == 0:
             raise ValueError("BoTorch domain_bounds are empty.")

        models = []
        for i in range(train_Y_raw.shape[-1]):
            outcome_transform_single = Standardize(m=1) 
            model_single = SingleTaskGP(
                train_X_normalized,
                train_Y_raw[:, i].unsqueeze(-1), 
                outcome_transform=outcome_transform_single
            ).to(device)
            mll_single = ExactMarginalLogLikelihood(model_single.likelihood, model_single)
            fit_gpytorch_mll(mll_single)
            models.append(model_single)
        model = ModelListGP(*models)
        
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]), seed=12345) 

        acqf = qLogNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=REF_POINT.to(device), 
            X_baseline=train_X_normalized, 
            sampler=sampler,
        )
        
        candidates_tensor_normalized, _ = optimize_acqf(
            acq_function=acqf,
            bounds=botorch_domain_bounds_normalized,
            q=num_suggestions,
            num_restarts=50, 
            raw_samples=512, 
            options={"batch_limit": 5, "maxiter": 100}
        )
        
        suggestions_list = []
        candidates_tensor_normalized_cpu = candidates_tensor_normalized.cpu()
        for i in range(candidates_tensor_normalized_cpu.shape[0]):
            candidate_row_norm = candidates_tensor_normalized_cpu[i]
            sugg = {}
            for mapping_info in tensor_col_map:
                col_name = mapping_info['name']
                start_idx, end_idx = mapping_info['start_idx'], mapping_info['end_idx']
                segment_norm = candidate_row_norm[start_idx:end_idx]

                if mapping_info['type'] == 'categorical':
                    details = mapping_info['details']
                    selected_idx = torch.argmax(segment_norm).item()
                    sugg[col_name] = details['idx_to_level'][selected_idx]
                elif mapping_info['type'] == 'continuous':
                    norm_val = segment_norm.item()
                    orig_bounds = mapping_info['original_bounds']
                    val = norm_val * (orig_bounds[1] - orig_bounds[0]) + orig_bounds[0]
                    sugg[col_name] = np.clip(val, orig_bounds[0], orig_bounds[1])
            suggestions_list.append(sugg)
        
        return pd.DataFrame(suggestions_list)

    except Exception:
        traceback.print_exc()
        print("Error in BoTorch suggestion generation. Falling back to LHS.")
        return _lhs_sample_suggestions(num_suggestions)


def _format_suggestion_for_ui(suggestion_series):
    """Formats a single suggestion Series into a list with appropriate rounding for the UI."""
    ui_outputs = []
    for col_name in ORDERED_INPUT_COLS_FOR_UI:
        val = suggestion_series.get(col_name)
        var_info = next((v for v in variables_spec if v['name'] == col_name), None)

        if var_info is None or val is None:
            ui_outputs.append(None)
            continue

        if var_info['type'] == 'categorical':
            ui_outputs.append(val)
        else:
            val_float = float(val)
            bounds = var_info['bounds']
            val_clamped = min(max(val_float, bounds[0]), bounds[1])

            if col_name == "freq":
                # interi 3–30
                ui_outputs.append(int(round(val_clamped)))
            elif col_name == "time":
                # minuti, step 5
                stepped = 5 * round(val_clamped / 5)
                stepped = int(min(max(stepped, bounds[0]), bounds[1]))
                ui_outputs.append(stepped)
            elif col_name == "eta":
                ui_outputs.append(round(val_clamped, 1))
            elif col_name == "base_eq":
                stepped = round(val_clamped * 2) / 2.0  # step 0.5
                stepped = min(max(stepped, bounds[0]), bounds[1])
                ui_outputs.append(stepped)
            else:
                ui_outputs.append(round(val_clamped, 2))
    return ui_outputs

def suggest_batch_experiments(num_exp):
    """
    Suggests multiple experiments using BoTorch and formats them as a flat list for the UI.
    Falls back to LHS if BoTorch fails or data is insufficient.
    """
    df = load_data()
    if len(df) < MIN_SAMPLES_FOR_BO:
        print(f"Suggest batch: Not enough samples ({len(df)} < {MIN_SAMPLES_FOR_BO}). Using LHS.")
        return suggest_multiple_lhs_experiments(num_experiments=num_exp) 
    
    try:
        suggestions_df = get_suggestions_botorch(df, num_suggestions=num_exp)
        all_suggestions_ui_flat_list = []
        for _, suggestion_series in suggestions_df.iterrows():
            all_suggestions_ui_flat_list.extend(_format_suggestion_for_ui(suggestion_series))
            
        return all_suggestions_ui_flat_list
        
    except Exception:
        traceback.print_exc()
        print("Suggest batch: Exception caught, falling back to LHS.")
        return suggest_multiple_lhs_experiments(num_experiments=num_exp)


def suggest_multiple_lhs_experiments(num_experiments=1):
    """
    Generates multiple LHS experiment suggestions, formatted as a flat list for the UI.
    """
    all_suggestions_flat_list = []
    lhs_suggestions_df = _lhs_sample_suggestions(num_experiments)

    for _, suggestion_series in lhs_suggestions_df.iterrows():
         all_suggestions_flat_list.extend(_format_suggestion_for_ui(suggestion_series))
    return all_suggestions_flat_list