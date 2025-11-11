"""
Handles Bayesian Optimization logic using BoTorch for suggesting new experiments.
"""
import torch
import pandas as pd
import numpy as np
import random
import traceback
from scipy.stats.qmc import LatinHypercube
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import qLogNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.models.transforms.outcome import Standardize
from botorch.sampling.normal import SobolQMCNormalSampler

from config import (
    variables_spec,
    cat_var_details, 
    MIN_SAMPLES_FOR_BO, ORDERED_INPUT_COLS_FOR_UI
)
from data_utils import load_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"Running on GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")


def _lhs_sample_suggestions(num_suggestions=1):
    """Generates suggestions using Latin Hypercube Sampling."""
    samples = []
    
    continuous_vars_spec = [s for s in variables_spec if s['name'] in ORDERED_INPUT_COLS_FOR_UI and s['type'] == 'continuous']
    categorical_vars_spec = [s for s in variables_spec if s['name'] in ORDERED_INPUT_COLS_FOR_UI and s['type'] == 'categorical']

    lhs_samples_cont_scaled = None
    if continuous_vars_spec:
        d_continuous = len(continuous_vars_spec)
        sampler_cont = LatinHypercube(d=d_continuous, seed=random.randint(0, 100000))
        lhs_samples_cont_norm = sampler_cont.random(n=num_suggestions)
        
        lhs_samples_cont_scaled = np.empty_like(lhs_samples_cont_norm)
        for i, var_spec in enumerate(continuous_vars_spec):
            bounds = var_spec['bounds']
            lhs_samples_cont_scaled[:, i] = bounds[0] + lhs_samples_cont_norm[:, i] * (bounds[1] - bounds[0])

    lhs_samples_cat_chosen_levels = {}
    for cat_spec in categorical_vars_spec:
        levels = cat_spec['levels']
        sampler_cat = LatinHypercube(d=1, seed=random.randint(0, 100000))
        lhs_norm_vals_cat = sampler_cat.random(n=num_suggestions).flatten() 
        chosen_indices = np.floor(lhs_norm_vals_cat * len(levels)).astype(int)
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
    """Generates experiment suggestions using BoTorch, falling back to LHS if needed."""
    if df.shape[0] < MIN_SAMPLES_FOR_BO:
        print(f"BoTorch fallback (not enough samples): {df.shape[0]} < {MIN_SAMPLES_FOR_BO}. Using LHS.")
        return _lhs_sample_suggestions(num_suggestions)

    try:
        df_bo = df.copy()
        objective_columns = [spec['name'] for spec in variables_spec if spec['type'] == 'objective']
        for obj_col in objective_columns + ['yield_', 'pmi']:
            df_bo[obj_col] = pd.to_numeric(df_bo[obj_col], errors='coerce')
        
        df_bo.dropna(subset=objective_columns + ['yield_', 'pmi'], inplace=True)
        
        if df_bo.shape[0] < MIN_SAMPLES_FOR_BO:
            print(f"BoTorch fallback (not enough valid samples): {df_bo.shape[0]} < {MIN_SAMPLES_FOR_BO}. Using LHS.")
            return _lhs_sample_suggestions(num_suggestions)

        # Use a NORMALIZED Euclidean distance for the optimizer to ensure balanced optimization,
        # even though the stored distance is de-normalized.
        yield_spec = next(v for v in variables_spec if v['name'] == 'yield_')
        pmi_spec = next(v for v in variables_spec if v['name'] == 'pmi')
        y_min, y_max = yield_spec['bounds']
        p_min, p_max = pmi_spec['bounds']

        yield_norm = (df_bo['yield_'] - y_min) / (y_max - y_min)
        pmi_norm = (df_bo['pmi'] - p_min) / (p_max - p_min)
        
        target_yield_norm = 1.0
        target_pmi_norm = 0.0
        
        normalized_distance = np.sqrt((yield_norm - target_yield_norm)**2 + (pmi_norm - target_pmi_norm)**2)
        
        train_Y = torch.tensor(normalized_distance.values, dtype=torch.float64).unsqueeze(-1)
        train_Y = -train_Y # We want to minimize the distance, BoTorch maximizes
        train_Y = train_Y.to(device)
        
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
                tensor_col_map.append({'name': col_name, 'type': 'categorical', 'start_idx': current_tensor_idx, 'end_idx': current_tensor_idx + num_levels, 'details': details})
                combined_bounds_list_normalized.extend([[0.0, 1.0]] * num_levels)
                current_tensor_idx += num_levels
            elif var_spec_item['type'] == 'continuous':
                bounds = var_spec_item['bounds']
                norm_values = (df_bo[col_name].astype(float).values - bounds[0]) / (bounds[1] - bounds[0])
                transformed_X_parts.append(torch.tensor(np.clip(norm_values, 0.0, 1.0), dtype=torch.float64).unsqueeze(-1))
                tensor_col_map.append({'name': col_name, 'type': 'continuous', 'start_idx': current_tensor_idx, 'end_idx': current_tensor_idx + 1, 'original_bounds': bounds})
                combined_bounds_list_normalized.extend([[0.0, 1.0]])
                current_tensor_idx += 1
        
        train_X_normalized = torch.cat(transformed_X_parts, dim=1).to(device)
        botorch_domain_bounds_normalized = torch.tensor(combined_bounds_list_normalized, dtype=torch.float64).T.to(device)

        if botorch_domain_bounds_normalized.numel() == 0:
             raise ValueError("BoTorch domain_bounds are empty.")

        model = SingleTaskGP(train_X_normalized, train_Y, outcome_transform=Standardize(m=train_Y.shape[-1])).to(device)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]), seed=12345) 
        acqf = qLogNoisyExpectedImprovement(model=model, X_baseline=train_X_normalized, sampler=sampler, prune_baseline=True)
        
        candidates_tensor_normalized, _ = optimize_acqf(
            acq_function=acqf,
            bounds=botorch_domain_bounds_normalized,
            q=num_suggestions,
            num_restarts=50, 
            raw_samples=512, 
            options={"batch_limit": 5, "maxiter": 100}
        )
        
        suggestions_list = []
        for i in range(candidates_tensor_normalized.shape[0]):
            candidate_row_norm = candidates_tensor_normalized.cpu()[i]
            sugg = {}
            for mapping_info in tensor_col_map:
                col_name, start_idx, end_idx = mapping_info['name'], mapping_info['start_idx'], mapping_info['end_idx']
                segment_norm = candidate_row_norm[start_idx:end_idx]

                if mapping_info['type'] == 'categorical':
                    sugg[col_name] = mapping_info['details']['idx_to_level'][torch.argmax(segment_norm).item()]
                else: # continuous
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
            val_clamped = np.clip(val_float, var_info['bounds'][0], var_info['bounds'][1])
            
            if col_name in ['eta', 'base_eq']:
                rounded_val = round(val_clamped, 2)
            elif col_name == 'freq':
                rounded_val = round(val_clamped * 2) / 2 # Round to nearest 0.5
            else: # time
                rounded_val = round(val_clamped)
            
            ui_outputs.append(rounded_val)
    return ui_outputs

def suggest_batch_experiments(num_exp):
    """Suggests a batch of experiments, falling back to LHS if needed."""
    df = load_data()
    if len(df) < MIN_SAMPLES_FOR_BO:
        print(f"Suggest batch: Not enough samples ({len(df)}). Using LHS.")
        suggestions_df = _lhs_sample_suggestions(num_exp)
    else:
        try:
            suggestions_df = get_suggestions_botorch(df, num_suggestions=num_exp)
        except Exception:
            traceback.print_exc()
            print("Suggest batch: Exception caught, falling back to LHS.")
            suggestions_df = _lhs_sample_suggestions(num_exp)

    all_suggestions_ui_flat_list = []
    for _, suggestion_series in suggestions_df.iterrows():
        all_suggestions_ui_flat_list.extend(_format_suggestion_for_ui(suggestion_series))
    return all_suggestions_ui_flat_list