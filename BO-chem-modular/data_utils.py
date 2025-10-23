"""
Utility functions for data loading, saving, and manipulation.

This module handles operations such as:
- Loading experimental data from a CSV file.
- Saving new experiment results to the CSV.
- Uploading and downloading dataset files.
- Normalizing and denormalizing data for BoTorch.
- Rounding values according to configuration.
"""
import pandas as pd
import os
import numpy as np
from config import DATA_FILE_PATH, columns, variables_spec, ORDERED_INPUT_COLS_FOR_UI, ORDERED_OBJECTIVE_COLS_FOR_UI, PMI_CONFIG

def calculate_pmi(base, base_eq, lag, eta, yield_):
    """Calculates the Process Mass Intensity (PMI)."""
    try:
        if yield_ is None or str(yield_).strip() == "":
            return None, "Yield value is required."
        yield_val = float(yield_)
        
        # If yield is 0 or less, cap PMI at 150 and return.
        if yield_val <= 0:
            return 150.0, "Yield is 0 or less; PMI capped at 150."

        if base is None or str(base).strip() == "" or base not in PMI_CONFIG['bases']:
            return None, f"Base '{base}' not found in PMI config."
        
        if lag is None or str(lag).strip() == "" or lag not in PMI_CONFIG['solvents']:
            return None, f"Solvent '{lag}' not found in PMI config."

        if base_eq is None or str(base_eq).strip() == "":
            return None, "Base Eq is required."
        base_eq_val = float(base_eq)

        if eta is None or str(eta).strip() == "":
            return None, "Eta is required."
        eta_val = float(eta)

        mass_chalcone = PMI_CONFIG['reagents']['chalcone']['mass_mg']
        mass_tmsoi = PMI_CONFIG['reagents']['TMSOI']['mass_mg']
        mw_base = PMI_CONFIG['bases'][base]['mw']
        mmol_scale = PMI_CONFIG['mmol_scale']
        density_lag = PMI_CONFIG['solvents'][lag]['density']
        
        mass_base = mmol_scale * base_eq_val * mw_base
        
        mass_reagents_and_base = mass_chalcone + mass_tmsoi + mass_base
        mass_solvent = eta_val * density_lag * mass_reagents_and_base
        
        total_mass_in = mass_reagents_and_base + mass_solvent
        mass_product = yield_val * PMI_CONFIG['product']['theoretical_mass_mg']
        
        # This check is technically redundant now with the yield_val <= 0 check, but kept for safety.
        if mass_product == 0:
             return 150.0, "Product mass is zero; PMI capped at 150."

        pmi = total_mass_in / mass_product
        
        # Cap PMI at 150 if it exceeds this value
        if pmi > 150:
            return 150.0, f"Calculated PMI ({round(pmi, 2)}) exceeds 150; capped."
        
        return round(pmi, 2), "PMI calculated successfully."

    except (ValueError, TypeError) as e:
        return None, f"Invalid input for calculation: {e}"
    except KeyError as e:
        return None, f"Configuration error: missing key {e}"
    except Exception as e:
        return None, f"An unexpected error occurred: {e}"

def load_data():
    """Loads data from CSV, ensuring column consistency, types, and recalculating objective."""
    if not os.path.exists(DATA_FILE_PATH):
        return pd.DataFrame(columns=columns)
    try:
        df = pd.read_csv(DATA_FILE_PATH)
        
        for col_name_config in columns:
            if col_name_config not in df.columns:
                df[col_name_config] = pd.NA
        
        for var_spec_item in variables_spec:
            if var_spec_item['name'] in df.columns:
                if var_spec_item['type'] in ['continuous', 'outcome', 'objective']:
                    df[var_spec_item['name']] = pd.to_numeric(df[var_spec_item['name']], errors='coerce')
                elif var_spec_item['type'] == 'categorical':
                    df[var_spec_item['name']] = df[var_spec_item['name']].astype(str)

        if 'euclidean_dist' in df.columns and 'yield_' in df.columns and 'pmi' in df.columns:
            yield_vals = pd.to_numeric(df['yield_'], errors='coerce')
            pmi_vals = pd.to_numeric(df['pmi'], errors='coerce')
            
            mask_calculable = yield_vals.notnull() & pmi_vals.notnull()

            if mask_calculable.any():
                new_distances = np.sqrt(((yield_vals[mask_calculable] * 100) - 100.0)**2 + (pmi_vals[mask_calculable] - 2.18)**2)
                
                if not df.loc[mask_calculable, 'euclidean_dist'].equals(new_distances):
                    df.loc[mask_calculable, 'euclidean_dist'] = new_distances
                    df[columns].to_csv(DATA_FILE_PATH, index=False)

    except (FileNotFoundError, pd.errors.EmptyDataError):
        df = pd.DataFrame(columns=columns)
    
    return df.reindex(columns=columns)

def save_suggestion_result(*args):
    """Saves the result of a suggested experiment to the CSV file."""
    num_input_params = len(ORDERED_INPUT_COLS_FOR_UI)
    num_objective_params = len(ORDERED_OBJECTIVE_COLS_FOR_UI)

    if len(args) != num_input_params + num_objective_params:
        return f"Error: Expected {num_input_params + num_objective_params} arguments, got {len(args)}"

    param_values = args[:num_input_params]
    objective_values_str = args[num_input_params:]

    new_row_dict = dict(zip(ORDERED_INPUT_COLS_FOR_UI, param_values))

    yield_val_str = str(objective_values_str[0]).strip() if objective_values_str[0] is not None else ""
    pmi_val_str = str(objective_values_str[1]).strip() if objective_values_str[1] is not None else ""

    if not yield_val_str or not pmi_val_str:
        return "Please enter values for both Yield and PMI."

    try:
        yield_val = float(yield_val_str)
        pmi_val = float(pmi_val_str)
    except ValueError:
        return "Invalid input for Yield or PMI: not a valid number."

    new_row_dict['yield_'] = yield_val
    new_row_dict['pmi'] = pmi_val
    new_row_dict['euclidean_dist'] = np.sqrt(((yield_val * 100) - 100.0)**2 + (pmi_val - 2.18)**2)

    for i, col_name in enumerate(ORDERED_INPUT_COLS_FOR_UI):
        val = param_values[i]
        var_spec_item = next(v for v in variables_spec if v['name'] == col_name)
        if val is None or str(val).strip() == "":
            return f"Please provide a value for {var_spec_item['description']}."
        
        if var_spec_item['type'] == 'continuous':
            try:
                val_float = float(val)
                bounds = var_spec_item['bounds']
                if not (bounds[0] <= val_float <= bounds[1]):
                    return f"{var_spec_item['description']} must be between {bounds[0]} and {bounds[1]}."
                new_row_dict[col_name] = val_float
            except (ValueError, TypeError):
                return f"Invalid input for {var_spec_item['description']}: '{val}' is not a number."
        elif var_spec_item['type'] == 'categorical':
            if val not in var_spec_item['levels']:
                return f"Invalid value for {var_spec_item['description']}: '{val}'."
            new_row_dict[col_name] = val

    df = load_data()
    
    eta_val_for_logic = new_row_dict.get('eta')
    
    if float(eta_val_for_logic) == 0.0: 
        all_lags = next(v['levels'] for v in variables_spec if v['name'] == 'lag')
        new_rows_list = [new_row_dict.copy() | {'lag': lag} for lag in all_lags]
        new_rows_df = pd.DataFrame(new_rows_list)
    else:
        new_rows_df = pd.DataFrame([new_row_dict])

    updated_df = pd.concat([df, new_rows_df], ignore_index=True)
    updated_df[columns].to_csv(DATA_FILE_PATH, index=False)
    return "Suggestion result(s) saved successfully!"

def upload_csv(file):
    """Handles uploading a new CSV dataset, replacing the existing one."""
    if file is None:
        return "No file uploaded."
    try:
        df_uploaded = pd.read_csv(file.name)
        
        for col_name_config in columns:
            if col_name_config not in df_uploaded.columns:
                df_uploaded[col_name_config] = pd.NA

        df_uploaded[columns].to_csv(DATA_FILE_PATH, index=False)
        load_data() # Trigger a reload to ensure types and objectives are calculated
        return "Dataset uploaded successfully. All views will refresh."
    except Exception as e:
        return f"Error uploading file: {e}"

def download_csv():
    """Prepares the current dataset for download by returning its file path."""
    load_data() # Ensure data is up-to-date before download
    return DATA_FILE_PATH