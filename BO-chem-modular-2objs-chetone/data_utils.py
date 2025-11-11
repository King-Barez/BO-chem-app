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
from config import DATA_FILE_PATH, columns, variables_spec, objective_columns, ORDERED_INPUT_COLS_FOR_UI, ORDERED_OBJECTIVE_COLS_FOR_UI, PMI_CONFIG

def calculate_pmi(s_salt, base, base_eq, solvent, eta, yield_):
    """
    Calculate Process Mass Intensity (PMI) for Ketone + S-Salt -> Epoxide.
    Assumptions:
      - Ketone: fixed scale (1 eq).
      - S-Salt: 1 eq (TMSI or TMSOI).
      - Base: 1–3 eq.
      - Solvent mass ≈ eta * density * (reagents + base).
    Returns (pmi_value, status_message).
    """
    try:
        if yield_ is None or str(yield_).strip() == "":
            return None, "Yield value required."
        yv = float(yield_)
        if yv <= 0:
            return 150.0, "Yield ≤ 0; PMI capped at 150."

        if s_salt not in PMI_CONFIG['reagents']['s_salt']:
            return None, f"S-Salt '{s_salt}' invalid. Choose {list(PMI_CONFIG['reagents']['s_salt'].keys())}."
        if base not in PMI_CONFIG['bases']:
            return None, f"Base '{base}' invalid."
        if solvent not in PMI_CONFIG['solvents']:
            return None, f"Solvent (LAG) '{solvent}' invalid."
        if base_eq is None or str(base_eq).strip() == "":
            return None, "Base equivalents required."
        beq = float(base_eq)
        if eta is None or str(eta).strip() == "":
            return None, "η required."
        eta_val = float(eta)

        mmol_scale = PMI_CONFIG['mmol_scale']
        mass_ketone = PMI_CONFIG['reagents']['ketone']['mass_mg']
        mw_salt = PMI_CONFIG['reagents']['s_salt'][s_salt]['mw']
        mass_salt = mmol_scale * mw_salt
        mw_base = PMI_CONFIG['bases'][base]['mw']
        mass_base = mmol_scale * beq * mw_base
        density = PMI_CONFIG['solvents'][solvent]['density']

        mass_reagents = mass_ketone + mass_salt + mass_base
        mass_solvent = eta_val * density * mass_reagents
        total_in = mass_reagents + mass_solvent
        mass_product = yv * PMI_CONFIG['product']['theoretical_mass_mg']
        if mass_product == 0:
            return 150.0, "Product mass zero; capped."

        pmi = total_in / mass_product
        if pmi > 150:
            return 150.0, f"PMI {round(pmi,2)} > 150; capped."
        return round(pmi, 2), "PMI calculated."
    except (ValueError, TypeError) as e:
        return None, f"Invalid numeric input: {e}"
    except KeyError as e:
        return None, f"Config key missing: {e}"
    except Exception as e:
        return None, f"Unexpected error: {e}"

def load_data():
    """Load data, migrate legacy column names, enforce schema."""
    if not os.path.exists(DATA_FILE_PATH):
        return pd.DataFrame(columns=columns)
    try:
        df = pd.read_csv(DATA_FILE_PATH)
        legacy_map = {
            'sulfonio': 's_salt',
            'lag': 'solvent',
            'chetone': 'ketone'  # if ever present
        }
        for old, new in legacy_map.items():
            if old in df.columns and new not in df.columns:
                df.rename(columns={old: new}, inplace=True)
        # Add missing columns
        changed = False
        for col in columns:
            if col not in df.columns:
                df[col] = pd.NA
                changed = True
        if changed:
            df = df[columns]
            df.to_csv(DATA_FILE_PATH, index=False)
    except Exception:
        df = pd.DataFrame(columns=columns)

    for spec in variables_spec:
        name = spec['name']
        if name in df.columns:
            if spec['type'] in ('continuous','objective'):
                df[name] = pd.to_numeric(df[name], errors='coerce')
            elif spec['type'] == 'categorical':
                df[name] = df[name].astype(str)
    return df

def save_suggestion_result(*args):
    """
    Saves the result of a suggested experiment to the CSV file.
    Validates inputs and handles special logic for 'eta' = 0.
    Args are ordered: input parameters, then objective parameters.
    """
    num_input_params = len(ORDERED_INPUT_COLS_FOR_UI)
    num_objective_params = len(ORDERED_OBJECTIVE_COLS_FOR_UI)

    if len(args) != num_input_params + num_objective_params:
        return f"Error: Expected {num_input_params + num_objective_params} arguments for saving, got {len(args)}"

    param_values = args[:num_input_params]
    objective_values_str = args[num_input_params : num_input_params + num_objective_params]

    new_row_dict = dict(zip(ORDERED_INPUT_COLS_FOR_UI, param_values))

    for i, obj_col_name in enumerate(ORDERED_OBJECTIVE_COLS_FOR_UI):
        val_str = str(objective_values_str[i]).strip() if objective_values_str[i] is not None else ""
        if not val_str:
            return f"Please enter a value for {obj_col_name}."
        try:
            val_float = float(val_str)
        except ValueError:
            return f"Invalid input for {obj_col_name}: '{val_str}' is not a valid number."
        
        var_spec_item = next(v for v in variables_spec if v['name'] == obj_col_name)
        bounds = var_spec_item['bounds']
        if not (bounds[0] <= val_float <= bounds[1]):
            return f"{var_spec_item['description']} must be between {bounds[0]} and {bounds[1]} (received {val_float})."
        new_row_dict[obj_col_name] = val_float

    for i, col_name in enumerate(ORDERED_INPUT_COLS_FOR_UI):
        val = param_values[i]
        var_spec_item = next(v for v in variables_spec if v['name'] == col_name)
        if var_spec_item['type'] == 'continuous':
            try:
                if val is None or str(val).strip() == "":
                     return f"Please ensure a value is provided or suggested for {var_spec_item['description']}."
                val_float = float(val)
                bounds = var_spec_item['bounds']
                if not (bounds[0] <= val_float <= bounds[1]):
                    return f"{var_spec_item['description']} must be between {bounds[0]} and {bounds[1]} (received {val_float})."
                new_row_dict[col_name] = val_float
            except (ValueError, TypeError):
                return f"Invalid input for {var_spec_item['description']}: '{val}' is not a valid number."
        elif var_spec_item['type'] == 'categorical':
            if val is None or str(val).strip() == "":
                return f"Please select a value for {var_spec_item['description']}."
            if val not in var_spec_item['levels']:
                return f"Invalid value for {var_spec_item['description']}: '{val}'. Must be one of {var_spec_item['levels']}."
            new_row_dict[col_name] = val

    df = load_data()
    
    eta_val_for_logic = new_row_dict.get('eta')
    try:
        eta_val_for_logic = float(eta_val_for_logic) if eta_val_for_logic is not None else None
    except ValueError: 
        eta_val_for_logic = None


    # eta==0 replicate across all solvents
    if eta_val_for_logic == 0.0:
        all_solvents = next(v['levels'] for v in variables_spec if v['name'] == 'solvent')
        base_row = new_row_dict.copy()
        rows = []
        for solv in all_solvents:
            r = base_row.copy()
            r['solvent'] = solv
            rows.append(r)
        new_rows_df = pd.DataFrame(rows)
    else:
        new_rows_df = pd.DataFrame([new_row_dict])

    updated_df = pd.concat([df, new_rows_df], ignore_index=True)
    
    for col_config in columns: 
        if col_config not in updated_df.columns:
            updated_df[col_config] = pd.NA
    
    updated_df = updated_df[columns] 

    updated_df.to_csv(DATA_FILE_PATH, index=False)
    return "Suggestion result(s) saved successfully!"

def upload_csv(file):
    """Handles uploading a new CSV dataset, replacing the existing one."""
    if file is None:
        return "No file uploaded."
    try:
        df_uploaded = pd.read_csv(file.name)
        
        for col_name_config in columns:
            if col_name_config not in df_uploaded.columns:
                if col_name_config == 'pmi': 
                    df_uploaded[col_name_config] = np.random.uniform(10, 80, size=len(df_uploaded)).round(2)
                else:
                    df_uploaded[col_name_config] = pd.NA

        df_uploaded = df_uploaded[columns]
        for var_spec_item in variables_spec:
            if var_spec_item['name'] in df_uploaded.columns:
                if var_spec_item['type'] == 'continuous' or var_spec_item['type'] == 'objective':
                    df_uploaded[var_spec_item['name']] = pd.to_numeric(df_uploaded[var_spec_item['name']], errors='coerce')
                elif var_spec_item['type'] == 'categorical':
                    df_uploaded[var_spec_item['name']] = df_uploaded[var_spec_item['name']].astype(str)


        df_uploaded.to_csv(DATA_FILE_PATH, index=False)
        return "Dataset uploaded successfully. All views will refresh."
    except Exception as e:
        return f"Error uploading file: {e}"

def download_csv():
    """Prepares the current dataset for download by returning its file path."""
    df = load_data()
    for col_name_config in columns:
        if col_name_config not in df.columns:
            df[col_name_config] = pd.NA 
    
    df = df[columns] 
    df.to_csv(DATA_FILE_PATH, index=False) 
    return DATA_FILE_PATH