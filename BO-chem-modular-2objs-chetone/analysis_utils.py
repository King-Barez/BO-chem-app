"""
Utility functions for generating statistical reports and analyses from the experiment data.
"""
import pandas as pd
from data_utils import load_data
from config import variables_spec, objective_columns, ORDERED_INPUT_COLS_FOR_UI

def get_statistics():
    """Generates descriptive statistics for the dataset for display in Markdown."""
    df = load_data()
    if df.empty:
        return "No data available to generate statistics."

    report = "### Dataset Overview\n"
    report += f"- **Total Experiments:** {len(df)}\n"
    
    input_descs = [v['description'] for v in variables_spec if v['name'] in ORDERED_INPUT_COLS_FOR_UI and v['name'] in df.columns]
    obj_descs = [v['description'] for v in variables_spec if v['name'] in objective_columns and v['name'] in df.columns]

    report += f"- **Input Variables:** {', '.join(input_descs) if input_descs else 'None defined or present in data'}\n"
    report += f"- **Objective Variables:** {', '.join(obj_descs) if obj_descs else 'None defined or present in data'}\n\n"

    report += "### Objective Statistics\n"
    for obj_col_name in objective_columns:
        if obj_col_name in df.columns and pd.api.types.is_numeric_dtype(df[obj_col_name].dropna()):
            obj_spec_item = next((v for v in variables_spec if v['name'] == obj_col_name), None)
            obj_desc_name = obj_spec_item['description'] if obj_spec_item else obj_col_name
            
            desc_stats = df[obj_col_name].dropna().describe()
            if desc_stats.empty:
                 report += f"#### {obj_desc_name} ({obj_col_name}):\n  - No valid numeric data.\n\n"
                 continue

            report += f"#### {obj_desc_name} ({obj_col_name}):\n"
            report += f"  - **Count:** {desc_stats.get('count', 0):.0f}\n"
            if obj_spec_item and not obj_spec_item['maximize']:
                report += f"  - **Best (Min):** {desc_stats.get('min', 'N/A'):.3f}\n"
                report += f"  - **Worst (Max):** {desc_stats.get('max', 'N/A'):.3f}\n"
            else:
                report += f"  - **Best (Max):** {desc_stats.get('max', 'N/A'):.3f}\n"
                report += f"  - **Worst (Min):** {desc_stats.get('min', 'N/A'):.3f}\n"
            report += f"  - **Average:** {desc_stats.get('mean', 'N/A'):.3f}\n"
            report += f"  - **Median:** {desc_stats.get('50%', 'N/A'):.3f}\n"
            report += f"  - **Std Dev:** {desc_stats.get('std', 'N/A'):.3f}\n\n"
        else:
            obj_spec_item = next((v for v in variables_spec if v['name'] == obj_col_name), None)
            obj_desc_name = obj_spec_item['description'] if obj_spec_item else obj_col_name
            report += f"#### {obj_desc_name} ({obj_col_name}):\n  - Data not available or not numeric.\n\n"


    report += "### Input Parameter Ranges Observed:\n"
    for var_spec_item_input in variables_spec:
        col_name_input = var_spec_item_input['name']
        if col_name_input in ORDERED_INPUT_COLS_FOR_UI and col_name_input in df.columns:
            report += f"- **{var_spec_item_input['description']} ({col_name_input}):** "
            if df[col_name_input].notna().any():
                if var_spec_item_input['type'] == 'continuous' and pd.api.types.is_numeric_dtype(df[col_name_input].dropna()):
                    desc_input_stats = df[col_name_input].dropna().describe()
                    report += f"Min: {desc_input_stats.get('min', 'N/A'):.2f}, Max: {desc_input_stats.get('max', 'N/A'):.2f}, Mean: {desc_input_stats.get('mean', 'N/A'):.2f}\n"
                elif var_spec_item_input['type'] == 'categorical':
                    unique_levels_input = df[col_name_input].dropna().unique()
                    report += f"Observed Levels: {', '.join(map(str,unique_levels_input)) if len(unique_levels_input) > 0 else 'None'}\n"
                else:
                    report += "N/A (Not numeric or no valid data)\n"
            else:
                report += "No data observed.\n"
    return report

def export_report():
    """Exports a summary report of the analysis to a text file."""
    df = load_data()
    report_filename = "analysis_report.txt"

    if df.empty:
        report_content = "No data available for reporting."
        with open(report_filename, "w") as f:
            f.write(report_content)
        return report_filename

    report_content = "========================================\n"
    report_content += "   Chemical Experiment Analysis Report   \n"
    report_content += "========================================\n\n"

    report_content += "Generated on: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + "\n\n"

    report_content += "--- Dataset Statistics ---\n"
    stats_text = get_statistics()
    stats_text = stats_text.replace("### ", "## ").replace("#### ", "### ")
    stats_text = stats_text.replace("- **", "  - ").replace("**", "")
    report_content += stats_text
    report_content += "\n\n"

    report_content += "--- Raw Data ---\n"
    try:
        report_content += df.to_string(index=False)
    except Exception as e:
        report_content += f"Could not convert raw data to string: {e}"
    report_content += "\n\n"

    report_content += "--- Notes ---\n"
    obj_descs_report = []
    for obj_name in objective_columns:
        spec = next((v for v in variables_spec if v['name'] == obj_name), None)
        if spec:
            direction = "(Minimize)" if not spec['maximize'] else "(Maximize)"
            obj_descs_report.append(f"{spec['description']} {direction}")
        elif obj_name in df.columns:
             obj_descs_report.append(f"{obj_name} (Direction unknown)")


    if len(obj_descs_report) >= 2:
        report_content += "- This is a multi-objective optimization problem.\n"
        report_content += f"- Objectives being optimized: {', '.join(obj_descs_report)}.\n"
        report_content += "- Pareto front analysis should be consulted for trade-offs between these objectives.\n"
    elif len(obj_descs_report) == 1:
        report_content += f"- This is a single-objective optimization problem for {obj_descs_report[0]}.\n"
    else:
        report_content += "- No objectives seem to be defined or present in the data for optimization analysis.\n"
    
    report_content += "\nEnd of Report.\n"

    with open(report_filename, "w") as f:
        f.write(report_content)
    return report_filename