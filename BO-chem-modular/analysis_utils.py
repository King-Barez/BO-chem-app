"""
Utility functions for generating statistical reports and analyses from the experiment data.
"""
import pandas as pd
from data_utils import load_data
from config import variables_spec, objective_columns, ORDERED_INPUT_COLS_FOR_UI

def get_statistics():
    """Generates descriptive statistics for the dataset."""
    df = load_data()
    if df.empty:
        return "No data available to generate statistics."

    report = "### Dataset Overview\n"
    report += f"- **Total Experiments:** {len(df)}\n"
    
    input_descs = [v['description'] for v in variables_spec if v['name'] in ORDERED_INPUT_COLS_FOR_UI and v['name'] in df.columns]
    obj_descs = [v['description'] for v in variables_spec if v['name'] in objective_columns and v['name'] in df.columns]

    report += f"- **Input Variables:** {', '.join(input_descs or ['None'])}\n"
    report += f"- **Objective Variables:** {', '.join(obj_descs or ['None'])}\n\n"

    report += "### Objective Statistics\n"
    for obj_col_name in objective_columns:
        if obj_col_name in df.columns and pd.api.types.is_numeric_dtype(df[obj_col_name].dropna()):
            obj_spec = next((v for v in variables_spec if v['name'] == obj_col_name), {})
            obj_desc_name = obj_spec.get('description', obj_col_name)
            
            desc_stats = df[obj_col_name].dropna().describe()
            if desc_stats.empty:
                 report += f"#### {obj_desc_name}:\n  - No valid numeric data.\n\n"
                 continue

            report += f"#### {obj_desc_name}:\n"
            report += f"  - **Count:** {desc_stats.get('count', 0):.0f}\n"
            best_val, worst_val = ('min', 'max') if not obj_spec.get('maximize', True) else ('max', 'min')
            report += f"  - **Best ({best_val.capitalize()}):** {desc_stats.get(best_val, 'N/A'):.3f}\n"
            report += f"  - **Worst ({worst_val.capitalize()}):** {desc_stats.get(worst_val, 'N/A'):.3f}\n"
            report += f"  - **Average:** {desc_stats.get('mean', 'N/A'):.3f}\n"
            report += f"  - **Median:** {desc_stats.get('50%', 'N/A'):.3f}\n\n"
        else:
            obj_spec = next((v for v in variables_spec if v['name'] == obj_col_name), {})
            obj_desc_name = obj_spec.get('description', obj_col_name)
            report += f"#### {obj_desc_name}:\n  - Data not available or not numeric.\n\n"

    report += "### Input Parameter Ranges Observed:\n"
    for var_spec in variables_spec:
        col_name = var_spec['name']
        if col_name in ORDERED_INPUT_COLS_FOR_UI and col_name in df.columns:
            report += f"- **{var_spec['description']}:** "
            if df[col_name].notna().any():
                if var_spec['type'] == 'continuous' and pd.api.types.is_numeric_dtype(df[col_name].dropna()):
                    stats = df[col_name].dropna().describe()
                    report += f"Min: {stats.get('min', 'N/A'):.2f}, Max: {stats.get('max', 'N/A'):.2f}, Mean: {stats.get('mean', 'N/A'):.2f}\n"
                elif var_spec['type'] == 'categorical':
                    levels = df[col_name].dropna().unique()
                    report += f"Observed Levels: {', '.join(map(str, levels)) or 'None'}\n"
            else:
                report += "No data observed.\n"
    return report

def export_report():
    """Exports a summary report of the analysis to a text file."""
    df = load_data()
    report_filename = "analysis_report.txt"

    if df.empty:
        report_content = "No data available for reporting."
    else:
        report_content = "========================================\n"
        report_content += "   Chemical Experiment Analysis Report   \n"
        report_content += f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report_content += "========================================\n\n"
        
        stats_text = get_statistics().replace("### ", "## ").replace("#### ", "### ").replace("- **", "  - ").replace("**", "")
        report_content += stats_text
        
        report_content += "\n\n--- Raw Data ---\n"
        report_content += df.to_string(index=False)
        report_content += "\n\nEnd of Report.\n"

    with open(report_filename, "w") as f:
        f.write(report_content)
    return report_filename