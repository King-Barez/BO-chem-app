"""
Plotting utilities for BO-chem data analysis.

Generates plots including:
- Objective progression (2 objectives, with trend).
- Correlation heatmaps.
- Feature importance (2 objectives).
- Feature distributions.
- Pareto front (2 objectives).
- Categorical analysis (2 objectives, with counts).
- Scatter analysis (2 objectives vs. most correlated feature).
- Suggestion frequency.

All functions return Gradio-compatible images.
"""
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import io
from PIL import Image
from sklearn.preprocessing import MinMaxScaler

from config import variables_spec, ORDERED_INPUT_COLS_FOR_UI, objective_columns 
from data_utils import load_data

def fig_to_gradio_image(fig):
    """Converts a Matplotlib figure to a Gradio-compatible image."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight') 
    plt.close(fig) 
    buf.seek(0)
    return Image.open(buf)

def plot_yield_progression():
    """Plots objective progression, cumulative best, and trend for two objectives."""
    df = load_data()
    if df.empty or not objective_columns or len(objective_columns) != 2:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Data requires exactly two objectives for this plot.", ha='center', va='center')
        return fig_to_gradio_image(fig)

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 10), sharex=True) # Fixed for 2 objectives

    for i, obj_name in enumerate(objective_columns): # Iterate through the two objectives
        ax = axes[i]
        if obj_name not in df.columns:
            ax.text(0.5, 0.5, f"Objective '{obj_name}' not in data.", ha='center', va='center')
            continue

        obj_data = pd.to_numeric(df[obj_name], errors='coerce').dropna()
        if obj_data.empty:
            ax.text(0.5, 0.5, f"No valid numeric data for {obj_name}", ha='center', va='center')
            continue
        
        obj_spec = next((v for v in variables_spec if v['name'] == obj_name), None)
        obj_desc = obj_spec['description'] if obj_spec else obj_name
        
        ax.plot(obj_data.index, obj_data.values, marker='o', linestyle='-', label=f'{obj_desc} (Actual)')
        
        if obj_spec:
            cumulative_best = obj_data.cummax() if obj_spec['maximize'] else obj_data.cummin()
            ax.plot(cumulative_best.index, cumulative_best.values, marker='.', linestyle='--', color='red', label=f'{obj_desc} (Cumulative Best)')

        # Add trend line
        if len(obj_data) >= 2:
            x_vals = obj_data.index.to_numpy().astype(float)
            y_vals = obj_data.values.astype(float)
            try:
                coeffs = np.polyfit(x_vals, y_vals, 1) # Linear fit
                trend_poly = np.poly1d(coeffs)
                ax.plot(obj_data.index, trend_poly(x_vals), color='green', linestyle=':', label='Trend')
            except (np.linalg.LinAlgError, ValueError):
                pass # Could not fit trend line

        ax.set_title(f'{obj_desc} Progression')
        ax.set_ylabel(obj_desc)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='best')
    
    axes[-1].set_xlabel('Experiment Number')
    fig.suptitle('Objective Progression & Cumulative Best (2 Objectives)', fontsize=16, y=1.02)
    
    sns.despine()
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig_to_gradio_image(fig)

def plot_correlation_heatmap():
    """Plots a correlation heatmap for numerical features and objectives."""
    df = load_data()
    numeric_cols_for_corr = [v['name'] for v in variables_spec if (v['type'] == 'continuous' or v['type'] == 'objective') and v['name'] in df.columns]
    
    if df.empty or not numeric_cols_for_corr:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Not enough numeric data for heatmap", ha='center', va='center', transform=ax.transAxes)
        return fig_to_gradio_image(fig)

    df_numeric = df[numeric_cols_for_corr].apply(pd.to_numeric, errors='coerce').dropna()

    if df_numeric.empty or df_numeric.shape[1] < 2:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Not enough valid numeric data for heatmap (need at least 2 columns)", ha='center', va='center', transform=ax.transAxes)
        return fig_to_gradio_image(fig)
        
    corr_matrix = df_numeric.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool)) 
    
    fig, ax = plt.subplots(figsize=(max(8, len(numeric_cols_for_corr)*0.9), max(6, len(numeric_cols_for_corr)*0.7)))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax, vmin=-1, vmax=1, annot_kws={"size": 8}, mask=mask)
    ax.set_title('Feature Correlation Heatmap (Lower Triangle)', fontsize=12)
    if len(numeric_cols_for_corr) > 5:
        ax.tick_params(axis='x', rotation=45, labelsize=8) 
    else:
        ax.tick_params(axis='x', rotation=0, labelsize=8)
    ax.tick_params(axis='y', rotation=0, labelsize=8)
    plt.tight_layout()
    return fig_to_gradio_image(fig)

def plot_feature_importance():
    """Plots feature importance (correlation) for two objectives."""
    df = load_data()
    if df.empty or not ORDERED_INPUT_COLS_FOR_UI or not objective_columns or len(objective_columns) != 2:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Data requires two objectives and inputs for feature importance.", ha='center', va='center')
        return fig_to_gradio_image(fig)

    continuous_inputs_present = [v['name'] for v in variables_spec if v['type'] == 'continuous' and v['name'] in ORDERED_INPUT_COLS_FOR_UI and v['name'] in df.columns]
    if not continuous_inputs_present:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No continuous input data for importance.", ha='center', va='center')
        return fig_to_gradio_image(fig)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, max(5, len(continuous_inputs_present) * 0.5)), sharey=True) # Fixed for 2 objectives

    for i, obj_name in enumerate(objective_columns): # Iterate through the two objectives
        ax = axes[i]
        if obj_name not in df.columns:
            ax.text(0.5,0.5, f"Objective '{obj_name}' not in data.", ha='center', va='center')
            continue
        obj_spec = next((v for v in variables_spec if v['name'] == obj_name), None)
        obj_desc = obj_spec['description'] if obj_spec else obj_name

        df_for_corr = df[continuous_inputs_present + [obj_name]].apply(pd.to_numeric, errors='coerce').dropna()
        if df_for_corr.empty or df_for_corr.shape[0] < 2:
            ax.text(0.5, 0.5, f"Not enough valid data for importance vs {obj_desc}", ha='center', va='center')
            continue
        
        correlations = df_for_corr.corr()[obj_name].abs().drop(obj_name, errors='ignore').sort_values(ascending=True) 
        if correlations.empty:
            ax.text(0.5, 0.5, f"Could not calculate correlations for {obj_desc}", ha='center', va='center')
            continue
        
        correlations.plot(kind='barh', ax=ax, color='skyblue')
        ax.set_title(f'Importance vs {obj_desc}')
        ax.set_xlabel(f'Absolute Correlation')
    
    if len(axes) > 0 and axes[0].get_ylabel() != '': 
         axes[0].set_ylabel("Input Features")

    fig.suptitle('Feature Importance (Correlation with Objectives)', fontsize=16, y=1.02)
    sns.despine()
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig_to_gradio_image(fig)

def plot_feature_distributions():
    """Plots normalized distributions of continuous input features."""
    df = load_data()
    plot_cols_desc = {v['name']: v['description'] for v in variables_spec}
    continuous_input_cols = [v['name'] for v in variables_spec if v['type'] == 'continuous' and v['name'] in ORDERED_INPUT_COLS_FOR_UI and v['name'] in df.columns and not df[v['name']].isnull().all()] 

    if df.empty or not continuous_input_cols:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No continuous input data for distributions", ha='center', va='center', transform=ax.transAxes)
        return fig_to_gradio_image(fig)

    df_to_plot = df[continuous_input_cols].apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')
    
    if df_to_plot.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No valid numeric data for distributions after cleaning", ha='center', va='center', transform=ax.transAxes)
        return fig_to_gradio_image(fig)

    scaler = MinMaxScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df_to_plot), columns=df_to_plot.columns)
    
    num_plots = df_normalized.shape[1]
    if num_plots == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No features to plot after normalization", ha='center', va='center', transform=ax.transAxes)
        return fig_to_gradio_image(fig)
        
    fig, ax = plt.subplots(figsize=(max(10, num_plots * 1.2), 6)) 
    sns.boxplot(data=df_normalized, ax=ax, palette="Set2", orient="v", width=0.5)
    ax.set_xticks(np.arange(len(df_normalized.columns))) 
    ax.set_xticklabels([plot_cols_desc.get(col, col) for col in df_normalized.columns], rotation=45, ha="right")
    ax.set_title('Normalized Continuous Feature Distributions (Box Plots)', fontsize=15)
    ax.set_ylabel('Normalized Value (0-1)')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    sns.despine()
    plt.tight_layout()
    return fig_to_gradio_image(fig)


def plot_pareto_front():
    """Plots the Pareto front for the two objectives."""
    df = load_data()

    if not objective_columns or len(objective_columns) != 2: # Check for exactly two objectives
        fig, ax = plt.subplots(figsize=(8,6))
        ax.text(0.5, 0.5, "Pareto plot requires exactly 2 objectives defined.", ha='center', va='center')
        ax.set_title('Pareto Front Analysis')
        return fig_to_gradio_image(fig)

    obj1_name = objective_columns[0] 
    obj2_name = objective_columns[1] 

    obj1_spec = next((v for v in variables_spec if v['name'] == obj1_name), None)
    obj2_spec = next((v for v in variables_spec if v['name'] == obj2_name), None)

    if not obj1_spec or not obj2_spec:
        fig, ax = plt.subplots(figsize=(8,6))
        ax.text(0.5, 0.5, "Objective specifications not found for Pareto plot.", ha='center', va='center')
        return fig_to_gradio_image(fig)

    if df.empty or obj1_name not in df.columns or obj2_name not in df.columns:
        fig, ax = plt.subplots(figsize=(8,6))
        ax.text(0.5, 0.5, "No data for Pareto front", ha='center', va='center')
        ax.set_xlabel(obj1_spec['description'])
        ax.set_ylabel(obj2_spec['description'])
        ax.set_title('Pareto Front Analysis')
        return fig_to_gradio_image(fig)

    # Store original indices before dropping NA
    df_objectives_with_indices = df[[obj1_name, obj2_name]].copy()
    df_objectives_with_indices['original_index'] = df_objectives_with_indices.index
    
    df_objectives_with_indices[obj1_name] = pd.to_numeric(df_objectives_with_indices[obj1_name], errors='coerce')
    df_objectives_with_indices[obj2_name] = pd.to_numeric(df_objectives_with_indices[obj2_name], errors='coerce')
    df_objectives_with_indices.dropna(subset=[obj1_name, obj2_name], inplace=True)


    if df_objectives_with_indices.empty or df_objectives_with_indices.shape[0] < 1:
        fig, ax = plt.subplots(figsize=(8,6))
        ax.text(0.5, 0.5, "No valid numeric data for Pareto front", ha='center', va='center')
        ax.set_xlabel(obj1_spec['description'])
        ax.set_ylabel(obj2_spec['description'])
        ax.set_title('Pareto Front Analysis')
        return fig_to_gradio_image(fig)

    points = df_objectives_with_indices[[obj1_name, obj2_name]].to_numpy()
    original_indices = df_objectives_with_indices['original_index'].to_numpy()
    
    is_pareto = np.ones(points.shape[0], dtype=bool)
    for i, p1 in enumerate(points):
        if not is_pareto[i]: 
            continue
        for j, p2 in enumerate(points):
            if i == j:
                continue
            
            p1_obj1_better = (p1[0] >= p2[0]) if obj1_spec['maximize'] else (p1[0] <= p2[0])
            p1_obj2_better = (p1[1] >= p2[1]) if obj2_spec['maximize'] else (p1[1] <= p2[1])
            p1_strictly_better_obj1 = (p1[0] > p2[0]) if obj1_spec['maximize'] else (p1[0] < p2[0])
            p1_strictly_better_obj2 = (p1[1] > p2[1]) if obj2_spec['maximize'] else (p1[1] < p2[1])

            if p1_obj1_better and p1_obj2_better and (p1_strictly_better_obj1 or p1_strictly_better_obj2):
                is_pareto[j] = False 

    pareto_points_arr = points[is_pareto]
    pareto_indices = original_indices[is_pareto]
    non_pareto_points_arr = points[~is_pareto]
    non_pareto_indices = original_indices[~is_pareto]

    fig, ax = plt.subplots(figsize=(9, 6)) # Adjusted figsize for colorbar
    cmap = plt.cm.viridis 

    scatter_kwargs_non_pareto = {'alpha': 0.6, 's': 30}
    scatter_kwargs_pareto = {'s': 60, 'edgecolors': 'black', 'zorder': 3}

    if non_pareto_points_arr.shape[0] > 0:
        sc_non_pareto = ax.scatter(non_pareto_points_arr[:, 0], non_pareto_points_arr[:, 1], 
                                   c=non_pareto_indices, cmap=cmap, 
                                   label='Dominated Points', **scatter_kwargs_non_pareto)
    if pareto_points_arr.shape[0] > 0:
        # Sort Pareto points for line plotting, but use original indices for color
        sort_indices_for_line = np.argsort(pareto_points_arr[:, 0])
        if not obj1_spec['maximize']: 
            sort_indices_for_line = sort_indices_for_line[::-1]
        
        sorted_pareto_points_for_line = pareto_points_arr[sort_indices_for_line]
        
        # Scatter plot uses unsorted pareto points with their original indices for color
        sc_pareto = ax.scatter(pareto_points_arr[:, 0], pareto_points_arr[:, 1], 
                               c=pareto_indices, cmap=cmap, 
                               label='Pareto Front', **scatter_kwargs_pareto)
        
        if sorted_pareto_points_for_line.shape[0] > 1:
            ax.plot(sorted_pareto_points_for_line[:, 0], sorted_pareto_points_for_line[:, 1], 'r--', alpha=0.6, zorder=2)

    ax.set_xlabel(obj1_spec['description'] + (" (Maximize)" if obj1_spec['maximize'] else " (Minimize)"))
    ax.set_ylabel(obj2_spec['description'] + (" (Maximize)" if obj2_spec['maximize'] else " (Minimize)"))
    ax.set_title(f'Pareto Front: {obj1_spec["description"]} vs. {obj2_spec["description"]}')
    
    # Add colorbar
    # Determine min/max for colorbar from all original indices plotted
    all_plotted_indices = np.concatenate((non_pareto_indices, pareto_indices)) if non_pareto_points_arr.shape[0] > 0 and pareto_points_arr.shape[0] > 0 else (pareto_indices if pareto_points_arr.shape[0] > 0 else (non_pareto_indices if non_pareto_points_arr.shape[0] > 0 else np.array([])))
    
    if all_plotted_indices.size > 0:
        norm = matplotlib.colors.Normalize(vmin=all_plotted_indices.min(), vmax=all_plotted_indices.max())
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([]) # You can also pass sc_pareto.get_array() or sc_non_pareto.get_array()
        cbar = fig.colorbar(sm, ax=ax, orientation='vertical', label='Experiment Number (Index)')

    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    sns.despine()
    plt.tight_layout(rect=[0, 0, 0.95, 0.96]) # Adjust rect for colorbar
    return fig_to_gradio_image(fig)

def plot_categorical_analysis():
    """Plots objective value by categories for categorical features, for two objectives, with counts."""
    df = load_data()
    plot_cols_desc = {v['name']: v['description'] for v in variables_spec}
    
    valid_categorical_inputs_for_plot = [
        v['name'] for v in variables_spec 
        if v['type'] == 'categorical' and v['name'] in ORDERED_INPUT_COLS_FOR_UI and 
           v['name'] in df.columns and not df[v['name']].isnull().all()
    ]

    if df.empty or not valid_categorical_inputs_for_plot or not objective_columns or len(objective_columns) != 2:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Data requires two objectives and categorical inputs.", ha='center', va='center')
        return fig_to_gradio_image(fig)

    num_cat_plots = len(valid_categorical_inputs_for_plot)
    
    fig, axes = plt.subplots(nrows=num_cat_plots, ncols=2, # Fixed for 2 objectives
                             figsize=(12, 5 * num_cat_plots), 
                             squeeze=False) 

    for i, cat_col in enumerate(valid_categorical_inputs_for_plot):
        cat_desc = plot_cols_desc.get(cat_col, cat_col)
        for j, obj_name in enumerate(objective_columns): # Iterate through the two objectives
            ax = axes[i, j]
            
            obj_spec = next((v for v in variables_spec if v['name'] == obj_name), None)
            obj_desc = obj_spec['description'] if obj_spec else obj_name

            if obj_name not in df.columns:
                ax.text(0.5, 0.5, f"Objective '{obj_name}' not in data.", ha='center', va='center')
                continue

            plot_data = df[[cat_col, obj_name]].copy()
            plot_data[obj_name] = pd.to_numeric(plot_data[obj_name], errors='coerce')
            plot_data.dropna(subset=[obj_name, cat_col], inplace=True)

            if not plot_data.empty and plot_data[cat_col].nunique() > 0:
                order = plot_data.groupby(cat_col)[obj_name].median().sort_values().index
                sns.boxplot(x=cat_col, y=obj_name, data=plot_data, ax=ax, hue=cat_col, palette="Pastel1", order=order, width=0.6, legend=False, dodge=False)
                
                category_counts = plot_data[cat_col].value_counts()
                new_xticklabels = []
                for cat_name_in_order in order: # Use the same order as sns.boxplot
                    count = category_counts.get(cat_name_in_order, 0)
                    new_xticklabels.append(f"{cat_name_in_order}\n(n={count})")
                ax.set_xticks(np.arange(len(order)))
                ax.set_xticklabels(new_xticklabels)

                ax.set_title(f'{obj_desc} by {cat_desc}')
                ax.set_xlabel(cat_desc if i == num_cat_plots -1 or num_cat_plots == 1 else '') 
                ax.set_ylabel(obj_desc)
                ax.tick_params(axis='x', rotation=30, labelsize=8) 
                ax.grid(axis='y', linestyle='--', alpha=0.7)
            else:
                ax.text(0.5, 0.5, f"No valid data for\n{cat_desc} vs {obj_desc}", ha='center', va='center')
    
    if num_cat_plots > 0: 
        fig.suptitle(f'Categorical Variable Analysis vs. Objectives', fontsize=16, y=1.02)
    
    sns.despine()
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig_to_gradio_image(fig)

def plot_scatter_analysis():
    """Plots two objectives vs. their most correlated numerical input feature."""
    df = load_data()
    plot_cols_desc = {v['name']: v['description'] for v in variables_spec}

    if df.empty or not ORDERED_INPUT_COLS_FOR_UI or not objective_columns or len(objective_columns) != 2:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Data requires two objectives and inputs for scatter analysis.", ha='center', va='center')
        return fig_to_gradio_image(fig)

    continuous_inputs_present = [v['name'] for v in variables_spec if v['type'] == 'continuous' and v['name'] in ORDERED_INPUT_COLS_FOR_UI and v['name'] in df.columns]
    if not continuous_inputs_present:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No continuous inputs for scatter analysis.", ha='center', va='center')
        return fig_to_gradio_image(fig)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6), sharey=False) # Fixed for 2 objectives

    for i, obj_name in enumerate(objective_columns): # Iterate through the two objectives
        ax = axes[i]
        if obj_name not in df.columns:
            ax.text(0.5,0.5, f"Objective '{obj_name}' not in data.", ha='center', va='center')
            continue
        obj_spec = next((v for v in variables_spec if v['name'] == obj_name), None)
        obj_desc = obj_spec['description'] if obj_spec else obj_name

        df_corr_scatter = df[continuous_inputs_present + [obj_name]].apply(pd.to_numeric, errors='coerce').dropna()
        if df_corr_scatter.empty or df_corr_scatter.shape[0] < 2:
            ax.text(0.5, 0.5, f"Not enough valid data for correlation for {obj_desc}", ha='center', va='center')
            continue

        correlations_scatter = df_corr_scatter.corr()[obj_name].abs().drop(obj_name, errors='ignore')
        if correlations_scatter.empty or correlations_scatter.isnull().all():
            ax.text(0.5, 0.5, f"Could not determine most correlated feature for {obj_desc}", ha='center', va='center')
            # Removed return here to allow other plot to still render if possible
            continue 
            
        most_correlated_feature_name = correlations_scatter.idxmax()
        most_correlated_feature_desc = plot_cols_desc.get(most_correlated_feature_name, most_correlated_feature_name)
        
        sns.regplot(x=most_correlated_feature_name, y=obj_name, data=df_corr_scatter, ax=ax, scatter_kws={'alpha':0.6, 's':40}, line_kws={'color':'red', 'linestyle':'--'})
        ax.set_title(f'{obj_desc} vs. {most_correlated_feature_desc}')
        ax.set_xlabel(most_correlated_feature_desc)
        ax.set_ylabel(obj_desc)
        ax.grid(True, linestyle='--', alpha=0.7)
        
    fig.suptitle('Objective vs. Most Correlated Continuous Input', fontsize=16, y=1.02)
    
    sns.despine()
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig_to_gradio_image(fig)

def plot_suggestion_frequency():
    """Plots frequency/distribution of suggested input parameters."""
    df = load_data()
    plot_cols_desc = {v['name']: v['description'] for v in variables_spec}

    if df.empty or not ORDERED_INPUT_COLS_FOR_UI:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data for suggestion frequency analysis", ha='center', va='center', transform=ax.transAxes)
        return fig_to_gradio_image(fig)

    input_cols_to_analyze = [col for col in ORDERED_INPUT_COLS_FOR_UI if col in df.columns]
    if not input_cols_to_analyze:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No input columns found in data for frequency analysis", ha='center', va='center', transform=ax.transAxes)
        return fig_to_gradio_image(fig)

    num_input_cols = len(input_cols_to_analyze)
    ncols_grid = 2
    nrows_grid = (num_input_cols + ncols_grid - 1) // ncols_grid 

    fig, axes = plt.subplots(nrows=nrows_grid, ncols=ncols_grid, figsize=(ncols_grid * 7, nrows_grid * 5), squeeze=False)
    axes_flat = axes.flatten()

    for i, col_name in enumerate(input_cols_to_analyze):
        ax = axes_flat[i]
        var_spec = next((v for v in variables_spec if v['name'] == col_name), None)
        col_desc = plot_cols_desc.get(col_name, col_name)

        if var_spec and var_spec['type'] == 'categorical':
            if not df[col_name].dropna().empty:
                sns.countplot(y=df[col_name], ax=ax, hue=df[col_name], palette="viridis", order=df[col_name].value_counts().index, legend=False, dodge=False) 
                ax.set_title(f'Frequency: {col_desc}')
                ax.set_xlabel('Count')
                ax.set_ylabel('') 
            else:
                ax.text(0.5, 0.5, f"No data for {col_desc}", ha='center', va='center')
        elif var_spec and var_spec['type'] == 'continuous':
            numeric_data = pd.to_numeric(df[col_name], errors='coerce').dropna()
            if not numeric_data.empty:
                sns.histplot(numeric_data, ax=ax, kde=True, color="skyblue", bins=15)
                ax.set_title(f'Distribution: {col_desc}')
                ax.set_xlabel(col_desc)
                ax.set_ylabel('Frequency')
            else:
                ax.text(0.5, 0.5, f"No numeric data for {col_desc}", ha='center', va='center')
        else:
            ax.text(0.5, 0.5, f"Unknown type or no spec\nfor {col_desc}", ha='center', va='center')
        ax.grid(axis='x', linestyle='--', alpha=0.7)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis('off')

    fig.suptitle('Frequency/Distribution of Suggested Input Parameters', fontsize=16, y=1.02)
    sns.despine()
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig_to_gradio_image(fig)

