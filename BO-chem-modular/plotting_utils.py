"""
Plotting utilities for BO-chem data analysis.
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
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight') 
    plt.close(fig) 
    buf.seek(0)
    return Image.open(buf)

def plot_yield_progression():
    df = load_data()
    if df.empty or not objective_columns:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data for objective progression plot.", ha='center', va='center')
        return fig_to_gradio_image(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    obj_name = objective_columns[0]

    if obj_name not in df.columns:
        ax.text(0.5, 0.5, f"Objective '{obj_name}' not in data.", ha='center', va='center')
        return fig_to_gradio_image(fig)

    obj_data = pd.to_numeric(df[obj_name], errors='coerce').dropna()
    if obj_data.empty:
        ax.text(0.5, 0.5, f"No valid numeric data for {obj_name}", ha='center', va='center')
        return fig_to_gradio_image(fig)
    
    obj_spec = next((v for v in variables_spec if v['name'] == obj_name), None)
    obj_desc = obj_spec['description'] if obj_spec else obj_name
    
    ax.plot(obj_data.index, obj_data.values, marker='o', linestyle='-', label=f'{obj_desc} (Actual)')
    
    if obj_spec:
        cumulative_best = obj_data.cummax() if obj_spec['maximize'] else obj_data.cummin()
        ax.plot(cumulative_best.index, cumulative_best.values, marker='.', linestyle='--', color='red', label='Cumulative Best')

    if len(obj_data) >= 2:
        x_vals, y_vals = obj_data.index.to_numpy(float), obj_data.values.astype(float)
        try:
            coeffs = np.polyfit(x_vals, y_vals, 1)
            ax.plot(obj_data.index, np.poly1d(coeffs)(x_vals), color='green', linestyle=':', label='Trend')
        except (np.linalg.LinAlgError, ValueError): pass

    ax.set_title(f'{obj_desc} Progression')
    ax.set_ylabel(obj_desc)
    ax.set_xlabel('Experiment Number')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='best')
    sns.despine()
    plt.tight_layout()
    return fig_to_gradio_image(fig)

def plot_correlation_heatmap():
    df = load_data()
    numeric_cols = [v['name'] for v in variables_spec if v['type'] in ['continuous', 'objective'] and v['name'] in df.columns]
    
    if df.empty or len(numeric_cols) < 2:
        fig, ax = plt.subplots(); ax.text(0.5, 0.5, "Not enough numeric data for heatmap.", ha='center'); return fig_to_gradio_image(fig)

    df_numeric = df[numeric_cols].apply(pd.to_numeric, errors='coerce').dropna()
    if df_numeric.shape[1] < 2:
        fig, ax = plt.subplots(); ax.text(0.5, 0.5, "Not enough valid numeric data.", ha='center'); return fig_to_gradio_image(fig)
        
    corr_matrix = df_numeric.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool)) 
    
    fig, ax = plt.subplots(figsize=(max(8, len(numeric_cols)*0.9), max(6, len(numeric_cols)*0.7)))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax, vmin=-1, vmax=1, annot_kws={"size": 8}, mask=mask)
    ax.set_title('Feature Correlation Heatmap')
    plt.tight_layout()
    return fig_to_gradio_image(fig)

def plot_feature_importance():
    df = load_data()
    continuous_inputs = [v['name'] for v in variables_spec if v['type'] == 'continuous' and v['name'] in ORDERED_INPUT_COLS_FOR_UI and v['name'] in df.columns]
    if df.empty or not continuous_inputs or not objective_columns:
        fig, ax = plt.subplots(); ax.text(0.5, 0.5, "Not enough data for feature importance.", ha='center'); return fig_to_gradio_image(fig)

    obj_name = objective_columns[0]
    fig, ax = plt.subplots(figsize=(8, max(5, len(continuous_inputs) * 0.5)))

    obj_spec = next((v for v in variables_spec if v['name'] == obj_name), None)
    obj_desc = obj_spec['description'] if obj_spec else obj_name

    df_for_corr = df[continuous_inputs + [obj_name]].apply(pd.to_numeric, errors='coerce').dropna()
    if df_for_corr.shape[0] < 2:
        ax.text(0.5, 0.5, f"Not enough valid data for importance vs {obj_desc}", ha='center'); return fig_to_gradio_image(fig)
    
    correlations = df_for_corr.corr()[obj_name].abs().drop(obj_name, errors='ignore').sort_values(ascending=True) 
    if correlations.empty:
        ax.text(0.5, 0.5, f"Could not calculate correlations for {obj_desc}", ha='center'); return fig_to_gradio_image(fig)
    
    correlations.plot(kind='barh', ax=ax, color='skyblue')
    ax.set_title(f'Input Importance (Correlation with {obj_desc})')
    ax.set_xlabel('Absolute Correlation')
    sns.despine()
    plt.tight_layout()
    return fig_to_gradio_image(fig)

def plot_feature_distributions():
    df = load_data()
    continuous_cols = [v['name'] for v in variables_spec if v['type'] == 'continuous' and v['name'] in ORDERED_INPUT_COLS_FOR_UI and v['name'] in df.columns and not df[v['name']].isnull().all()] 

    if df.empty or not continuous_cols:
        fig, ax = plt.subplots(); ax.text(0.5, 0.5, "No continuous input data for distributions.", ha='center'); return fig_to_gradio_image(fig)

    df_to_plot = df[continuous_cols].apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')
    if df_to_plot.empty:
        fig, ax = plt.subplots(); ax.text(0.5, 0.5, "No valid numeric data for distributions.", ha='center'); return fig_to_gradio_image(fig)

    df_normalized = pd.DataFrame(MinMaxScaler().fit_transform(df_to_plot), columns=df_to_plot.columns)
    
    fig, ax = plt.subplots(figsize=(max(10, df_normalized.shape[1] * 1.2), 6)) 
    sns.boxplot(data=df_normalized, ax=ax, palette="Set2", orient="v", width=0.5)
    
    # Fix for UserWarning: Explicitly set tick positions before labels
    tick_labels = [v['description'] for v in variables_spec if v['name'] in df_normalized.columns]
    ax.set_xticks(np.arange(len(tick_labels)))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")

    ax.set_title('Normalized Continuous Feature Distributions')
    ax.set_ylabel('Normalized Value (0-1)')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    sns.despine()
    plt.tight_layout()
    return fig_to_gradio_image(fig)

def plot_pareto_front():
    fig, ax = plt.subplots(figsize=(8,6))
    ax.text(0.5, 0.5, "Pareto Front plot is for multi-objective optimization only.", ha='center', va='center')
    ax.set_title('Pareto Front Analysis')
    return fig_to_gradio_image(fig)

def plot_categorical_analysis():
    df = load_data()
    categorical_inputs = [v for v in variables_spec if v['type'] == 'categorical' and v['name'] in ORDERED_INPUT_COLS_FOR_UI and v['name'] in df.columns and not df[v['name']].isnull().all()]

    if df.empty or not categorical_inputs or not objective_columns:
        fig, ax = plt.subplots(); ax.text(0.5, 0.5, "Not enough data for categorical analysis.", ha='center'); return fig_to_gradio_image(fig)

    obj_name = objective_columns[0]
    obj_spec = next((v for v in variables_spec if v['name'] == obj_name), None)
    obj_desc = obj_spec['description'] if obj_spec else obj_name
    
    fig, axes = plt.subplots(nrows=len(categorical_inputs), ncols=1, figsize=(8, 5 * len(categorical_inputs)), squeeze=False) 

    for i, cat_spec in enumerate(categorical_inputs):
        ax = axes[i, 0]
        cat_col = cat_spec['name']
        plot_data = df[[cat_col, obj_name]].copy()
        plot_data[obj_name] = pd.to_numeric(plot_data[obj_name], errors='coerce')
        plot_data.dropna(inplace=True)

        if not plot_data.empty and plot_data[cat_col].nunique() > 0:
            order = plot_data.groupby(cat_col)[obj_name].median().sort_values().index
            sns.boxplot(x=cat_col, y=obj_name, data=plot_data, ax=ax, hue=cat_col, palette="Pastel1", order=order, legend=False, dodge=False)
            
            counts = plot_data[cat_col].value_counts()
            
            # Fix for UserWarning: Explicitly set tick positions before labels
            tick_labels = [f"{cat}\n(n={counts.get(cat, 0)})" for cat in order]
            ax.set_xticks(np.arange(len(tick_labels)))
            ax.set_xticklabels(tick_labels)

            ax.set_title(f'{obj_desc} by {cat_spec["description"]}')
            ax.set_xlabel(None)
            ax.set_ylabel(obj_desc)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
        else:
            ax.text(0.5, 0.5, f"No valid data for {cat_spec['description']}", ha='center', va='center')
    
    sns.despine()
    plt.tight_layout()
    return fig_to_gradio_image(fig)

def plot_scatter_analysis():
    df = load_data()
    continuous_inputs = [v['name'] for v in variables_spec if v['type'] == 'continuous' and v['name'] in ORDERED_INPUT_COLS_FOR_UI and v['name'] in df.columns]
    if df.empty or not continuous_inputs or not objective_columns:
        fig, ax = plt.subplots(); ax.text(0.5, 0.5, "Not enough data for scatter analysis.", ha='center'); return fig_to_gradio_image(fig)

    obj_name = objective_columns[0]
    obj_spec = next((v for v in variables_spec if v['name'] == obj_name), None)
    obj_desc = obj_spec['description'] if obj_spec else obj_name

    df_corr = df[continuous_inputs + [obj_name]].apply(pd.to_numeric, errors='coerce').dropna()
    if df_corr.shape[0] < 2:
        fig, ax = plt.subplots(); ax.text(0.5, 0.5, f"Not enough valid data for correlation.", ha='center'); return fig_to_gradio_image(fig)

    correlations = df_corr.corr()[obj_name].abs().drop(obj_name, errors='ignore')
    if correlations.empty or correlations.isnull().all():
        fig, ax = plt.subplots(); ax.text(0.5, 0.5, f"Could not determine most correlated feature.", ha='center'); return fig_to_gradio_image(fig)
        
    most_corr_feature_name = correlations.idxmax()
    most_corr_feature_spec = next((v for v in variables_spec if v['name'] == most_corr_feature_name), None)
    most_corr_feature_desc = most_corr_feature_spec['description'] if most_corr_feature_spec else most_corr_feature_name
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.regplot(x=most_corr_feature_name, y=obj_name, data=df_corr, ax=ax, scatter_kws={'alpha':0.6}, line_kws={'color':'red', 'linestyle':'--'})
    ax.set_title(f'{obj_desc} vs. Most Correlated Input ({most_corr_feature_desc})')
    ax.set_xlabel(most_corr_feature_desc)
    ax.set_ylabel(obj_desc)
    ax.grid(True, linestyle='--', alpha=0.7)
    sns.despine()
    plt.tight_layout()
    return fig_to_gradio_image(fig)

def plot_yield_vs_pmi():
    """Plots the Pareto front for Yield vs. PMI, colored by experiment number."""
    df = load_data()
    
    yield_spec = next((v for v in variables_spec if v['name'] == 'yield_'), None)
    pmi_spec = next((v for v in variables_spec if v['name'] == 'pmi'), None)

    if df.empty or not yield_spec or not pmi_spec or 'yield_' not in df.columns or 'pmi' not in df.columns:
        fig, ax = plt.subplots(figsize=(8,6))
        ax.text(0.5, 0.5, "Data for 'yield_' and/or 'pmi' not found.", ha='center', va='center')
        ax.set_title('Yield vs. PMI Pareto Front')
        return fig_to_gradio_image(fig)

    df_objectives = df[['yield_', 'pmi']].copy()
    df_objectives['original_index'] = df_objectives.index
    
    df_objectives['yield_'] = pd.to_numeric(df_objectives['yield_'], errors='coerce')
    df_objectives['pmi'] = pd.to_numeric(df_objectives['pmi'], errors='coerce')
    df_objectives.dropna(subset=['yield_', 'pmi'], inplace=True)

    if df_objectives.empty:
        fig, ax = plt.subplots(figsize=(8,6))
        ax.text(0.5, 0.5, "No valid numeric data for Yield vs. PMI plot.", ha='center', va='center')
        ax.set_xlabel(yield_spec.get('description', 'yield_'))
        ax.set_ylabel(pmi_spec.get('description', 'pmi'))
        ax.set_title('Yield vs. PMI Pareto Front')
        return fig_to_gradio_image(fig)

    points = df_objectives[['yield_', 'pmi']].to_numpy()
    original_indices = df_objectives['original_index'].to_numpy()
    
    is_pareto = np.ones(points.shape[0], dtype=bool)
    for i, p1 in enumerate(points):
        if not is_pareto[i]: continue
        for j, p2 in enumerate(points):
            if i == j: continue
            p1_yield_better = p1[0] >= p2[0]  # Maximize
            p1_pmi_better = p1[1] <= p2[1]    # Minimize
            p1_strictly_better = (p1[0] > p2[0]) or (p1[1] < p2[1])
            if p1_yield_better and p1_pmi_better and p1_strictly_better:
                is_pareto[j] = False 

    pareto_points = points[is_pareto]
    pareto_indices = original_indices[is_pareto]
    non_pareto_points = points[~is_pareto]
    non_pareto_indices = original_indices[~is_pareto]

    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.cm.viridis 

    if non_pareto_points.shape[0] > 0:
        ax.scatter(non_pareto_points[:, 0], non_pareto_points[:, 1], c=non_pareto_indices, cmap=cmap, 
                   label='Dominated Points', alpha=0.6, s=30)
    
    if pareto_points.shape[0] > 0:
        sort_indices = np.argsort(pareto_points[:, 0])
        
        ax.scatter(pareto_points[:, 0], pareto_points[:, 1], c=pareto_indices, cmap=cmap, 
                   label='Pareto Front', s=60, edgecolors='black', zorder=3)
        
        if pareto_points.shape[0] > 1:
            ax.plot(pareto_points[sort_indices, 0], pareto_points[sort_indices, 1], 'r--', alpha=0.6, zorder=2)

    ax.set_xlabel(f"{yield_spec.get('description', 'yield_')} (Maximize)")
    ax.set_ylabel(f"{pmi_spec.get('description', 'pmi')} (Minimize)")
    ax.set_title('Yield vs. PMI Pareto Front')
    
    if original_indices.size > 0:
        norm = matplotlib.colors.Normalize(vmin=original_indices.min(), vmax=original_indices.max())
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        fig.colorbar(sm, ax=ax, label='Experiment Number (Index)')

    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    sns.despine()
    plt.tight_layout(rect=[0, 0, 0.95, 1])
    return fig_to_gradio_image(fig)

def plot_suggestion_frequency():
    df = load_data()
    input_cols = [col for col in ORDERED_INPUT_COLS_FOR_UI if col in df.columns]
    if df.empty or not input_cols:
        fig, ax = plt.subplots(); ax.text(0.5, 0.5, "No data for suggestion frequency analysis.", ha='center'); return fig_to_gradio_image(fig)

    ncols = 2
    nrows = (len(input_cols) + ncols - 1) // ncols 
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 7, nrows * 5), squeeze=False)
    axes_flat = axes.flatten()

    for i, col_name in enumerate(input_cols):
        ax = axes_flat[i]
        var_spec = next((v for v in variables_spec if v['name'] == col_name), None)
        col_desc = var_spec['description'] if var_spec else col_name

        if var_spec and var_spec['type'] == 'categorical':
            if not df[col_name].dropna().empty:
                sns.countplot(y=df[col_name], ax=ax, hue=df[col_name], palette="viridis", order=df[col_name].value_counts().index, legend=False, dodge=False) 
                ax.set_title(f'Frequency: {col_desc}')
            else: ax.text(0.5, 0.5, f"No data for {col_desc}", ha='center')
        elif var_spec and var_spec['type'] == 'continuous':
            numeric_data = pd.to_numeric(df[col_name], errors='coerce').dropna()
            if not numeric_data.empty:
                sns.histplot(numeric_data, ax=ax, kde=True, color="skyblue", bins=15)
                ax.set_title(f'Distribution: {col_desc}')
            else: ax.text(0.5, 0.5, f"No numeric data for {col_desc}", ha='center')
        ax.grid(axis='x', linestyle='--', alpha=0.7)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis('off')

    fig.suptitle('Frequency/Distribution of Input Parameters', fontsize=16)
    sns.despine()
    plt.tight_layout()
    return fig_to_gradio_image(fig)

