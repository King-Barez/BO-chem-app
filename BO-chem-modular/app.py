"""
Gradio application for Chemical Experiment Optimization using BoTorch.

This application provides a user interface for:
- Viewing existing experimental data.
- Suggesting new experiment parameters using Bayesian Optimization.
- Saving results of new experiments.
- Visualizing data through various analytical plots.
- Managing the dataset (upload/download).
"""
import gradio as gr
import pandas as pd
import os

from config import (
    variables_spec, DATA_FILE_PATH, columns,
    ORDERED_INPUT_COLS_FOR_UI, objective_columns, 
    ORDERED_OBJECTIVE_COLS_FOR_UI, MAX_SUGGESTIONS_DISPLAY,
    MIN_SAMPLES_FOR_BO
)
from data_utils import save_suggestion_result, upload_csv, download_csv, load_data, calculate_pmi
from bo_optimizer import suggest_batch_experiments
from plotting_utils import (
    plot_yield_progression, plot_correlation_heatmap, plot_feature_importance,
    plot_feature_distributions, plot_categorical_analysis,
    plot_scatter_analysis, plot_suggestion_frequency, plot_yield_vs_pmi
)
from analysis_utils import get_statistics, export_report

app = gr.Blocks(title="Chemical Experiment Optimizer").queue()

if not os.path.exists('data'):
    os.makedirs('data', exist_ok=True)
if not os.path.exists(DATA_FILE_PATH):
    pd.DataFrame(columns=columns).to_csv(DATA_FILE_PATH, index=False)
else:
    load_data()


with app:
    gr.Markdown("# 🧪 Chemical Experiment Optimization Interface")

    with gr.Tab("View Data 📄"):
        data_display = gr.DataFrame(interactive=False, value=load_data, wrap=True)
        refresh_btn = gr.Button("🔄 Refresh Data")
        refresh_btn.click(load_data, outputs=data_display)

    with gr.Tab("Suggest & Save Results 💡"):
        gr.Markdown("## 1. Get New Experiment Suggestion")
        
        with gr.Row():
            num_experiments_input = gr.Number(minimum=1, precision=0, value=1, label="Number of experiments to suggest")
            suggest_params_btn = gr.Button("🤖 Suggest Parameters", variant="primary")

        def update_pmi_ui(base, base_eq, lag, eta, yield_val):
            if yield_val is None: return None
            pmi_value, _ = calculate_pmi(base, base_eq, lag, eta, yield_val)
            return pmi_value

        suggestions_ui_container = gr.Column(visible=False) 
        with suggestions_ui_container:
            experiment_suggestion_sets_ui = []
            for exp_idx_ui in range(MAX_SUGGESTIONS_DISPLAY): 
                with gr.Group(visible=(exp_idx_ui == 0)) as exp_group_ui: 
                    gr.Markdown(f"### Experiment {exp_idx_ui + 1}")
                    exp_input_param_comps_map_ui = {}
                    with gr.Row():
                        for col_name_param_ui in ORDERED_INPUT_COLS_FOR_UI:
                            var_spec = next(v for v in variables_spec if v['name'] == col_name_param_ui)
                            label = f"{var_spec['description']}"
                            if var_spec['type'] == 'categorical':
                                comp = gr.Dropdown(choices=var_spec['levels'], label=label, interactive=True)
                            else:
                                if col_name_param_ui in ['eta', 'base_eq']:
                                    step = 0.01
                                elif col_name_param_ui == 'freq':
                                    step = 0.5
                                else: # time
                                    step = 1
                                comp = gr.Number(label=label, step=step, precision=None, interactive=True)
                            exp_input_param_comps_map_ui[col_name_param_ui] = comp
                    
                    exp_objective_input_comps_map_ui = {}
                    with gr.Row(): 
                        for obj_col_name_ui in ORDERED_OBJECTIVE_COLS_FOR_UI:
                            spec = next(v for v in variables_spec if v['name'] == obj_col_name_ui)
                            comp = gr.Number(label=spec['description'], minimum=spec['bounds'][0], maximum=spec['bounds'][1], step=0.001, precision=None, interactive=(obj_col_name_ui != 'pmi'))
                            exp_objective_input_comps_map_ui[obj_col_name_ui] = comp
                    
                    exp_save_btn_elem_ui = gr.Button(f"💾 Save Result for Experiment {exp_idx_ui + 1}", variant="secondary")
                    exp_save_status_elem_ui = gr.Textbox(label="Save Status", lines=1, interactive=False)

                    ui_set = {'group': exp_group_ui, 'inputs': exp_input_param_comps_map_ui, 'objectives': exp_objective_input_comps_map_ui, 'save_btn': exp_save_btn_elem_ui, 'save_status': exp_save_status_elem_ui}
                    experiment_suggestion_sets_ui.append(ui_set)
                    
                    # Link PMI calculation
                    pmi_deps = [ui_set['inputs'][name] for name in ['base', 'base_eq', 'lag', 'eta']] + [ui_set['objectives']['yield_']]
                    ui_set['objectives']['yield_'].change(fn=update_pmi_ui, inputs=pmi_deps, outputs=[ui_set['objectives']['pmi']])
        
        def update_suggestions_visibility_ui(num_exp):
            try: num_exp = int(num_exp)
            except (ValueError, TypeError): num_exp = 0
            updates = [gr.update(visible=(num_exp > 0))]
            for i in range(MAX_SUGGESTIONS_DISPLAY):
                updates.append(gr.update(visible=(i < num_exp)))
            return updates

        all_suggestion_groups = [s['group'] for s in experiment_suggestion_sets_ui]
        num_experiments_input.change(fn=update_suggestions_visibility_ui, inputs=[num_experiments_input], outputs=[suggestions_ui_container] + all_suggestion_groups)

        def call_suggest_batch_experiments_for_ui(num_exp):
            try: num_exp = int(num_exp)
            except (ValueError, TypeError): num_exp = 0
            if num_exp <= 0: return [None] * (MAX_SUGGESTIONS_DISPLAY * len(ORDERED_INPUT_COLS_FOR_UI))
            
            suggestions = suggest_batch_experiments(num_exp)
            outputs = [None] * (MAX_SUGGESTIONS_DISPLAY * len(ORDERED_INPUT_COLS_FOR_UI))
            outputs[:len(suggestions)] = suggestions
            return outputs

        all_ui_input_outputs = [comp for s in experiment_suggestion_sets_ui for comp in s['inputs'].values()]
        suggest_params_btn.click(fn=call_suggest_batch_experiments_for_ui, inputs=[num_experiments_input], outputs=all_ui_input_outputs, show_progress="full")

        for exp_set in experiment_suggestion_sets_ui:
            save_inputs = list(exp_set['inputs'].values()) + list(exp_set['objectives'].values())
            exp_set['save_btn'].click(fn=save_suggestion_result, inputs=save_inputs, outputs=exp_set['save_status']).then(fn=load_data, outputs=data_display)

    def load_all_analysis_plots_for_ui(): 
        return (get_statistics(), plot_yield_progression(), plot_correlation_heatmap(), plot_feature_importance(), plot_feature_distributions(), plot_categorical_analysis(), plot_scatter_analysis(), plot_yield_vs_pmi(), plot_suggestion_frequency()) 

    with gr.Tab("📊 Data Analysis & Visualization"):
        gr.Markdown("## 🔬 Comprehensive Analysis")
        load_analysis_btn = gr.Button("🚀 Load/Refresh All Visualizations")
        
        with gr.Row():
            stats_markdown_ui = gr.Markdown("Statistics will appear here.")
        
        with gr.Row():
            yield_prog_plot_ui = gr.Image(label="Objective Progression", interactive=False) 
            corr_heatmap_plot_ui = gr.Image(label="Correlation Heatmap", interactive=False)
        with gr.Row():
            feat_importance_plot_ui = gr.Image(label="Feature Importance", interactive=False)
            feat_dist_plot_ui = gr.Image(label="Feature Distributions", interactive=False)
        with gr.Row(): 
            cat_analysis_plot_ui = gr.Image(label="Categorical Analysis", interactive=False)
            scatter_analysis_plot_ui = gr.Image(label="Scatter Analysis", interactive=False) 
        with gr.Row():
            yield_vs_pmi_plot_ui = gr.Image(label="Yield vs. PMI", interactive=False)
            suggestion_freq_plot_ui = gr.Image(label="Suggestion Frequency", interactive=False) 

        analysis_outputs_list_ui = [stats_markdown_ui, yield_prog_plot_ui, corr_heatmap_plot_ui, feat_importance_plot_ui, feat_dist_plot_ui, cat_analysis_plot_ui, scatter_analysis_plot_ui, yield_vs_pmi_plot_ui, suggestion_freq_plot_ui]
        load_analysis_btn.click(fn=load_all_analysis_plots_for_ui, inputs=None, outputs=analysis_outputs_list_ui, show_progress="full")

    with gr.Tab("🗃️ Dataset Management"):
        gr.Markdown("## 💾 Manage Your Dataset")
        with gr.Row():
            upload_button = gr.UploadButton("📁 Upload CSV Dataset", file_types=[".csv"])
            download_button = gr.DownloadButton(label="📄 Download Current Dataset as CSV", value=download_csv)
            export_report_btn = gr.DownloadButton(label="📝 Export Analysis Report", value=export_report)
        upload_status_text = gr.Textbox(label="Upload Status", interactive=False)

        def upload_and_refresh(file_obj):
            status = upload_csv(file_obj) 
            return status, load_data()

        upload_button.upload(fn=upload_and_refresh, inputs=[upload_button], outputs=[upload_status_text, data_display])

    def initial_load_actions():
        initial_num_exp_val = 1
        visibility_updates = update_suggestions_visibility_ui(initial_num_exp_val)
        analysis_updates = load_all_analysis_plots_for_ui()
        return tuple([gr.update(value=initial_num_exp_val)] + visibility_updates + list(analysis_updates))

    app.load(fn=initial_load_actions, outputs=[num_experiments_input, suggestions_ui_container] + all_suggestion_groups + analysis_outputs_list_ui)

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7862, share=False)