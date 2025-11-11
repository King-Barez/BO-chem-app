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
    ORDERED_OBJECTIVE_COLS_FOR_UI, REF_POINT, MAX_SUGGESTIONS_DISPLAY,
    MIN_SAMPLES_FOR_BO
)
from data_utils import (
    save_suggestion_result,
    upload_csv, download_csv, load_data,
    calculate_pmi
)
from bo_optimizer import suggest_batch_experiments, suggest_multiple_lhs_experiments
from plotting_utils import (
    plot_yield_progression, 
    plot_correlation_heatmap, plot_feature_importance,
    plot_feature_distributions, plot_categorical_analysis,
    plot_scatter_analysis, plot_pareto_front,
    plot_suggestion_frequency 
)
from analysis_utils import get_statistics

app = gr.Blocks(title="Chemical Experiment Optimizer").queue()

if not os.path.exists('data'):
    os.makedirs('data', exist_ok=True)
if not os.path.exists(DATA_FILE_PATH):
    pd.DataFrame(columns=columns).to_csv(DATA_FILE_PATH, index=False)
else:
    _ = load_data()


with app:
    gr.Markdown("# 🧪 Ketone Reaction Optimization Interface (BoTorch - Multi-Objective)")

    with gr.Tab("View Data 📄"):
        data_display = gr.DataFrame(interactive=False, value=load_data)
        refresh_btn = gr.Button("🔄 Refresh Data")
        refresh_btn.click(load_data, outputs=data_display)

    with gr.Tab("Suggest & Save Results 💡"):
        gr.Markdown("## 1. Get New Experiment Suggestion")
        
        with gr.Row():
            num_experiments_input = gr.Number(
                minimum=1, precision=0, value=1, # Initial value will be updated by app.load
                label="Number of experiments to suggest",
                info="Enter the number of experiment suggestions you want."
            )
            suggest_params_btn = gr.Button("🤖 Suggest Next Experiment Parameters", variant="primary")

        def update_pmi_ui(base, base_eq, solvent, eta, s_salt, yield_val):
            if yield_val is None:
                return None
            pmi_value, _ = calculate_pmi(s_salt, base, base_eq, solvent, eta, yield_val)
            return pmi_value

        suggestions_ui_container = gr.Column(visible=False) 
        
        with suggestions_ui_container:
            experiment_suggestion_sets_ui = [] 
            
            for exp_idx_ui in range(MAX_SUGGESTIONS_DISPLAY): 
                with gr.Group(visible=(exp_idx_ui == 0)) as exp_group_ui: 
                    gr.Markdown(f"### Experiment {exp_idx_ui + 1}")
                    
                    exp_input_param_comps_map_ui = {}
                    current_row_comps_for_layout_ui = []

                    for i, col_name_param_ui in enumerate(ORDERED_INPUT_COLS_FOR_UI):
                        if i > 0 and i % 3 == 0 and current_row_comps_for_layout_ui: 
                            with gr.Row(): pass 
                            current_row_comps_for_layout_ui = [] 
                        
                        var_spec_param_ui = next(v for v in variables_spec if v['name'] == col_name_param_ui)
                        label_param_ui = f"{var_spec_param_ui['description']} (Exp {exp_idx_ui + 1})"
                        
                        if var_spec_param_ui['type'] == 'categorical':
                            comp_ui_elem = gr.Dropdown(choices=var_spec_param_ui['levels'], label=label_param_ui, value=(var_spec_param_ui['levels'][0] if var_spec_param_ui['levels'] else None), interactive=True)
                        else: 
                            # Precision/step secondo specifiche
                            if col_name_param_ui == 'eta':
                                precision_val_ui = 1   # 1 decimale
                            elif col_name_param_ui == 'base_eq':
                                precision_val_ui = 1   # step 0.5 gestito in rounding
                            elif col_name_param_ui in ('freq', 'time'):
                                precision_val_ui = 0   # interi
                            else:
                                precision_val_ui = 2
                            comp_ui_elem = gr.Number(label=label_param_ui, precision=precision_val_ui, value=None, interactive=True)
                        
                        exp_input_param_comps_map_ui[col_name_param_ui] = comp_ui_elem
                        current_row_comps_for_layout_ui.append(comp_ui_elem)
                    
                    if current_row_comps_for_layout_ui: 
                        with gr.Row(): pass 
                    
                    exp_objective_input_comps_map_ui = {}
                    with gr.Row(): 
                        for obj_col_name_ui in ORDERED_OBJECTIVE_COLS_FOR_UI:
                            obj_var_spec_ui_elem = next(v for v in variables_spec if v['name'] == obj_col_name_ui)
                            label_ui_obj = f"{obj_var_spec_ui_elem['description']} (Exp {exp_idx_ui + 1})" 
                            
                            is_interactive_ui = obj_col_name_ui != 'pmi'

                            obj_input_comp_ui = gr.Number(
                                label=label_ui_obj, 
                                minimum=obj_var_spec_ui_elem['bounds'][0], 
                                maximum=obj_var_spec_ui_elem['bounds'][1], 
                                precision=3, 
                                interactive=is_interactive_ui
                            )
                            exp_objective_input_comps_map_ui[obj_col_name_ui] = obj_input_comp_ui
                    
                    exp_save_btn_elem_ui = gr.Button(f"💾 Save Result for Experiment {exp_idx_ui + 1}", variant="secondary")
                    exp_save_status_elem_ui = gr.Textbox(label=f"Save Status Exp {exp_idx_ui + 1}", lines=1, interactive=False)

                    experiment_suggestion_sets_ui.append({
                        'group': exp_group_ui,
                        'input_params': exp_input_param_comps_map_ui,    
                        'objective_inputs': exp_objective_input_comps_map_ui, 
                        'save_btn': exp_save_btn_elem_ui,
                        'save_status': exp_save_status_elem_ui
                    })

            for exp_set in experiment_suggestion_sets_ui:
                yield_comp = exp_set['objective_inputs']['yield_']
                pmi_comp = exp_set['objective_inputs']['pmi']
                base_comp = exp_set['input_params']['base']
                base_eq_comp = exp_set['input_params']['base_eq']
                solvent_comp = exp_set['input_params']['solvent']
                eta_comp = exp_set['input_params']['eta']
                s_salt_comp = exp_set['input_params']['s_salt']
                yield_comp.change(
                    fn=update_pmi_ui,
                    inputs=[base_comp, base_eq_comp, solvent_comp, eta_comp, s_salt_comp, yield_comp],
                    outputs=[pmi_comp]
                )
        
        def update_suggestions_visibility_ui(num_exp_to_show_ui_str):
            """Updates visibility of suggestion input groups based on the number input value."""
            updates = []
            try:
                num_exp_to_show_ui = int(num_exp_to_show_ui_str)
                if num_exp_to_show_ui <= 0:
                    num_exp_to_show_ui = 0 # Hide all if invalid or zero
            except (ValueError, TypeError):
                num_exp_to_show_ui = 0 # Hide all if not a valid number

            updates.append(gr.update(visible=(num_exp_to_show_ui > 0))) 
            for i in range(MAX_SUGGESTIONS_DISPLAY):
                updates.append(gr.update(visible=(i < num_exp_to_show_ui))) 
            for i in range(MAX_SUGGESTIONS_DISPLAY):
                for _ in range(len(ORDERED_OBJECTIVE_COLS_FOR_UI)): 
                    updates.append(gr.update(value=None)) 
            return updates

        all_objective_outputs_for_vis = []
        for exp_set in experiment_suggestion_sets_ui:
            for obj_name in ORDERED_OBJECTIVE_COLS_FOR_UI:
                all_objective_outputs_for_vis.append(exp_set['objective_inputs'][obj_name])

        input_change_outputs = [suggestions_ui_container] + \
                                [exp_set['group'] for exp_set in experiment_suggestion_sets_ui] + \
                                all_objective_outputs_for_vis

        num_experiments_input.change(
            fn=update_suggestions_visibility_ui,
            inputs=[num_experiments_input],
            outputs=input_change_outputs
        )

        def call_suggest_batch_experiments_for_ui(num_exp_val_ui_str):
            """Calls the BO backend to get suggestions and formats them for the UI."""
            try:
                num_exp_val_ui = int(num_exp_val_ui_str)
            except (ValueError, TypeError):
                num_exp_val_ui = 0
            
            num_input_cols = len(ORDERED_INPUT_COLS_FOR_UI)
            # num_objective_cols = len(ORDERED_OBJECTIVE_COLS_FOR_UI) # Not directly used for output list length calculation here

            total_input_param_slots_in_ui = MAX_SUGGESTIONS_DISPLAY * num_input_cols
            # The output list for Gradio should match the number of input components for suggestions
            # These are all_ui_input_param_outputs_list + objective fields.
            # suggest_btn_outputs_list defines the target components.
            # The length of output_list_for_gradio should match len(suggest_btn_outputs_list)
            
            # Number of input parameter components per experiment suggestion UI block
            num_ui_input_fields_per_exp = len(ORDERED_INPUT_COLS_FOR_UI)
            # Number of objective input components per experiment suggestion UI block
            num_ui_objective_fields_per_exp = len(ORDERED_OBJECTIVE_COLS_FOR_UI)

            # Total slots in UI for input parameters across all MAX_SUGGESTIONS_DISPLAY blocks
            total_ui_input_param_slots = MAX_SUGGESTIONS_DISPLAY * num_ui_input_fields_per_exp
            # Total slots in UI for objective parameters across all MAX_SUGGESTIONS_DISPLAY blocks
            total_ui_objective_param_slots = MAX_SUGGESTIONS_DISPLAY * num_ui_objective_fields_per_exp
            
            gradio_expected_total_outputs = total_ui_input_param_slots + total_ui_objective_param_slots


            if not isinstance(num_exp_val_ui, int) or num_exp_val_ui <= 0:
                # Return a list of Nones matching the number of output components for suggestions
                return [None] * gradio_expected_total_outputs

            actual_suggestions_flat_list = suggest_batch_experiments(num_exp_val_ui)

            output_list_for_gradio = [None] * gradio_expected_total_outputs
            
            # actual_suggestions_flat_list contains only input parameters, num_exp_val_ui * num_input_cols
            num_values_from_suggestions = len(actual_suggestions_flat_list)
            
            # Fill the input parameter slots in the UI
            # Iterate up to num_exp_val_ui (number of requested suggestions)
            # and up to MAX_SUGGESTIONS_DISPLAY (number of available UI slots)
            # and up to num_values_from_suggestions (number of actual values returned)
            
            current_suggestion_value_idx = 0
            # Populate only up to MAX_SUGGESTIONS_DISPLAY UI slots, even if more were requested/generated
            for exp_idx_ui in range(min(num_exp_val_ui, MAX_SUGGESTIONS_DISPLAY)):
                for input_idx_in_exp in range(num_ui_input_fields_per_exp):
                    if current_suggestion_value_idx < num_values_from_suggestions:
                        # Calculate the correct index in output_list_for_gradio
                        # This corresponds to the flattened list of all input_params components
                        ui_component_idx = exp_idx_ui * num_ui_input_fields_per_exp + input_idx_in_exp
                        output_list_for_gradio[ui_component_idx] = actual_suggestions_flat_list[current_suggestion_value_idx]
                        current_suggestion_value_idx += 1
                    else:
                        break 
                if current_suggestion_value_idx >= num_values_from_suggestions:
                    break
            
            # The remaining slots (other input params beyond num_exp_val_ui, and all objective inputs) remain None
            return output_list_for_gradio

        all_ui_input_param_outputs_list = []
        for exp_set in experiment_suggestion_sets_ui:
            for input_name in ORDERED_INPUT_COLS_FOR_UI:
                all_ui_input_param_outputs_list.append(exp_set['input_params'][input_name])
        
        all_ui_objective_outputs_list = []
        for exp_set in experiment_suggestion_sets_ui:
            for obj_name in ORDERED_OBJECTIVE_COLS_FOR_UI:
                all_ui_objective_outputs_list.append(exp_set['objective_inputs'][obj_name])

        suggest_btn_outputs_list = all_ui_input_param_outputs_list + all_ui_objective_outputs_list

        suggest_params_btn.click(
            fn=call_suggest_batch_experiments_for_ui,
            inputs=[num_experiments_input],
            outputs=suggest_btn_outputs_list,
            show_progress="full" 
        )

        gr.Markdown("---")
        gr.Markdown("## 2. Bulk Save All Visible Results")
        save_all_btn_ui = gr.Button("💾 Save All Visible Experiment Results", variant="primary")
        bulk_save_status_ui = gr.Textbox(label="Bulk Save Status", lines=3, interactive=False)

        def collect_and_save_all_visible_results_ui(num_requested_experiments_str, *all_field_values_from_ui):
            """Collects data from visible UI fields and saves them using save_suggestion_result."""
            num_experiments_to_process = 0
            try:
                num_requested_experiments = int(num_requested_experiments_str)
                # Determine how many are actually visible and potentially filled in the UI
                num_experiments_to_process = min(num_requested_experiments, MAX_SUGGESTIONS_DISPLAY)
                if num_experiments_to_process <= 0:
                     return "No experiments selected or visible for bulk save.", load_data()
            except (ValueError, TypeError):
                return "Error: Invalid number of experiments input.", load_data()


            all_statuses = []
            num_input_cols = len(ORDERED_INPUT_COLS_FOR_UI)
            num_objective_cols = len(ORDERED_OBJECTIVE_COLS_FOR_UI)
            params_per_experiment_slot = num_input_cols + num_objective_cols

            saved_any = False
            for exp_idx in range(num_experiments_to_process): # Iterate up to the number of visible/processed experiments
                if exp_idx >= MAX_SUGGESTIONS_DISPLAY: # Should not happen if slider is configured correctly
                    all_statuses.append(f"Experiment {exp_idx + 1}: Index out of bounds for UI components.")
                    continue

                start_offset = exp_idx * params_per_experiment_slot
                
                if start_offset + params_per_experiment_slot > len(all_field_values_from_ui):
                    all_statuses.append(f"Experiment {exp_idx + 1}: Not enough data fields provided for this experiment index.")
                    continue
                    
                current_exp_args = list(all_field_values_from_ui[start_offset : start_offset + params_per_experiment_slot])
                
                current_exp_input_params = current_exp_args[:num_input_cols]
                current_exp_objective_params = current_exp_args[num_input_cols:] # Corrected variable name from num_inputCols

                if any(val is None or str(val).strip() == "" for val in current_exp_input_params):
                    all_statuses.append(f"Experiment {exp_idx + 1}: Skipped (missing one or more input parameter values).")
                    continue
                
                if any(val is None or str(val).strip() == "" for val in current_exp_objective_params):
                    all_statuses.append(f"Experiment {exp_idx + 1}: Skipped (missing one or more objective values).")
                    continue
                
                status = save_suggestion_result(*current_exp_args)
                all_statuses.append(f"Experiment {exp_idx + 1}: {status}")
                if "successfully" in status.lower():
                    saved_any = True
            
            if not all_statuses:
                return "No experiments were processed for saving (e.g., all were skipped due to missing data).", load_data()

            final_status_message = "\n".join(all_statuses)
            if saved_any:
                 final_status_message += "\n\nData table refreshed."
            
            return final_status_message, load_data()

        bulk_save_input_components = [num_experiments_input]
        for i in range(MAX_SUGGESTIONS_DISPLAY):
            exp_set = experiment_suggestion_sets_ui[i]
            for input_name in ORDERED_INPUT_COLS_FOR_UI:
                bulk_save_input_components.append(exp_set['input_params'][input_name])
            for obj_name in ORDERED_OBJECTIVE_COLS_FOR_UI:
                bulk_save_input_components.append(exp_set['objective_inputs'][obj_name])

        save_all_btn_ui.click(
            fn=collect_and_save_all_visible_results_ui,
            inputs=bulk_save_input_components,
            outputs=[bulk_save_status_ui, data_display]
        )

        for exp_idx, exp_set in enumerate(experiment_suggestion_sets_ui):
            current_save_inputs = []
            for input_name in ORDERED_INPUT_COLS_FOR_UI:
                current_save_inputs.append(exp_set['input_params'][input_name])
            for obj_name in ORDERED_OBJECTIVE_COLS_FOR_UI:
                current_save_inputs.append(exp_set['objective_inputs'][obj_name])
            
            exp_set['save_btn'].click(
                fn=save_suggestion_result,
                inputs=current_save_inputs, 
                outputs=exp_set['save_status']
            )

    def load_all_analysis_plots_for_ui(): 
        """Loads all data analysis plots and statistics for the UI."""
        stats = get_statistics()
        plot_yield = plot_yield_progression()
        plot_pareto = plot_pareto_front()
        plot_corr = plot_correlation_heatmap()
        plot_feat_imp = plot_feature_importance()
        plot_feat_dist = plot_feature_distributions()
        plot_cat_an = plot_categorical_analysis()
        plot_scatter = plot_scatter_analysis()
        plot_sugg_freq = plot_suggestion_frequency() 
        return (stats, plot_yield, plot_pareto, plot_corr, plot_feat_imp, 
                plot_feat_dist, plot_cat_an, plot_scatter, plot_sugg_freq) 

    with gr.Tab("📊 Data Analysis & Visualization"):
        gr.Markdown("## 🔬 Comprehensive Chemical Reaction Analysis")
        load_analysis_btn = gr.Button("🚀 Load/Refresh All Analysis Visualizations")
        
        with gr.Row():
            stats_markdown_ui = gr.Markdown("Statistics will appear here.")
        
        gr.Markdown("### Objective Progression & Pareto Front")
        with gr.Row():
            yield_prog_plot_ui = gr.Image(label="Objective Progression & Cumulative Best", interactive=False) 
            pareto_plot_ui = gr.Image(label="Pareto Front", interactive=False)
        
        gr.Markdown("### Correlations & Feature Insights") 
        with gr.Row():
            corr_heatmap_plot_ui = gr.Image(label="Correlation Heatmap", interactive=False)
            feat_importance_plot_ui = gr.Image(label="Feature Importance", interactive=False)

        with gr.Row(): 
            feat_dist_plot_ui = gr.Image(label="Feature Distributions", interactive=False)
            cat_analysis_plot_ui = gr.Image(label="Categorical Analysis", interactive=False)
        
        gr.Markdown("### Parameter Analysis") 
        with gr.Row():
            scatter_analysis_plot_ui = gr.Image(label="Scatter Analysis (Objective vs. Correlated Input)", interactive=False) 
            suggestion_freq_plot_ui = gr.Image(label="Suggested Parameter Frequency/Distribution", interactive=False) 

        analysis_outputs_list_ui = [
            stats_markdown_ui, 
            yield_prog_plot_ui, 
            pareto_plot_ui, 
            corr_heatmap_plot_ui, 
            feat_importance_plot_ui,
            feat_dist_plot_ui, 
            cat_analysis_plot_ui, 
            scatter_analysis_plot_ui,
            suggestion_freq_plot_ui 
        ]
        
        load_analysis_btn.click(
            fn=load_all_analysis_plots_for_ui,
            inputs=None,
            outputs=analysis_outputs_list_ui,
            show_progress="full"
        )

    with gr.Tab("🗃️ Dataset Management"):
        gr.Markdown("## 💾 Manage Your Dataset")
        
        gr.Markdown("### Upload New Dataset")
        gr.Markdown("Upload a CSV file. Ensure it has the correct columns. This will **replace** the current dataset.")
        upload_button = gr.UploadButton("📁 Upload CSV Dataset", file_types=[".csv"])
        upload_status_text = gr.Textbox(label="Upload Status", interactive=False)
        
        gr.Markdown("### Download Current Dataset")
        download_button = gr.DownloadButton(label="📄 Download Current Dataset as CSV", value=download_csv)

        def upload_and_refresh(file_obj):
            """Handles CSV upload and refreshes the data display table."""
            status = upload_csv(file_obj) 
            new_df = load_data() 
            return status, new_df

        upload_button.upload(
            fn=upload_and_refresh, 
            inputs=[upload_button], 
            outputs=[upload_status_text, data_display] 
        )

    def initial_load_actions():
        """Performs initial actions on app load: sets number input, updates suggestion visibility, and loads analysis plots."""
        current_df_for_load = load_data()
        num_current_experiments = len(current_df_for_load)
        
        initial_num_exp_val = MIN_SAMPLES_FOR_BO - num_current_experiments
        if initial_num_exp_val < 1:
            initial_num_exp_val = 1
        # initial_num_exp_val = min(initial_num_exp_val, MAX_SUGGESTIONS_DISPLAY) # No longer cap initial value by MAX_SUGGESTIONS_DISPLAY for the input field itself

        suggestion_visibility_updates = update_suggestions_visibility_ui(initial_num_exp_val)
        analysis_plot_updates = load_all_analysis_plots_for_ui()
        
        # Order: Number input update, then visibility updates, then plot updates
        return tuple([gr.update(value=initial_num_exp_val)] + suggestion_visibility_updates + list(analysis_plot_updates))

    app.load(
        fn=initial_load_actions,
        inputs=None, # No direct inputs from UI for this initial load
        outputs=[num_experiments_input] + input_change_outputs + analysis_outputs_list_ui
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7864, share=False)