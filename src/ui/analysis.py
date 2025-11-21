"""JSON log analysis tab for Gradio UI."""

import gradio as gr
import json
import numpy as np
from typing import Optional, List
from visualization.plots import plot_probability_visualizations
from core.tracer import TokenStep


def create_analysis_tab() -> gr.Tab:
    """
    Create the JSON log analysis tab interface.

    Returns:
        Gradio Tab component
    """
    with gr.Tab("Log Analysis") as tab:
        gr.Markdown("Load and analyze previously exported generation logs.")

        # File Upload
        json_file_input = gr.File(
            label="Upload JSON Log", file_types=[".json"], type="filepath"
        )

        # Visualization Settings
        gr.Markdown("### Visualization Settings")
        with gr.Row():
            heatmap_ranks = gr.Number(
                minimum=1,
                value=10,
                precision=0,
                label="Heatmap Ranks",
                info="Ranks to show in visualization",
            )
        with gr.Row():
            visualize_all_ranks_checkbox = gr.Checkbox(
                label="Visualize All Ranks (Override Heatmap Ranks)",
                value=False,
                info="WARNING: Shows ALL logged tokens in heatmap. May CRASH YOUR BROWSER with large vocabularies!",
            )
        with gr.Row():
            probability_mode = gr.Radio(
                choices=["Adjusted (Post-Temperature)", "Raw (Pre-Temperature)"],
                value="Adjusted (Post-Temperature)",
                label="Probability Display Mode",
                info="Adjusted shows sampling distribution (with temperature), Raw shows model's true confidence. Hover over heatmap cells to see both values.",
            )

        # Output Section
        gr.Markdown("### Log Information")

        log_info_output = gr.Code(
            label="Log Metadata", language="json", interactive=False
        )

        gr.Markdown("### Statistics")

        stats_output = gr.Textbox(
            label="Probability Statistics", lines=10, interactive=False
        )

        # Visualizations
        gr.Markdown("### Visualizations")
        viz_plot_heatmap = gr.Plot(label="Probability Heatmap")
        viz_plot_confidence = gr.Plot(label="Confidence Analysis")

        def analyze_json_log(filepath, heatmap_r, visualize_all_ranks_check, prob_mode):
            """Analyze an uploaded JSON log file."""
            if filepath is None:
                return None, "No file uploaded", None, None

            # Convert probability mode from UI choice to internal format
            internal_prob_mode = "raw" if "Raw" in prob_mode else "adjusted"

            # Determine actual visualization parameters based on checkbox
            actual_heatmap_ranks = int(heatmap_r) if heatmap_r else 10

            if visualize_all_ranks_check:
                # Will be set to the number of logged tokens after loading the data
                actual_heatmap_ranks = None  # Placeholder, will be updated below

            try:
                with open(filepath, "r") as f:
                    data = json.load(f)

                # Extract metadata
                metadata = {
                    "mode": data.get("mode", "unknown"),
                    "prompt": (
                        data.get("prompt", "")[:200] + "..."
                        if len(data.get("prompt", "")) > 200
                        else data.get("prompt", "")
                    ),
                    "generated_text": (
                        data.get("generated_text", "")[:200] + "..."
                        if len(data.get("generated_text", "")) > 200
                        else data.get("generated_text", "")
                    ),
                    "timestamp": data.get("timestamp", "unknown"),
                    "num_steps": data.get("num_steps", 0),
                }

                # Extract temperature from metadata if available (default to 1.0)
                temperature = data.get("temperature", 1.0)

                # Calculate statistics
                history = data.get("history", [])
                if not history:
                    return (
                        json.dumps(metadata, indent=2),
                        "No history data found in log",
                        None,
                        None,
                    )

                probs = [step["probability"] for step in history]

                stats_text = "Probability Statistics:\n"
                stats_text += f"  Mean: {np.mean(probs):.4f}\n"
                stats_text += f"  Std Dev: {np.std(probs):.4f}\n"
                stats_text += f"  Min: {np.min(probs):.4f}\n"
                stats_text += f"  Max: {np.max(probs):.4f}\n"
                stats_text += f"  Median: {np.median(probs):.4f}\n"
                stats_text += f"\n"
                stats_text += f"Total Steps: {len(history)}\n"

                # Reconstruct tracer history for visualization
                reconstructed_history = []
                for step_data in history:
                    token_step = TokenStep(
                        step=step_data["step"],
                        token_id=step_data["token_id"],
                        token_text=step_data["token_text"],
                        probability=step_data["probability"],
                        top_k_tokens=step_data["top_k_tokens"],
                        top_k_probs=step_data["top_k_probs"],
                        top_k_texts=step_data["top_k_texts"],
                        raw_probability=step_data.get("raw_probability", step_data["probability"]),  # Backward compatibility
                        top_k_raw_probs=step_data.get("top_k_raw_probs", step_data["top_k_probs"]),  # Backward compatibility
                        all_logits=None,  # Not needed for visualization
                        token_text_raw=step_data.get(
                            "token_text_raw"
                        ),  # Backward compatibility
                        top_k_texts_raw=step_data.get(
                            "top_k_texts_raw"
                        ),  # Backward compatibility
                    )
                    reconstructed_history.append(token_step)

                # Create a mock tracer object with history
                class MockTracer:
                    def __init__(self, history):
                        self.history = history

                mock_tracer = MockTracer(reconstructed_history)

                # Finalize visualization parameters
                if visualize_all_ranks_check and reconstructed_history:
                    # Use all logged tokens (length of top_k_tokens in first step)
                    actual_heatmap_ranks = len(reconstructed_history[0].top_k_tokens)

                # Generate visualizations
                figures = plot_probability_visualizations(
                    mock_tracer,
                    top_k=actual_heatmap_ranks,
                    probability_mode=internal_prob_mode,
                    temperature=temperature,
                )
                plot_heatmap = figures[0] if len(figures) > 0 else None
                plot_confidence = figures[1] if len(figures) > 1 else None

                return (
                    json.dumps(metadata, indent=2),
                    stats_text,
                    plot_heatmap,
                    plot_confidence,
                )

            except Exception as e:
                error_msg = f"Error analyzing log:\n\n{str(e)}"
                import traceback

                error_msg += f"\n\nTraceback:\n{traceback.format_exc()}"
                return "", error_msg, None, None

        # Visibility control function
        def update_number_visibility(viz_all_check):
            """Hide heatmap_ranks number input when Visualize All Ranks checkbox is enabled."""
            return gr.update(visible=not viz_all_check)

        # Wire up number input visibility control
        visualize_all_ranks_checkbox.change(
            fn=update_number_visibility,
            inputs=[visualize_all_ranks_checkbox],
            outputs=[heatmap_ranks],
        )

        json_file_input.upload(
            fn=analyze_json_log,
            inputs=[json_file_input, heatmap_ranks, visualize_all_ranks_checkbox, probability_mode],
            outputs=[
                log_info_output,
                stats_output,
                viz_plot_heatmap,
                viz_plot_confidence,
            ],
        )

        # Also update visualization when settings change
        heatmap_ranks.change(
            fn=analyze_json_log,
            inputs=[json_file_input, heatmap_ranks, visualize_all_ranks_checkbox, probability_mode],
            outputs=[
                log_info_output,
                stats_output,
                viz_plot_heatmap,
                viz_plot_confidence,
            ],
        )

        visualize_all_ranks_checkbox.change(
            fn=analyze_json_log,
            inputs=[json_file_input, heatmap_ranks, visualize_all_ranks_checkbox, probability_mode],
            outputs=[
                log_info_output,
                stats_output,
                viz_plot_heatmap,
                viz_plot_confidence,
            ],
        )

        probability_mode.change(
            fn=analyze_json_log,
            inputs=[json_file_input, heatmap_ranks, visualize_all_ranks_checkbox, probability_mode],
            outputs=[
                log_info_output,
                stats_output,
                viz_plot_heatmap,
                viz_plot_confidence,
            ],
        )

    return tab
