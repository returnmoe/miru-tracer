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

        def analyze_json_log(filepath):
            """Analyze an uploaded JSON log file."""
            if filepath is None:
                return None, "No file uploaded", None, None

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
                        all_logits=None,  # Not needed for visualization
                        token_text_raw=step_data.get("token_text_raw"),  # Backward compatibility
                        top_k_texts_raw=step_data.get("top_k_texts_raw"),  # Backward compatibility
                    )
                    reconstructed_history.append(token_step)

                # Create a mock tracer object with history
                class MockTracer:
                    def __init__(self, history):
                        self.history = history

                mock_tracer = MockTracer(reconstructed_history)

                # Generate visualizations
                figures = plot_probability_visualizations(mock_tracer, top_k=5)
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

        json_file_input.upload(
            fn=analyze_json_log,
            inputs=[json_file_input],
            outputs=[
                log_info_output,
                stats_output,
                viz_plot_heatmap,
                viz_plot_confidence,
            ],
        )

    return tab
