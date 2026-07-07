"""JSON log analysis tab: visualize previously exported generation logs."""

from __future__ import annotations

import json
import traceback

import gradio as gr
import numpy as np

from miru_tracer.core.logging_config import get_logger
from miru_tracer.core.schema import parse_log
from miru_tracer.ui.helpers import prob_mode_key
from miru_tracer.visualization.plots import plot_probability_visualizations

logger = get_logger(__name__)


def create_analysis_tab() -> gr.Tab:
    """Create the JSON log analysis tab interface."""
    with gr.Tab("Log Analysis") as tab:
        gr.Markdown(
            "Load and analyze previously exported generation logs. "
            "Both current (v2) and older log formats are supported."
        )

        json_file_input = gr.File(
            label="Upload JSON Log", file_types=[".json"], type="filepath"
        )

        gr.Markdown("### Visualization Settings")
        with gr.Row():
            heatmap_ranks = gr.Number(
                minimum=1,
                value=10,
                precision=0,
                label="Heatmap ranks",
                info="Ranks to show in visualization.",
            )
            probability_mode = gr.Radio(
                choices=["Adjusted (post-temperature)", "Raw (pre-temperature)"],
                value="Adjusted (post-temperature)",
                label="Probability display mode",
                info=(
                    "Adjusted shows the sampling distribution (with temperature), "
                    "Raw shows the model's true confidence."
                ),
            )

        gr.Markdown("### Information")
        log_info_output = gr.Code(label="Metadata", language="json", interactive=False)
        stats_output = gr.Textbox(label="Statistics", lines=6, interactive=False)

        gr.Markdown("### Visualizations")
        viz_plot_heatmap = gr.Plot(label="Probability heatmap")
        viz_plot_confidence = gr.Plot(label="Confidence analysis")

        def analyze_json_log(filepath, heatmap_r, prob_mode):
            """Analyze an uploaded JSON log file (schema v1 or v2)."""
            if filepath is None:
                return None, "No file uploaded", None, None

            def truncate(text, limit=200):
                return text[:limit] + "..." if len(text) > limit else text

            try:
                with open(filepath) as f:
                    data = json.load(f)
                log = parse_log(data)

                metadata = {
                    "schema_version": log.schema_version,
                    "mode": log.mode,
                    "prompt": truncate(log.prompt),
                    "generated_text": truncate(log.generated_text),
                    "timestamp": log.timestamp,
                    "num_steps": log.num_steps,
                    "sampling_params": log.sampling_params,
                }

                if not log.history:
                    return (
                        json.dumps(metadata, indent=2),
                        "No history data found in log",
                        None,
                        None,
                    )

                probs = [step.probability for step in log.history]
                stats_text = (
                    f"Mean: {np.mean(probs):.4f}\n"
                    f"Std Dev: {np.std(probs):.4f}\n"
                    f"Min: {np.min(probs):.4f}\n"
                    f"Max: {np.max(probs):.4f}\n"
                    f"Median: {np.median(probs):.4f}\n"
                    f"Total steps: {len(log.history)}\n"
                )

                # Cap heatmap ranks at what was actually logged
                ranks = min(
                    int(heatmap_r) if heatmap_r else 10,
                    len(log.history[0].top_k_tokens),
                )
                figures = plot_probability_visualizations(
                    log.history,
                    top_k=ranks,
                    probability_mode=prob_mode_key(prob_mode),
                    temperature=log.temperature,
                )
                return (
                    json.dumps(metadata, indent=2),
                    stats_text,
                    figures[0] if figures else None,
                    figures[1] if len(figures) > 1 else None,
                )

            except ValueError as e:
                # parse_log rejects files that aren't Miru Tracer logs
                return "", f"Error: {e}", None, None
            except Exception as e:
                logger.error(f"Log analysis error: {e}", exc_info=True)
                return (
                    "",
                    f"Error analyzing log:\n\n{e}\n\nTraceback:\n{traceback.format_exc()}",
                    None,
                    None,
                )

        for trigger in (
            json_file_input.upload,
            heatmap_ranks.change,
            probability_mode.change,
        ):
            trigger(
                fn=analyze_json_log,
                inputs=[json_file_input, heatmap_ranks, probability_mode],
                outputs=[
                    log_info_output,
                    stats_output,
                    viz_plot_heatmap,
                    viz_plot_confidence,
                ],
            )

    return tab
