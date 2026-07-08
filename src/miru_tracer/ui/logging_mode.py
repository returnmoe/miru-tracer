"""Logging mode tab: autonomous generation with full probability logging."""

from __future__ import annotations

import json
import time
import traceback

import gradio as gr

from miru_tracer.core.logging_config import get_logger
from miru_tracer.core.model_manager import ModelManager
from miru_tracer.core.tracer import LLMTracer
from miru_tracer.ui.helpers import (
    DEFAULT_CHAT_JSON,
    ChatValidationError,
    ExportManager,
    parse_chat_messages,
    prob_mode_key,
    toggle_mode_visibility,
    ui_sampling_params,
)
from miru_tracer.visualization.plots import (
    get_generation_stats,
    plot_probability_visualizations,
)

logger = get_logger(__name__)


def create_logging_mode_tab(model_manager: ModelManager) -> gr.Tab:
    """Create the logging mode tab interface."""

    exports = ExportManager("log")

    with gr.Tab("Logging Mode") as tab, gr.Column(elem_classes="miru-narrow"):
        gr.Markdown(
            "Generate text with complete token probability logging and visualization."
        )

        mode_selector = gr.Radio(
            choices=["Completion", "Chat"],
            value="Completion",
            label="Mode",
            info="Choose between direct text completion or chat format.",
        )

        with gr.Group() as completion_inputs:
            prompt_input = gr.Textbox(
                label="Prompt",
                placeholder="Enter your prompt",
                lines=2,
                value="The future of artificial intelligence is",
            )

        with gr.Group(visible=False) as chat_inputs:
            gr.Markdown(
                "Edit the JSON to add/remove/modify messages. "
                "Supported roles: system, user, assistant"
            )
            chat_messages = gr.Code(
                label="Chat (JSON)",
                language="json",
                lines=10,
                value=DEFAULT_CHAT_JSON,
            )

        with gr.Group():
            with gr.Row():
                max_tokens = gr.Number(
                    minimum=1, value=20, label="Maximum new tokens", precision=0
                )
                strategy = gr.Radio(
                    choices=["greedy", "sampling"], value="greedy", label="Strategy"
                )
            with gr.Row():
                temperature = gr.Slider(0.1, 2.0, value=1.0, step=0.1, label="Temperature")
                top_k = gr.Slider(1, 100, value=50, step=1, label="Top-K")
                top_p = gr.Slider(0.01, 1.0, value=0.9, step=0.01, label="Top-P")
            with gr.Accordion("Logging & visualization options", open=False):
                with gr.Row():
                    log_top_k = gr.Number(
                        minimum=1,
                        value=10,
                        precision=0,
                        label="Log Top-K tokens",
                        info="Top candidates recorded per step.",
                    )
                    heatmap_ranks = gr.Number(
                        minimum=1,
                        value=10,
                        precision=0,
                        label="Heatmap ranks",
                        info="Ranks shown in the heatmap.",
                    )
                with gr.Row():
                    log_full_probs_checkbox = gr.Checkbox(
                        label="Log full probabilities",
                        value=False,
                        info="Whole-vocabulary distribution per step (large "
                        "exports; enables exact entropy).",
                    )
                    stop_at_eos_checkbox = gr.Checkbox(
                        label="Stop at EOS",
                        value=True,
                        info="Stop when an end-of-sequence token appears.",
                    )

        with gr.Row():
            generate_button = gr.Button("Generate", variant="primary", size="lg")
            continue_button = gr.Button(
                "Continue", variant="secondary", size="lg", interactive=False
            )
            stop_button = gr.Button("Stop", variant="stop", size="lg", interactive=False)

        gr.Markdown("### Output")
        generated_text_output = gr.Textbox(
            label="Generated Text", lines=8, interactive=False, buttons=["copy"]
        )
        generation_stats = gr.Code(
            label="Generation Statistics", language="json", interactive=False
        )

        gr.Markdown("### Visualizations")
        with gr.Row():
            probability_mode = gr.Radio(
                choices=["Adjusted (post-temperature)", "Raw (pre-temperature)"],
                value="Adjusted (post-temperature)",
                label="Probability display mode",
                info=(
                    "Adjusted shows the sampling distribution (with temperature), "
                    "Raw shows the model's true confidence. Hover over heatmap "
                    "cells to see both values."
                ),
            )
        viz_plot_heatmap = gr.Plot(label="Probability heatmap")
        viz_plot_confidence = gr.Plot(label="Confidence analysis")

        gr.Markdown("### Export")
        download_button = gr.DownloadButton(
            label="Download JSON",
            interactive=False,
            variant="secondary",
            size="lg",
        )

        # State: the live tracer plus the inputs it was started from
        # (used to invalidate Continue when the user edits the prompt).
        tracer_state = gr.State(value=None)
        original_inputs_state = gr.State(value=None)  # (mode, prompt, messages)

        # ------------------------------------------------------------ helpers

        def finalize(tracer, params, heatmap_r, prob_mode):
            """Stats, plots, and download for a finished (or stopped) run."""
            stats_json = json.dumps(get_generation_stats(tracer.history), indent=2)
            ranks = int(heatmap_r) if heatmap_r else 10
            if tracer.history:
                ranks = min(ranks, len(tracer.history[0].top_k_tokens))
            figures = plot_probability_visualizations(
                tracer.history,
                top_k=ranks,
                probability_mode=prob_mode_key(prob_mode),
                temperature=params.temperature,
            )
            heatmap = figures[0] if figures else None
            confidence = figures[1] if len(figures) > 1 else None
            download = exports.prepare(tracer.export_to_dict(params))
            return stats_json, heatmap, confidence, download

        def run_generation(tracer, params, max_new_tokens, log_topk, log_full, stop_at_eos):
            """Stream steps; yields (text, progress_json) per token."""
            tokens_before = len(tracer.history)
            for step_data in tracer.generate_stream(
                max_new_tokens=int(max_new_tokens),
                params=params,
                log_top_k=max(int(log_topk or 10), 1),
                log_full_probs=bool(log_full),
                stop_at_eos=stop_at_eos,
            ):
                progress = {
                    "step": len(tracer.history),
                    "new_tokens": len(tracer.history) - tokens_before,
                    "last_token": step_data.token_text,
                }
                yield tracer.get_generated_text(), json.dumps(progress, indent=2)

        def error_yield(message, keep_tracer=None, keep_originals=None):
            return (
                message,
                None,
                None,
                None,
                keep_tracer,
                gr.update() if keep_tracer else gr.update(interactive=False),
                gr.update(interactive=keep_tracer is not None),
                gr.update(interactive=False),
                keep_originals,
            )

        # ----------------------------------------------------------- handlers

        def generate_handler(
            mode,
            prompt,
            chat_msgs,
            max_new_tokens,
            strat,
            temp,
            topk,
            topp,
            stop_at_eos,
            log_topk,
            heatmap_r,
            log_full,
            prob_mode,
        ):
            model = model_manager.get_model()
            tokenizer = model_manager.get_tokenizer()
            device = model_manager.get_device()

            if model is None or tokenizer is None:
                yield error_yield(
                    "Error: No model loaded. Please load a model in the "
                    "Model Loader tab first."
                )
                return
            if max_new_tokens is None or max_new_tokens < 1:
                yield error_yield("Error: Maximum new tokens must be at least 1")
                return

            start_time = time.time()
            try:
                params = ui_sampling_params(strat, temp, topk, topp)
                tracer = LLMTracer(model, tokenizer, device)

                if mode == "Chat":
                    try:
                        messages = parse_chat_messages(chat_msgs)
                    except ChatValidationError as e:
                        yield error_yield(f"Error: {e}")
                        return
                    tracer.reset(messages=messages, mode="chat")
                else:
                    tracer.reset(prompt=prompt, mode="completion")

                logger.info(
                    f"Logging mode generation started: mode={mode}, "
                    f"max_tokens={max_new_tokens}, strategy={strat}"
                )
                originals = (mode, prompt, chat_msgs)

                for text, progress in run_generation(
                    tracer, params, max_new_tokens, log_topk, log_full, stop_at_eos
                ):
                    yield (
                        text,
                        progress,
                        gr.update(),
                        gr.update(),
                        tracer,  # available to the Stop handler mid-run
                        gr.update(),
                        gr.update(),
                        gr.update(interactive=True),  # stop enabled while running
                        originals,
                    )

                stats_json, heatmap, confidence, download = finalize(
                    tracer, params, heatmap_r, prob_mode
                )
                logger.info(
                    f"Generation complete: {len(tracer.history)} tokens in "
                    f"{time.time() - start_time:.2f}s"
                )
                yield (
                    tracer.get_generated_text(),
                    stats_json,
                    heatmap,
                    confidence,
                    tracer,
                    download,
                    gr.update(interactive=True),  # continue available
                    gr.update(interactive=False),
                    originals,
                )
            except Exception as e:
                logger.error(f"Generation error: {e}", exc_info=True)
                yield error_yield(
                    f"Error during generation:\n\n{e}\n\n"
                    f"Traceback:\n{traceback.format_exc()}"
                )

        def continue_handler(
            tracer,
            originals,
            max_new_tokens,
            strat,
            temp,
            topk,
            topp,
            stop_at_eos,
            log_topk,
            heatmap_r,
            log_full,
            prob_mode,
        ):
            if tracer is None:
                yield error_yield("Error: No previous generation to continue from.")
                return
            if max_new_tokens is None or max_new_tokens < 1:
                yield error_yield(
                    "Error: Maximum new tokens must be at least 1", tracer, originals
                )
                return
            if model_manager.get_model() is None or tracer.model is not model_manager.get_model():
                yield error_yield(
                    "Error: Model has been unloaded or changed.\n\n"
                    "Please start a new generation."
                )
                return

            start_time = time.time()
            try:
                params = ui_sampling_params(strat, temp, topk, topp)
                logger.info(f"Continuing generation: max_tokens={max_new_tokens}")

                for text, progress in run_generation(
                    tracer, params, max_new_tokens, log_topk, log_full, stop_at_eos
                ):
                    yield (
                        text,
                        progress,
                        gr.update(),
                        gr.update(),
                        tracer,
                        gr.update(),
                        gr.update(),
                        gr.update(interactive=True),
                        originals,
                    )

                stats_json, heatmap, confidence, download = finalize(
                    tracer, params, heatmap_r, prob_mode
                )
                logger.info(
                    f"Continuation complete: {len(tracer.history)} total tokens "
                    f"in {time.time() - start_time:.2f}s"
                )
                yield (
                    tracer.get_generated_text(),
                    stats_json,
                    heatmap,
                    confidence,
                    tracer,
                    download,
                    gr.update(interactive=True),
                    gr.update(interactive=False),
                    originals,
                )
            except Exception as e:
                logger.error(f"Continuation error: {e}", exc_info=True)
                yield error_yield(
                    f"Error during continuation:\n\n{e}\n\n"
                    f"Traceback:\n{traceback.format_exc()}",
                    tracer,
                    originals,
                )

        def stop_handler(tracer, originals, temp, heatmap_r, prob_mode, strat, topk, topp):
            """Finalize UI state after cancelling a running generation."""
            if tracer is None:
                logger.warning("Stop clicked but no tracer in state")
                return (
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    None,
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    originals,
                )

            tracer.request_stop()
            logger.info(f"Stop requested - finalizing after {len(tracer.history)} tokens")
            try:
                params = ui_sampling_params(strat, temp, topk, topp)
                if tracer.history:
                    stats_json, heatmap, confidence, download = finalize(
                        tracer, params, heatmap_r, prob_mode
                    )
                else:
                    stats_json, heatmap, confidence, download = (
                        "{}",
                        None,
                        None,
                        exports.disabled(),
                    )
                return (
                    tracer.get_generated_text(),
                    stats_json,
                    heatmap,
                    confidence,
                    tracer,  # keep it so Continue works
                    download,
                    gr.update(interactive=True),
                    gr.update(interactive=False),
                    originals,
                )
            except Exception as e:
                logger.error(f"Error in stop_handler: {e}", exc_info=True)
                return (
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    tracer,
                    gr.update(),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    originals,
                )

        def refresh_visualizations(tracer, heatmap_r, prob_mode, temp):
            """Re-render plots with new display settings, no regeneration."""
            if tracer is None or not tracer.history:
                return None, None
            try:
                ranks = min(
                    int(heatmap_r) if heatmap_r else 10,
                    len(tracer.history[0].top_k_tokens),
                )
                figures = plot_probability_visualizations(
                    tracer.history,
                    top_k=ranks,
                    probability_mode=prob_mode_key(prob_mode),
                    temperature=temp,
                )
                return (
                    figures[0] if figures else None,
                    figures[1] if len(figures) > 1 else None,
                )
            except Exception as e:
                logger.error(f"Error refreshing visualizations: {e}", exc_info=True)
                return None, None

        def check_continue_availability(mode, prompt, chat_msgs, originals, tracer):
            """Disable Continue when the inputs no longer match the run."""
            if tracer is None or originals is None:
                return gr.update(interactive=False)
            original_mode, original_prompt, original_messages = originals
            unchanged = mode == original_mode and (
                prompt == original_prompt
                if mode == "Completion"
                else chat_msgs == original_messages
            )
            return gr.update(interactive=unchanged)

        # -------------------------------------------------------------- wiring

        outputs = [
            generated_text_output,
            generation_stats,
            viz_plot_heatmap,
            viz_plot_confidence,
            tracer_state,
            download_button,
            continue_button,
            stop_button,
            original_inputs_state,
        ]

        mode_selector.change(
            fn=toggle_mode_visibility,
            inputs=[mode_selector],
            outputs=[completion_inputs, chat_inputs],
        )

        for viz_input in (probability_mode, heatmap_ranks):
            viz_input.change(
                fn=refresh_visualizations,
                inputs=[tracer_state, heatmap_ranks, probability_mode, temperature],
                outputs=[viz_plot_heatmap, viz_plot_confidence],
            )

        generate_event = generate_button.click(
            fn=generate_handler,
            inputs=[
                mode_selector,
                prompt_input,
                chat_messages,
                max_tokens,
                strategy,
                temperature,
                top_k,
                top_p,
                stop_at_eos_checkbox,
                log_top_k,
                heatmap_ranks,
                log_full_probs_checkbox,
                probability_mode,
            ],
            outputs=outputs,
        )
        continue_event = continue_button.click(
            fn=continue_handler,
            inputs=[
                tracer_state,
                original_inputs_state,
                max_tokens,
                strategy,
                temperature,
                top_k,
                top_p,
                stop_at_eos_checkbox,
                log_top_k,
                heatmap_ranks,
                log_full_probs_checkbox,
                probability_mode,
            ],
            outputs=outputs,
        )
        stop_button.click(
            fn=stop_handler,
            inputs=[
                tracer_state,
                original_inputs_state,
                temperature,
                heatmap_ranks,
                probability_mode,
                strategy,
                top_k,
                top_p,
            ],
            outputs=outputs,
            cancels=[generate_event, continue_event],
        )

        for input_component in (mode_selector, prompt_input, chat_messages):
            input_component.change(
                fn=check_continue_availability,
                inputs=[
                    mode_selector,
                    prompt_input,
                    chat_messages,
                    original_inputs_state,
                    tracer_state,
                ],
                outputs=[continue_button],
            )

    return tab
