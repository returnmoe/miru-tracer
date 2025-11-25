"""Logging mode tab for Gradio UI."""

import gradio as gr
import json
import time
from typing import Dict, List
from core.models import ModelManager
from core.tracer import LLMTracer
from visualization.plots import plot_probability_visualizations, get_generation_stats
from core.logging_config import get_logger

logger = get_logger(__name__)


def create_logging_mode_tab(model_manager: ModelManager) -> gr.Tab:
    """
    Create the logging mode tab interface.

    Args:
        model_manager: Singleton ModelManager instance

    Returns:
        Gradio Tab component
    """
    with gr.Tab("Logging Mode") as tab:
        gr.Markdown(
            "Generate text with complete token probability logging and visualization."
        )

        # Mode Selection
        mode_selector = gr.Radio(
            choices=["Completion", "Chat"],
            value="Completion",
            label="Mode",
            info="Choose between direct text completion or chat format.",
        )

        # Input Section
        with gr.Group() as completion_inputs:
            prompt_input = gr.Textbox(
                label="Prompt",
                placeholder="Enter your prompt",
                lines=2,
                value="The future of artificial intelligence is",
            )

        with gr.Group(visible=False) as chat_inputs:
            gr.Markdown(
                "Edit the JSON to add/remove/modify messages. Supported roles: system, user, assistant"
            )
            chat_messages = gr.Code(
                label="Chat Messages (JSON)",
                language="json",
                lines=10,
                value=json.dumps(
                    [
                        {
                            "role": "system",
                            "content": "You are a helpful assistant.",
                        },
                        {
                            "role": "user",
                            "content": "What is the future of artificial intelligence?",
                        },
                    ],
                    indent=2,
                ),
            )

        # Settings
        gr.Markdown("### Settings")

        with gr.Row():
            max_tokens = gr.Number(
                minimum=1,
                value=20,
                label="Maximum new tokens",
                precision=0,
            )
            strategy = gr.Radio(
                choices=["greedy", "sampling"], value="greedy", label="Strategy"
            )

        with gr.Row():
            temperature = gr.Slider(
                minimum=0.1, maximum=2.0, value=1.0, step=0.1, label="Temperature"
            )
            top_k = gr.Slider(minimum=1, maximum=100, value=50, step=1, label="Top-K")
            top_p = gr.Slider(
                minimum=0.0, maximum=1.0, value=0.9, step=0.05, label="Top-P"
            )

        with gr.Row():
            stop_at_eos_checkbox = gr.Checkbox(
                label="Stop at EOS",
                value=True,
                info="Stop generation when end-of-sequence token is encountered.",
            )

        # Logging Settings
        gr.Markdown("### Logging & Visualization")

        with gr.Row():
            log_all_logits_checkbox = gr.Checkbox(
                label="Log all logits",
                value=False,
                info="Warning: Logs ENTIRE vocabulary (~600KB per step for 150K vocab). May cause memory issues!",
            )

        with gr.Row():
            log_top_k = gr.Number(
                minimum=1,
                value=10,
                precision=0,
                label="Log Top-K tokens",
                info="Number of top candidates to log per step.",
            )
            heatmap_ranks = gr.Number(
                minimum=1,
                value=10,
                precision=0,
                label="Heatmap ranks",
                info="Ranks to show in visualization.",
            )

        # Generate Buttons
        with gr.Row():
            generate_button = gr.Button("Generate", variant="primary", size="lg")
            continue_button = gr.Button(
                "Continue",
                variant="secondary",
                size="lg",
                visible=True,
                interactive=False,
            )
            stop_button = gr.Button(
                "Stop",
                variant="stop",
                size="lg",
                visible=True,
                interactive=False,
            )

        # Output Section
        gr.Markdown("### Output")

        generated_text_output = gr.Textbox(
            label="Generated Text", lines=8, interactive=False, show_copy_button=True
        )

        generation_stats = gr.Code(
            label="Generation Statistics", language="json", interactive=False
        )

        # Visualizations
        gr.Markdown("### Visualizations")

        with gr.Row():
            probability_mode = gr.Radio(
                choices=["Adjusted (post-temperature)", "Raw (pre-temperature)"],
                value="Adjusted (post-temperature)",
                label="Probability display mode",
                info="Adjusted shows sampling distribution (with temperature), Raw shows model's true confidence. Hover over heatmap cells to see both values.",
            )

        viz_plot_heatmap = gr.Plot(label="Probability Heatmap")
        viz_plot_confidence = gr.Plot(label="Confidence Analysis")

        # Export Section
        gr.Markdown("### Export")

        download_button = gr.DownloadButton(
            label="Download JSON",
            visible=True,
            interactive=False,
            variant="secondary",
            size="lg",
        )

        # State to persist tracer object and original inputs for change detection
        tracer_state = gr.State(value=None)
        original_mode_state = gr.State(value=None)
        original_prompt_state = gr.State(value=None)
        original_messages_state = gr.State(value=None)

        def update_mode_visibility(mode):
            """Update input visibility based on mode."""
            if mode == "Completion":
                return gr.update(visible=True), gr.update(visible=False)
            else:
                return gr.update(visible=False), gr.update(visible=True)

        def update_number_visibility(log_all_check):
            """Hide Log Top-K input when Log All Logits is enabled."""
            # Hide log_top_k if Log All Logits is checked
            log_visible = not log_all_check
            # Always show heatmap_ranks (user controls visualization)
            return gr.update(visible=log_visible), gr.update(visible=True)

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
            log_all_logits_check,
            prob_mode,
        ):
            """Handle text generation with streaming updates."""
            model = model_manager.get_model()
            tokenizer = model_manager.get_tokenizer()
            device = model_manager.get_device()

            # Use selected probability mode for initial visualization (also hot-toggleable after)
            internal_prob_mode = "raw" if "Raw" in prob_mode else "adjusted"

            # Validate max_new_tokens
            if max_new_tokens is None or max_new_tokens < 1:
                error_msg = "Error: Maximum new tokens must be at least 1"
                logger.warning(f"Invalid max_new_tokens: {max_new_tokens}")
                yield (
                    error_msg,
                    None,
                    None,
                    None,
                    None,
                    gr.update(),
                    gr.update(),
                    gr.update(visible=False),
                    None,
                    None,
                    None,
                )
                return

            # Determine actual logging and visualization parameters based on checkboxes
            actual_log_top_k = int(log_topk) if log_topk else 10
            actual_log_all_logits = log_all_logits_check

            if log_all_logits_check:
                # Override: log entire vocabulary
                actual_log_all_logits = True
                # When logging all logits, log all tokens for visualization
                actual_log_top_k = len(tokenizer) if tokenizer else 50

            # Cap heatmap ranks at what was actually logged
            actual_heatmap_ranks = min(
                int(heatmap_r) if heatmap_r else 10, actual_log_top_k
            )

            if model is None or tokenizer is None:
                error_msg = "Error: No model loaded. Please load a model in the Model Loader tab first."
                logger.warning("Generation attempted without loaded model")
                yield (
                    error_msg,
                    None,
                    None,
                    None,
                    None,
                    gr.update(),
                    gr.update(),  # Continue button
                    gr.update(visible=False),  # Stop button
                    None,  # original_mode
                    None,  # original_prompt
                    None,  # original_messages
                )
                return

            start_time = time.time()
            logger.info(
                f"Logging mode generation started: mode={mode}, max_tokens={max_new_tokens}, strategy={strat}"
            )

            try:
                # Create tracer
                tracer = LLMTracer(model, tokenizer, device)

                # Clear any previous stop flag
                tracer.clear_stop_flag()

                # Setup input based on mode
                if mode == "Chat":
                    try:
                        messages = json.loads(chat_msgs)
                        if not isinstance(messages, list):
                            logger.error("Chat messages must be a JSON array")
                            yield (
                                "Error: Chat messages must be a JSON array",
                                None,
                                None,
                                None,
                                None,
                                gr.update(),
                                gr.update(),
                                gr.update(visible=False),
                                None,
                                None,
                                None,
                            )
                            return
                        for msg in messages:
                            if (
                                not isinstance(msg, dict)
                                or "role" not in msg
                                or "content" not in msg
                            ):
                                logger.error(
                                    "Invalid chat message format (missing role or content)"
                                )
                                yield (
                                    "Error: Each message must have 'role' and 'content' fields",
                                    None,
                                    None,
                                    None,
                                    None,
                                    gr.update(),
                                    gr.update(),
                                    gr.update(visible=False),
                                    None,
                                    None,
                                    None,
                                )
                                return
                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON in chat messages: {str(e)}")
                        yield (
                            f"Error: Invalid JSON in chat messages: {str(e)}",
                            None,
                            None,
                            None,
                            None,
                            gr.update(),
                            gr.update(),
                            gr.update(visible=False),
                            None,
                            None,
                            None,
                        )
                        return
                    tracer.reset(messages=messages, mode="chat")
                    logger.debug(f"Chat mode: {len(messages)} messages")
                else:
                    tracer.reset(prompt=prompt, mode="completion")
                    logger.debug(f"Completion mode: prompt_length={len(prompt)} chars")

                # Generate with streaming
                first_step = True
                for current_text, step_num, is_complete in tracer.generate_stream(
                    max_new_tokens=max_new_tokens,
                    strategy=strat,
                    temperature=temp,
                    top_k=topk,
                    top_p=topp,
                    log_top_k=actual_log_top_k,
                    log_all_logits=actual_log_all_logits,
                    stop_at_eos=stop_at_eos,
                ):
                    # Get generated text
                    generated_text = tracer.get_generated_text()

                    # Update with current progress
                    progress_stats = {
                        "step": step_num,
                        "total_tokens": len(tracer.history),
                        "is_complete": is_complete,
                    }

                    logger.debug(
                        f"Generation progress: step={step_num}, complete={is_complete}"
                    )

                    # Set original state on first yield so it's captured before Stop can be clicked
                    # Also yield the tracer so stop_handler can receive it
                    yield (
                        generated_text,
                        json.dumps(progress_stats, indent=2),
                        None,
                        None,
                        tracer,  # Yield tracer so it's available when Stop is clicked
                        gr.update(),
                        gr.update(),  # Continue button (no change during generation)
                        gr.update(
                            interactive=True
                        ),  # Stop button (enable during generation)
                        (
                            mode if first_step else None
                        ),  # original_mode (set on first step)
                        (
                            prompt if first_step else None
                        ),  # original_prompt (set on first step)
                        (
                            chat_msgs if first_step else None
                        ),  # original_messages (set on first step)
                    )
                    first_step = False

                # After completion, generate final stats and visualizations
                generation_time = time.time() - start_time
                logger.info(
                    f"Generation complete: {len(tracer.history)} tokens in {generation_time:.2f}s"
                )

                stats = get_generation_stats(tracer)

                # Create visualizations
                logger.debug(
                    f"Creating visualizations (heatmap_ranks={actual_heatmap_ranks}, mode={internal_prob_mode})"
                )
                figures = plot_probability_visualizations(
                    tracer,
                    top_k=actual_heatmap_ranks,
                    probability_mode=internal_prob_mode,
                    temperature=temp,
                )
                logger.info(f"Visualizations generated: {len(figures)} plots")

                # Return both plots (heatmap and confidence charts)
                plot_output_heatmap = figures[0] if len(figures) > 0 else None
                plot_output_confidence = figures[1] if len(figures) > 1 else None

                # Convert stats to JSON string
                stats_json = json.dumps(stats, indent=2)

                # Prepare download file
                download_update = prepare_download(tracer)

                # Show download button, store tracer and original inputs in state
                # Also enable Continue button
                yield (
                    generated_text,
                    stats_json,
                    plot_output_heatmap,
                    plot_output_confidence,
                    tracer,  # Store tracer in state
                    download_update,  # Show download button with file path
                    gr.update(visible=True, interactive=True),  # Enable Continue button
                    gr.update(
                        interactive=False
                    ),  # Disable Stop button (generation complete)
                    mode,  # Store original mode
                    prompt,  # Store original prompt
                    chat_msgs,  # Store original messages
                )

            except Exception as e:
                error_msg = f"Error during generation:\n\n{str(e)}"
                import traceback

                error_msg += f"\n\nTraceback:\n{traceback.format_exc()}"
                logger.error(f"Generation error: {str(e)}", exc_info=True)
                yield (
                    error_msg,
                    None,
                    None,
                    None,
                    None,
                    gr.update(),
                    gr.update(),  # Continue button
                    gr.update(interactive=False),  # Stop button
                    None,  # original_mode
                    None,  # original_prompt
                    None,  # original_messages
                )

        def continue_handler(
            tracer,
            max_new_tokens,
            strat,
            temp,
            topk,
            topp,
            stop_at_eos,
            log_topk,
            heatmap_r,
            log_all_logits_check,
            prob_mode,
        ):
            """Handle continuation of existing generation with new parameters."""
            # Use selected probability mode for visualization (also hot-toggleable after)
            internal_prob_mode = "raw" if "Raw" in prob_mode else "adjusted"

            # Validate max_new_tokens
            if max_new_tokens is None or max_new_tokens < 1:
                error_msg = "Error: Maximum new tokens must be at least 1"
                logger.warning(
                    f"Invalid max_new_tokens in continue_handler: {max_new_tokens}"
                )
                yield (
                    error_msg,
                    None,
                    None,
                    None,
                    None,
                    gr.update(),
                    gr.update(),
                    gr.update(visible=False),
                )
                return

            if tracer is None:
                error_msg = "Error: No previous generation to continue from."
                logger.warning("Continue attempted without existing tracer")
                yield (
                    error_msg,
                    None,
                    None,
                    None,
                    None,
                    gr.update(),
                    gr.update(),
                    gr.update(visible=False),
                )
                return

            # Check if model has been unloaded or changed
            current_model = model_manager.get_model()
            if current_model is None or tracer.model is not current_model:
                error_msg = "Error: Model has been unloaded or changed.\n\nPlease start a new generation."
                logger.warning("Continue attempted with unloaded/changed model")
                yield (
                    error_msg,
                    None,
                    None,
                    None,
                    None,  # Clear tracer state
                    gr.update(interactive=False),  # Disable download
                    gr.update(interactive=False),  # Disable continue
                    gr.update(visible=False),  # Hide stop
                )
                return

            # Determine actual logging and visualization parameters based on checkboxes
            actual_log_top_k = int(log_topk) if log_topk else 10
            actual_log_all_logits = log_all_logits_check

            if log_all_logits_check:
                # Override: log entire vocabulary
                actual_log_all_logits = True
                # When logging all logits, log all tokens for visualization
                actual_log_top_k = len(tracer.tokenizer) if tracer.tokenizer else 50

            # Cap heatmap ranks at what was actually logged
            actual_heatmap_ranks = min(
                int(heatmap_r) if heatmap_r else 10, actual_log_top_k
            )

            start_time = time.time()
            logger.info(
                f"Continuing generation: max_tokens={max_new_tokens}, strategy={strat}"
            )

            try:
                # Clear any previous stop flag
                tracer.clear_stop_flag()

                # Store how many tokens we had before continuation
                tokens_before = len(tracer.history)

                # Continue generation with NEW parameters
                for current_text, step_num, is_complete in tracer.generate_stream(
                    max_new_tokens=max_new_tokens,
                    strategy=strat,
                    temperature=temp,
                    top_k=topk,
                    top_p=topp,
                    log_top_k=actual_log_top_k,
                    log_all_logits=actual_log_all_logits,
                    stop_at_eos=stop_at_eos,
                ):
                    # Get generated text
                    generated_text = tracer.get_generated_text()

                    # Update with current progress
                    progress_stats = {
                        "step": tokens_before + step_num,
                        "total_tokens": len(tracer.history),
                        "new_tokens_this_continuation": step_num,
                        "is_complete": is_complete,
                    }

                    logger.debug(
                        f"Continuation progress: step={step_num}, total={len(tracer.history)}, complete={is_complete}"
                    )

                    yield (
                        generated_text,
                        json.dumps(progress_stats, indent=2),
                        None,
                        None,
                        tracer,  # Yield tracer so it's available when Stop is clicked
                        gr.update(),
                        gr.update(),
                        gr.update(
                            interactive=True
                        ),  # Enable Stop button during generation
                    )

                # After completion, generate final stats and visualizations
                generation_time = time.time() - start_time
                logger.info(
                    f"Continuation complete: {len(tracer.history)} total tokens ({step_num} new tokens in {generation_time:.2f}s)"
                )

                stats = get_generation_stats(tracer)

                # Create visualizations (shows ALL steps including previous)
                logger.debug(
                    f"Creating visualizations (heatmap_ranks={heatmap_r}, mode={internal_prob_mode})"
                )
                figures = plot_probability_visualizations(
                    tracer,
                    top_k=heatmap_r,
                    probability_mode=internal_prob_mode,
                    temperature=temp,
                )
                logger.info(f"Visualizations generated: {len(figures)} plots")

                # Return both plots
                plot_output_heatmap = figures[0] if len(figures) > 0 else None
                plot_output_confidence = figures[1] if len(figures) > 1 else None

                # Convert stats to JSON string
                stats_json = json.dumps(stats, indent=2)

                # Prepare download file
                download_update = prepare_download(tracer)

                # Return updated tracer and download button
                yield (
                    generated_text,
                    stats_json,
                    plot_output_heatmap,
                    plot_output_confidence,
                    tracer,  # Return updated tracer
                    download_update,
                    gr.update(),  # Continue button stays visible
                    gr.update(
                        interactive=False
                    ),  # Disable Stop button (generation complete)
                )

            except Exception as e:
                error_msg = f"Error during continuation:\n\n{str(e)}"
                import traceback

                error_msg += f"\n\nTraceback:\n{traceback.format_exc()}"
                logger.error(f"Continuation error: {str(e)}", exc_info=True)
                yield (
                    error_msg,
                    None,
                    None,
                    None,
                    None,
                    gr.update(),
                    gr.update(),
                    gr.update(interactive=False),
                )

        def stop_handler(
            tracer,
            heatmap_ranks,
            prob_mode,
            temp,
            current_mode,
            current_prompt,
            current_messages,
            original_mode,
            original_prompt,
            original_messages,
        ):
            """Handle stop request during generation - finalize UI state."""
            if tracer is None:
                logger.warning("Stop clicked but no tracer in state")
                return (
                    gr.update(),  # generated_text_output
                    gr.update(),  # generation_stats
                    gr.update(),  # viz_plot_heatmap
                    gr.update(),  # viz_plot_confidence
                    None,  # tracer_state
                    gr.update(interactive=False),  # download_button
                    gr.update(interactive=False),  # continue_button
                    gr.update(interactive=False),  # stop_button
                    original_mode,  # original_mode_state (preserve)
                    original_prompt,  # original_prompt_state (preserve)
                    original_messages,  # original_messages_state (preserve)
                )

            logger.info(
                f"Stop button clicked - finalizing after {len(tracer.history)} tokens"
            )

            try:
                # Use selected probability mode for visualization (also hot-toggleable after)
                internal_prob_mode = "raw" if "Raw" in prob_mode else "adjusted"

                # Generate final stats and visualizations from current state
                generated_text = tracer.get_generated_text()
                stats = get_generation_stats(tracer)
                stats_json = json.dumps(stats, indent=2)

                # Create visualizations if we have any generated tokens
                if len(tracer.history) > 0:
                    figures = plot_probability_visualizations(
                        tracer,
                        top_k=heatmap_ranks,
                        probability_mode=internal_prob_mode,
                        temperature=temp,
                    )
                    plot_output_heatmap = figures[0] if len(figures) > 0 else None
                    plot_output_confidence = figures[1] if len(figures) > 1 else None
                else:
                    plot_output_heatmap = None
                    plot_output_confidence = None

                # Prepare download
                download_update = prepare_download(tracer)

                # Preserve originals if they exist, otherwise set them now
                final_original_mode = (
                    original_mode if original_mode is not None else current_mode
                )
                final_original_prompt = (
                    original_prompt if original_prompt is not None else current_prompt
                )
                final_original_messages = (
                    original_messages
                    if original_messages is not None
                    else current_messages
                )

                return (
                    generated_text,  # generated_text_output
                    stats_json,  # generation_stats
                    plot_output_heatmap,  # viz_plot_heatmap
                    plot_output_confidence,  # viz_plot_confidence
                    tracer,  # tracer_state (keep it so Continue works)
                    download_update,  # download_button
                    gr.update(
                        visible=True, interactive=True
                    ),  # continue_button (enable!)
                    gr.update(interactive=False),  # stop_button
                    final_original_mode,  # original_mode_state
                    final_original_prompt,  # original_prompt_state
                    final_original_messages,  # original_messages_state
                )

            except Exception as e:
                logger.error(f"Error in stop_handler: {e}")
                import traceback

                logger.error(traceback.format_exc())

                # Preserve originals even in error case
                final_original_mode = (
                    original_mode if original_mode is not None else current_mode
                )
                final_original_prompt = (
                    original_prompt if original_prompt is not None else current_prompt
                )
                final_original_messages = (
                    original_messages
                    if original_messages is not None
                    else current_messages
                )

                # Return safe defaults on error
                return (
                    gr.update(),  # generated_text_output
                    gr.update(),  # generation_stats
                    gr.update(),  # viz_plot_heatmap
                    gr.update(),  # viz_plot_confidence
                    tracer,  # tracer_state
                    gr.update(),  # download_button
                    gr.update(
                        visible=True, interactive=False
                    ),  # continue_button (visible but disabled)
                    gr.update(interactive=False),  # stop_button
                    final_original_mode,  # original_mode_state
                    final_original_prompt,  # original_prompt_state
                    final_original_messages,  # original_messages_state
                )

        def refresh_visualizations(tracer, heatmap_r, prob_mode, temp):
            """Refresh visualizations with selected probability mode without regenerating text."""
            if (
                tracer is None
                or not hasattr(tracer, "history")
                or len(tracer.history) == 0
            ):
                return None, None

            # Convert probability mode from UI choice to internal format
            internal_prob_mode = "raw" if "Raw" in prob_mode else "adjusted"

            # Determine actual heatmap ranks
            actual_heatmap_ranks = int(heatmap_r) if heatmap_r else 10

            try:
                # Regenerate visualizations from existing tracer with selected mode
                figures = plot_probability_visualizations(
                    tracer,
                    top_k=actual_heatmap_ranks,
                    probability_mode=internal_prob_mode,
                    temperature=temp,
                )
                plot_heatmap = figures[0] if len(figures) > 0 else None
                plot_confidence = figures[1] if len(figures) > 1 else None

                return plot_heatmap, plot_confidence

            except Exception as e:
                logger.error(f"Error refreshing visualizations: {e}")
                import traceback

                logger.error(traceback.format_exc())
                return None, None

        def check_continue_availability(
            current_mode,
            current_prompt,
            current_messages,
            original_mode,
            original_prompt,
            original_messages,
            tracer,
        ):
            """
            Check if Continue button should be enabled based on input changes.

            Disables Continue button if:
            - No tracer exists
            - Mode has changed
            - Prompt has changed (in Completion mode)
            - Messages have changed (in Chat mode)
            """
            # No tracer = can't continue (keep button visible but disabled)
            if tracer is None or original_mode is None:
                return gr.update(visible=True, interactive=False)

            # Mode changed = can't continue
            if current_mode != original_mode:
                logger.debug("Continue disabled: mode changed")
                return gr.update(visible=True, interactive=False)

            # Check if inputs changed based on mode
            if current_mode == "Completion":
                if current_prompt != original_prompt:
                    logger.debug("Continue disabled: prompt changed")
                    return gr.update(visible=True, interactive=False)
            elif current_mode == "Chat":
                if current_messages != original_messages:
                    logger.debug("Continue disabled: messages changed")
                    return gr.update(visible=True, interactive=False)

            # All good - enable continue
            return gr.update(visible=True, interactive=True)

        def prepare_download(tracer):
            """Prepare JSON file for browser download."""
            if tracer is None:
                return gr.update(interactive=False)

            try:
                import tempfile
                from datetime import datetime

                # Get data dictionary
                data = tracer.export_to_dict()

                # Create timestamp for filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                prefix = f"miru_log_{timestamp}_"

                # Create temporary file
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".json", delete=False, prefix=prefix
                ) as f:
                    json.dump(data, f, indent=2)
                    temp_path = f.name

                # Return file path for Gradio to serve
                return gr.update(value=temp_path, interactive=True)

            except Exception as e:
                logger.error(f"Export error: {e}")
                return gr.update(interactive=False)

        mode_selector.change(
            fn=update_mode_visibility,
            inputs=[mode_selector],
            outputs=[completion_inputs, chat_inputs],
        )

        # Wire up number input visibility controls
        log_all_logits_checkbox.change(
            fn=update_number_visibility,
            inputs=[log_all_logits_checkbox],
            outputs=[log_top_k, heatmap_ranks],
        )

        # Hot-toggle probability mode without regenerating
        probability_mode.change(
            fn=refresh_visualizations,
            inputs=[tracer_state, heatmap_ranks, probability_mode, temperature],
            outputs=[viz_plot_heatmap, viz_plot_confidence],
        )

        # Also refresh when heatmap_ranks changes
        heatmap_ranks.change(
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
                log_all_logits_checkbox,
                probability_mode,
            ],
            outputs=[
                generated_text_output,
                generation_stats,
                viz_plot_heatmap,
                viz_plot_confidence,
                tracer_state,
                download_button,
                continue_button,
                stop_button,
                original_mode_state,
                original_prompt_state,
                original_messages_state,
            ],
        )

        continue_event = continue_button.click(
            fn=continue_handler,
            inputs=[
                tracer_state,
                max_tokens,
                strategy,
                temperature,
                top_k,
                top_p,
                stop_at_eos_checkbox,
                log_top_k,
                heatmap_ranks,
                log_all_logits_checkbox,
                probability_mode,
            ],
            outputs=[
                generated_text_output,
                generation_stats,
                viz_plot_heatmap,
                viz_plot_confidence,
                tracer_state,
                download_button,
                continue_button,
                stop_button,
            ],
        )

        stop_button.click(
            fn=stop_handler,
            inputs=[
                tracer_state,
                heatmap_ranks,
                probability_mode,
                temperature,
                mode_selector,
                prompt_input,
                chat_messages,
                original_mode_state,
                original_prompt_state,
                original_messages_state,
            ],
            outputs=[
                generated_text_output,
                generation_stats,
                viz_plot_heatmap,
                viz_plot_confidence,
                tracer_state,
                download_button,
                continue_button,
                stop_button,
                original_mode_state,
                original_prompt_state,
                original_messages_state,
            ],
            cancels=[generate_event, continue_event],
        )

        # Monitor input changes to enable/disable Continue button
        for input_component in [mode_selector, prompt_input, chat_messages]:
            input_component.change(
                fn=check_continue_availability,
                inputs=[
                    mode_selector,
                    prompt_input,
                    chat_messages,
                    original_mode_state,
                    original_prompt_state,
                    original_messages_state,
                    tracer_state,
                ],
                outputs=[continue_button],
            )

    return tab
