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
                            "content": "You are a helpful AI assistant.",
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
            max_tokens = gr.Slider(
                minimum=1, maximum=500, value=50, step=1, label="Max New Tokens"
            )
            strategy = gr.Radio(
                choices=["greedy", "sampling"], value="sampling", label="Strategy"
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
            stop_at_token_enabled = gr.Checkbox(
                label="Stop at Token ID",
                value=False,
                info="Stop generation when specific token ID is encountered",
            )

        stop_token_id = gr.Number(
            label="Token ID",
            placeholder="151643 (Example stop token for Qwen3)",
            visible=False,
            precision=0,
            info="Stop generation at",
        )

        # Logging Settings
        gr.Markdown("### Logging & Visualization")
        with gr.Row():
            log_top_k = gr.Slider(
                minimum=1,
                maximum=50,
                value=10,
                step=1,
                label="Log Top-K Tokens",
                info="Number of top candidates to log per step",
            )
            heatmap_ranks = gr.Slider(
                minimum=1,
                maximum=20,
                value=5,
                step=1,
                label="Heatmap Ranks",
                info="Ranks to show in visualization",
            )

        # Generate Buttons
        with gr.Row():
            generate_button = gr.Button("Generate", variant="primary", size="lg")
            continue_button = gr.Button(
                "Continue",
                variant="secondary",
                size="lg",
                visible=False,
                interactive=False,
            )

        # Output Section
        gr.Markdown("---")
        gr.Markdown("### Output")

        generated_text_output = gr.Textbox(
            label="Generated Text", lines=8, interactive=False, show_copy_button=True
        )

        generation_stats = gr.Code(
            label="Generation Statistics", language="json", interactive=False
        )

        # Visualizations
        gr.Markdown("### Visualizations")
        viz_plot_heatmap = gr.Plot(label="Probability Heatmap")
        viz_plot_confidence = gr.Plot(label="Confidence Analysis")

        # Export Section (appears after generation)
        gr.Markdown("### Export")
        download_button = gr.DownloadButton(
            label="Download JSON", visible=False, variant="secondary", size="lg"
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

        def generate_handler(
            mode,
            prompt,
            chat_msgs,
            max_new_tokens,
            strat,
            temp,
            topk,
            topp,
            log_topk,
            heatmap_r,
            stop_enabled,
            stop_token,
        ):
            """Handle text generation with streaming updates."""
            model = model_manager.get_model()
            tokenizer = model_manager.get_tokenizer()
            device = model_manager.get_device()

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
                    None,         # original_mode
                    None,         # original_prompt
                    None,         # original_messages
                )
                return

            start_time = time.time()
            logger.info(
                f"Logging mode generation started: mode={mode}, max_tokens={max_new_tokens}, strategy={strat}"
            )

            try:
                # Create tracer
                tracer = LLMTracer(model, tokenizer, device)

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
                # Only pass stop_token_id if enabled and value is provided
                stop_token_param = None
                if stop_enabled and stop_token is not None:
                    stop_token_param = int(stop_token)
                    logger.debug(f"Stop token enabled: {stop_token_param}")

                for current_text, step_num, is_complete in tracer.generate_stream(
                    max_new_tokens=max_new_tokens,
                    strategy=strat,
                    temperature=temp,
                    top_k=topk,
                    top_p=topp,
                    log_top_k=log_topk,
                    log_all_logits=False,
                    stop_token_id=stop_token_param,
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

                    yield (
                        generated_text,
                        json.dumps(progress_stats, indent=2),
                        None,
                        None,
                        None,
                        gr.update(),
                        gr.update(),  # Continue button (no change during generation)
                        None,         # original_mode
                        None,         # original_prompt
                        None,         # original_messages
                    )

                # After completion, generate final stats and visualizations
                generation_time = time.time() - start_time
                logger.info(
                    f"Generation complete: {len(tracer.history)} tokens in {generation_time:.2f}s"
                )

                stats = get_generation_stats(tracer)

                # Create visualizations
                logger.debug(f"Creating visualizations (heatmap_ranks={heatmap_r})")
                figures = plot_probability_visualizations(tracer, top_k=heatmap_r)
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
                    gr.update(visible=True, interactive=True),  # Show Continue button
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
            log_topk,
            heatmap_r,
            stop_enabled,
            stop_token,
        ):
            """Handle continuation of existing generation with new parameters."""
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
                )
                return

            start_time = time.time()
            logger.info(
                f"Continuing generation: max_tokens={max_new_tokens}, strategy={strat}"
            )

            try:
                # Store how many tokens we had before continuation
                tokens_before = len(tracer.history)

                # Continue generation with NEW parameters
                stop_token_param = None
                if stop_enabled and stop_token is not None:
                    stop_token_param = int(stop_token)
                    logger.debug(f"Stop token enabled: {stop_token_param}")

                for current_text, step_num, is_complete in tracer.generate_stream(
                    max_new_tokens=max_new_tokens,
                    strategy=strat,
                    temperature=temp,
                    top_k=topk,
                    top_p=topp,
                    log_top_k=log_topk,
                    log_all_logits=False,
                    stop_token_id=stop_token_param,
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
                        None,
                        gr.update(),
                        gr.update(),
                    )

                # After completion, generate final stats and visualizations
                generation_time = time.time() - start_time
                logger.info(
                    f"Continuation complete: {len(tracer.history)} total tokens ({step_num} new tokens in {generation_time:.2f}s)"
                )

                stats = get_generation_stats(tracer)

                # Create visualizations (shows ALL steps including previous)
                logger.debug(f"Creating visualizations (heatmap_ranks={heatmap_r})")
                figures = plot_probability_visualizations(tracer, top_k=heatmap_r)
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
                )

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
            # No tracer = can't continue
            if tracer is None or original_mode is None:
                return gr.update(visible=False, interactive=False)

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
                return gr.update(visible=False)

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
                return gr.update(value=temp_path, visible=True)

            except Exception as e:
                logger.error(f"Export error: {e}")
                return gr.update(visible=False)

        mode_selector.change(
            fn=update_mode_visibility,
            inputs=[mode_selector],
            outputs=[completion_inputs, chat_inputs],
        )

        stop_at_token_enabled.change(
            fn=lambda x: gr.update(visible=x),
            inputs=[stop_at_token_enabled],
            outputs=[stop_token_id],
        )

        generate_button.click(
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
                log_top_k,
                heatmap_ranks,
                stop_at_token_enabled,
                stop_token_id,
            ],
            outputs=[
                generated_text_output,
                generation_stats,
                viz_plot_heatmap,
                viz_plot_confidence,
                tracer_state,
                download_button,
                continue_button,
                original_mode_state,
                original_prompt_state,
                original_messages_state,
            ],
        )

        continue_button.click(
            fn=continue_handler,
            inputs=[
                tracer_state,
                max_tokens,
                strategy,
                temperature,
                top_k,
                top_p,
                log_top_k,
                heatmap_ranks,
                stop_at_token_enabled,
                stop_token_id,
            ],
            outputs=[
                generated_text_output,
                generation_stats,
                viz_plot_heatmap,
                viz_plot_confidence,
                tracer_state,
                download_button,
                continue_button,
            ],
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
