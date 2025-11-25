"""Interactive mode tab for Gradio UI with session-based state management."""

import gradio as gr
import pandas as pd
from typing import Optional, Tuple
from core.models import ModelManager
from core.tracer import LLMTracer
from core.session_manager import get_session_manager
from core.logging_config import get_logger

logger = get_logger(__name__)


def create_interactive_mode_tab(model_manager: ModelManager) -> gr.Tab:
    """
    Create the interactive debugging tab interface.

    Args:
        model_manager: Singleton ModelManager instance

    Returns:
        Gradio Tab component
    """
    with gr.Tab("Interactive Mode") as tab:
        gr.Markdown(
            "Generate text step-by-step with full control over each token selection."
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
            import json as json_module

            gr.Markdown(
                "Edit the JSON to add/remove/modify messages. Supported roles: system, user, assistant"
            )
            chat_messages = gr.Code(
                label="Chat (JSON)",
                language="json",
                lines=8,
                value=json_module.dumps(
                    [
                        {
                            "role": "system",
                            "content": "You are a helpful assistant.",
                        },
                        {"role": "user", "content": "Tell me about the future of AI."},
                    ],
                    indent=2,
                ),
            )

        # Settings
        gr.Markdown("### Settings")
        with gr.Row():
            strategy = gr.Radio(
                choices=["greedy", "sampling"],
                value="greedy",
                label="Strategy",
                info="Greedy picks top-1, Sampling randomly samples. Preview shown below - you can change selection before clicking 'Next Step'.",
            )
        with gr.Row():
            temperature = gr.Slider(0.1, 2.0, value=1.0, step=0.1, label="Temperature")
            top_k = gr.Slider(1, 100, value=50, step=1, label="Top-K")
            top_p = gr.Slider(0.0, 1.0, value=0.9, step=0.05, label="Top-P")

        gr.Markdown("### Logging")
        with gr.Row():
            log_top_k = gr.Number(
                minimum=1,
                value=10,
                precision=0,
                label="Log Top-K Tokens",
                info="Number of top candidates to log per step.",
            )
        with gr.Row():
            log_all_logits_checkbox = gr.Checkbox(
                label="Log all logits",
                value=False,
                info="Warning: Logs ENTIRE vocabulary (~600KB per step for 150K vocab). May cause memory issues!",
            )

        gr.Markdown("### Generation Control")
        with gr.Row():
            stop_at_eos_checkbox = gr.Checkbox(
                label="Stop at EOS",
                value=True,
                info="Stop generation when end-of-sequence token is encountered.",
            )

        with gr.Row():
            init_button = gr.Button("Initialize", variant="primary")
            reset_button = gr.Button("Reset", variant="secondary")

        # Step Controls
        gr.Markdown("### Output")

        status_output = gr.Textbox(label="Status", interactive=False, lines=2)

        current_text_output = gr.Textbox(
            label="Output",
            lines=6,
            interactive=False,
            show_copy_button=True,
        )

        with gr.Row():
            continue_tokens = gr.Number(
                minimum=1,
                maximum=1000,
                value=10,
                precision=0,
                label="Continue for N tokens",
                info="Number of tokens to generate (1-1000)",
            )

        with gr.Row():
            continue_button = gr.Button("Run", variant="secondary")
            stop_button = gr.Button(
                "Stop",
                variant="stop",
                size="lg",
                visible=True,
                interactive=False,
            )

        # Token Selection
        gr.Markdown("### Token Selection")

        # Next Token Probabilities
        next_token_probs = gr.Dataframe(
            headers=["Rank", "Token ID", "Token", "Probability"],
            datatype=["number", "number", "str", "number"],
            label="Next Token Candidates",
            interactive=False,
        )

        with gr.Row():
            use_override = gr.Checkbox(
                label="Token ID override",
                value=False,
                info="Enable to specify an arbitrary token ID instead of selecting from the list.",
            )

        with gr.Row():
            token_selector = gr.Radio(
                choices=["0"],
                value="0",
                label="Next Token Preview",
                info="Shows the token that will be generated. You can change this selection before clicking 'Next Step'.",
            )
            token_override = gr.Number(
                label="Token ID",
                value=0,
                precision=0,
                info="The ID for the token to be used in this step.",
                visible=False,
            )

        with gr.Row():
            undo_button = gr.Button("Step Back (Undo)", variant="secondary")
            step_button = gr.Button("Next Step", variant="primary", size="lg")

        gr.Markdown("### Go to Specific Step")
        with gr.Row():
            go_to_step_input = gr.Number(
                minimum=0,
                value=0,
                precision=0,
                label="Target Step Number",
                info="Go back to a specific step (0 = initial state)",
            )
            current_step_display = gr.Textbox(
                label="Current Step",
                value="0",
                interactive=False,
                scale=0,
                min_width=100,
            )

        with gr.Row():
            go_to_step_button = gr.Button("Go to Step", variant="secondary")

        # Export Section
        gr.Markdown("### Export")

        download_button = gr.DownloadButton(
            label="Download JSON",
            visible=True,
            interactive=False,
            variant="secondary",
            size="lg",
        )

        # Session state - only stores simple session ID string
        session_state = gr.State(value=None)

        def prepare_download(session_id: Optional[str]):
            """Prepare JSON file for browser download."""
            if session_id is None:
                return gr.update(interactive=False)

            session_manager = get_session_manager()
            tracer = session_manager.get_tracer(session_id)

            if tracer is None or tracer.input_ids is None:
                return gr.update(interactive=False)

            try:
                import tempfile
                import json
                from datetime import datetime

                # Get data dictionary
                data = tracer.export_to_dict()

                # Create timestamp for filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                prefix = f"miru_interactive_{timestamp}_"

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

        def initialize_tracer(
            mode,
            prompt,
            chat_msgs,
            temp,
            topk,
            topp,
            strat,
            log_topk,
            log_all_logits_check,
        ):
            """Initialize the tracer with a prompt."""
            model = model_manager.get_model()
            tokenizer = model_manager.get_tokenizer()
            device = model_manager.get_device()

            if model is None or tokenizer is None:
                return (
                    "Error: No model loaded",
                    "",
                    None,
                    gr.update(choices=[("Error", "0")], value="0"),
                    None,
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(value="0"),
                )

            try:
                import json as json_module

                session_manager = get_session_manager()

                # Create new session
                session_id = session_manager.create_session(model, tokenizer, device)
                tracer = session_manager.get_tracer(session_id)

                if mode == "Chat":
                    try:
                        messages = json_module.loads(chat_msgs)
                        if not isinstance(messages, list):
                            return (
                                "Error: Chat messages must be a JSON array",
                                "",
                                None,
                                gr.update(choices=[("Error", "0")], value="0"),
                                None,
                                gr.update(interactive=False),
                                gr.update(interactive=False),
                                gr.update(value="0"),
                            )
                        for msg in messages:
                            if (
                                not isinstance(msg, dict)
                                or "role" not in msg
                                or "content" not in msg
                            ):
                                return (
                                    "Error: Each message must have 'role' and 'content' fields",
                                    "",
                                    None,
                                    gr.update(choices=[("Error", "0")], value="0"),
                                    None,
                                    gr.update(interactive=False),
                                    gr.update(interactive=False),
                                    gr.update(value="0"),
                                )
                    except json_module.JSONDecodeError as e:
                        return (
                            f"Error: Invalid JSON: {str(e)}",
                            "",
                            None,
                            gr.update(choices=[("Error", "0")], value="0"),
                            None,
                            gr.update(interactive=False),
                            gr.update(interactive=False),
                            gr.update(value="0"),
                        )

                    tracer.reset(messages=messages, mode="chat")
                    input_text = "\n".join(
                        [f"{m['role']}: {m['content'][:50]}..." for m in messages]
                    )
                else:
                    tracer.reset(prompt=prompt, mode="completion")
                    input_text = prompt

                status = f"Initialized in {mode} mode"

                # Determine actual logging parameters based on checkbox
                actual_log_top_k = int(log_topk) if log_topk else 10
                if log_all_logits_check:
                    actual_log_top_k = len(tokenizer)

                # Get first set of probabilities and CACHE them
                prob_data = tracer.cache_next_probabilities(
                    top_k=actual_log_top_k, temperature=temp
                )

                # Preview what token would be selected based on strategy
                previewed_token_id, preview_method = _preview_next_token(
                    tracer, prob_data, strat, temp, topk, topp
                )

                # Build dataframe
                df, radio_choices = _build_prob_display(prob_data)

                # Prepare download
                download_update = prepare_download(session_id)

                # Select the previewed token in the radio
                radio_update = gr.update(
                    choices=radio_choices, value=str(previewed_token_id)
                )

                logger.info(
                    f"Interactive session initialized: {session_id} (mode={mode})"
                )
                logger.debug(f"Session {session_id} state: {tracer.get_state_info()}")
                logger.debug(
                    f"Previewed next token ({preview_method}): {previewed_token_id}"
                )

                current_step = str(len(tracer.history))
                return (
                    status,
                    "",
                    df,
                    radio_update,
                    session_id,
                    download_update,
                    gr.update(interactive=False),
                    gr.update(value=current_step),
                )

            except Exception as e:
                import traceback

                error = f"Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
                return (
                    error,
                    "",
                    None,
                    gr.update(choices=[("Error", "0")], value="0"),
                    None,
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(value="0"),
                )

        def reset_tracer(session_id):
            """Reset the current tracer."""
            if session_id is not None:
                session_manager = get_session_manager()
                session_manager.delete_session(session_id)
                logger.info(f"Interactive session reset: {session_id}")

            return (
                "Reset complete. Click 'Initialize' to start a new generation.",
                "",
                None,
                gr.update(choices=[("Reset", "0")], value="0"),
                None,
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(value="0"),
            )

        def undo_step(
            session_id, temp, topk, topp, strat, log_topk, log_all_logits_check
        ):
            """Undo the last generation step."""
            if session_id is None:
                return (
                    "Error: Not initialized. Click 'Initialize' first.",
                    "",
                    None,
                    gr.update(),
                    session_id,
                    gr.update(),
                    gr.update(interactive=False),
                    gr.update(value="0"),
                )

            session_manager = get_session_manager()
            tracer = session_manager.get_tracer(session_id)
            lock = session_manager.get_lock(session_id)

            if tracer is None or lock is None:
                return (
                    "Error: Session not found. Please reinitialize.",
                    "",
                    None,
                    gr.update(),
                    None,
                    gr.update(),
                    gr.update(interactive=False),
                    gr.update(value="0"),
                )

            # Acquire lock to prevent race conditions
            with lock:
                try:
                    # Validate state before operation
                    is_valid, error_msg = tracer.validate_state()
                    if not is_valid:
                        return (
                            f"Error: State corrupted: {error_msg}\nPlease reset and reinitialize.",
                            "",
                            None,
                            gr.update(),
                            session_id,
                            gr.update(),
                            gr.update(interactive=False),
                            gr.update(value="0"),
                        )

                    success = tracer.undo_step()

                    if not success:
                        return (
                            "Error: No steps to undo",
                            "",
                            None,
                            gr.update(),
                            session_id,
                            gr.update(),
                            gr.update(interactive=False),
                            gr.update(value="0"),
                        )

                    # Get current text
                    current_text = (
                        tracer.get_full_text() if len(tracer.history) > 0 else ""
                    )

                    # Determine actual logging parameters based on checkbox
                    actual_log_top_k = int(log_topk) if log_topk else 10
                    if log_all_logits_check:
                        actual_log_top_k = len(tracer.tokenizer)

                    # Get next probabilities and CACHE them
                    prob_data = tracer.cache_next_probabilities(
                        top_k=actual_log_top_k, temperature=temp
                    )

                    # Preview what token would be selected based on strategy
                    previewed_token_id, preview_method = _preview_next_token(
                        tracer, prob_data, strat, temp, topk, topp
                    )

                    # Build display
                    df, radio_choices = _build_prob_display(prob_data)

                    status = f"Undone last step. Current steps: {len(tracer.history)}"

                    # Prepare download
                    download_update = prepare_download(session_id)

                    # Select the previewed token
                    radio_update = gr.update(
                        choices=radio_choices, value=str(previewed_token_id)
                    )

                    logger.debug(
                        f"Undo successful for session {session_id}, state: {tracer.get_state_info()}"
                    )

                    current_step = str(len(tracer.history))
                    return (
                        status,
                        current_text,
                        df,
                        radio_update,
                        session_id,
                        download_update,
                        gr.update(interactive=False),
                        gr.update(value=current_step),
                    )

                except Exception as e:
                    import traceback

                    error = f"Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
                    current_step = (
                        str(len(tracer.history)) if tracer and tracer.history else "0"
                    )
                    return (
                        error,
                        "",
                        None,
                        gr.update(),
                        session_id,
                        gr.update(),
                        gr.update(interactive=False),
                        gr.update(value=current_step),
                    )

        def continue_generation(
            session_id,
            strat,
            temp,
            topk,
            topp,
            n_tokens,
            log_topk,
            log_all_logits_check,
            stop_at_eos,
        ):
            """Continue generating for N tokens autonomously."""
            if session_id is None:
                yield "Error: Not initialized. Click 'Initialize' first.", "", None, gr.update(), session_id, gr.update(), gr.update(
                    interactive=False
                ), gr.update()
                return

            session_manager = get_session_manager()
            tracer = session_manager.get_tracer(session_id)
            lock = session_manager.get_lock(session_id)

            if tracer is None or lock is None:
                logger.warning(
                    f"Continue generation failed: session {session_id} not found"
                )
                yield "Error: Session not found. Please reinitialize.", "", None, gr.update(), None, gr.update(), gr.update(
                    interactive=False
                ), gr.update()
                return

            # Validate n_tokens
            if n_tokens is None or n_tokens < 1:
                error_msg = "Error: Number of tokens must be at least 1"
                logger.warning(f"Invalid n_tokens in continue_generation: {n_tokens}")
                yield error_msg, "", None, gr.update(), session_id, gr.update(), gr.update(
                    interactive=False
                ), gr.update()
                return

            logger.info(
                f"Continue generation started: session={session_id}, n_tokens={n_tokens}, strategy={strat}"
            )

            # Acquire lock for the entire generation process
            with lock:
                try:
                    # Clear stop flag before starting
                    tracer.clear_stop_flag()

                    # Validate state before operation
                    is_valid, error_msg = tracer.validate_state()
                    if not is_valid:
                        yield f"Error: State corrupted: {error_msg}\nPlease reset and reinitialize.", "", None, gr.update(), session_id, gr.update(), gr.update(
                            interactive=False
                        ), gr.update()
                        return

                    # Determine actual logging parameters based on checkbox
                    actual_log_top_k = int(log_topk) if log_topk else 10
                    if log_all_logits_check:
                        actual_log_top_k = len(tracer.tokenizer)

                    actual_log_all_logits = log_all_logits_check

                    for i in range(int(n_tokens)):
                        # Check if stop was requested (highest priority)
                        if tracer._stop_requested:
                            logger.info(
                                f"Continue generation stopped by user request after {i} tokens, session={session_id}"
                            )
                            status = f"Generation stopped by user\nGenerated {i} tokens. Total steps: {len(tracer.history)}"
                            current_text = tracer.get_full_text()

                            # Cache next probabilities for continuation
                            prob_data = tracer.cache_next_probabilities(
                                top_k=actual_log_top_k, temperature=temp
                            )
                            previewed_token_id, preview_method = _preview_next_token(
                                tracer, prob_data, strat, temp, topk, topp
                            )
                            df, radio_choices = _build_prob_display(prob_data)
                            radio_update = gr.update(
                                choices=radio_choices, value=str(previewed_token_id)
                            )
                            download_update = prepare_download(session_id)
                            current_step = str(len(tracer.history))

                            yield status, current_text, df, radio_update, session_id, download_update, gr.update(
                                interactive=False
                            ), gr.update(
                                value=current_step
                            )
                            return

                        # Generate next token
                        step_data = tracer.step(
                            strategy=strat,
                            temperature=temp,
                            top_k=topk,
                            top_p=topp,
                            log_top_k=actual_log_top_k,
                            log_all_logits=actual_log_all_logits,
                        )

                        # Get current text
                        current_text = tracer.get_full_text()

                        # Check if we hit EOS (if enabled)
                        if (
                            stop_at_eos
                            and step_data.token_id == tracer.tokenizer.eos_token_id
                        ):
                            # Get final probabilities
                            logger.info(
                                f"Continue generation complete (EOS): session={session_id}, total_steps={len(tracer.history)}"
                            )
                            status = f"Generation complete (EOS reached)\nTotal steps: {len(tracer.history)}"
                            download_update = prepare_download(session_id)
                            current_step = str(len(tracer.history))
                            yield status, current_text, None, gr.update(
                                choices=[("EOS", "0")], value="0"
                            ), session_id, download_update, gr.update(
                                interactive=False
                            ), gr.update(
                                value=current_step
                            )
                            return

                        # Update status
                        status = f"Generating... Step {len(tracer.history)}/{len(tracer.history) + n_tokens - i - 1}"
                        download_update = prepare_download(session_id)
                        current_step = str(len(tracer.history))
                        yield status, current_text, None, gr.update(), session_id, download_update, gr.update(
                            interactive=True
                        ), gr.update(
                            value=current_step
                        )

                    # After completion, get next probabilities and CACHE them
                    prob_data = tracer.cache_next_probabilities(
                        top_k=actual_log_top_k, temperature=temp
                    )

                    # Preview what token would be selected based on strategy
                    previewed_token_id, preview_method = _preview_next_token(
                        tracer, prob_data, strat, temp, topk, topp
                    )

                    # Build display
                    df, radio_choices = _build_prob_display(prob_data)

                    status = f"Continue complete. Total steps: {len(tracer.history)}\n"
                    status += f"Next token preview ({preview_method}) shown below."

                    current_text = tracer.get_full_text()

                    # Prepare download
                    download_update = prepare_download(session_id)

                    # Select the previewed token
                    radio_update = gr.update(
                        choices=radio_choices, value=str(previewed_token_id)
                    )

                    logger.info(
                        f"Continue generation complete: session={session_id}, total_steps={len(tracer.history)}"
                    )
                    logger.debug(
                        f"Session {session_id} state: {tracer.get_state_info()}"
                    )

                    current_step = str(len(tracer.history))
                    yield status, current_text, df, radio_update, session_id, download_update, gr.update(
                        interactive=False
                    ), gr.update(
                        value=current_step
                    )

                except Exception as e:
                    import traceback

                    error = f"Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
                    current_step = str(len(tracer.history)) if tracer.history else "0"
                    yield error, "", None, gr.update(), session_id, gr.update(), gr.update(
                        interactive=False
                    ), gr.update(
                        value=current_step
                    )

        def step_generation(
            session_id,
            strat,
            temp,
            topk,
            topp,
            rank_selection,
            use_override,
            override_id,
            log_topk,
            log_all_logits_check,
            stop_at_eos,
        ):
            """Generate the next token."""
            if session_id is None:
                return (
                    "Error: Not initialized. Click 'Initialize' first.",
                    "",
                    None,
                    gr.update(),
                    session_id,
                    gr.update(),
                    gr.update(interactive=False),
                    gr.update(value="0"),
                )

            session_manager = get_session_manager()
            tracer = session_manager.get_tracer(session_id)
            lock = session_manager.get_lock(session_id)

            if tracer is None or lock is None:
                return (
                    "Error: Session not found. Please reinitialize.",
                    "",
                    None,
                    gr.update(),
                    None,
                    gr.update(),
                    gr.update(interactive=False),
                    gr.update(value="0"),
                )

            # Acquire lock to prevent race conditions
            with lock:
                try:
                    # Validate state before operation
                    is_valid, error_msg = tracer.validate_state()
                    if not is_valid:
                        return (
                            f"Error: State corrupted: {error_msg}\nPlease reset and reinitialize.",
                            "",
                            None,
                            gr.update(),
                            session_id,
                            gr.update(),
                            gr.update(interactive=False),
                            gr.update(value="0"),
                        )

                    logger.debug(
                        f"Step generation: session={session_id}, strategy={strat}, rank_selection={repr(rank_selection)}, use_override={use_override}"
                    )
                    state_info = tracer.get_state_info()
                    logger.debug(
                        f"Current position (history length): {len(tracer.history)}, state_info={state_info}"
                    )

                    # Determine actual logging parameters based on checkbox
                    actual_log_top_k = int(log_topk) if log_topk else 10
                    if log_all_logits_check:
                        actual_log_top_k = len(tracer.tokenizer)

                    actual_log_all_logits = log_all_logits_check

                    # Determine which token to commit (what's currently selected in radio)
                    if use_override:
                        # User manually specified a token ID via override field
                        if override_id is None:
                            return (
                                "Error: Override enabled but no token ID provided",
                                "",
                                None,
                                gr.update(),
                                session_id,
                                gr.update(),
                                gr.update(interactive=False),
                                gr.update(value="0"),
                            )
                        selected_token_id = int(override_id)
                        if selected_token_id < 0 or selected_token_id >= len(
                            tracer.tokenizer
                        ):
                            return (
                                f"Error: Token ID {selected_token_id} is out of range (vocab size: {len(tracer.tokenizer)})",
                                "",
                                None,
                                gr.update(),
                                session_id,
                                gr.update(),
                                gr.update(interactive=False),
                                gr.update(value="0"),
                            )
                    else:
                        # Use whatever is currently selected in the radio (from preview or user change)
                        try:
                            selected_token_id = int(rank_selection)
                        except (ValueError, TypeError):
                            # Fallback: if parsing fails, use Rank 0
                            prob_data = tracer.get_next_token_probabilities(
                                top_k=10, temperature=temp, use_cache=True
                            )
                            selected_token_id = prob_data["top_k_tokens"][0]

                    # Commit the selected token
                    step_data = tracer.step(
                        token_id=selected_token_id,
                        strategy="greedy",  # Always greedy since we're specifying the token
                        temperature=temp,
                        top_k=topk,
                        top_p=topp,
                        log_top_k=actual_log_top_k,
                        log_all_logits=actual_log_all_logits,
                    )

                    logger.debug(
                        f"Selected token: id={step_data.token_id}, text={repr(step_data.token_text)}, prob={step_data.probability:.4f}"
                    )

                    # Get updated text
                    current_text = tracer.get_full_text()

                    # Check if we hit EOS (if enabled)
                    if (
                        stop_at_eos
                        and step_data.token_id == tracer.tokenizer.eos_token_id
                    ):
                        logger.info(
                            f"EOS token reached: session={session_id}, total_steps={len(tracer.history)}"
                        )
                        status = f"Generation complete (EOS reached)\nTotal steps: {len(tracer.history)}"
                        download_update = prepare_download(session_id)
                        current_step = str(len(tracer.history))
                        return (
                            status,
                            current_text,
                            None,
                            gr.update(choices=[("EOS", "0")], value="0"),
                            session_id,
                            download_update,
                            gr.update(interactive=False),
                            gr.update(value=current_step),
                        )

                    # Get next probabilities and CACHE them (optimization!)
                    prob_data_next = tracer.cache_next_probabilities(
                        top_k=actual_log_top_k, temperature=temp
                    )

                    # Preview what the NEXT token would be based on strategy
                    previewed_token_id, preview_method = _preview_next_token(
                        tracer, prob_data_next, strat, temp, topk, topp
                    )

                    top3 = list(
                        zip(
                            prob_data_next["top_k_texts"][:3],
                            prob_data_next["top_k_probs"][:3],
                        )
                    )
                    logger.debug(f"Top 3 next tokens: {top3}")
                    logger.debug(
                        f"Previewed next token ({preview_method}): {previewed_token_id}"
                    )

                    # Build display
                    df, radio_choices = _build_prob_display(prob_data_next)

                    # Build status message
                    status = f"Step {len(tracer.history)} complete\n"
                    status += f"Generated: {step_data.token_text} (p={step_data.probability:.4f})\n"
                    status += f"Next token preview ({preview_method}) shown below. You can change it before clicking 'Next Step'."

                    # Prepare download
                    download_update = prepare_download(session_id)

                    # Select the previewed token in the radio
                    radio_update = gr.update(
                        choices=radio_choices, value=str(previewed_token_id)
                    )

                    current_step = str(len(tracer.history))
                    return (
                        status,
                        current_text,
                        df,
                        radio_update,
                        session_id,
                        download_update,
                        gr.update(interactive=False),
                        gr.update(value=current_step),
                    )

                except Exception as e:
                    import traceback

                    error = f"Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
                    logger.error(
                        f"Error in step_generation for session {session_id}: {str(e)}",
                        exc_info=True,
                    )
                    current_step = (
                        str(len(tracer.history)) if tracer and tracer.history else "0"
                    )
                    return (
                        error,
                        "",
                        None,
                        gr.update(),
                        session_id,
                        gr.update(),
                        gr.update(interactive=False),
                        gr.update(value=current_step),
                    )

        def stop_handler(session_id):
            """Handle stop button click - request stop and disable button."""
            if session_id is None:
                return gr.update(interactive=False)

            session_manager = get_session_manager()
            tracer = session_manager.get_tracer(session_id)

            if tracer is not None:
                tracer.request_stop()
                logger.info(f"Stop requested for session {session_id}")

            return gr.update(interactive=False)

        def go_to_step(
            session_id,
            target_step,
            temp,
            topk,
            topp,
            strat,
            log_topk,
            log_all_logits_check,
        ):
            """Go back to a specific step number."""
            if session_id is None:
                return (
                    "Error: Not initialized. Click 'Initialize' first.",
                    "",
                    None,
                    gr.update(),
                    session_id,
                    gr.update(),
                    gr.update(interactive=False),
                    gr.update(value="0"),
                )

            session_manager = get_session_manager()
            tracer = session_manager.get_tracer(session_id)
            lock = session_manager.get_lock(session_id)

            if tracer is None or lock is None:
                return (
                    "Error: Session not found. Please reinitialize.",
                    "",
                    None,
                    gr.update(),
                    None,
                    gr.update(),
                    gr.update(interactive=False),
                    gr.update(value="0"),
                )

            # Acquire lock to prevent race conditions
            with lock:
                try:
                    # Validate state before operation
                    is_valid, error_msg = tracer.validate_state()
                    if not is_valid:
                        return (
                            f"Error: State corrupted: {error_msg}\nPlease reset and reinitialize.",
                            "",
                            None,
                            gr.update(),
                            session_id,
                            gr.update(),
                            gr.update(interactive=False),
                            gr.update(value="0"),
                        )

                    current_steps = len(tracer.history)

                    # Validate target_step
                    if target_step is None or target_step < 0:
                        return (
                            f"Error: Target step must be 0 or greater. Current step: {current_steps}",
                            tracer.get_full_text() if current_steps > 0 else "",
                            None,
                            gr.update(),
                            session_id,
                            gr.update(),
                            gr.update(interactive=False),
                            gr.update(value=str(current_steps)),
                        )

                    if target_step > current_steps:
                        return (
                            f"Error: Target step {target_step} is beyond current step {current_steps}",
                            tracer.get_full_text() if current_steps > 0 else "",
                            None,
                            gr.update(),
                            session_id,
                            gr.update(),
                            gr.update(interactive=False),
                            gr.update(value=str(current_steps)),
                        )

                    # Calculate how many steps to undo
                    steps_to_undo = current_steps - int(target_step)

                    if steps_to_undo == 0:
                        # Already at target step
                        current_text = (
                            tracer.get_full_text() if current_steps > 0 else ""
                        )

                        # Determine actual logging parameters based on checkbox
                        actual_log_top_k = int(log_topk) if log_topk else 10
                        if log_all_logits_check:
                            actual_log_top_k = len(tracer.tokenizer)

                        # Get next probabilities and CACHE them
                        prob_data = tracer.cache_next_probabilities(
                            top_k=actual_log_top_k, temperature=temp
                        )

                        # Preview what token would be selected based on strategy
                        previewed_token_id, preview_method = _preview_next_token(
                            tracer, prob_data, strat, temp, topk, topp
                        )

                        # Build display
                        df, radio_choices = _build_prob_display(prob_data)

                        status = f"Already at step {target_step}"

                        # Prepare download
                        download_update = prepare_download(session_id)

                        # Select the previewed token
                        radio_update = gr.update(
                            choices=radio_choices, value=str(previewed_token_id)
                        )

                        return (
                            status,
                            current_text,
                            df,
                            radio_update,
                            session_id,
                            download_update,
                            gr.update(interactive=False),
                            gr.update(value=str(target_step)),
                        )

                    # Undo multiple steps
                    logger.info(
                        f"Going back from step {current_steps} to step {target_step} (undoing {steps_to_undo} steps)"
                    )

                    for i in range(steps_to_undo):
                        success = tracer.undo_step()
                        if not success:
                            # This shouldn't happen due to validation above, but handle it anyway
                            current_step = str(len(tracer.history))
                            return (
                                f"Error: Failed to undo step {current_steps - i}. Currently at step {len(tracer.history)}",
                                (
                                    tracer.get_full_text()
                                    if len(tracer.history) > 0
                                    else ""
                                ),
                                None,
                                gr.update(),
                                session_id,
                                gr.update(),
                                gr.update(interactive=False),
                                gr.update(value=current_step),
                            )

                    # Get current text
                    current_text = (
                        tracer.get_full_text() if len(tracer.history) > 0 else ""
                    )

                    # Determine actual logging parameters based on checkbox
                    actual_log_top_k = int(log_topk) if log_topk else 10
                    if log_all_logits_check:
                        actual_log_top_k = len(tracer.tokenizer)

                    # Get next probabilities and CACHE them
                    prob_data = tracer.cache_next_probabilities(
                        top_k=actual_log_top_k, temperature=temp
                    )

                    # Preview what token would be selected based on strategy
                    previewed_token_id, preview_method = _preview_next_token(
                        tracer, prob_data, strat, temp, topk, topp
                    )

                    # Build display
                    df, radio_choices = _build_prob_display(prob_data)

                    status = (
                        f"Went back to step {target_step} (undid {steps_to_undo} steps)"
                    )

                    # Prepare download
                    download_update = prepare_download(session_id)

                    # Select the previewed token
                    radio_update = gr.update(
                        choices=radio_choices, value=str(previewed_token_id)
                    )

                    logger.debug(
                        f"Go to step successful for session {session_id}, now at step {len(tracer.history)}"
                    )

                    current_step = str(len(tracer.history))
                    return (
                        status,
                        current_text,
                        df,
                        radio_update,
                        session_id,
                        download_update,
                        gr.update(interactive=False),
                        gr.update(value=current_step),
                    )

                except Exception as e:
                    import traceback

                    error = f"Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
                    current_step = (
                        str(len(tracer.history)) if tracer and tracer.history else "0"
                    )
                    return (
                        error,
                        "",
                        None,
                        gr.update(),
                        session_id,
                        gr.update(),
                        gr.update(interactive=False),
                        gr.update(value=current_step),
                    )

        def update_mode_visibility(mode):
            """Update input visibility based on mode."""
            if mode == "Completion":
                return gr.update(visible=True), gr.update(visible=False)
            else:
                return gr.update(visible=False), gr.update(visible=True)

        def update_number_visibility(log_all_check):
            """Hide log_top_k number input when Log All Logits checkbox is enabled."""
            return gr.update(visible=not log_all_check)

        def update_token_selection_visibility(use_override_enabled):
            """Toggle visibility between token selector radio and token ID number input."""
            return (
                gr.update(
                    visible=not use_override_enabled
                ),  # token_selector (hide when override enabled)
                gr.update(
                    visible=use_override_enabled
                ),  # token_override (show when override enabled)
            )

        def _build_prob_display(prob_data):
            """Build dataframe and radio choices from probability data."""
            rows = []
            for i, (tok_id, prob, text) in enumerate(
                zip(
                    prob_data["top_k_tokens"],
                    prob_data["top_k_probs"],
                    prob_data["top_k_texts"],
                )
            ):
                decoded = text[1]  # Use raw token representation
                rows.append([i, tok_id, decoded, f"{prob:.4f}"])

            df = pd.DataFrame(
                rows, columns=["Rank", "Token ID", "Token", "Probability"]
            )

            # Create radio choices (label, value) tuples with token IDs as values
            radio_choices = []
            for i, (tok_id, prob, text) in enumerate(
                zip(
                    prob_data["top_k_tokens"][:10],
                    prob_data["top_k_probs"][:10],
                    prob_data["top_k_texts"][:10],
                )
            ):
                decoded = text[1]  # Use raw token representation
                label = f"Rank {i}: {decoded} (p={prob:.4f})"
                radio_choices.append((label, str(tok_id)))

            return df, radio_choices

        def _preview_next_token(tracer, prob_data, strat, temp, topk, topp):
            """
            Preview what token would be selected based on the strategy.

            Returns:
                token_id: The previewed token ID
                method: String describing how it was picked ("greedy" or "sampled")
            """
            import torch

            if strat == "sampling":
                # Actually sample a token to preview
                next_token_logits = prob_data["logits"].clone()

                # Apply top-k filtering
                if topk > 0:
                    indices_to_remove = (
                        next_token_logits
                        < torch.topk(next_token_logits, topk)[0][..., -1, None]
                    )
                    next_token_logits[indices_to_remove] = float("-inf")

                # Apply top-p (nucleus) filtering
                if topp < 1.0:
                    sorted_logits, sorted_indices = torch.sort(
                        next_token_logits, descending=True
                    )
                    cumulative_probs = torch.cumsum(
                        torch.softmax(sorted_logits, dim=-1), dim=-1
                    )

                    sorted_indices_to_remove = cumulative_probs > topp
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                        ..., :-1
                    ].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    next_token_logits[indices_to_remove] = float("-inf")

                # Sample
                probs = torch.softmax(next_token_logits, dim=-1)
                token_id = torch.multinomial(probs[0], num_samples=1).item()

                return token_id, "sampled"
            else:
                # Greedy: pick top-1
                return prob_data["top_k_tokens"][0], "greedy"

        # Wire up event handlers
        mode_selector.change(
            fn=update_mode_visibility,
            inputs=[mode_selector],
            outputs=[completion_inputs, chat_inputs],
        )

        # Wire up number input visibility control
        log_all_logits_checkbox.change(
            fn=update_number_visibility,
            inputs=[log_all_logits_checkbox],
            outputs=[log_top_k],
        )

        # Wire up token selection visibility control
        use_override.change(
            fn=update_token_selection_visibility,
            inputs=[use_override],
            outputs=[token_selector, token_override],
        )

        init_button.click(
            fn=initialize_tracer,
            inputs=[
                mode_selector,
                prompt_input,
                chat_messages,
                temperature,
                top_k,
                top_p,
                strategy,
                log_top_k,
                log_all_logits_checkbox,
            ],
            outputs=[
                status_output,
                current_text_output,
                next_token_probs,
                token_selector,
                session_state,
                download_button,
                stop_button,
                current_step_display,
            ],
        )

        reset_button.click(
            fn=reset_tracer,
            inputs=[session_state],
            outputs=[
                status_output,
                current_text_output,
                next_token_probs,
                token_selector,
                session_state,
                download_button,
                stop_button,
                current_step_display,
            ],
        )

        step_button.click(
            fn=step_generation,
            inputs=[
                session_state,
                strategy,
                temperature,
                top_k,
                top_p,
                token_selector,
                use_override,
                token_override,
                log_top_k,
                log_all_logits_checkbox,
                stop_at_eos_checkbox,
            ],
            outputs=[
                status_output,
                current_text_output,
                next_token_probs,
                token_selector,
                session_state,
                download_button,
                stop_button,
                current_step_display,
            ],
        )

        undo_button.click(
            fn=undo_step,
            inputs=[
                session_state,
                temperature,
                top_k,
                top_p,
                strategy,
                log_top_k,
                log_all_logits_checkbox,
            ],
            outputs=[
                status_output,
                current_text_output,
                next_token_probs,
                token_selector,
                session_state,
                download_button,
                stop_button,
                current_step_display,
            ],
        )

        continue_event = continue_button.click(
            fn=continue_generation,
            inputs=[
                session_state,
                strategy,
                temperature,
                top_k,
                top_p,
                continue_tokens,
                log_top_k,
                log_all_logits_checkbox,
                stop_at_eos_checkbox,
            ],
            outputs=[
                status_output,
                current_text_output,
                next_token_probs,
                token_selector,
                session_state,
                download_button,
                stop_button,
                current_step_display,
            ],
        )

        stop_button.click(
            fn=stop_handler,
            inputs=[session_state],
            outputs=[stop_button],
            cancels=[continue_event],
        )

        go_to_step_button.click(
            fn=go_to_step,
            inputs=[
                session_state,
                go_to_step_input,
                temperature,
                top_k,
                top_p,
                strategy,
                log_top_k,
                log_all_logits_checkbox,
            ],
            outputs=[
                status_output,
                current_text_output,
                next_token_probs,
                token_selector,
                session_state,
                download_button,
                stop_button,
                current_step_display,
            ],
        )

    return tab
