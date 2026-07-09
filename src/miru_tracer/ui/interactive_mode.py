"""Interactive mode tab: step-through generation with session-based state.

Gradio state carries only the session id; the live tracer lives in the
SessionManager. Every handler resolves the session, takes its lock, mutates
the tracer, and renders the common output tuple via ``render_state``.
"""

from __future__ import annotations

import traceback

import gradio as gr

from miru_tracer.core.lens import compute_lens_slice, get_lens_store
from miru_tracer.core.logging_config import get_logger
from miru_tracer.core.model_manager import ModelManager
from miru_tracer.core.sampling import select_token
from miru_tracer.core.session_manager import get_session_manager
from miru_tracer.ui.helpers import (
    CHAT_MODE_HELP,
    DEFAULT_CHAT_JSON,
    GENERATION_MODES,
    RAW_MODE_HELP,
    RAW_MODE_PLACEHOLDER,
    TEMPERATURE_GREEDY_INFO,
    THINK_PREFILL_INFO,
    THINKING_CHOICES,
    ChatValidationError,
    ExportManager,
    build_prob_table,
    build_radio_choices,
    parse_chat_messages,
    thinking_key,
    toggle_mode_visibility,
    toggle_temperature,
    toggle_think_prefill,
    ui_sampling_params,
)
from miru_tracer.ui.lens_common import (
    LENS_MODE_CHOICES,
    get_active_interventions,
    layer_selection,
    lens_mode_key,
)
from miru_tracer.visualization.plots import plot_lens_heatmap

logger = get_logger(__name__)


def create_interactive_mode_tab(model_manager: ModelManager) -> gr.Tab:
    """Create the interactive debugging tab interface."""

    exports = ExportManager("interactive")

    with gr.Tab("Interactive Mode") as tab, gr.Column(elem_classes="miru-narrow"):
        gr.Markdown(
            "Generate text step-by-step with full control over each token selection."
        )

        mode_selector = gr.Radio(
            choices=list(GENERATION_MODES),
            value="Completion",
            label="Mode",
            info="Direct text completion, chat format, or raw text with "
            "explicit special tokens.",
        )

        with gr.Group() as completion_inputs:
            prompt_input = gr.Textbox(
                label="Prompt",
                placeholder="Enter your prompt",
                lines=2,
                value="The future of artificial intelligence is",
            )

        with gr.Group(visible=False) as chat_inputs:
            gr.Markdown(CHAT_MODE_HELP)
            chat_messages = gr.Code(
                label="Chat (JSON)",
                language="json",
                lines=8,
                value=DEFAULT_CHAT_JSON,
            )
            thinking_selector = gr.Radio(
                choices=list(THINKING_CHOICES),
                value=THINKING_CHOICES[0],
                label="Thinking",
            )
            think_prefill_box = gr.Textbox(
                label="Thought prefill",
                visible=False,
                lines=2,
                placeholder="Okay, the user wants…",
                info=THINK_PREFILL_INFO,
            )

        with gr.Group(visible=False) as raw_inputs:
            raw_input = gr.Textbox(
                label="Raw text",
                lines=4,
                placeholder=RAW_MODE_PLACEHOLDER,
                info=RAW_MODE_HELP,
                elem_classes=["miru-textbox-mono"],
            )

        with gr.Group():
            with gr.Row():
                strategy = gr.Radio(
                    choices=["greedy", "sampling"],
                    value="greedy",
                    label="Strategy",
                    info="A preview of the next token is shown below either way.",
                )
            with gr.Row():
                temperature = gr.Slider(
                    0.1,
                    2.0,
                    value=1.0,
                    step=0.1,
                    label="Temperature",
                    interactive=False,  # default strategy is greedy
                    info=TEMPERATURE_GREEDY_INFO,
                )
                top_k = gr.Slider(1, 100, value=50, step=1, label="Top-K")
                top_p = gr.Slider(0.01, 1.0, value=0.9, step=0.01, label="Top-P")
            with gr.Accordion("Advanced (logging & stopping)", open=False), gr.Row():
                log_top_k = gr.Number(
                    minimum=1,
                    value=10,
                    precision=0,
                    label="Log Top-K Tokens",
                    info="Top candidates recorded per step.",
                )
                log_full_probs_checkbox = gr.Checkbox(
                    label="Log full probabilities",
                    value=False,
                    info="Whole-vocabulary distribution per step (large exports).",
                )
                stop_at_eos_checkbox = gr.Checkbox(
                    label="Stop at EOS",
                    value=True,
                    info="Stop when an end-of-sequence token appears.",
                )

        with gr.Row():
            init_button = gr.Button("Initialize", variant="primary")
            reset_button = gr.Button("Reset", variant="secondary")

        gr.Markdown("### Output")

        status_output = gr.Textbox(label="Status", interactive=False, lines=2)
        current_text_output = gr.Textbox(
            label="Output",
            lines=6,
            interactive=False,
            buttons=["copy"],
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
            stop_button = gr.Button("Stop", variant="stop", size="lg", interactive=False)

        gr.Markdown("### Token Selection")

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
                info=(
                    "Shows the token that will be generated. You can change this "
                    "selection before clicking 'Next Step'."
                ),
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

        with gr.Accordion("Layer Lens (current position)", open=False):
            gr.Markdown(
                "Per-layer readout of the **next-token** position via the "
                "logit or Jacobian lens. Refresh runs one extra forward pass. "
                "Interventions added in the Lens tab can be applied to this "
                "session here. ⚠️ *Experimental — still being tested; "
                "readouts may currently yield nonsense.*"
            )
            with gr.Row():
                lens_mode_choice = gr.Radio(
                    choices=list(LENS_MODE_CHOICES), value="Logit", label="Lens"
                )
                lens_stride = gr.Number(
                    minimum=1, value=1, precision=0, label="Layer stride"
                )
                lens_top_k = gr.Number(
                    minimum=1, value=50, precision=0, label="Readouts per layer"
                )
            with gr.Row():
                lens_refresh_button = gr.Button("Refresh lens", variant="secondary")
                lens_apply_iv_button = gr.Button(
                    "Apply Lens-tab interventions to this session", size="sm"
                )
            lens_status = gr.Textbox(label="Lens status", interactive=False, lines=1)
            lens_plot = gr.Plot(label="Layer readouts at the next-token position")

        gr.Markdown("### Navigation & Export")
        with gr.Row():
            go_to_step_input = gr.Number(
                minimum=0,
                value=0,
                precision=0,
                label="Target step",
                info="0 = initial state",
            )
            current_step_display = gr.Textbox(
                label="Current Step",
                value="0",
                interactive=False,
                scale=0,
                min_width=100,
            )
            go_to_step_button = gr.Button("Go to Step", variant="secondary")
        download_button = gr.DownloadButton(
            label="Download JSON",
            interactive=False,
            variant="secondary",
        )

        # Session state - only stores the session ID string
        session_state = gr.State(value=None)

        # -------------------------------------------------------- lens panel

        def refresh_lens(session_id, mode_choice, stride, top_k):
            session, error = resolve_session(session_id)
            if error:
                return None, error[0]  # error tuple's status message

            with session.lock:
                tracer = session.tracer
                if tracer.input_ids is None:
                    return None, "Initialize a prompt first."
                model_name = model_manager.get_model_name()
                mode = lens_mode_key(mode_choice)
                jlens = get_lens_store().get(model_name)
                if mode in ("jacobian", "diff") and jlens is None:
                    return None, (
                        f"No fitted Jacobian lens for {model_name}. Fit one in "
                        f"the Lens tab or run: miru-tracer-fit-lens {model_name}"
                    )
                try:
                    n_layers = tracer.model.config.get_text_config().num_hidden_layers
                    layers = layer_selection(n_layers, 0, -1, stride)
                    if mode in ("jacobian", "diff") and jlens is not None:
                        fitted = set(jlens.source_layers) | {n_layers - 1}
                        layers = [layer for layer in layers if layer in fitted]
                    slice_ = compute_lens_slice(
                        tracer.model,
                        tracer.tokenizer,
                        tracer.input_ids,
                        layers=layers,
                        positions=[tracer.seq_len - 1],
                        mode=mode,
                        jlens=jlens,
                        top_k=int(top_k),
                        interventions=tracer._intervention_set,
                    )
                    active = len(tracer.interventions)
                    status = (
                        f"{mode} lens over {len(layers)} layers at position "
                        f"{tracer.seq_len - 1}."
                    )
                    if active:
                        status += f" {active} intervention(s) active on this session."
                    return plot_lens_heatmap(slice_), status
                except Exception as e:
                    logger.error(f"Interactive lens error: {e}", exc_info=True)
                    return None, f"Error: {e}"

        def apply_lens_interventions(session_id):
            session, error = resolve_session(session_id)
            if error:
                return error[0]
            interventions = get_active_interventions()
            with session.lock:
                jlens = get_lens_store().get(model_manager.get_model_name())
                try:
                    session.tracer.set_interventions(interventions or None, jlens=jlens)
                except ValueError as e:
                    return f"Error: {e}"
            if not interventions:
                return "No active interventions in the Lens tab — session cleared."
            return (
                f"Applied {len(interventions)} intervention(s) to this session. "
                "They affect all subsequent steps (KV cache was rebuilt)."
            )

        # ------------------------------------------------------------ helpers

        def render_state(session_id, tracer, status, params, log_topk):
            """The canonical output tuple every successful handler returns."""
            dist = tracer.peek(
                top_k=max(int(log_topk or 10), 1), temperature=params.temperature
            )
            # Preview exactly the token step() would commit; for sampling this
            # draws the sample now — clicking 'Next Step' locks it in.
            preview_id = select_token(dist.raw_logits, params)
            current_text = tracer.get_full_text() if tracer.history else ""
            return (
                status,
                current_text,
                build_prob_table(dist),
                gr.update(choices=build_radio_choices(dist), value=str(preview_id)),
                session_id,
                exports.prepare(tracer.export_to_dict(params)),
                gr.update(interactive=False),  # stop button idle
                gr.update(value=str(len(tracer.history))),
            )

        def error_state(message, session_id, steps="0"):
            return (
                message,
                "",
                None,
                gr.update(),
                session_id,
                gr.update(),
                gr.update(interactive=False),
                gr.update(value=str(steps)),
            )

        def resolve_session(session_id):
            """Common session lookup; returns (session, error_tuple_or_None)."""
            if session_id is None:
                return None, error_state(
                    "Error: Not initialized. Click 'Initialize' first.", None
                )
            session = get_session_manager().get_session(session_id)
            if session is None:
                return None, error_state(
                    "Error: Session not found. Please reinitialize.", None
                )
            return session, None

        # ----------------------------------------------------------- handlers

        def initialize_tracer(
            mode, prompt, chat_msgs, raw_text, think_choice, think_text,
            temp, topk, topp, strat, log_topk,
        ):
            model = model_manager.get_model()
            tokenizer = model_manager.get_tokenizer()
            device = model_manager.get_device()
            if model is None or tokenizer is None:
                return error_state("Error: No model loaded", None)

            try:
                if mode == "Chat":
                    try:
                        messages = parse_chat_messages(chat_msgs)
                    except ChatValidationError as e:
                        return error_state(f"Error: {e}", None)
                else:
                    messages = None

                session_manager = get_session_manager()
                session_id = session_manager.create_session(model, tokenizer, device)
                session = session_manager.get_session(session_id)

                with session.lock:
                    if messages is not None:
                        session.tracer.reset(
                            messages=messages,
                            mode="chat",
                            thinking=thinking_key(think_choice),
                            think_prefill=think_text or "",
                        )
                    elif mode == "Raw":
                        session.tracer.reset(prompt=raw_text, mode="raw")
                    else:
                        session.tracer.reset(prompt=prompt, mode="completion")

                    params = ui_sampling_params(strat, temp, topk, topp)
                    logger.info(
                        f"Interactive session initialized: {session_id} (mode={mode})"
                    )
                    return render_state(
                        session_id,
                        session.tracer,
                        f"Initialized in {mode} mode",
                        params,
                        log_topk,
                    )
            except Exception as e:
                logger.error(f"Initialize error: {e}", exc_info=True)
                return error_state(
                    f"Error: {e}\n\nTraceback:\n{traceback.format_exc()}", None
                )

        def reset_tracer(session_id):
            if session_id is not None:
                get_session_manager().delete_session(session_id)
                logger.info(f"Interactive session reset: {session_id}")
            return (
                "Reset complete. Click 'Initialize' to start a new generation.",
                "",
                None,
                gr.update(choices=[("Reset", "0")], value="0"),
                None,
                exports.disabled(),
                gr.update(interactive=False),
                gr.update(value="0"),
            )

        def step_generation(
            session_id,
            strat,
            temp,
            topk,
            topp,
            rank_selection,
            override_enabled,
            override_id,
            log_topk,
            log_full,
            stop_at_eos,
        ):
            session, error = resolve_session(session_id)
            if error:
                return error

            with session.lock:
                tracer = session.tracer
                try:
                    params = ui_sampling_params(strat, temp, topk, topp)

                    if override_enabled:
                        if override_id is None:
                            return error_state(
                                "Error: Override enabled but no token ID provided",
                                session_id,
                                len(tracer.history),
                            )
                        token_id = int(override_id)
                        if not 0 <= token_id < len(tracer.tokenizer):
                            return error_state(
                                f"Error: Token ID {token_id} is out of range "
                                f"(vocab size: {len(tracer.tokenizer)})",
                                session_id,
                                len(tracer.history),
                            )
                    else:
                        try:
                            token_id = int(rank_selection)
                        except (ValueError, TypeError):
                            token_id = None  # let the tracer pick per strategy

                    step_data = tracer.step(
                        params,
                        token_id=token_id,
                        log_top_k=max(int(log_topk or 10), 1),
                        log_full_probs=bool(log_full),
                    )

                    if stop_at_eos and tracer.is_eos(step_data.token_id):
                        logger.info(
                            f"EOS reached: session={session_id}, steps={len(tracer.history)}"
                        )
                        return (
                            f"Generation complete (EOS reached)\n"
                            f"Total steps: {len(tracer.history)}",
                            tracer.get_full_text(),
                            None,
                            gr.update(choices=[("EOS", "0")], value="0"),
                            session_id,
                            exports.prepare(tracer.export_to_dict(params)),
                            gr.update(interactive=False),
                            gr.update(value=str(len(tracer.history))),
                        )

                    status = (
                        f"Step {len(tracer.history)} complete\n"
                        f"Generated: {step_data.token_text} (p={step_data.probability:.4f})"
                    )
                    return render_state(session_id, tracer, status, params, log_topk)
                except Exception as e:
                    logger.error(f"Step error: {e}", exc_info=True)
                    return error_state(
                        f"Error: {e}\n\nTraceback:\n{traceback.format_exc()}",
                        session_id,
                        len(tracer.history),
                    )

        def undo_step(session_id, temp, topk, topp, strat, log_topk):
            session, error = resolve_session(session_id)
            if error:
                return error

            with session.lock:
                tracer = session.tracer
                try:
                    if not tracer.undo():
                        return error_state("Error: No steps to undo", session_id)
                    params = ui_sampling_params(strat, temp, topk, topp)
                    return render_state(
                        session_id,
                        tracer,
                        f"Undone last step. Current steps: {len(tracer.history)}",
                        params,
                        log_topk,
                    )
                except Exception as e:
                    logger.error(f"Undo error: {e}", exc_info=True)
                    return error_state(
                        f"Error: {e}\n\nTraceback:\n{traceback.format_exc()}",
                        session_id,
                        len(tracer.history),
                    )

        def go_to_step(session_id, target_step, temp, topk, topp, strat, log_topk):
            session, error = resolve_session(session_id)
            if error:
                return error

            with session.lock:
                tracer = session.tracer
                current_steps = len(tracer.history)
                try:
                    if target_step is None or target_step < 0:
                        return error_state(
                            f"Error: Target step must be 0 or greater. "
                            f"Current step: {current_steps}",
                            session_id,
                            current_steps,
                        )
                    if target_step > current_steps:
                        return error_state(
                            f"Error: Target step {int(target_step)} is beyond "
                            f"current step {current_steps}",
                            session_id,
                            current_steps,
                        )

                    tracer.goto_step(int(target_step))
                    params = ui_sampling_params(strat, temp, topk, topp)
                    undone = current_steps - int(target_step)
                    status = (
                        f"Already at step {int(target_step)}"
                        if undone == 0
                        else f"Went back to step {int(target_step)} (undid {undone} steps)"
                    )
                    return render_state(session_id, tracer, status, params, log_topk)
                except Exception as e:
                    logger.error(f"Go-to-step error: {e}", exc_info=True)
                    return error_state(
                        f"Error: {e}\n\nTraceback:\n{traceback.format_exc()}",
                        session_id,
                        len(tracer.history),
                    )

        def continue_generation(
            session_id,
            strat,
            temp,
            topk,
            topp,
            n_tokens,
            log_topk,
            log_full,
            stop_at_eos,
        ):
            session, error = resolve_session(session_id)
            if error:
                yield error
                return
            if n_tokens is None or n_tokens < 1:
                yield error_state(
                    "Error: Number of tokens must be at least 1", session_id
                )
                return

            with session.lock:
                tracer = session.tracer
                try:
                    params = ui_sampling_params(strat, temp, topk, topp)
                    tracer.clear_stop_flag()
                    logger.info(
                        f"Continue generation: session={session_id}, n_tokens={n_tokens}"
                    )

                    stopped_reason = None
                    for i in range(int(n_tokens)):
                        if tracer._stop_requested:
                            stopped_reason = f"Generation stopped by user after {i} tokens"
                            break

                        step_data = tracer.step(
                            params,
                            log_top_k=max(int(log_topk or 10), 1),
                            log_full_probs=bool(log_full),
                        )

                        if stop_at_eos and tracer.is_eos(step_data.token_id):
                            stopped_reason = "Generation complete (EOS reached)"
                            break

                        # Intermediate progress: only text/status/step change.
                        # The candidates table and download stay untouched
                        # (no flicker, no per-token export files).
                        yield (
                            f"Generating... Step {len(tracer.history)} "
                            f"({i + 1}/{int(n_tokens)})",
                            tracer.get_full_text(),
                            gr.update(),
                            gr.update(),
                            session_id,
                            gr.update(),
                            gr.update(interactive=True),
                            gr.update(value=str(len(tracer.history))),
                        )

                    status = stopped_reason or "Continue complete"
                    status += f"\nTotal steps: {len(tracer.history)}"
                    yield render_state(session_id, tracer, status, params, log_topk)
                except Exception as e:
                    logger.error(f"Continue error: {e}", exc_info=True)
                    yield error_state(
                        f"Error: {e}\n\nTraceback:\n{traceback.format_exc()}",
                        session_id,
                        len(tracer.history),
                    )

        def stop_handler(session_id):
            """Request stop; the running generator finalizes the UI."""
            if session_id is not None:
                session = get_session_manager().get_session(session_id)
                if session is not None:
                    session.tracer.request_stop()
                    logger.info(f"Stop requested for session {session_id}")
            return gr.update(interactive=False)

        def toggle_override_visibility(enabled):
            return gr.update(visible=not enabled), gr.update(visible=enabled)

        # -------------------------------------------------------------- wiring

        outputs = [
            status_output,
            current_text_output,
            next_token_probs,
            token_selector,
            session_state,
            download_button,
            stop_button,
            current_step_display,
        ]

        mode_selector.change(
            fn=toggle_mode_visibility,
            inputs=[mode_selector],
            outputs=[completion_inputs, chat_inputs, raw_inputs],
        )
        strategy.change(
            fn=toggle_temperature, inputs=[strategy], outputs=[temperature]
        )
        thinking_selector.change(
            fn=toggle_think_prefill,
            inputs=[thinking_selector],
            outputs=[think_prefill_box],
        )
        use_override.change(
            fn=toggle_override_visibility,
            inputs=[use_override],
            outputs=[token_selector, token_override],
        )

        init_button.click(
            fn=initialize_tracer,
            inputs=[
                mode_selector,
                prompt_input,
                chat_messages,
                raw_input,
                thinking_selector,
                think_prefill_box,
                temperature,
                top_k,
                top_p,
                strategy,
                log_top_k,
            ],
            outputs=outputs,
        )
        reset_button.click(fn=reset_tracer, inputs=[session_state], outputs=outputs)
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
                log_full_probs_checkbox,
                stop_at_eos_checkbox,
            ],
            outputs=outputs,
        )
        undo_button.click(
            fn=undo_step,
            inputs=[session_state, temperature, top_k, top_p, strategy, log_top_k],
            outputs=outputs,
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
                log_full_probs_checkbox,
                stop_at_eos_checkbox,
            ],
            outputs=outputs,
        )
        stop_button.click(
            fn=stop_handler,
            inputs=[session_state],
            outputs=[stop_button],
            cancels=[continue_event],
        )
        lens_refresh_button.click(
            fn=refresh_lens,
            inputs=[session_state, lens_mode_choice, lens_stride, lens_top_k],
            outputs=[lens_plot, lens_status],
        )
        lens_apply_iv_button.click(
            fn=apply_lens_interventions,
            inputs=[session_state],
            outputs=[lens_status],
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
            ],
            outputs=outputs,
        )

    return tab
