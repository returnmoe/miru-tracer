"""Lens tab: layer-by-layer readouts (logit / Jacobian / diff) with
position and layer selection, aggregated readout browsing, multi-intervention
steering, and fit-file management (fitting itself runs offline via the
``miru-tracer-fit-lens`` CLI — see docs/lens-tutorial.md).

Semantics: interventions are applied at generation time. "Update readouts"
re-slices the existing sequence under the interventions it was generated
with; changing the intervention list requires Generate & Analyze again (the
status line says so). This keeps the displayed text and the displayed
readouts always consistent with each other.
"""

from __future__ import annotations

import traceback

import gradio as gr

from miru_tracer.core._jlens import JacobianLens
from miru_tracer.core.interventions import Intervention
from miru_tracer.core.lens import (
    aggregate_readouts,
    compute_lens_slice,
    decode_token,
    get_lens_store,
)
from miru_tracer.core.logging_config import get_logger
from miru_tracer.core.model_manager import ModelManager
from miru_tracer.core.sampling import SamplingParams
from miru_tracer.core.tracer import LLMTracer
from miru_tracer.ui.helpers import (
    DEFAULT_CHAT_JSON,
    ChatValidationError,
    parse_chat_messages,
    toggle_mode_visibility,
)
from miru_tracer.ui.lens_common import (
    LENS_MODE_CHOICES,
    highlighted_tokens,
    interventions_dataframe,
    layer_selection,
    lens_mode_key,
    parse_token_refs,
    readouts_dataframe,
    set_active_interventions,
    toggle_position,
    token_ref_to_id,
)
from miru_tracer.visualization.plots import (
    plot_lens_heatmap,
    plot_pinned_token_ranks,
    plot_readout_distribution,
)

logger = get_logger(__name__)


def create_lens_tab(model_manager: ModelManager) -> gr.Tab:
    """Create the Lens analysis tab."""

    with gr.Tab("Lens") as tab:
        gr.Markdown(
            "Read out what intermediate layers are 'thinking' with the logit "
            "lens and the **Jacobian lens** (Anthropic, 2026), and steer / "
            "swap / ablate those readouts during generation. "
            "See `docs/lens-tutorial.md` for fitting and usage.\n\n"
            "⚠️ *Experimental: this lens/intervention implementation is new "
            "and still being tested — readouts and steering effects may "
            "currently yield nonsense. The final layer is the sanity anchor: "
            "it always equals the model's real output distribution.*"
        )

        # ------------------------------------------------------------- input
        mode_selector = gr.Radio(
            choices=["Completion", "Chat"], value="Completion", label="Mode"
        )
        with gr.Group() as completion_inputs:
            prompt_input = gr.Textbox(
                label="Prompt",
                lines=2,
                value="The capital of France is",
            )
        with gr.Group(visible=False) as chat_inputs:
            chat_messages = gr.Code(
                label="Chat (JSON)", language="json", lines=8, value=DEFAULT_CHAT_JSON
            )

        with gr.Row():
            max_tokens = gr.Number(minimum=0, value=12, precision=0, label="New tokens")
            strategy = gr.Radio(
                choices=["greedy", "sampling"], value="greedy", label="Strategy"
            )
            temperature = gr.Slider(0.1, 2.0, value=1.0, step=0.1, label="Temperature")

        generate_button = gr.Button("Generate & Analyze", variant="primary", size="lg")
        status_output = gr.Textbox(label="Status", interactive=False, lines=2)
        text_output = gr.Textbox(label="Text", lines=4, interactive=False, buttons=["copy"])

        # --------------------------------------------------------- selection
        gr.Markdown("### Selection")
        tokens_display = gr.HighlightedText(
            label="Sequence — click tokens to toggle position selection "
            "(none selected = all positions)",
            value=[],
            color_map={"sel": "orange"},
            combine_adjacent=False,
        )
        with gr.Row():
            select_generated_button = gr.Button("Select generated", size="sm")
            select_clear_button = gr.Button("Clear selection", size="sm")

        with gr.Row():
            lens_mode = gr.Radio(
                choices=list(LENS_MODE_CHOICES),
                value="Logit",
                label="Lens",
                info="Jacobian/Diff need a fitted lens for the loaded model.",
            )
            layer_start = gr.Number(minimum=0, value=0, precision=0, label="From layer")
            layer_end = gr.Number(value=-1, precision=0, label="To layer (-1 = last)")
            layer_stride = gr.Number(minimum=1, value=2, precision=0, label="Stride")
        with gr.Row():
            readouts_per_cell = gr.Slider(
                1, 16, value=8, step=1, label="Readouts per layer+pos"
            )
            skip_non_words = gr.Checkbox(label="Hide non-word tokens", value=False)
            pinned_tokens = gr.Textbox(
                label="Pinned tokens",
                placeholder="comma-separated: Paris, 12345, ...",
                info="Track these tokens' ranks across layers.",
            )
        update_button = gr.Button("Update readouts", variant="secondary")

        # ---------------------------------------------------------- readouts
        gr.Markdown("### Readouts")
        readout_table = gr.Dataframe(
            headers=["Token", "ID", "Count", "By layer"],
            datatype=["str", "number", "number", "str"],
            label="Aggregated readouts over the selected cells",
            interactive=False,
        )
        dist_plot = gr.Plot(label="Readout counts by layer")
        heatmap_plot = gr.Plot(label="Position × layer heatmap")
        pinned_plot = gr.Plot(label="Pinned token ranks")

        # ------------------------------------------------------ interventions
        gr.Markdown("### Interventions")
        gr.Markdown(
            "Steer, swap, or ablate readout directions during generation. "
            "Any number can be active at once; they take effect on the next "
            "**Generate & Analyze**."
        )
        with gr.Row():
            iv_kind = gr.Radio(
                choices=["steer", "swap", "ablate"], value="steer", label="Kind"
            )
            iv_token = gr.Textbox(label="Token", placeholder="text or id, e.g. Paris")
            iv_swap_to = gr.Textbox(
                label="Swap to", placeholder="target token", visible=False
            )
            iv_layer = gr.Number(minimum=0, value=0, precision=0, label="Layer")
            iv_strength = gr.Slider(
                -4.0, 4.0, value=1.0, step=0.1, label="Strength (steer)"
            )
            iv_basis = gr.Radio(
                choices=["jacobian", "logit"], value="jacobian", label="Basis"
            )
        with gr.Row():
            iv_add_button = gr.Button("Add intervention", variant="secondary")
            iv_remove_index = gr.Number(
                minimum=0, value=0, precision=0, label="#", scale=0, min_width=80
            )
            iv_remove_button = gr.Button("Remove #", size="sm")
            iv_clear_button = gr.Button("Clear all", size="sm")
        iv_table = gr.Dataframe(
            headers=["#", "Intervention", "Basis"],
            datatype=["number", "str", "str"],
            label="Active interventions (applied on next generate)",
            interactive=False,
        )

        # ----------------------------------------------------------- fit file
        with gr.Accordion("Jacobian lens fit file", open=False):
            gr.Markdown(
                "The Jacobian lens needs per-model `J_ℓ` matrices, fitted once "
                "per model by averaging ∂h_final/∂h_ℓ over a text corpus. "
                "Fitting is compute-heavy — run it on a **GPU instance** with\n\n"
                "```\nmiru-tracer-fit-lens <model-name> --dim-batch 32\n```\n\n"
                "then load the resulting `lens.pt` here (or drop it into the "
                "lens cache directory yourself). Logit-lens mode never needs a "
                "fit file."
            )
            lens_file_status = gr.Textbox(
                label="Fit file status", interactive=False, lines=3
            )
            with gr.Row():
                lens_file_upload = gr.File(
                    label="Load fit file (lens.pt)", file_types=[".pt"], type="filepath"
                )
                lens_file_check_button = gr.Button("Check status", size="sm")

        # ------------------------------------------------------------ states
        # analysis_state: dict(input_ids, model_name, iset, n_layers, prompt_len)
        analysis_state = gr.State(None)
        positions_state = gr.State([])  # [] = all positions
        interventions_state = gr.State([])  # list[Intervention]

        readout_outputs = [
            readout_table,
            dist_plot,
            heatmap_plot,
            pinned_plot,
            tokens_display,
            status_output,
        ]

        # ----------------------------------------------------------- helpers

        def _empty_readouts(status):
            return None, None, None, None, gr.update(), status

        def render_readouts(
            analysis, positions, mode_choice, l_start, l_end, l_stride,
            per_cell, skip_nw, pinned_text,
        ):
            """Compute a slice for the stored sequence and render everything."""
            if analysis is None:
                return _empty_readouts("Generate first.")
            model = model_manager.get_model()
            tokenizer = model_manager.get_tokenizer()
            if model is None or analysis["model_name"] != model_manager.get_model_name():
                return _empty_readouts(
                    "Error: the model changed since this sequence was generated. "
                    "Generate again."
                )

            mode = lens_mode_key(mode_choice)
            jlens = get_lens_store().get(analysis["model_name"])
            if mode in ("jacobian", "diff") and jlens is None:
                return _empty_readouts(
                    "Error: no fitted Jacobian lens for "
                    f"{analysis['model_name']}. Fit one below or run:\n"
                    f"  miru-tracer-fit-lens {analysis['model_name']}"
                )

            try:
                layers = layer_selection(analysis["n_layers"], l_start, l_end, l_stride)
                if mode in ("jacobian", "diff") and jlens is not None:
                    fitted = set(jlens.source_layers) | {analysis["n_layers"] - 1}
                    dropped = [layer for layer in layers if layer not in fitted]
                    layers = [layer for layer in layers if layer in fitted]
                    if not layers:
                        return _empty_readouts(
                            f"Error: none of the selected layers are fitted "
                            f"(lens covers {jlens.source_layers})."
                        )
                else:
                    dropped = []
                pinned_ids = parse_token_refs(pinned_text, tokenizer)
                seq_len = int(analysis["input_ids"].shape[1])
                selected = [p for p in positions if 0 <= p < seq_len] or None

                slice_ = compute_lens_slice(
                    model,
                    tokenizer,
                    analysis["input_ids"],
                    layers=layers,
                    positions=selected,
                    mode=mode,
                    jlens=jlens,
                    top_k=int(per_cell),
                    skip_non_words=bool(skip_nw),
                    pinned_token_ids=pinned_ids,
                    interventions=analysis["iset"],
                )
                rows = aggregate_readouts(slice_)

                # Highlight over the FULL sequence, not just selected positions
                full_texts = analysis["position_texts"]
                highlight = highlighted_tokens(full_texts, positions)

                n_cells = len(slice_.layers) * len(slice_.positions)
                status = (
                    f"{slice_.mode} lens: {len(slice_.layers)} layers × "
                    f"{len(slice_.positions)} positions = {n_cells} cells, "
                    f"{len(rows)} distinct readout tokens."
                )
                if dropped:
                    status += f" (skipped unfitted layers: {dropped})"
                if analysis["iset"] is not None:
                    status += f" Interventions active: {len(analysis['iset'])}."
                return (
                    readouts_dataframe(rows),
                    plot_readout_distribution(rows, slice_.layers),
                    plot_lens_heatmap(slice_),
                    plot_pinned_token_ranks(slice_, tokenizer),
                    highlight,
                    status,
                )
            except Exception as e:
                logger.error(f"Lens readout error: {e}", exc_info=True)
                return _empty_readouts(
                    f"Error: {e}\n\nTraceback:\n{traceback.format_exc()}"
                )

        # ---------------------------------------------------------- handlers

        def generate_and_analyze(
            mode, prompt, chat_msgs, n_tokens, strat, temp,
            interventions, mode_choice, l_start, l_end, l_stride,
            per_cell, skip_nw, pinned_text,
        ):
            model = model_manager.get_model()
            tokenizer = model_manager.get_tokenizer()
            device = model_manager.get_device()
            if model is None or tokenizer is None:
                yield (
                    *_empty_readouts("Error: No model loaded. Use the Model Loader tab."),
                    "",
                    None,
                    gr.update(),
                )
                return

            try:
                tracer = LLMTracer(model, tokenizer, device)
                jlens = get_lens_store().get(model_manager.get_model_name())
                try:
                    tracer.set_interventions(interventions or None, jlens=jlens)
                except ValueError as e:
                    yield (
                        *_empty_readouts(
                            f"Error in interventions: {e}\n"
                            "(jacobian basis needs a fitted lens covering that layer "
                            "— fit one below, or use the logit basis)"
                        ),
                        "",
                        None,
                        gr.update(),
                    )
                    return

                if mode == "Chat":
                    tracer.reset(messages=parse_chat_messages(chat_msgs), mode="chat")
                else:
                    tracer.reset(prompt=prompt, mode="completion")

                params = SamplingParams(strategy=strat, temperature=float(temp))
                if n_tokens and int(n_tokens) > 0:
                    for _step in tracer.generate_stream(
                        max_new_tokens=int(n_tokens), params=params
                    ):
                        yield (
                            *_empty_readouts(
                                f"Generating... {len(tracer.history)}/{int(n_tokens)}"
                            ),
                            tracer.get_full_text(),
                            None,
                            gr.update(),
                        )

                position_texts = [
                    decode_token(tokenizer, int(t)) for t in tracer.input_ids[0]
                ]
                analysis = {
                    "input_ids": tracer.input_ids.clone(),
                    "model_name": model_manager.get_model_name(),
                    "iset": tracer._intervention_set,
                    "n_layers": model.config.get_text_config().num_hidden_layers,
                    "prompt_len": tracer._prompt_len,
                    "position_texts": position_texts,
                }
                positions: list[int] = []  # all
                outputs = render_readouts(
                    analysis, positions, mode_choice, l_start, l_end, l_stride,
                    per_cell, skip_nw, pinned_text,
                )
                yield (*outputs, tracer.get_full_text(), analysis, positions)
            except ChatValidationError as e:
                yield (*_empty_readouts(f"Error: {e}"), "", None, gr.update())
            except Exception as e:
                logger.error(f"Lens generate error: {e}", exc_info=True)
                yield (
                    *_empty_readouts(f"Error: {e}\n\nTraceback:\n{traceback.format_exc()}"),
                    "",
                    None,
                    gr.update(),
                )

        def on_token_select(positions, analysis, evt: gr.SelectData):
            if analysis is None:
                return positions, gr.update()
            updated = toggle_position(positions, evt.index)
            return updated, highlighted_tokens(analysis["position_texts"], updated)

        def select_generated(analysis):
            if analysis is None:
                return [], gr.update()
            seq_len = int(analysis["input_ids"].shape[1])
            selected = list(range(analysis["prompt_len"], seq_len))
            return selected, highlighted_tokens(analysis["position_texts"], selected)

        def clear_selection(analysis):
            if analysis is None:
                return [], gr.update()
            return [], highlighted_tokens(analysis["position_texts"], [])

        def add_intervention(
            interventions, kind, token_ref, swap_to_ref, layer, strength, basis
        ):
            tokenizer = model_manager.get_tokenizer()
            if tokenizer is None:
                return interventions, gr.update(), "Error: No model loaded."
            try:
                token_id = token_ref_to_id(token_ref, tokenizer)
                token_id_to = (
                    token_ref_to_id(swap_to_ref, tokenizer) if kind == "swap" else None
                )
                iv = Intervention(
                    kind=kind,
                    layer=int(layer),
                    token_id=token_id,
                    strength=float(strength),
                    token_id_to=token_id_to,
                    basis=basis,
                )
                updated = [*interventions, iv]
                set_active_interventions(updated)
                return (
                    updated,
                    interventions_dataframe(updated, tokenizer),
                    f"Added: {iv.describe(tokenizer)}. "
                    f"{len(updated)} intervention(s) — regenerate to apply.",
                )
            except ValueError as e:
                return interventions, gr.update(), f"Error: {e}"

        def remove_intervention(interventions, index):
            tokenizer = model_manager.get_tokenizer()
            index = int(index) if index is not None else -1
            if not 0 <= index < len(interventions):
                return interventions, gr.update(), f"Error: no intervention #{index}"
            updated = [iv for i, iv in enumerate(interventions) if i != index]
            set_active_interventions(updated)
            return (
                updated,
                interventions_dataframe(updated, tokenizer),
                f"Removed #{index}. {len(updated)} intervention(s) — regenerate to apply.",
            )

        def clear_interventions(_interventions):
            set_active_interventions([])
            return [], interventions_dataframe([], None), (
                "Cleared all interventions — regenerate to apply."
            )

        def toggle_swap_field(kind):
            return gr.update(visible=kind == "swap")

        # ----------------------------------------------------------- fit file

        def fit_file_status():
            """Describe the fit-file situation for the loaded model."""
            model_name = model_manager.get_model_name()
            if model_name is None:
                return "No model loaded."
            store = get_lens_store()
            lens = store.get(model_name)
            path = store.lens_path(model_name)
            if lens is None:
                return (
                    f"No fitted lens for {model_name}.\n"
                    f"Expected at: {path}\n"
                    f"Fit one on a GPU instance: miru-tracer-fit-lens {model_name}"
                )
            return (
                f"Fitted lens loaded for {model_name}: "
                f"{len(lens.source_layers)} layers "
                f"(L{lens.source_layers[0]}..L{lens.source_layers[-1]}), "
                f"averaged over {lens.n_prompts} prompts.\nPath: {path}"
            )

        def install_fit_file(filepath):
            """Validate an uploaded lens.pt and install it for the loaded model."""
            if filepath is None:
                return fit_file_status()
            model = model_manager.get_model()
            model_name = model_manager.get_model_name()
            if model is None:
                return "Error: Load a model first — the fit file is stored per model."
            try:
                lens = JacobianLens.load(filepath)
            except Exception as e:
                return f"Error: not a valid lens file: {e}"

            d_model = model.config.get_text_config().hidden_size
            n_layers = model.config.get_text_config().num_hidden_layers
            if lens.d_model != d_model:
                return (
                    f"Error: fit file has d_model={lens.d_model}, but "
                    f"{model_name} has d_model={d_model}. This lens was fitted "
                    "for a different model."
                )
            if lens.source_layers[-1] >= n_layers:
                return (
                    f"Error: fit file covers layer {lens.source_layers[-1]}, but "
                    f"{model_name} only has {n_layers} layers."
                )

            path = get_lens_store().lens_path(model_name)
            path.parent.mkdir(parents=True, exist_ok=True)
            lens.save(str(path))
            logger.info(f"Installed fit file for {model_name} at {path}")
            return f"Installed.\n{fit_file_status()}"

        # ------------------------------------------------------------ wiring

        mode_selector.change(
            fn=toggle_mode_visibility,
            inputs=[mode_selector],
            outputs=[completion_inputs, chat_inputs],
        )
        iv_kind.change(fn=toggle_swap_field, inputs=[iv_kind], outputs=[iv_swap_to])

        lens_controls = [
            lens_mode, layer_start, layer_end, layer_stride,
            readouts_per_cell, skip_non_words, pinned_tokens,
        ]

        generate_button.click(
            fn=generate_and_analyze,
            inputs=[
                mode_selector, prompt_input, chat_messages,
                max_tokens, strategy, temperature,
                interventions_state, *lens_controls,
            ],
            outputs=[*readout_outputs, text_output, analysis_state, positions_state],
        )
        update_button.click(
            fn=render_readouts,
            inputs=[analysis_state, positions_state, *lens_controls],
            outputs=readout_outputs,
        )
        tokens_display.select(
            fn=on_token_select,
            inputs=[positions_state, analysis_state],
            outputs=[positions_state, tokens_display],
        )
        select_generated_button.click(
            fn=select_generated,
            inputs=[analysis_state],
            outputs=[positions_state, tokens_display],
        )
        select_clear_button.click(
            fn=clear_selection,
            inputs=[analysis_state],
            outputs=[positions_state, tokens_display],
        )

        iv_add_button.click(
            fn=add_intervention,
            inputs=[
                interventions_state, iv_kind, iv_token, iv_swap_to,
                iv_layer, iv_strength, iv_basis,
            ],
            outputs=[interventions_state, iv_table, status_output],
        )
        iv_remove_button.click(
            fn=remove_intervention,
            inputs=[interventions_state, iv_remove_index],
            outputs=[interventions_state, iv_table, status_output],
        )
        iv_clear_button.click(
            fn=clear_interventions,
            inputs=[interventions_state],
            outputs=[interventions_state, iv_table, status_output],
        )

        lens_file_upload.upload(
            fn=install_fit_file, inputs=[lens_file_upload], outputs=[lens_file_status]
        )
        lens_file_check_button.click(
            fn=fit_file_status, inputs=[], outputs=[lens_file_status]
        )

    return tab
