"""Lens tab: layer-by-layer readouts (logit / Jacobian / diff) with
position and layer selection, aggregated readout browsing, multi-intervention
steering, and fit-file management (fitting itself runs offline via the
``miru-tracer-fit-lens`` CLI — see docs/lens-tutorial.md).

Semantics: interventions are applied at generation time. "Update readouts"
re-slices the existing sequence under the interventions it was generated
with; changing the intervention list requires Generate & Analyze again (the
status line says so). This keeps the displayed text and the displayed
readouts always consistent with each other.

Rendering: "Update readouts" (and Generate & Analyze) compute ONE lens slice
and cache it in ``slice_state``; the result views (Readouts / Heatmap /
Pinned ranks) render lazily from that cache when their tab is opened. Only
the currently open view is rendered eagerly — building Plotly figures inside
hidden tabs is what historically triggered relayout/freeze trouble (see
theme.py's ResizeObserver note).
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
    record_lens_activations,
)
from miru_tracer.core.logging_config import get_logger
from miru_tracer.core.model_manager import ModelManager
from miru_tracer.core.sampling import SamplingParams
from miru_tracer.core.tracer import LLMTracer
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
    parse_chat_messages,
    thinking_key,
    toggle_mode_visibility,
    toggle_temperature,
    toggle_think_prefill,
)
from miru_tracer.ui.lens_common import (
    LENS_MODE_CHOICES,
    TOKEN_COLOR_MAP,
    highlighted_tokens,
    interventions_dataframe,
    layer_selection,
    lens_mode_key,
    parse_layer_refs,
    parse_token_refs,
    selection_summary,
    set_active_interventions,
    toggle_position,
    token_ref_to_id,
)
from miru_tracer.ui.lens_views import (
    distribution_html,
    heatmap_html,
    readouts_table_html,
)
from miru_tracer.visualization.plots import plot_pinned_token_ranks

logger = get_logger(__name__)


def create_lens_tab(model_manager: ModelManager) -> gr.Tab:
    """Create the Lens analysis tab."""

    with gr.Tab("Lens") as tab:
        gr.Markdown(
            "Layer-by-layer readouts (logit lens / **Jacobian lens**) with "
            "steer/swap/ablate interventions — see `docs/lens-tutorial.md`. "
            "⚠️ *Experimental: readouts and steering may currently yield "
            "nonsense; the final layer always equals the model's real output.*"
        )

        with gr.Row(equal_height=False):
            # ------------------------------------------ left: workspace
            with gr.Column(scale=3):
                with gr.Group():
                    with gr.Row():
                        mode_selector = gr.Radio(
                            choices=list(GENERATION_MODES),
                            value="Completion",
                            label="Mode",
                            scale=2,
                        )
                        max_tokens = gr.Number(
                            minimum=0, value=12, precision=0, label="New tokens", scale=1
                        )
                    with gr.Group() as completion_inputs:
                        prompt_input = gr.Textbox(
                            label="Prompt",
                            lines=2,
                            value="The capital of France is",
                        )
                    with gr.Group(visible=False) as chat_inputs:
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
                        gr.Markdown(CHAT_MODE_HELP)
                    with gr.Group(visible=False) as raw_inputs:
                        raw_input = gr.Textbox(
                            label="Raw text",
                            lines=4,
                            placeholder=RAW_MODE_PLACEHOLDER,
                            info=RAW_MODE_HELP,
                            elem_classes=["miru-textbox-mono"],
                        )
                    generate_button = gr.Button(
                        "Generate & Analyze", variant="primary", size="lg"
                    )
                    with gr.Accordion("Generation settings", open=False), gr.Row():
                        strategy = gr.Radio(
                            choices=["greedy", "sampling"],
                            value="greedy",
                            label="Strategy",
                        )
                        temperature = gr.Slider(
                            0.1,
                            2.0,
                            value=1.0,
                            step=0.1,
                            label="Temperature",
                            interactive=False,
                            info=TEMPERATURE_GREEDY_INFO,
                        )

                status_output = gr.Textbox(
                    label="Status", interactive=False, lines=2, max_lines=4
                )
                text_output = gr.Textbox(
                    label="Text", lines=4, interactive=False, buttons=["copy"]
                )

                tokens_display = gr.HighlightedText(
                    label="Sequence — click tokens to select positions "
                    "(none = all). A position's readout predicts its NEXT "
                    "token: select “is” to see where “Paris” "
                    "emerges.",
                    value=[],
                    color_map=TOKEN_COLOR_MAP,
                    show_inline_category=False,
                    combine_adjacent=False,
                    elem_classes=["miru-token-select"],
                )
                selection_info = gr.Markdown("")
                with gr.Row():
                    select_generated_button = gr.Button("Select generated", size="sm")
                    select_clear_button = gr.Button("Clear selection", size="sm")

            # ------------------------------------------ right: lens controls
            with gr.Column(scale=2):
                with gr.Group():
                    lens_mode = gr.Radio(
                        choices=list(LENS_MODE_CHOICES),
                        value="Logit",
                        label="Lens",
                        info="Jacobian/Diff need a fitted lens.",
                    )
                    with gr.Row():
                        layer_start = gr.Number(
                            minimum=0, value=0, precision=0, label="From layer"
                        )
                        layer_end = gr.Number(
                            value=-1, precision=0, label="To (-1 = last)"
                        )
                        layer_stride = gr.Number(
                            minimum=1, value=1, precision=0, label="Stride"
                        )
                    with gr.Accordion("Display options", open=False):
                        readouts_per_cell = gr.Slider(
                            1, 50, value=20, step=1, label="Readouts per layer+pos"
                        )
                        skip_non_words = gr.Checkbox(
                            label="Hide non-word tokens", value=False
                        )
                        pinned_tokens = gr.Textbox(
                            label="Pinned tokens",
                            placeholder="comma-separated: Paris, 12345, ...",
                            info="Track these tokens' ranks across layers.",
                        )
                    update_button = gr.Button("Update readouts", variant="secondary")
                    gr.Markdown(
                        "Recomputes the readouts below for the current sequence, "
                        "using the token selection on the left. Interventions "
                        "only change on **Generate & Analyze**."
                    )

                with gr.Accordion("Interventions", open=False):
                    gr.Markdown(
                        "Steer, swap, or ablate readout directions during "
                        "generation — any number at once. Applied on the next "
                        "**Generate & Analyze**."
                    )
                    with gr.Row():
                        iv_kind = gr.Radio(
                            choices=["steer", "swap", "ablate"],
                            value="steer",
                            label="Kind",
                        )
                        iv_token = gr.Textbox(
                            label="Token", placeholder="text or id, e.g. Paris",
                            info="Text or a numeric token id. Leading spaces "
                            "count: ' Paris' ≠ 'Paris'.",
                        )
                        iv_swap_to = gr.Textbox(
                            label="Swap to", placeholder="text or id",
                            visible=False,
                            info="Text or a numeric token id. Leading spaces "
                            "count: ' Paris' ≠ 'Paris'.",
                        )
                    with gr.Row():
                        iv_layer = gr.Textbox(
                            value="0", label="Layer(s)",
                            info="List/ranges allowed: 11,12-15,18 adds one "
                            "intervention per layer.",
                        )
                        iv_strength = gr.Slider(
                            -4.0, 4.0, value=1.0, step=0.1, label="Strength (steer)"
                        )
                        iv_basis = gr.Radio(
                            choices=["jacobian", "logit"],
                            value="jacobian",
                            label="Basis",
                        )
                    with gr.Row():
                        iv_add_button = gr.Button(
                            "Add intervention", variant="secondary"
                        )
                        iv_remove_index = gr.Number(
                            minimum=0, value=0, precision=0,
                            label="#", scale=0, min_width=80,
                        )
                        iv_remove_button = gr.Button("Remove #", size="sm")
                        iv_clear_button = gr.Button("Clear all", size="sm")
                    iv_table = gr.Dataframe(
                        headers=["#", "Intervention", "Basis"],
                        datatype=["number", "str", "str"],
                        label="Active interventions",
                        interactive=False,
                    )

                with gr.Accordion("Jacobian lens fit file", open=False):
                    gr.Markdown(
                        "Fit `J_ℓ` matrices once per model — on a **GPU "
                        "instance**: `miru-tracer-fit-lens <model> "
                        "--dim-batch 32` — then load the `lens.pt` here. "
                        "Logit-lens mode never needs a fit file."
                    )
                    lens_file_status = gr.Textbox(
                        label="Fit file status", interactive=False, lines=3
                    )
                    lens_file_upload = gr.File(
                        label="Load fit file (lens.pt)",
                        file_types=[".pt"],
                        type="filepath",
                    )
                    lens_file_check_button = gr.Button("Check status", size="sm")

        # ------------------------------------- full-width results, lazy views
        # Readouts table and heatmap are server-rendered static HTML (see
        # lens_views.py) — gr.Dataframe and per-cell Plotly text both fall
        # over in the browser at layers x positions scale.
        with gr.Tabs():
            with gr.Tab("Summary") as summary_tab:
                readout_table = gr.HTML("")
            with gr.Tab("Readouts") as readouts_tab:
                dist_plot = gr.HTML("")
            with gr.Tab("Heatmap") as heatmap_tab:
                heatmap_plot = gr.HTML("")
            with gr.Tab("Pinned ranks") as pinned_tab:
                pinned_plot = gr.Plot(label="Pinned token ranks")

        # ------------------------------------------------------------ states
        # analysis_state: dict(input_ids, model_name, iset, n_layers,
        # prompt_len, position_texts)
        analysis_state = gr.State(None)
        positions_state = gr.State([])  # [] = all positions
        interventions_state = gr.State([])  # list[Intervention]
        # slice_state: dict(slice=LensSlice, rows=list[ReadoutRow]) — the
        # cached compute the result views render from.
        slice_state = gr.State(None)
        active_view_state = gr.State("summary")  # which result tab is open

        view_components = [readout_table, dist_plot, heatmap_plot, pinned_plot]

        # ----------------------------------------------------------- compute

        def compute_slice_bundle(
            analysis, positions, mode_choice, l_start, l_end, l_stride,
            per_cell, skip_nw, pinned_text,
        ):
            """One forward pass over the stored sequence -> (bundle, status)."""
            if analysis is None:
                return None, "Generate first."
            model = model_manager.get_model()
            tokenizer = model_manager.get_tokenizer()
            if model is None or analysis["model_name"] != model_manager.get_model_name():
                return None, (
                    "Error: the model changed since this sequence was generated. "
                    "Generate again."
                )

            mode = lens_mode_key(mode_choice)
            jlens = get_lens_store().get(analysis["model_name"])
            if mode in ("jacobian", "diff") and jlens is None:
                return None, (
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
                        return None, (
                            f"Error: none of the selected layers are fitted "
                            f"(lens covers {jlens.source_layers})."
                        )
                else:
                    dropped = []
                pinned_ids = parse_token_refs(pinned_text, tokenizer)
                seq_len = int(analysis["input_ids"].shape[1])
                selected = [p for p in positions if 0 <= p < seq_len] or None

                # The residuals only depend on (sequence, interventions), both
                # frozen per analysis — record once, then every Update is
                # unembed + top-k with no model forward.
                activations = analysis.get("activations")
                if activations is None:
                    activations = record_lens_activations(
                        model,
                        tokenizer,
                        analysis["input_ids"],
                        interventions=analysis["iset"],
                    )
                    analysis["activations"] = activations

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
                    activations=activations,
                )
                rows = aggregate_readouts(slice_)

                n_cells = len(slice_.layers) * len(slice_.positions)
                where = (
                    f"{len(slice_.positions)} selected positions"
                    if selected is not None
                    else f"all {len(slice_.positions)} positions"
                )
                status = (
                    f"{slice_.mode} lens: {len(slice_.layers)} layers × {where} "
                    f"= {n_cells} cells, {len(rows)} distinct readout tokens."
                )
                if dropped:
                    status += f" (skipped unfitted layers: {dropped})"
                if analysis["iset"] is not None:
                    status += f" Interventions active: {len(analysis['iset'])}."
                return {"slice": slice_, "rows": rows}, status
            except Exception as e:
                logger.error(f"Lens readout error: {e}", exc_info=True)
                return None, f"Error: {e}\n\nTraceback:\n{traceback.format_exc()}"

        # ------------------------------------------------------ view renders

        def show_summary_view(bundle):
            return "" if bundle is None else readouts_table_html(bundle["rows"])

        def show_readouts_view(bundle):
            if bundle is None:
                return ""
            return distribution_html(bundle["rows"], bundle["slice"].layers)

        def show_heatmap_view(bundle):
            return "" if bundle is None else heatmap_html(bundle["slice"])

        def show_pinned_view(bundle):
            if bundle is None:
                return None
            return plot_pinned_token_ranks(
                bundle["slice"], model_manager.get_tokenizer()
            )

        def render_active_view(bundle, active_view):
            """One value per view component; only the open view is built, the
            other components are cleared so no stale render survives."""
            table = dist = heat = pin = None
            if bundle is not None:
                if active_view == "readouts":
                    dist = show_readouts_view(bundle)
                elif active_view == "heatmap":
                    heat = show_heatmap_view(bundle)
                elif active_view == "pinned":
                    pin = show_pinned_view(bundle)
                else:
                    table = show_summary_view(bundle)
            return table, dist, heat, pin

        def open_summary_view(bundle):
            return "summary", show_summary_view(bundle)

        def open_readouts_view(bundle):
            return "readouts", show_readouts_view(bundle)

        def open_heatmap_view(bundle):
            return "heatmap", show_heatmap_view(bundle)

        def open_pinned_view(bundle):
            return "pinned", show_pinned_view(bundle)

        # ---------------------------------------------------------- handlers

        def update_readouts(
            analysis, positions, active_view, mode_choice, l_start, l_end,
            l_stride, per_cell, skip_nw, pinned_text,
        ):
            bundle, status = compute_slice_bundle(
                analysis, positions, mode_choice, l_start, l_end, l_stride,
                per_cell, skip_nw, pinned_text,
            )
            return (bundle, *render_active_view(bundle, active_view), status)

        def generate_and_analyze(
            mode, prompt, chat_msgs, raw_text, think_choice, think_text,
            n_tokens, strat, temp,
            interventions, active_view, mode_choice, l_start, l_end, l_stride,
            per_cell, skip_nw, pinned_text,
        ):
            def progress(status, text=""):
                """Streaming yield: only status/text change while generating."""
                return (
                    gr.update(),  # slice_state
                    *[gr.update()] * len(view_components),
                    status,
                    text,
                    gr.update(),  # tokens_display
                    gr.update(),  # selection_info
                    gr.update(),  # analysis_state
                    gr.update(),  # positions_state
                )

            def failed(status):
                return (
                    None,
                    *[None] * len(view_components),
                    status,
                    "",
                    gr.update(),
                    gr.update(),
                    None,
                    gr.update(),
                )

            model = model_manager.get_model()
            tokenizer = model_manager.get_tokenizer()
            device = model_manager.get_device()
            if model is None or tokenizer is None:
                yield failed("Error: No model loaded. Use the Model Loader tab.")
                return

            try:
                tracer = LLMTracer(model, tokenizer, device)
                jlens = get_lens_store().get(model_manager.get_model_name())
                try:
                    tracer.set_interventions(interventions or None, jlens=jlens)
                except ValueError as e:
                    yield failed(
                        f"Error in interventions: {e}\n"
                        "(jacobian basis needs a fitted lens covering that layer "
                        "— fit one below, or use the logit basis)"
                    )
                    return

                if mode == "Chat":
                    tracer.reset(
                        messages=parse_chat_messages(chat_msgs),
                        mode="chat",
                        thinking=thinking_key(think_choice),
                        think_prefill=think_text or "",
                    )
                elif mode == "Raw":
                    tracer.reset(prompt=raw_text, mode="raw")
                else:
                    tracer.reset(prompt=prompt, mode="completion")

                params = SamplingParams(strategy=strat, temperature=float(temp))
                if n_tokens and int(n_tokens) > 0:
                    for _step in tracer.generate_stream(
                        max_new_tokens=int(n_tokens), params=params
                    ):
                        yield progress(
                            f"Generating... {len(tracer.history)}/{int(n_tokens)}",
                            tracer.get_full_text(),
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
                bundle, status = compute_slice_bundle(
                    analysis, positions, mode_choice, l_start, l_end, l_stride,
                    per_cell, skip_nw, pinned_text,
                )
                yield (
                    bundle,
                    *render_active_view(bundle, active_view),
                    status,
                    tracer.get_full_text(),
                    highlighted_tokens(position_texts, positions),
                    selection_summary(positions, len(position_texts)),
                    analysis,
                    positions,
                )
            except ChatValidationError as e:
                yield failed(f"Error: {e}")
            except Exception as e:
                logger.error(f"Lens generate error: {e}", exc_info=True)
                yield failed(f"Error: {e}\n\nTraceback:\n{traceback.format_exc()}")

        def on_token_select(positions, analysis, evt: gr.SelectData):
            if analysis is None:
                return positions, gr.update(), gr.update()
            texts = analysis["position_texts"]
            updated = toggle_position(positions, evt.index)
            return (
                updated,
                highlighted_tokens(texts, updated),
                selection_summary(updated, len(texts)),
            )

        def select_generated(analysis):
            if analysis is None:
                return [], gr.update(), gr.update()
            texts = analysis["position_texts"]
            seq_len = int(analysis["input_ids"].shape[1])
            selected = list(range(analysis["prompt_len"], seq_len))
            return (
                selected,
                highlighted_tokens(texts, selected),
                selection_summary(selected, len(texts)),
            )

        def clear_selection(analysis):
            if analysis is None:
                return [], gr.update(), gr.update()
            texts = analysis["position_texts"]
            return [], highlighted_tokens(texts, []), selection_summary([], len(texts))

        def add_intervention(
            interventions, kind, token_ref, swap_to_ref, layer_refs, strength, basis
        ):
            tokenizer = model_manager.get_tokenizer()
            if tokenizer is None:
                return interventions, gr.update(), "Error: No model loaded."
            try:
                token_id = token_ref_to_id(token_ref, tokenizer)
                token_id_to = (
                    token_ref_to_id(swap_to_ref, tokenizer) if kind == "swap" else None
                )
                added = [
                    Intervention(
                        kind=kind,
                        layer=layer,
                        token_id=token_id,
                        strength=float(strength),
                        token_id_to=token_id_to,
                        basis=basis,
                    )
                    for layer in parse_layer_refs(layer_refs)
                ]
                updated = [*interventions, *added]
                set_active_interventions(updated)
                described = added[0].describe(tokenizer)
                if len(added) > 1:
                    described += f" … (+{len(added) - 1} more layers)"
                return (
                    updated,
                    interventions_dataframe(updated, tokenizer),
                    f"Added: {described}. "
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
        iv_kind.change(fn=toggle_swap_field, inputs=[iv_kind], outputs=[iv_swap_to])

        lens_controls = [
            lens_mode, layer_start, layer_end, layer_stride,
            readouts_per_cell, skip_non_words, pinned_tokens,
        ]

        generate_button.click(
            fn=generate_and_analyze,
            inputs=[
                mode_selector, prompt_input, chat_messages, raw_input,
                thinking_selector, think_prefill_box,
                max_tokens, strategy, temperature,
                interventions_state, active_view_state, *lens_controls,
            ],
            outputs=[
                slice_state, *view_components, status_output, text_output,
                tokens_display, selection_info, analysis_state, positions_state,
            ],
        )
        update_button.click(
            fn=update_readouts,
            inputs=[analysis_state, positions_state, active_view_state, *lens_controls],
            outputs=[slice_state, *view_components, status_output],
        )
        summary_tab.select(
            fn=open_summary_view,
            inputs=[slice_state],
            outputs=[active_view_state, readout_table],
        )
        readouts_tab.select(
            fn=open_readouts_view,
            inputs=[slice_state],
            outputs=[active_view_state, dist_plot],
        )
        heatmap_tab.select(
            fn=open_heatmap_view,
            inputs=[slice_state],
            outputs=[active_view_state, heatmap_plot],
        )
        pinned_tab.select(
            fn=open_pinned_view,
            inputs=[slice_state],
            outputs=[active_view_state, pinned_plot],
        )
        tokens_display.select(
            fn=on_token_select,
            inputs=[positions_state, analysis_state],
            outputs=[positions_state, tokens_display, selection_info],
        )
        select_generated_button.click(
            fn=select_generated,
            inputs=[analysis_state],
            outputs=[positions_state, tokens_display, selection_info],
        )
        select_clear_button.click(
            fn=clear_selection,
            inputs=[analysis_state],
            outputs=[positions_state, tokens_display, selection_info],
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
