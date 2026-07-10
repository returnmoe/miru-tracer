"""Lens tab: layer-by-layer Logit/Jacobian readouts and visual comparison with
position and layer selection, aggregated readout browsing, multi-intervention
steering, and fit-file management (fitting itself runs offline via the
``miru-tracer-fit-lens`` CLI — see docs/lens-tutorial.md).

Semantics: interventions are applied at generation time. "Update readouts"
re-slices the existing sequence under the interventions it was generated
with; changing the intervention list requires Generate & Analyze again (the
status line says so). This keeps the displayed text and the displayed
readouts always consistent with each other.

Rendering: "Update readouts" (and Generate & Analyze) records activations once
and caches one ordinary slice per requested lens. Compare mode contains an
independent Jacobian slice and Logit slice; it never subtracts their scores.
Result views render lazily from that cache when their tab is opened. Only the
currently open view is rendered eagerly — building Plotly figures inside hidden
tabs is what historically triggered relayout/freeze trouble (see theme.py's
ResizeObserver note).
"""

from __future__ import annotations

import traceback

import gradio as gr

from miru_tracer.config import Settings
from miru_tracer.core.interventions import Intervention
from miru_tracer.core.lens import (
    LEGACY_LENS_FILENAME,
    aggregate_readouts,
    compute_lens_slice,
    decode_token,
    get_lens_store,
    record_lens_activations,
)
from miru_tracer.core.lens_fit import validate_lens_provenance
from miru_tracer.core.lens_io import load_lens, save_lens
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
    INTERVENTIONS_TABLE_JS,
    JACOBIAN_DEFAULT_LAYER_FRACTION,
    LENS_MODE_CHOICES,
    TOKEN_COLOR_MAP,
    add_pinned_token,
    apply_intervention_table_action,
    enabled_intervention_group_count,
    enabled_interventions,
    format_layer_refs,
    highlighted_tokens,
    intervened_layer_titles,
    intervention_description,
    intervention_group,
    intervention_visibility_warning,
    interventions_summary,
    interventions_table_html,
    lens_layer_selection,
    lens_mode_key,
    parse_layer_refs,
    pinned_token_choices,
    pinned_tokens_table_html,
    remove_pinned_tokens,
    selection_summary,
    set_active_interventions,
    toggle_position,
    token_mode_key,
    token_ref_to_id,
)
from miru_tracer.ui.lens_views import (
    READOUT_INSPECTOR_JS,
    comparison_heatmap_html,
    heatmap_html,
    readout_inspector_html,
)
from miru_tracer.visualization.plots import (
    plot_pinned_token_ranks,
    plot_pinned_token_ranks_comparison,
)

logger = get_logger(__name__)


def create_lens_tab(model_manager: ModelManager, settings: Settings) -> gr.Tab:
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
                            minimum=0,
                            maximum=settings.max_new_tokens,
                            value=12,
                            precision=0,
                            label="New tokens",
                            scale=1,
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

                    with gr.Accordion(
                        "Interventions",
                        open=False,
                        elem_classes=["miru-interventions-panel"],
                    ):
                        gr.Markdown(
                            "Steer, swap, or ablate readout directions during generation."
                        )
                        with gr.Row(equal_height=True):
                            iv_kind = gr.Radio(
                                choices=["steer", "swap", "ablate"],
                                value="steer",
                                label="Kind",
                                scale=3,
                            )
                            iv_basis = gr.Radio(
                                choices=["jacobian", "logit"],
                                value="jacobian",
                                label="Basis",
                                scale=2,
                            )
                        with gr.Row(
                            equal_height=True, elem_classes=["miru-iv-pair-row"]
                        ):
                            iv_token = gr.Textbox(
                                label="Token",
                                placeholder="e.g. Paris",
                                scale=5,
                                min_width=260,
                            )
                            iv_token_mode = gr.Radio(
                                choices=["Text", "ID"],
                                value="Text",
                                label="Interpret as",
                                scale=2,
                                min_width=160,
                            )
                        with gr.Row(
                            equal_height=True, elem_classes=["miru-iv-pair-row"]
                        ):
                            iv_swap_to = gr.Textbox(
                                label="Swap to",
                                placeholder="e.g. Paris",
                                interactive=False,
                                scale=5,
                                min_width=260,
                            )
                            iv_swap_to_mode = gr.Radio(
                                choices=["Text", "ID"],
                                value="Text",
                                label="Interpret as",
                                interactive=False,
                                scale=2,
                                min_width=160,
                            )
                        with gr.Row(equal_height=True):
                            iv_layer = gr.Textbox(
                                value="0",
                                label="Layer(s)",
                                scale=3,
                            )
                            iv_strength = gr.Slider(
                                -2.0,
                                2.0,
                                value=0.0,
                                step=0.05,
                                label="Strength (steer)",
                                scale=4,
                            )
                        iv_add_button = gr.Button("Add intervention", variant="secondary")
                        gr.Markdown(
                            "#### Active interventions",
                            elem_classes=["miru-iv-section-title"],
                        )
                        iv_table = gr.HTML(
                            interventions_table_html([]),
                            elem_id="miru-iv-table",
                            js_on_load=INTERVENTIONS_TABLE_JS,
                        )
                        with gr.Row():
                            iv_clear_button = gr.Button("Clear all", size="sm")

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
                    generate_button = gr.Button("Generate", variant="primary", size="lg")

                status_output = gr.Textbox(
                    label="Status", interactive=False, lines=2, max_lines=4
                )
                text_output = gr.Textbox(
                    label="Text", lines=4, interactive=False, buttons=["copy"]
                )

            # ------------------------------------------ right: lens controls
            with gr.Column(scale=2):
                with gr.Accordion("Jacobian lens fit file", open=True):
                    gr.Markdown(
                        "Fit `J_ℓ` matrices once per model — on a **GPU "
                        "instance**: `miru-tracer-fit-lens <model> "
                        "--dim-batch 32` — then load the `lens.safetensors` "
                        "here. Logit-lens mode never needs a fit file."
                    )
                    lens_file_status = gr.Textbox(
                        label="Fit file status", interactive=False, lines=3
                    )
                    lens_file_upload = gr.File(
                        label="Load fit file (lens.safetensors or legacy lens.pt)",
                        file_types=[".safetensors", ".pt"],
                        type="filepath",
                    )
                    allow_legacy_lens = gr.Checkbox(
                        label="Allow unverified legacy lens",
                        value=False,
                        info="Required only for artifacts without model/tokenizer provenance.",
                    )
                    lens_file_check_button = gr.Button("Check status", size="sm")

                with gr.Group():
                    lens_mode = gr.Radio(
                        choices=list(LENS_MODE_CHOICES),
                        value="Logit",
                        label="Lens",
                        info="Jacobian/Compare need a fitted lens.",
                    )
                    with gr.Row():
                        layer_start = gr.Number(
                            minimum=-1,
                            value=-1,
                            precision=0,
                            label="From layer (-1 = recommended)",
                        )
                        layer_end = gr.Number(
                            value=-1, precision=0, label="To (-1 = last)"
                        )
                        layer_stride = gr.Number(
                            minimum=1, value=1, precision=0, label="Stride"
                        )
                    with gr.Accordion("Display options", open=False):
                        readouts_per_cell = gr.Number(
                            minimum=1,
                            maximum=50,
                            value=8,
                            precision=0,
                            label="Candidates per layer+position",
                        )
                        readout_rows = gr.Number(
                            minimum=1,
                            maximum=200,
                            value=100,
                            precision=0,
                            label="All Layers rows",
                        )
                        skip_non_words = gr.Checkbox(
                            label="Hide non-word tokens", value=True
                        )
                        gr.Markdown("#### Pinned tokens")
                        with gr.Row():
                            pinned_token_ref = gr.Textbox(
                                label="Token",
                                placeholder="e.g. Paris",
                                info="Add one pinned token at a time.",
                            )
                            pinned_token_mode = gr.Radio(
                                choices=["Text", "ID"],
                                value="Text",
                                label="Interpret as",
                            )
                        pinned_add_button = gr.Button("Add pinned token", size="sm")
                        pinned_tokens_table = gr.HTML(pinned_tokens_table_html([]))
                        pinned_select = gr.Dropdown(
                            choices=[],
                            multiselect=True,
                            label="Pinned tokens to remove",
                        )
                        with gr.Row():
                            pinned_remove_button = gr.Button("Remove selected", size="sm")
                            pinned_clear_button = gr.Button("Clear pinned", size="sm")

                tokens_display = gr.HighlightedText(
                    label="Sequence — click tokens to select positions "
                    "(none = all). Readouts are aligned to the selected token: "
                    "token p uses the preceding causal state p−1 that produced it.",
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

                update_button = gr.Button("Update readouts", variant="secondary")
                gr.Markdown(
                    "Recomputes the readouts below for the current sequence and token selection. "
                    "Interventions only change on **Generate**."
                )

        # ------------------------------------- full-width results, lazy views
        # Readouts and heatmap are server-rendered static HTML (see lens_views.py)
        # — gr.Dataframe and per-cell Plotly text both fall over in the browser
        # at layers x positions scale.
        with gr.Tabs():
            with gr.Tab("Readouts") as readouts_tab:
                dist_plot = gr.HTML("", js_on_load=READOUT_INSPECTOR_JS)
            with gr.Tab("Heatmap") as heatmap_tab:
                heatmap_plot = gr.HTML("")
            with gr.Tab("Pinned ranks") as pinned_tab:
                pinned_plot = gr.Plot(label="Pinned token ranks")

        # ------------------------------------------------------------ states
        # analysis_state: dict(input_ids, model_name, iset, n_layers,
        # prompt_len, position_texts)
        analysis_state = gr.State(None)
        positions_state = gr.State([])  # [] = all positions
        pinned_tokens_state = gr.State([])  # list[token_id]
        # One row per Add click; each row owns its concrete per-layer edits.
        interventions_state = gr.State([])  # list[dict(enabled, interventions)]
        # slice_state: dict(slice=LensSlice, rows=list[ReadoutRow]) — the
        # cached compute the result views render from.
        slice_state = gr.State(None)
        active_view_state = gr.State("readouts")  # which result tab is open

        view_components = [dist_plot, heatmap_plot, pinned_plot]

        # ----------------------------------------------------------- compute

        def compute_slice_bundle(
            analysis, positions, mode_choice, l_start, l_end, l_stride,
            per_cell, row_limit, skip_nw, pinned_ids,
        ):
            """One forward pass over the stored sequence -> (bundle, status)."""
            if analysis is None:
                return None, "Generate first."
            snapshot = model_manager.snapshot()
            if snapshot is None:
                return None, "Error: no model is currently loaded."
            model, tokenizer, _device, generation = snapshot
            if (
                analysis["model_name"] != model_manager.get_model_name()
                or analysis.get("model_generation") != generation
            ):
                return None, (
                    "Error: the model changed since this sequence was generated. "
                    "Generate again."
                )

            mode = lens_mode_key(mode_choice)
            jlens = get_lens_store().get(
                analysis["model_name"], model=model, tokenizer=tokenizer
            )
            if mode in ("jacobian", "compare") and jlens is None:
                return None, (
                    "Error: no fitted Jacobian lens for "
                    f"{analysis['model_name']}. Fit one below or run:\n"
                    f"  miru-tracer-fit-lens {analysis['model_name']}"
                )

            try:
                layers = lens_layer_selection(
                    analysis["n_layers"], l_start, l_end, l_stride, mode
                )
                if mode in ("jacobian", "compare") and jlens is not None:
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
                seq_len = int(analysis["input_ids"].shape[1])
                selected = [p for p in positions if 0 < p < seq_len] or None
                if positions and selected is None:
                    return None, (
                        "Error: token position 0 has no preceding causal state. "
                        "Select a later token."
                    )

                cell_positions = len(selected) if selected is not None else max(seq_len - 1, 0)
                if len(layers) * cell_positions > settings.max_lens_cells:
                    return None, (
                        "Error: requested lens grid exceeds the configured limit of "
                        f"{settings.max_lens_cells} layer-position cells."
                    )

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
                        offload_to_cpu=True,
                    )
                    analysis["activations"] = activations

                readout_limit = max(1, int(per_cell))
                aggregate_limit = max(1, int(row_limit))
                requested_modes = (
                    ("jacobian", "logit") if mode == "compare" else (mode,)
                )
                slices = {}
                rows_by_mode = {}
                all_rows_by_mode = {}
                for readout_mode in requested_modes:
                    slice_ = compute_lens_slice(
                        model,
                        tokenizer,
                        analysis["input_ids"],
                        layers=layers,
                        positions=selected,
                        mode=readout_mode,
                        jlens=jlens,
                        top_k=readout_limit,
                        skip_non_words=bool(skip_nw),
                        pinned_token_ids=list(pinned_ids or []),
                        interventions=analysis["iset"],
                        activations=activations,
                        token_aligned=True,
                    )
                    slices[readout_mode] = slice_
                    all_rows = aggregate_readouts(slice_, limit=None)
                    all_rows_by_mode[readout_mode] = all_rows
                    rows_by_mode[readout_mode] = all_rows[:aggregate_limit]

                representative = slices[requested_modes[0]]
                n_cells = len(representative.layers) * len(representative.positions)
                where = (
                    f"{len(representative.positions)} selected positions"
                    if selected is not None
                    else f"all {len(representative.positions)} positions"
                )
                if mode == "compare":
                    status = (
                        f"comparison: Jacobian and Logit lenses over "
                        f"{len(representative.layers)} layers × {where} = "
                        f"{n_cells} cells each; "
                        f"showing {len(rows_by_mode['jacobian'])} / "
                        f"{len(rows_by_mode['logit'])} of "
                        f"{len(all_rows_by_mode['jacobian'])} / "
                        f"{len(all_rows_by_mode['logit'])} distinct readout tokens."
                    )
                else:
                    status = (
                        f"{representative.mode} lens: "
                        f"{len(representative.layers)} layers × {where} "
                        f"= {n_cells} cells, "
                        f"showing {len(rows_by_mode[mode])} of "
                        f"{len(all_rows_by_mode[mode])} distinct readout tokens."
                    )
                if dropped:
                    status += f" (skipped unfitted layers: {dropped})"
                recommended_start = int(
                    analysis["n_layers"] * JACOBIAN_DEFAULT_LAYER_FRACTION
                )
                explicit_start = int(l_start) if l_start is not None else -1
                if (
                    mode in ("jacobian", "compare")
                    and 0 <= explicit_start < recommended_start
                ):
                    status += (
                        f" Early J-lens layers below L{recommended_start} are often "
                        "degenerate; they were included because the range was explicit."
                    )
                intervened: dict[int, str] = {}
                if analysis["iset"] is not None:
                    ivs = analysis["iset"].interventions
                    intervened = intervened_layer_titles(ivs, tokenizer)
                    status += f" Interventions: {interventions_summary(ivs, tokenizer)}."
                    warning = intervention_visibility_warning(
                        ivs, mode, analysis["n_layers"], tokenizer
                    )
                    if warning:
                        status += f"\n{warning}"
                return {
                    "mode": mode,
                    "slices": slices,
                    "rows": rows_by_mode,
                    "all_rows": all_rows_by_mode,
                    "recommended_start": recommended_start,
                    "intervened": intervened,
                }, status
            except Exception as e:
                logger.error(f"Lens readout error: {e}", exc_info=True)
                return None, (
                    f"Error: {e}"
                    + (f"\n\nTraceback:\n{traceback.format_exc()}" if settings.debug else "")
                )

        # ------------------------------------------------------ view renders

        def show_readouts_view(bundle):
            if bundle is None:
                return ""
            return readout_inspector_html(
                mode=bundle["mode"],
                slices=bundle["slices"],
                rows=bundle["rows"],
                all_rows=bundle["all_rows"],
                recommended_start=bundle["recommended_start"],
                intervened=bundle.get("intervened"),
            )

        def show_heatmap_view(bundle):
            if bundle is None:
                return ""
            if bundle["mode"] == "compare":
                return comparison_heatmap_html(
                    bundle["slices"]["jacobian"],
                    bundle["slices"]["logit"],
                    bundle.get("intervened"),
                )
            return heatmap_html(
                bundle["slices"][bundle["mode"]], bundle.get("intervened")
            )

        def show_pinned_view(bundle):
            if bundle is None:
                return None
            if bundle["mode"] == "compare":
                return plot_pinned_token_ranks_comparison(
                    bundle["slices"]["jacobian"],
                    bundle["slices"]["logit"],
                    model_manager.get_tokenizer(),
                )
            return plot_pinned_token_ranks(
                bundle["slices"][bundle["mode"]], model_manager.get_tokenizer()
            )

        def render_active_view(bundle, active_view):
            """One value per view component; only the open view is built, the
            other components are cleared so no stale render survives."""
            dist = heat = pin = None
            if bundle is not None:
                if active_view == "heatmap":
                    heat = show_heatmap_view(bundle)
                elif active_view == "pinned":
                    pin = show_pinned_view(bundle)
                else:
                    dist = show_readouts_view(bundle)
            return dist, heat, pin

        def open_readouts_view(bundle):
            return "readouts", show_readouts_view(bundle)

        def open_heatmap_view(bundle):
            return "heatmap", show_heatmap_view(bundle)

        def open_pinned_view(bundle):
            return "pinned", show_pinned_view(bundle)

        # ---------------------------------------------------------- handlers

        def update_readouts(
            analysis, positions, active_view, mode_choice, l_start, l_end,
            l_stride, per_cell, row_limit, skip_nw, pinned_ids,
        ):
            bundle, status = compute_slice_bundle(
                analysis, positions, mode_choice, l_start, l_end, l_stride,
                per_cell, row_limit, skip_nw, pinned_ids,
            )
            return (bundle, *render_active_view(bundle, active_view), status)

        def generate_and_analyze(
            mode, prompt, chat_msgs, raw_text, think_choice, think_text,
            n_tokens, strat, temp,
            interventions, active_view, mode_choice, l_start, l_end, l_stride,
            per_cell, row_limit, skip_nw, pinned_ids,
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

            snapshot = model_manager.snapshot()
            if snapshot is None:
                yield failed("Error: No model loaded. Use the Model Loader tab.")
                return
            model, tokenizer, device, generation = snapshot
            if n_tokens is None or not 0 <= int(n_tokens) <= settings.max_new_tokens:
                yield failed(
                    f"Error: New tokens must be between 0 and {settings.max_new_tokens}."
                )
                return

            tracer = None
            try:
                tracer = LLMTracer(model, tokenizer, device)
                jlens = get_lens_store().get(
                    model_manager.get_model_name(), model=model, tokenizer=tokenizer
                )
                active_interventions = enabled_interventions(interventions)
                try:
                    tracer.set_interventions(active_interventions or None, jlens=jlens)
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
                    "input_ids": tracer.input_ids.detach().cpu().clone(),
                    "model_name": model_manager.get_model_name(),
                    "model_generation": generation,
                    "iset": tracer._intervention_set,
                    "n_layers": model.config.get_text_config().num_hidden_layers,
                    "prompt_len": tracer._prompt_len,
                    "position_texts": position_texts,
                }
                positions: list[int] = []  # all
                bundle, status = compute_slice_bundle(
                    analysis, positions, mode_choice, l_start, l_end, l_stride,
                    per_cell, row_limit, skip_nw, pinned_ids,
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
                yield failed(
                    f"Error: {e}"
                    + (f"\n\nTraceback:\n{traceback.format_exc()}" if settings.debug else "")
                )
            finally:
                if tracer is not None:
                    tracer.close()

        def on_token_select(positions, analysis, evt: gr.SelectData):
            if analysis is None:
                return positions, gr.update(), gr.update()
            texts = analysis["position_texts"]
            if int(evt.index) == 0:
                return (
                    positions,
                    highlighted_tokens(texts, positions),
                    "**Selection:** position 0 cannot be read token-aligned because "
                    "the captured sequence has no preceding causal state.",
                )
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
            interventions, kind, token_ref, token_mode, swap_to_ref,
            swap_to_mode, layer_refs, strength, basis,
        ):
            tokenizer = model_manager.get_tokenizer()
            if tokenizer is None:
                return interventions, gr.update(), "Error: No model loaded."
            try:
                token_id = token_ref_to_id(
                    token_ref, tokenizer, token_mode_key(token_mode)
                )
                token_id_to = (
                    token_ref_to_id(swap_to_ref, tokenizer, token_mode_key(swap_to_mode))
                    if kind == "swap"
                    else None
                )
                effective_strength = float(strength) if kind == "steer" else 0.0
                concrete = [
                    Intervention(
                        kind=kind,
                        layer=layer,
                        token_id=token_id,
                        strength=effective_strength,
                        token_id_to=token_id_to,
                        basis=basis,
                    )
                    for layer in parse_layer_refs(layer_refs)
                ]
                group = intervention_group(concrete)
                updated = [*(interventions or []), group]
                set_active_interventions(enabled_interventions(updated))
                described = intervention_description(concrete[0], tokenizer)
                layers = format_layer_refs([iv.layer for iv in concrete])
                action = f"Added: {described} on layers {layers}."
                return (
                    updated,
                    interventions_table_html(updated, tokenizer),
                    f"{action} "
                    f"{enabled_intervention_group_count(updated)} enabled "
                    "intervention group(s) — regenerate to apply.",
                )
            except ValueError as e:
                return interventions, gr.update(), f"Error: {e}"

        def apply_intervention_action(interventions, evt: gr.EventData):
            tokenizer = model_manager.get_tokenizer()
            updated, action_status = apply_intervention_table_action(
                interventions, getattr(evt, "_data", {})
            )
            set_active_interventions(enabled_interventions(updated))
            return (
                updated,
                interventions_table_html(updated, tokenizer),
                f"{action_status} {enabled_intervention_group_count(updated)} enabled "
                "intervention group(s) — regenerate to apply.",
            )

        def clear_interventions(_interventions):
            set_active_interventions([])
            return [], interventions_table_html([], None), (
                "Cleared all interventions — regenerate to apply."
            )

        def add_pinned(pinned_ids, token_ref, token_mode):
            tokenizer = model_manager.get_tokenizer()
            if tokenizer is None:
                return pinned_ids, gr.update(), gr.update(), "Error: No model loaded."
            try:
                updated = add_pinned_token(
                    pinned_ids, token_ref, tokenizer, token_mode_key(token_mode)
                )
                return (
                    updated,
                    pinned_tokens_table_html(updated, tokenizer),
                    gr.update(choices=pinned_token_choices(updated, tokenizer), value=[]),
                    f"{len(updated)} pinned token(s).",
                )
            except ValueError as e:
                return pinned_ids, gr.update(), gr.update(), f"Error: {e}"

        def remove_pinned(pinned_ids, selected):
            tokenizer = model_manager.get_tokenizer()
            updated = remove_pinned_tokens(pinned_ids, selected)
            return (
                updated,
                pinned_tokens_table_html(updated, tokenizer),
                gr.update(choices=pinned_token_choices(updated, tokenizer), value=[]),
                f"{len(updated)} pinned token(s).",
            )

        def clear_pinned(_pinned_ids):
            return [], pinned_tokens_table_html([]), gr.update(choices=[], value=[]), (
                "Cleared pinned tokens."
            )

        def toggle_intervention_fields(kind):
            swap_update = gr.update(interactive=kind == "swap")
            strength_update = gr.update(interactive=kind == "steer")
            return swap_update, swap_update, strength_update

        # ----------------------------------------------------------- fit file

        def fit_file_status():
            """Describe the fit-file situation for the loaded model."""
            model_name = model_manager.get_model_name()
            if model_name is None:
                return "No model loaded."
            store = get_lens_store()
            lens = store.get(
                model_name,
                model=model_manager.get_model(),
                tokenizer=model_manager.get_tokenizer(),
            )
            if lens is None:
                return (
                    f"No fitted lens for {model_name}.\n"
                    f"Expected at: {store.lens_path(model_name)}\n"
                    f"Fit one on a GPU instance: miru-tracer-fit-lens {model_name}"
                )
            return (
                f"Fitted lens loaded for {model_name}: "
                f"{len(lens.source_layers)} layers "
                f"(L{lens.source_layers[0]}..L{lens.source_layers[-1]}), "
                f"averaged over {lens.n_prompts} prompts.\n"
                f"Path: {store.existing_lens_path(model_name)}"
            )

        def install_fit_file(filepath, allow_legacy):
            """Validate an uploaded fit file and install it for the loaded model."""
            if filepath is None:
                return fit_file_status()
            model = model_manager.get_model()
            model_name = model_manager.get_model_name()
            if model is None:
                return "Error: Load a model first — the fit file is stored per model."
            try:
                lens = load_lens(filepath)
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

            tokenizer = model_manager.get_tokenizer()
            provenance_status, provenance_detail = validate_lens_provenance(
                lens, model, tokenizer
            )
            if provenance_status == "mismatch":
                return f"Error: lens provenance mismatch: {provenance_detail}."
            if provenance_status == "legacy" and not allow_legacy:
                return (
                    "Error: this legacy lens has no verifiable model/tokenizer "
                    "provenance. Enable 'Allow unverified legacy lens' to install it."
                )

            path = get_lens_store().lens_path(model_name)
            path.parent.mkdir(parents=True, exist_ok=True)
            save_lens(lens, path)
            # A stale legacy pickle next to the fresh safetensors would keep
            # the unsafe copy around — drop it.
            path.with_name(LEGACY_LENS_FILENAME).unlink(missing_ok=True)
            logger.info(f"Installed fit file for {model_name} at {path}")
            warning = "\nWarning: installed without provenance verification." if provenance_status == "legacy" else ""
            return f"Installed ({provenance_detail}).{warning}\n{fit_file_status()}"

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
        iv_kind.change(
            fn=toggle_intervention_fields,
            inputs=[iv_kind],
            outputs=[iv_swap_to, iv_swap_to_mode, iv_strength],
        )

        lens_controls = [
            lens_mode, layer_start, layer_end, layer_stride,
            readouts_per_cell, readout_rows, skip_non_words, pinned_tokens_state,
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
                interventions_state, iv_kind, iv_token, iv_token_mode,
                iv_swap_to, iv_swap_to_mode, iv_layer, iv_strength, iv_basis,
            ],
            outputs=[interventions_state, iv_table, status_output],
        )
        iv_table.click(
            fn=apply_intervention_action,
            inputs=[interventions_state],
            outputs=[interventions_state, iv_table, status_output],
        )
        iv_clear_button.click(
            fn=clear_interventions,
            inputs=[interventions_state],
            outputs=[interventions_state, iv_table, status_output],
        )

        pinned_add_button.click(
            fn=add_pinned,
            inputs=[pinned_tokens_state, pinned_token_ref, pinned_token_mode],
            outputs=[
                pinned_tokens_state, pinned_tokens_table, pinned_select, status_output,
            ],
        )
        pinned_remove_button.click(
            fn=remove_pinned,
            inputs=[pinned_tokens_state, pinned_select],
            outputs=[
                pinned_tokens_state, pinned_tokens_table, pinned_select, status_output,
            ],
        )
        pinned_clear_button.click(
            fn=clear_pinned,
            inputs=[pinned_tokens_state],
            outputs=[
                pinned_tokens_state, pinned_tokens_table, pinned_select, status_output,
            ],
        )

        lens_file_upload.upload(
            fn=install_fit_file,
            inputs=[lens_file_upload, allow_legacy_lens],
            outputs=[lens_file_status],
        )
        lens_file_check_button.click(
            fn=fit_file_status, inputs=[], outputs=[lens_file_status]
        )

    return tab
