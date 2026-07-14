"""Model loader tab for Gradio UI."""

from __future__ import annotations

import json

import gradio as gr
import torch

from miru_tracer.config import Settings
from miru_tracer.core.logging_config import get_logger
from miru_tracer.core.model_manager import ModelManager

logger = get_logger(__name__)

QUICK_MODEL_CHOICES = (
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-4B",
    "Other...",
)


CUSTOM_MODEL_CHOICE = "Other..."


def resolve_model_name(quick_choice: str | None, custom_model: str | None) -> str:
    """Resolve the model name from the preset selector plus optional textbox."""
    if quick_choice == CUSTOM_MODEL_CHOICE:
        return (custom_model or "").strip()
    return (quick_choice or "").strip()


def toggle_custom_model_field(quick_choice: str | None):
    """Show the freeform HF model field only for the custom option."""
    if quick_choice == CUSTOM_MODEL_CHOICE:
        return gr.update(visible=True, value="")
    return gr.update(visible=False, value=quick_choice or "")


def create_model_loader_tab(model_manager: ModelManager, settings: Settings):
    """
    Create the model loader tab interface.

    Returns:
        (tab, (load_state_fn, load_state_outputs)) — the tab plus the handler
        the app wires to its load event so a page refresh shows current state.
    """

    def get_memory_usage() -> str:
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            percentage = (allocated / total) * 100 if total > 0 else 0
            return f"{allocated:.2f} GB / {total:.2f} GB ({percentage:.1f}%)"
        return "CPU mode (no GPU)"

    def get_current_model_display() -> str:
        return model_manager.get_model_name() or "No model loaded"

    with gr.Tab("Model Loader") as tab, gr.Column(elem_classes="miru-narrow"):
        gr.Markdown("Load a language model from HuggingFace.")

        with gr.Row():
            current_model_display = gr.Textbox(
                label="Current",
                value=get_current_model_display(),
                interactive=False,
                scale=3,
            )
            memory_usage_display = gr.Textbox(
                label="VRAM usage",
                value=get_memory_usage(),
                interactive=False,
                scale=2,
            )

        with gr.Row():
            model_info_display = gr.Code(
                label="Details", language="json", interactive=False, value=""
            )

        with gr.Row(), gr.Column(scale=3):
            quick_model = gr.Dropdown(
                choices=list(QUICK_MODEL_CHOICES),
                value=QUICK_MODEL_CHOICES[0],
                label="Quick select",
                info="Pick a common model, or choose Other... for a HuggingFace identifier.",
            )
            model_name = gr.Textbox(
                label="HuggingFace model",
                placeholder="e.g., Qwen/Qwen3-0.6B",
                value=QUICK_MODEL_CHOICES[0],
                visible=False,
                info="Load a model using its HF identifier",
            )
            with gr.Row():
                quantization = gr.Radio(
                    choices=["none", "4bit", "8bit"],
                    value="none",
                    label="Quantization",
                    info="Reduce memory usage (requires CUDA)",
                )
                trust_remote_code = gr.Checkbox(
                    label="Trust remote code",
                    value=False,
                    interactive=settings.allow_remote_code,
                    info=(
                        "Enabled by MIRU_ALLOW_REMOTE_CODE=1; repository code can "
                        "execute inside the server."
                        if settings.allow_remote_code
                        else "Disabled by server policy (MIRU_ALLOW_REMOTE_CODE=0)."
                    ),
                )
                minimize_ram = gr.Checkbox(
                    label="Minimize RAM usage",
                    value=False,
                    info="Load directly to GPU with minimal RAM (slower loading)",
                )

        with gr.Row():
            status_output = gr.Textbox(
                label="Status",
                lines=5,
                interactive=False,
                buttons=["copy"],
            )

        with gr.Row():
            load_button = gr.Button("Load Model", variant="primary", size="lg")
            unload_button = gr.Button("Unload Model", variant="stop", size="lg")

        outputs = [
            status_output,
            model_info_display,
            current_model_display,
            memory_usage_display,
            load_button,
            unload_button,
        ]

        def buttons_enabled():
            return gr.update(interactive=True), gr.update(interactive=True)

        def buttons_disabled():
            return gr.update(interactive=False), gr.update(interactive=False)

        def load_model_handler(
            quick_model_val, model_name_val, quant_val, trust_val, minimize_ram_val
        ):
            """Load a model, streaming status so the user sees progress."""
            model_name_val = resolve_model_name(quick_model_val, model_name_val)
            if not model_name_val:
                logger.warning("Model load attempted with empty model name")
                yield (
                    "Error: Please enter a model name",
                    "",
                    get_current_model_display(),
                    get_memory_usage(),
                    *buttons_enabled(),
                )
                return

            logger.info(
                f"Model load requested via UI: {model_name_val} "
                f"(quantization={quant_val}, trust_remote_code={trust_val}, "
                f"minimize_ram={minimize_ram_val})"
            )
            if trust_val:
                if not settings.allow_remote_code:
                    logger.warning("Blocked trust_remote_code request by server policy")
                    yield (
                        "Error: Remote repository code is disabled by server policy.",
                        "",
                        get_current_model_display(),
                        get_memory_usage(),
                        *buttons_enabled(),
                    )
                    return
                logger.warning("trust_remote_code=True enabled (security risk)")

            # Immediate feedback while the (long) download/load runs.
            progress_lines = [f"Loading model: {model_name_val}"]
            if quant_val != "none":
                progress_lines.append(f"Quantization: {quant_val}")
            if trust_val:
                progress_lines.append("Warning: trust_remote_code=True")
            if minimize_ram_val:
                progress_lines.append("RAM optimization: enabled (slower loading)")
            progress_lines.append("")
            progress_lines.append(
                "Downloading/loading weights — this can take a while "
                "for large models. Please wait..."
            )
            yield (
                "\n".join(progress_lines),
                gr.update(),
                gr.update(),
                gr.update(),
                *buttons_disabled(),
            )

            try:
                from miru_tracer.ui.lens_common import set_active_interventions

                set_active_interventions([])
                model, tokenizer, device, info = model_manager.load_model(
                    model_name=model_name_val,
                    quantization=quant_val,
                    trust_remote_code=trust_val,
                    minimize_ram_usage=minimize_ram_val,
                )

                success_lines = [
                    "Model loaded successfully.",
                    f"Device: {info['device_name']}",
                    f"Vocabulary size: {info['vocab_size']:,}",
                    f"Parameters: {info['num_parameters_b']:.2f}B",
                ]
                if info["device"] == "cuda":
                    success_lines.append(f"VRAM: {info['vram_gb']:.2f} GB")
                if info.get("quantization_note"):
                    success_lines.append(f"\n⚠️ {info['quantization_note']}")
                if info.get("is_vlm"):
                    success_lines.append(f"\n⚠️ {info['vlm_warning']}")

                logger.info(f"Model load successful via UI: {model_name_val}")
                yield (
                    "\n".join(success_lines),
                    json.dumps(info, indent=2),
                    model_name_val,
                    get_memory_usage(),
                    *buttons_enabled(),
                )

            except RuntimeError as e:
                # Concurrent load/unload in progress
                logger.warning(f"Load blocked: {e}")
                yield (
                    f"Error: {e}",
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    *buttons_enabled(),
                )
            except Exception as e:
                logger.error(
                    f"Model load failed via UI: {model_name_val} - {e}", exc_info=True
                )
                yield (
                    f"Error loading model:\n\n{e}",
                    "",
                    get_current_model_display(),
                    get_memory_usage(),
                    *buttons_enabled(),
                )

        def unload_model_handler():
            try:
                result = model_manager.unload_model()
                from miru_tracer.ui.lens_common import set_active_interventions

                set_active_interventions([])
                status_msg = result["message"]
                cleared_count = result.get("cleared_sessions", 0)
                if cleared_count:
                    status_msg += (
                        f"\n\n{cleared_count} active Interactive Mode session(s) "
                        "were cleared.\nAny in-progress generation work has been lost."
                    )
                if torch.cuda.is_available():
                    status_msg += "\n\nGPU memory has been freed."
                status_msg += (
                    "\n\nNote: Any Logging Mode sessions in other tabs will be "
                    "invalidated.\nYou will need to start a new generation in those tabs."
                )

                logger.info("Model unload completed via UI")
                return (
                    status_msg,
                    "",
                    "No model loaded",
                    get_memory_usage(),
                    *buttons_enabled(),
                )
            except RuntimeError as e:
                logger.warning(f"Unload blocked: {e}")
                return (
                    f"Error: {e}",
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    *buttons_enabled(),
                )

        quick_model.change(
            fn=toggle_custom_model_field,
            inputs=[quick_model],
            outputs=[model_name],
        )
        load_button.click(
            fn=load_model_handler,
            inputs=[
                quick_model, model_name, quantization, trust_remote_code, minimize_ram,
            ],
            outputs=outputs,
        )
        unload_button.click(fn=unload_model_handler, inputs=[], outputs=outputs)

        def load_current_state():
            """Restore displays after a page refresh."""
            status = ""
            model_info = ""
            if model_manager.is_loaded():
                model = model_manager.get_model()
                tokenizer = model_manager.get_tokenizer()
                num_params = model.num_parameters() / 1e9
                status = (
                    f"Device: {model_manager.get_device()}\n"
                    f"Parameters: {num_params:.2f}B\n"
                    f"Vocabulary size: {len(tokenizer):,}"
                )
                model_info = json.dumps(
                    {
                        "model_name": model_manager.get_model_name(),
                        "device": model_manager.get_device(),
                        "vocab_size": len(tokenizer),
                        "num_parameters_b": num_params,
                    },
                    indent=2,
                )
            else:
                status = (
                    "No model currently loaded.\n\n"
                    "Load a model using the controls above."
                )
            return status, get_current_model_display(), get_memory_usage(), model_info

    return tab, (
        load_current_state,
        [status_output, current_model_display, memory_usage_display, model_info_display],
    )
