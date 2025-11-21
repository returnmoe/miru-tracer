"""Model loader tab for Gradio UI."""

import gradio as gr
import json
from core.models import ModelManager
from core.logging_config import get_logger

logger = get_logger(__name__)


def create_model_loader_tab(model_manager: ModelManager) -> gr.Tab:
    """
    Create the model loader tab interface.

    Args:
        model_manager: Singleton ModelManager instance

    Returns:
        Gradio Tab component
    """
    with gr.Tab("Model Loader") as tab:
        gr.Markdown("Load a language model from HuggingFace.")

        with gr.Row():
            with gr.Column(scale=3):
                model_name = gr.Textbox(
                    label="Model Name",
                    placeholder="e.g., Qwen/Qwen3-1.7B",
                    value="Qwen/Qwen3-1.7B",
                    info="HuggingFace model identifier",
                )

                with gr.Row():
                    quantization = gr.Radio(
                        choices=["none", "4bit", "8bit"],
                        value="none",
                        label="Quantization",
                        info="Reduce memory usage (requires CUDA)",
                    )

                    trust_remote_code = gr.Checkbox(
                        label="Trust Remote Code",
                        value=False,
                        info="Security risk: allows model to execute arbitrary code",
                    )

                    minimize_ram = gr.Checkbox(
                        label="Minimize RAM Usage",
                        value=False,
                        info="Load directly to GPU with minimal RAM (slower loading)",
                    )

            with gr.Column(scale=1, elem_classes="fill-height"):
                status_output = gr.Textbox(
                    label="Status",
                    lines=5,
                    interactive=False,
                    show_copy_button=True,
                )

        with gr.Row():
            model_info_display = gr.Code(
                label="Loaded Model Details",
                language="json",
                interactive=False,
                value="",
            )

        with gr.Row():
            load_button = gr.Button("Load Model", variant="primary", size="lg")

        def load_model_handler(model_name_val, quant_val, trust_val, minimize_ram_val):
            """Handle model loading."""
            if not model_name_val:
                logger.warning("Model load attempted with empty model name")
                return "Error: Please enter a model name", ""

            logger.info(
                f"Model load requested via UI: {model_name_val} (quantization={quant_val}, trust_remote_code={trust_val}, minimize_ram={minimize_ram_val})"
            )

            if trust_val:
                logger.warning("trust_remote_code=True enabled (security risk)")

            try:
                status = f"Loading model: {model_name_val}\n"
                status += f"Quantization: {quant_val}\n"
                if trust_val:
                    status += "SECURITY WARNING: trust_remote_code=True\n"
                if minimize_ram_val:
                    status += "RAM optimization: enabled (slower loading)\n"
                status += "\nPlease wait...\n"

                model, tokenizer, device, info = model_manager.load_model(
                    model_name=model_name_val,
                    quantization=quant_val,
                    trust_remote_code=trust_val,
                    minimize_ram_usage=minimize_ram_val,
                )

                success_msg = f"Model loaded successfully.\n"
                success_msg += f"Device: {info['device_name']}\n"
                success_msg += f"Vocabulary size: {info['vocab_size']:,}\n"
                success_msg += f"Parameters: {info['num_parameters_b']:.2f}B"

                if info["device"] == "cuda":
                    success_msg += f"\nVRAM: {info['vram_gb']:.2f} GB"

                # Display VLM warning
                if info.get("is_vlm"):
                    success_msg += f"\n\n⚠️ {info['vlm_warning']}"

                logger.info(f"Model load successful via UI: {model_name_val}")

                return success_msg, json.dumps(info, indent=2)

            except Exception as e:
                error_msg = f"Error loading model:\n\n{str(e)}"
                logger.error(
                    f"Model load failed via UI: {model_name_val} - {str(e)}",
                    exc_info=True,
                )
                return error_msg, ""

        load_button.click(
            fn=load_model_handler,
            inputs=[model_name, quantization, trust_remote_code, minimize_ram],
            outputs=[status_output, model_info_display],
        )

    return tab
