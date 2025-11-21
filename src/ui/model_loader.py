"""Model loader tab for Gradio UI."""

import gradio as gr
import json
import torch
from core.models import ModelManager
from core.session_manager import get_session_manager
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

    # Helper functions for getting current state
    def get_memory_usage() -> str:
        """Get current GPU memory usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1e9  # GB
            total = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
            percentage = (allocated / total) * 100 if total > 0 else 0
            return f"{allocated:.2f} GB / {total:.2f} GB ({percentage:.1f}%)"
        else:
            return "CPU mode (no GPU)"

    def get_current_model_display() -> str:
        """Get display text for currently loaded model."""
        model_name = model_manager.get_model_name()
        if model_name:
            return model_name
        else:
            return "No model loaded"

    # Get initial values
    initial_model = get_current_model_display()
    initial_memory = get_memory_usage()

    with gr.Tab("Model Loader") as tab:
        gr.Markdown("Load a language model from HuggingFace.")

        # Status display row
        with gr.Row():
            current_model_display = gr.Textbox(
                label="Currently Loaded Model",
                value=initial_model,
                interactive=False,
                scale=3,
            )
            memory_usage_display = gr.Textbox(
                label="VRAM usage",
                value=initial_memory,
                interactive=False,
                scale=2,
            )

        with gr.Row():
            with gr.Column(scale=3):
                model_name = gr.Textbox(
                    label="Load a new model",
                    placeholder="e.g., Qwen/Qwen3-1.7B",
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
                        info="Security risk: allows model to execute arbitrary code",
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
            unload_button = gr.Button("Unload Model", variant="stop", size="lg")

        def unload_model_handler():
            """Handle model unloading."""
            try:
                session_manager = get_session_manager()
                session_count = session_manager.get_session_count()

                # Check if there are active sessions
                if session_count > 0:
                    # Clear sessions with warning
                    logger.warning(
                        f"Clearing {session_count} active session(s) before unload"
                    )
                    cleared_count = session_manager.clear_all_sessions()

                    # Unload model
                    result = model_manager.unload_model()

                    status_msg = result["message"]
                    status_msg += f"\n\n{cleared_count} active Interactive Mode session(s) were cleared."
                    status_msg += "\nAny in-progress generation work has been lost."
                else:
                    # No sessions, just unload
                    result = model_manager.unload_model()
                    status_msg = result["message"]

                # Add memory freed info if available
                if torch.cuda.is_available():
                    status_msg += f"\n\nGPU memory has been freed."

                # Warn about Logging Mode
                status_msg += "\n\nNote: Any Logging Mode sessions in other tabs will be invalidated."
                status_msg += "\nYou will need to start a new generation in those tabs."

                logger.info(f"Model unload completed via UI")

                # Return updated displays
                return (
                    status_msg,  # status_output
                    "",  # model_info_display (clear it)
                    "No model loaded",  # current_model_display
                    "N/A",  # memory_usage_display
                    gr.update(interactive=True),  # load_button (re-enable)
                    gr.update(interactive=True),  # unload_button (re-enable)
                )

            except RuntimeError as e:
                # Handle concurrent operation error
                error_msg = f"Error: {str(e)}"
                logger.warning(f"Unload blocked: {str(e)}")
                return (
                    error_msg,
                    gr.update(),  # model_info_display (no change)
                    gr.update(),  # current_model_display (no change)
                    gr.update(),  # memory_usage_display (no change)
                    gr.update(interactive=True),  # load_button (re-enable)
                    gr.update(interactive=True),  # unload_button (re-enable)
                )

        def load_model_handler(model_name_val, quant_val, trust_val, minimize_ram_val):
            """Handle model loading."""
            if not model_name_val:
                logger.warning("Model load attempted with empty model name")
                return (
                    "Error: Please enter a model name",
                    "",
                    get_current_model_display(),
                    get_memory_usage(),
                    gr.update(interactive=True),  # load_button
                    gr.update(interactive=True),  # unload_button
                )

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

                return (
                    success_msg,
                    json.dumps(info, indent=2),
                    model_name_val,  # current_model_display
                    get_memory_usage(),  # memory_usage_display
                    gr.update(interactive=True),  # load_button (re-enable)
                    gr.update(interactive=True),  # unload_button (re-enable)
                )

            except RuntimeError as e:
                # Handle concurrent operation error
                error_msg = f"Error: {str(e)}"
                logger.warning(f"Load blocked: {str(e)}")
                return (
                    error_msg,
                    gr.update(),  # model_info_display (no change)
                    gr.update(),  # current_model_display (no change)
                    gr.update(),  # memory_usage_display (no change)
                    gr.update(interactive=True),  # load_button (re-enable)
                    gr.update(interactive=True),  # unload_button (re-enable)
                )

            except Exception as e:
                error_msg = f"Error loading model:\n\n{str(e)}"
                logger.error(
                    f"Model load failed via UI: {model_name_val} - {str(e)}",
                    exc_info=True,
                )
                return (
                    error_msg,
                    "",
                    get_current_model_display(),
                    get_memory_usage(),
                    gr.update(interactive=True),  # load_button (re-enable)
                    gr.update(interactive=True),  # unload_button (re-enable)
                )

        load_button.click(
            fn=load_model_handler,
            inputs=[model_name, quantization, trust_remote_code, minimize_ram],
            outputs=[
                status_output,
                model_info_display,
                current_model_display,
                memory_usage_display,
                load_button,
                unload_button,
            ],
        )

        unload_button.click(
            fn=unload_model_handler,
            inputs=[],
            outputs=[
                status_output,
                model_info_display,
                current_model_display,
                memory_usage_display,
                load_button,
                unload_button,
            ],
        )

        def load_current_state():
            """Load current model state when page loads/reloads."""
            current_model = get_current_model_display()
            current_memory = get_memory_usage()

            # Generate status message based on current state
            status = ""
            model_info = ""

            if model_manager.is_loaded():
                model = model_manager.get_model()
                tokenizer = model_manager.get_tokenizer()
                device = model_manager.get_device()
                model_name = model_manager.get_model_name()

                if model is not None:
                    num_params = model.num_parameters() / 1e9

                    # Build status message
                    status += f"Device: {device}\n"
                    status += f"Parameters: {num_params:.2f}B\n"
                    status += f"Vocabulary size: {len(tokenizer):,}"

                    # Build model info JSON
                    info = {
                        "model_name": model_name,
                        "device": device,
                        "vocab_size": len(tokenizer) if tokenizer else 0,
                        "num_parameters_b": num_params,
                    }
                    model_info = json.dumps(info, indent=2)
            else:
                status = "No model currently loaded.\n\nLoad a model using the controls above."

            return status, current_model, current_memory, model_info

    # Return tab and state components for app-level load event
    return tab, (
        load_current_state,
        [
            status_output,
            current_model_display,
            memory_usage_display,
            model_info_display,
        ],
    )
