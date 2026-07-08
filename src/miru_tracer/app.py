"""
Miru Tracer

An experimental tool for interactive analysis of LLM text generation, token by
token.
"""

from __future__ import annotations

import gradio as gr
from dotenv import load_dotenv

from miru_tracer import __version__
from miru_tracer.config import Settings

# Load environment variables from .env file before anything reads them
load_dotenv()

# Setup logging before importing other modules
from miru_tracer.core.logging_config import get_logger, setup_logging  # noqa: E402

setup_logging()
logger = get_logger(__name__)

from miru_tracer.core.model_manager import ModelManager  # noqa: E402
from miru_tracer.ui.analysis import create_analysis_tab  # noqa: E402
from miru_tracer.ui.interactive_mode import create_interactive_mode_tab  # noqa: E402
from miru_tracer.ui.lens_tab import create_lens_tab  # noqa: E402
from miru_tracer.ui.logging_mode import create_logging_mode_tab  # noqa: E402
from miru_tracer.ui.model_loader import create_model_loader_tab  # noqa: E402
from miru_tracer.ui.theme import launch_kwargs  # noqa: E402
from miru_tracer.ui.token_lookup import create_token_lookup_tab  # noqa: E402
from miru_tracer.ui.tokenize_text import create_tokenize_text_tab  # noqa: E402


def create_app() -> gr.Blocks:
    """Create the main Gradio application."""
    model_manager = ModelManager()

    with gr.Blocks(title="Miru Tracer", analytics_enabled=False) as app:
        with gr.Tabs():
            _, model_loader_state = create_model_loader_tab(model_manager)
            create_tokenize_text_tab(model_manager)
            create_token_lookup_tab(model_manager)
            create_logging_mode_tab(model_manager)
            create_interactive_mode_tab(model_manager)
            create_lens_tab(model_manager)
            create_analysis_tab()

        # Restore Model Loader displays on page (re)load
        load_fn, load_outputs = model_loader_state
        app.load(fn=load_fn, inputs=[], outputs=load_outputs)

    return app


def main() -> None:
    """Launch the Miru Tracer web application."""
    settings = Settings.from_env()

    logger.info("Miru Tracer application starting...")
    logger.info(
        f"Server configuration: host={settings.server_name}, port={settings.server_port}"
    )
    logger.info(f"Debug mode: {'enabled' if settings.debug else 'disabled'}")

    demo = create_app()

    # Enable queue for event cancellation support
    demo.queue()

    demo.launch(
        server_name=settings.server_name,
        server_port=settings.server_port,
        share=False,
        show_error=True,
        debug=settings.debug,
        **launch_kwargs(__version__),
    )


if __name__ == "__main__":
    main()
