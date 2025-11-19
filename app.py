"""
Miru Tracer

An experimental tool for interactive analysis of LLM text generation, token by
token.
"""

from __future__ import annotations
from typing import Iterable
import os
from dotenv import load_dotenv
import gradio as gr

# Load environment variables from .env file
load_dotenv()

# Setup logging before importing other modules
from src.core.logging_config import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)

from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes
from src.core.models import ModelManager
from src.ui.model_loader import create_model_loader_tab
from src.ui.tokenize_text import create_tokenize_text_tab
from src.ui.token_lookup import create_token_lookup_tab
from src.ui.logging_mode import create_logging_mode_tab
from src.ui.interactive_mode import create_interactive_mode_tab
from src.ui.analysis import create_analysis_tab

# Application version
__version__ = "0.0.1"


class MiruTheme(Base):
    def __init__(self):
        super().__init__(
            primary_hue="pink",
            secondary_hue="sky",
            neutral_hue="gray",
            text_size="lg",
            font=[
                gr.themes.GoogleFont("Inter"),
                "ui-sans-serif",
                "system-ui",
                "sans-serif",
            ],
            font_mono=[
                gr.themes.GoogleFont("IBM Plex Mono"),
                "ui-monospace",
                "Consolas",
                "monospace",
            ],
        )


def create_app() -> gr.Blocks:
    """Create the main Gradio application."""

    footer_mod = f"""
    function() {{
        const footer = document.querySelector('footer');

        if (footer) {{
        // Create version display
        const version = document.createElement('span');
        version.className = 'footer-version';
        version.innerText = 'v{__version__}';

        // Create separator after version
        const separator1 = document.createElement('div');
        separator1.className = 'divider show-api-divider';
        separator1.style.marginLeft = 'var(--size-2)';
        separator1.style.marginRight = 'var(--size-2)';
        separator1.innerHTML = '·';

        // Create powered-by link
        const moe = document.createElement('a');
        moe.href = 'https://return.moe';
        moe.className = 'powered-by';
        moe.innerText = 'Made by return moe; 💜';
        moe.target = '_blank';

        // Create separator after powered-by
        const separator2 = document.createElement('div');
        separator2.className = 'divider show-api-divider';
        separator2.style.marginLeft = 'var(--size-2)';
        separator2.style.marginRight = 'var(--size-2)';
        separator2.innerHTML = '·';

        footer.prepend(separator2);
        footer.prepend(moe);
        footer.prepend(separator1);
        footer.prepend(version);
        }}
    }}
    """

    # Initialize singleton model manager
    model_manager = ModelManager()

    # Create the main app
    with gr.Blocks(
        title="Miru Tracer",
        theme=MiruTheme(),
        js=footer_mod,
        css="""
        /* Custom layout only - theme handles colors, fonts, and component styling */
        main {
            max-width: 1024px !important;
        }

        footer::before {
            content: "";
        }

        /* Monospace textboxes - use theme's monospace font */
        .miru-textbox-mono textarea {
            font-family: var(--font-mono);
        }

        /* Footer items styling */
        .footer-version {
            color: inherit;
        }

        .powered-by {
            color: inherit;
        }

        .powered-by:hover {
            color: var(--body-text-color) !important;
        }
        """,
    ) as app:
        # Create tabs - main center content
        with gr.Tabs():
            # Tab 1: Model Loader
            create_model_loader_tab(model_manager)

            # Tab 2: Tokenize Text
            create_tokenize_text_tab(model_manager)

            # Tab 3: Token Lookup
            create_token_lookup_tab(model_manager)

            # Tab 4: Logging Mode
            create_logging_mode_tab(model_manager)

            # Tab 5: Interactive Mode
            create_interactive_mode_tab(model_manager)

            # Tab 6: Log Analysis
            create_analysis_tab()

    return app


# Create demo instance for Gradio CLI hot reloading
demo = create_app()

if __name__ == "__main__":
    # Check if debug mode is enabled via environment variable
    debug_mode = os.getenv("MIRU_DEBUG", "0") == ("1" or "true")

    logger.info("Miru Tracer application starting...")
    logger.info(f"Server configuration: host=127.0.0.1, port=7860")
    logger.info(f"Debug mode: {'enabled' if debug_mode else 'disabled'}")

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        debug=debug_mode,
    )
