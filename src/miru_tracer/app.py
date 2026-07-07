"""
Miru Tracer

An experimental tool for interactive analysis of LLM text generation, token by
token.
"""

from __future__ import annotations

from dotenv import load_dotenv
import gradio as gr

from miru_tracer import __version__
from miru_tracer.config import Settings

# Load environment variables from .env file
load_dotenv()

# Setup logging before importing other modules
from miru_tracer.core.logging_config import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)

from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes
from miru_tracer.core.model_manager import ModelManager
from miru_tracer.ui.model_loader import create_model_loader_tab
from miru_tracer.ui.tokenize_text import create_tokenize_text_tab
from miru_tracer.ui.token_lookup import create_token_lookup_tab
from miru_tracer.ui.logging_mode import create_logging_mode_tab
from miru_tracer.ui.interactive_mode import create_interactive_mode_tab
from miru_tracer.ui.analysis import create_analysis_tab


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
            # Tab 1: Model Loader (returns tab and load state components)
            model_loader_tab, model_loader_state = create_model_loader_tab(
                model_manager
            )

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

        # Wire up app-level load event for Model Loader state
        load_fn, outputs = model_loader_state
        app.load(
            fn=load_fn,
            inputs=[],
            outputs=outputs,
        )

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
    )


if __name__ == "__main__":
    main()
