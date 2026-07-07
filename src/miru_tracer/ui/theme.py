"""Theme, CSS, and footer script for the Miru Tracer UI.

Gradio 6 moved theme/css/js from the Blocks constructor to launch();
``launch_kwargs()`` bundles everything so app entry points and tests stay in
sync.
"""

from __future__ import annotations

import gradio as gr
from gradio.themes.base import Base


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


MIRU_CSS = """
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
"""


def footer_js(version: str) -> str:
    """Inject version and attribution into the Gradio footer.

    Feature-detects the footer DOM: if Gradio's internal structure changes,
    this silently does nothing instead of breaking the app.
    """
    return f"""
    function() {{
        const footer = document.querySelector('footer');
        if (!footer) return;

        const version = document.createElement('span');
        version.className = 'footer-version';
        version.innerText = 'v{version}';

        const separator1 = document.createElement('div');
        separator1.className = 'divider show-api-divider';
        separator1.style.marginLeft = 'var(--size-2)';
        separator1.style.marginRight = 'var(--size-2)';
        separator1.innerHTML = '·';

        const moe = document.createElement('a');
        moe.href = 'https://return.moe';
        moe.className = 'powered-by';
        moe.innerText = 'Made by return moe; 💜';
        moe.target = '_blank';

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
    """


def launch_kwargs(version: str) -> dict:
    """Keyword arguments for Blocks.launch() carrying the app's look."""
    return {
        "theme": MiruTheme(),
        "css": MIRU_CSS,
        "js": footer_js(version),
    }
