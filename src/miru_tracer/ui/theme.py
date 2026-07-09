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
/* Custom layout only - theme handles colors, fonts, and component styling.
   IMPORTANT: never put width constraints on Gradio-owned layout elements
   (e.g. <main>) — Gradio 6's tab bar re-measures available width on every
   tab switch, and a capped+centered main feeds it a smaller number each
   time (ratcheting shrink). We constrain our own wrapper column instead. */
#miru-shell {
    max-width: 1400px;
    width: 100%;
    margin-left: auto;
    margin-right: auto;
}

/* Standard page content fills the same 1400px shell as the Lens tab. */
.miru-narrow {
    width: 100%;
    margin-left: auto !important;
    margin-right: auto !important;
}

/* NOTE: do not wrap plots in overflow-x containers with fixed-width figures —
   Gradio's responsive plot wrapper + a toggling scrollbar creates a
   ResizeObserver feedback loop that freezes the browser. Wide plots use
   Plotly-native pan/zoom over a windowed axis range instead. */

/* Monospace textboxes - use theme's monospace font */
.miru-textbox-mono textarea {
    font-family: var(--font-mono);
}

/* Token sequence (Lens tab): spans toggle position selection on click */
.miru-token-select .token-container {
    cursor: pointer;
}

.miru-interventions-panel .block,
.miru-interventions-panel .form,
.miru-interventions-panel fieldset {
    border-color: rgba(127, 127, 127, 0.18) !important;
}

.miru-interventions-panel .wrap {
    gap: 0.45rem;
}

.miru-interventions-panel .miru-iv-pair-row {
    align-items: end;
}

.miru-interventions-panel .miru-iv-pair-row > .form {
    align-items: end;
}

.miru-hidden-bridge {
    position: absolute !important;
    width: 1px !important;
    height: 1px !important;
    overflow: hidden !important;
    opacity: 0 !important;
    pointer-events: none !important;
    left: -10000px !important;
    top: auto !important;
}

.miru-iv-table-wrap {
    width: 100%;
    overflow-x: auto;
}

#miru-iv-table,
#miru-iv-table .html-container,
#miru-iv-table .prose {
    border-color: rgba(127, 127, 127, 0.18) !important;
    box-shadow: none !important;
}

#miru-iv-table table,
#miru-iv-table thead,
#miru-iv-table tbody,
#miru-iv-table tr,
#miru-iv-table th,
#miru-iv-table td {
    border-color: rgba(127, 127, 127, 0.18) !important;
}

.miru-iv-table {
    width: 100%;
    border: 1px solid rgba(127, 127, 127, 0.18) !important;
    border-collapse: collapse;
    table-layout: fixed;
    font-size: 0.92rem;
}

.miru-iv-section-title {
    padding-top: 0.75rem;
}

.miru-iv-table th,
.miru-iv-table td {
    border: 1px solid rgba(127, 127, 127, 0.18) !important;
    padding: 0.35rem 0.45rem;
    vertical-align: middle;
    overflow-wrap: anywhere;
}

.miru-iv-table th {
    color: var(--body-text-color-subdued);
    font-weight: 600;
    text-align: left;
}

.miru-iv-table th:nth-child(1),
.miru-iv-table td:nth-child(1) {
    width: 3.2rem;
    text-align: center;
}

.miru-iv-table th:nth-child(2),
.miru-iv-table td:nth-child(2),
.miru-iv-table th:nth-child(6),
.miru-iv-table td:nth-child(6) {
    width: 3.6rem;
}

.miru-iv-table th:nth-child(3),
.miru-iv-table td:nth-child(3),
.miru-iv-table th:nth-child(5),
.miru-iv-table td:nth-child(5) {
    width: 5.5rem;
}

.miru-iv-table th:nth-child(7),
.miru-iv-table td:nth-child(7) {
    width: 5.6rem;
    text-align: right;
}

.miru-iv-row-disabled {
    opacity: 0.58;
}

.miru-iv-toggle {
    cursor: pointer;
}

.miru-iv-delete {
    border: 1px solid var(--border-color-primary);
    border-radius: 6px;
    background: var(--button-secondary-background-fill);
    color: var(--body-text-color);
    cursor: pointer;
    font: inherit;
    padding: 0.2rem 0.45rem;
}

.miru-iv-delete:hover {
    border-color: var(--color-accent);
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
    () => {{
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
