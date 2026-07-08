"""Server-rendered static HTML for the Lens result views.

These replace the Plotly heatmaps and the gr.Dataframe readouts table in the
Lens tab: at (layers x positions) scale, Plotly's per-cell SVG text and
Gradio's virtualized dataframe both misbehave in the browser (relayout
freezes; the table blanking while scrolling). Static HTML has neither
failure mode, renders instantly, and — unlike self-resizing Plotly figures
(see the ResizeObserver note in theme.py) — is safe inside a scroll
container.

Layout notes: each view lives in an ``overflow:auto`` div capped at 75vh, so
wide grids scroll horizontally inside it (the scrollbar sits on the div's
edge, always reachable). Header rows stick to the top and label columns
stick to the left while panning. Colors are self-contained bg+fg pairs or
theme-variable-based, so both light and dark themes stay readable.

The pinned-ranks line chart stays Plotly (small, no per-cell text), as does
the Interactive Mode lens panel (single position).
"""

from __future__ import annotations

import html

from miru_tracer.core.lens import LensSlice, ReadoutRow
from miru_tracer.core.tokenizer_utils import visible_whitespace
from miru_tracer.ui.lens_common import sparkline

# YlOrRd (matches the previous Plotly colorscale)
_YLORRD = [
    (255, 255, 204), (255, 237, 160), (254, 217, 118), (254, 178, 76),
    (253, 141, 60), (252, 78, 42), (227, 26, 28), (189, 0, 38), (128, 0, 38),
]

_BG = "var(--body-background-fill, #fff)"

# One accent encodes "how many readouts" everywhere: counts-grid cell fill
# AND the sparkline glyphs in the readouts table, so more ink = more
# readouts in both themes (the old grayscale grid read inverted vs the
# sparklines in dark mode).
_ACCENT = "79,70,229"  # indigo
_ACCENT_SPARK = "#7c7ff2"  # mid indigo, legible on light and dark

_SCROLL_DIV = (
    '<div style="overflow:auto; max-height:75vh; max-width:100%; '
    'padding-bottom:6px; margin-top:6px;">'
)
# Grids (heatmap, counts): spaced rounded cells. List (readouts): collapsed
# rows with subtle separators.
_GRID_STYLE = (
    "border-collapse:separate; border-spacing:3px; font-size:0.8em; "
    "line-height:1.35; font-family:var(--font-mono, monospace); "
    "white-space:nowrap;"
)
_LIST_STYLE = (
    "border-collapse:collapse; font-size:0.8em; line-height:1.4; "
    "font-family:var(--font-mono, monospace); white-space:nowrap;"
)
_TH = f"background:{_BG}; padding:3px 10px; font-weight:600;"
_COL_TH = f'style="position:sticky; top:0; z-index:1; text-align:left; {_TH}"'
_ROW_TH = f'style="position:sticky; left:0; z-index:1; text-align:right; {_TH}"'
_CORNER_TH = f'style="position:sticky; top:0; left:0; z-index:2; {_TH}"'
_CAPTION = 'style="margin:2px 0 4px 0; font-size:0.85em; opacity:0.9;"'


def _color(v: float) -> tuple[int, int, int]:
    """Interpolate the YlOrRd scale at v in [0, 1]."""
    v = min(max(v, 0.0), 1.0)
    x = v * (len(_YLORRD) - 1)
    i = min(int(x), len(_YLORRD) - 2)
    f = x - i
    lo, hi = _YLORRD[i], _YLORRD[i + 1]
    return tuple(round(a + (b - a) * f) for a, b in zip(lo, hi, strict=True))


def _cell_style(v: float) -> str:
    r, g, b = _color(v)
    # Explicit fg per bg keeps cells readable in light AND dark themes.
    fg = "#111" if (0.299 * r + 0.587 * g + 0.114 * b) > 140 else "#fff"
    return (
        f"background:rgb({r},{g},{b}); color:{fg}; padding:3px 8px; "
        "border-radius:4px; text-align:center;"
    )


def _tok(text: str) -> str:
    return html.escape(visible_whitespace(text))


def heatmap_html(slice_: LensSlice) -> str:
    """Position x layer grid of top-1 readouts; hover (title) lists the top-k."""
    if not slice_.layers or not slice_.positions:
        return ""

    header = "".join(
        f"<th {_COL_TH}>{p}<br>{_tok(text)}</th>"
        for p, text in zip(slice_.positions, slice_.position_texts, strict=True)
    )

    # Contrast-normalize like Plotly's autoscaled colorbar ("diff" mode can
    # be negative).
    values = [
        slice_.probs[i][j][0] if slice_.probs[i][j] else 0.0
        for i in range(len(slice_.layers))
        for j in range(len(slice_.positions))
    ]
    vmin, vmax = min(values), max(values)
    span = (vmax - vmin) or 1.0

    body = []
    for i in reversed(range(len(slice_.layers))):  # final layer on top
        cells = [f"<th {_ROW_TH}>L{slice_.layers[i]}</th>"]
        for j in range(len(slice_.positions)):
            probs, texts = slice_.probs[i][j], slice_.texts[i][j]
            top1 = _tok(texts[0]) if texts else ""
            hover = "&#10;".join(
                f"{rank + 1}. {_tok(t)} ({p:.3f})"
                for rank, (t, p) in enumerate(zip(texts, probs, strict=True))
            )
            v = ((probs[0] if probs else 0.0) - vmin) / span
            cells.append(f'<td style="{_cell_style(v)}" title="{hover}">{top1}</td>')
        body.append("<tr>" + "".join(cells) + "</tr>")

    value_name = "Δprob (J-lens − logit)" if slice_.mode == "diff" else "probability"
    caption = (
        f"<b>Lens readouts — {slice_.mode}</b> · each cell predicts the NEXT "
        f"token after the column's input token · color = top-1 {value_name} · "
        "hover a cell for its top-k · scroll sideways for more positions"
    )
    return (
        f"<p {_CAPTION}>{caption}</p>"
        f'{_SCROLL_DIV}<table style="{_GRID_STYLE}">'
        f"<tr><th {_CORNER_TH}></th>{header}</tr>{''.join(body)}"
        "</table></div>"
    )


def readouts_table_html(rows: list[ReadoutRow]) -> str:
    """Aggregated readouts table with a per-layer sparkline column."""
    if not rows:
        return ""
    cell = (
        'style="padding:4px 14px; '
        'border-bottom:1px solid rgba(127,127,127,0.18);"'
    )
    cell_num = (
        'style="padding:4px 14px; text-align:right; '
        'border-bottom:1px solid rgba(127,127,127,0.18);"'
    )
    body = "".join(
        "<tr>"
        f"<td {cell}>{_tok(row.text)}</td>"
        f"<td {cell_num}>{row.token_id}</td>"
        f"<td {cell_num}>{row.count}</td>"
        f'<td {cell} title="count per layer, low to high">'
        f'<span style="color:{_ACCENT_SPARK};">{sparkline(row.count_by_layer)}'
        "</span></td>"
        "</tr>"
        for row in rows
    )
    header = "".join(
        f"<th {_COL_TH}>{name}</th>" for name in ("Token", "ID", "Count", "By layer")
    )
    return (
        f'{_SCROLL_DIV}<table style="{_LIST_STYLE}">'
        f"<tr>{header}</tr>{body}</table></div>"
    )


def distribution_html(
    rows: list[ReadoutRow], layers: list[int], *, limit: int = 20
) -> str:
    """Token x layer grid of readout counts (theme-aware accent scale)."""
    rows = rows[:limit]
    if not rows or not layers:
        return ""
    peak = max(max(row.count_by_layer) for row in rows) or 1

    header = "".join(f"<th {_COL_TH}>L{layer}</th>" for layer in layers)
    body = []
    for row in rows:
        cells = [f"<th {_ROW_TH}>{_tok(row.text)} ({row.count})</th>"]
        for layer, count in zip(layers, row.count_by_layer, strict=True):
            if count:
                alpha = 0.15 + 0.85 * count / peak
                fg = "#fff" if alpha > 0.55 else "var(--body-text-color, inherit)"
                style = (
                    f"background:rgba({_ACCENT},{alpha:.2f}); color:{fg}; "
                    "padding:3px 8px; border-radius:4px; text-align:center; "
                    "min-width:1.6em;"
                )
                label = str(count)
            else:
                style = (
                    "background:rgba(127,127,127,0.08); padding:3px 8px; "
                    "border-radius:4px; min-width:1.6em;"
                )
                label = ""
            cells.append(
                f'<td style="{style}" title="L{layer}: {count} cells">{label}</td>'
            )
        body.append("<tr>" + "".join(cells) + "</tr>")

    return (
        f"<p {_CAPTION}><b>Readout counts by layer</b> · how often each "
        "token appears in the selected cells' top-k</p>"
        f'{_SCROLL_DIV}<table style="{_GRID_STYLE}">'
        f"<tr><th {_CORNER_TH}></th>{header}</tr>{''.join(body)}"
        "</table></div>"
    )
