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

The Interactive Mode lens panel still uses Plotly for its single-position
view; the Lens tab result views are static HTML.
"""

from __future__ import annotations

import html

from miru_tracer.core.lens import LensSlice, ReadoutRow

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
_MUTED_BORDER = "1px solid rgba(127,127,127,0.18)"

_SCROLL_DIV = (
    '<div style="overflow:auto; max-height:75vh; max-width:100%; '
    'padding-bottom:6px; margin-top:6px;">'
)
# Grids (heatmap, counts): flush borderless cells, uniform column width
# (content wrapped in fixed-width divs — max-width on td is ignored in auto
# table layout). List (readouts): collapsed rows with subtle separators.
_GRID_STYLE = (
    "border:0; border-collapse:collapse; font-size:0.8em; line-height:1.5; "
    "font-family:var(--font-mono, monospace); white-space:nowrap;"
)
_LIST_STYLE = (
    "border:0 !important; border-collapse:collapse; font-size:0.95em; line-height:1.55; "
    "font-family:var(--font), var(--body-font), system-ui, sans-serif; white-space:nowrap;"
)
_TH = f"background:{_BG}; padding:3px 10px; font-weight:600; border:0 !important;"
_HEAT_COL_W = "5.5em"  # heatmap column width (top-1 tokens, ellipsized)
_COUNT_COL_W = "2.2em"  # counts-grid column width (small integers)
_COL_TH = f'style="position:sticky; top:0; z-index:1; text-align:left; {_TH}"'
_ROW_TH = f'style="position:sticky; left:0; z-index:1; text-align:right; {_TH}"'
_CORNER_TH = f'style="position:sticky; top:0; left:0; z-index:2; {_TH}"'
_CAPTION = 'style="margin:2px 0 4px 0; font-size:0.85em; opacity:0.9;"'

_COMPARISON_STYLE = """
<style>
.miru-lens-comparison {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
  gap: 16px;
  align-items: start;
  width: 100%;
}
.miru-lens-comparison-panel { min-width: 0; }
.miru-lens-comparison-title {
  margin: 4px 0 8px;
  font-size: 1em;
  font-weight: 650;
}
@media (max-width: 900px) {
  .miru-lens-comparison { grid-template-columns: minmax(0, 1fr); }
}
</style>
""".strip()


def _color(v: float) -> tuple[int, int, int]:
    """Interpolate the YlOrRd scale at v in [0, 1]."""
    v = min(max(v, 0.0), 1.0)
    x = v * (len(_YLORRD) - 1)
    i = min(int(x), len(_YLORRD) - 2)
    f = x - i
    lo, hi = _YLORRD[i], _YLORRD[i + 1]
    return tuple(round(a + (b - a) * f) for a, b in zip(lo, hi, strict=True))


def _cell_colors(v: float) -> tuple[str, str]:
    """(background, foreground) for a normalized value.

    Explicit fg per bg keeps cells readable in light AND dark themes. Black
    text beats white on this palette up to the brightest reds (contrast ~7:1
    vs ~3:1 on #FC4E2A), so only the darkest cells go white.
    """
    r, g, b = _color(v)
    fg = "#111" if (0.299 * r + 0.587 * g + 0.114 * b) > 110 else "#fff"
    return f"rgb({r},{g},{b})", fg


def _fixed(content: str, width: str, color: str | None = None) -> str:
    """Uniform-width cell content; overflow shows as an ellipsis (full text
    stays in the cell's title attribute).

    The foreground color must live on THIS innermost element: Gradio's prose
    CSS colors inner elements directly, so a color set on the surrounding td
    is not inherited.
    """
    style = (
        f"width:{width}; overflow:hidden; text-overflow:ellipsis; "
        "text-align:center; margin:0 auto;"  # center the div in wider cells
    )
    if color:
        style += f" color:{color};"
    return f'<div style="{style}">{content}</div>'


def _count_bar(counts: list[int], layers: list[int] | None = None) -> str:
    peak = max(counts) if counts else 0
    if peak <= 0:
        return ""
    layer_labels = layers or list(range(len(counts)))
    cells = []
    for layer, count in zip(layer_labels, counts, strict=False):
        opacity = 0.08 + (0.82 * count / peak if count else 0)
        noun = "occurrence" if count == 1 else "occurrences"
        cells.append(
            '<span style="display:inline-block; width:8px; height:16px; '
            f'margin-right:2px; border-radius:2px; background:rgba({_ACCENT},{opacity:.3f});" '
            f'title="Layer {layer}: {count} {noun}"></span>'
        )
    return "".join(cells)


def _tok(text: str) -> str:
    return html.escape(text)


def _layer_label(
    layer: int, intervened: dict[int, str] | None, *, glyph: bool = True
) -> str:
    """Layer label, marked and hoverable when the layer carries an intervention.

    Color lives on the innermost span (Gradio's prose CSS defeats inherited
    color); ``quote=True`` because ``describe()`` uses ``repr`` and token text
    can contain quotes or angle brackets.
    """
    if not intervened or layer not in intervened:
        return f"L{layer}"
    title = html.escape(intervened[layer], quote=True)
    mark = "⚡" if glyph else ""
    return (
        f'<span style="color:{_ACCENT_SPARK}; font-weight:600;" '
        f'title="{title}">{mark}L{layer}</span>'
    )


def comparison_html(jacobian_html: str, logit_html: str) -> str:
    """Place independent Jacobian and Logit views in a responsive grid.

    The inputs are already-rendered views.  This helper only arranges them;
    it never combines or derives values from either lens.  The order is
    intentionally fixed so comparison views are consistent throughout the
    UI: Jacobian first, Logit second.
    """
    if not jacobian_html and not logit_html:
        return ""
    return (
        f"{_COMPARISON_STYLE}"
        '<div class="miru-lens-comparison" data-lens-comparison="true">'
        '<section class="miru-lens-comparison-panel" data-lens-mode="jacobian">'
        '<h3 class="miru-lens-comparison-title">Jacobian Lens</h3>'
        f"{jacobian_html}</section>"
        '<section class="miru-lens-comparison-panel" data-lens-mode="logit">'
        '<h3 class="miru-lens-comparison-title">Logit Lens</h3>'
        f"{logit_html}</section></div>"
    )


def _top1_values(slice_: LensSlice) -> list[float]:
    return [
        slice_.probs[i][j][0] if slice_.probs[i][j] else 0.0
        for i in range(len(slice_.layers))
        for j in range(len(slice_.positions))
    ]


def heatmap_html(
    slice_: LensSlice,
    intervened: dict[int, str] | None = None,
    *,
    value_range: tuple[float, float] | None = None,
) -> str:
    """Position x layer grid of top-1 readouts; hover (title) lists the top-k.

    ``intervened`` maps edited layer indices to a hover description, which
    marks those layers' row labels.
    """
    if not slice_.layers or not slice_.positions:
        return ""

    header = "".join(
        f"<th {_COL_TH} title=\"{p}: {_tok(text)}\">"
        f"{_fixed(f'{p}<br>{_tok(text)}', _HEAT_COL_W)}</th>"
        for p, text in zip(slice_.positions, slice_.position_texts, strict=True)
    )

    # Contrast-normalize like Plotly's autoscaled colorbar. Comparison mode
    # supplies one shared range so colors have the same meaning in both panes.
    values = _top1_values(slice_)
    vmin, vmax = value_range or (min(values), max(values))
    span = (vmax - vmin) or 1.0

    body = []
    for i in reversed(range(len(slice_.layers))):  # final layer on top
        cells = [f"<th {_ROW_TH}>{_layer_label(slice_.layers[i], intervened)}</th>"]
        for j in range(len(slice_.positions)):
            probs, texts = slice_.probs[i][j], slice_.texts[i][j]
            top1 = _tok(texts[0]) if texts else ""
            hover = "&#10;".join(
                f"{rank + 1}. {_tok(t)} ({p:.3f})"
                for rank, (t, p) in enumerate(zip(texts, probs, strict=True))
            )
            v = ((probs[0] if probs else 0.0) - vmin) / span
            bg, fg = _cell_colors(v)
            cells.append(
                f'<td style="background:{bg}; padding:5px 4px; border:0;" title="{hover}">'
                f"{_fixed(top1, _HEAT_COL_W, fg)}</td>"
            )
        body.append('<tr style="border:0;">' + "".join(cells) + "</tr>")

    caption = (
        f"<b>Lens readouts — {slice_.mode}</b> · each cell predicts the NEXT "
        "token after the column's input token · color = top-1 probability · "
        "hover a cell for its top-k · scroll sideways for more positions"
    )
    if intervened and any(layer in intervened for layer in slice_.layers):
        caption += " · ⚡ = intervened layer (hover for details)"
    return (
        f"<p {_CAPTION}>{caption}</p>"
        f'{_SCROLL_DIV}<table style="{_GRID_STYLE}">'
        f'<tr style="border:0;"><th {_CORNER_TH}></th>{header}</tr>{"".join(body)}'
        "</table></div>"
    )


def comparison_heatmap_html(
    jacobian_slice: LensSlice,
    logit_slice: LensSlice,
    intervened: dict[int, str] | None = None,
) -> str:
    """Render two ordinary heatmaps with one shared probability scale."""
    if jacobian_slice.mode != "jacobian" or logit_slice.mode != "logit":
        raise ValueError(
            "comparison heatmaps require a jacobian slice followed by a logit slice"
        )

    values = _top1_values(jacobian_slice) + _top1_values(logit_slice)
    shared_range = (0.0, max(values, default=0.0))
    jacobian = heatmap_html(
        jacobian_slice, intervened, value_range=shared_range
    )
    logit = heatmap_html(logit_slice, intervened, value_range=shared_range)
    return comparison_html(jacobian, logit)


def readouts_table_html(
    rows: list[ReadoutRow],
    intervened: dict[int, str] | None = None,
    *,
    layers: list[int] | None = None,
) -> str:
    """Aggregated readouts table with a per-layer sparkline column.

    Rows are tokens (not layers), so ``intervened`` edits are surfaced as a
    caption above the table rather than per-cell markers.
    """
    if not rows:
        return ""
    caption = ""
    if intervened:
        items = "; ".join(
            f"L{layer}: {html.escape(desc)}" for layer, desc in sorted(intervened.items())
        )
        caption = f"<p {_CAPTION}>⚡ <b>Interventions</b> — {items}</p>"
    cell = (
        f'style="padding:9px 16px; border:{_MUTED_BORDER} !important;"'
    )
    cell_token = (
        f'style="padding:9px 16px; border:{_MUTED_BORDER} !important; '
        'font-family:var(--font-mono, monospace);"'
    )
    cell_num = (
        f'style="padding:9px 16px; text-align:left; border:{_MUTED_BORDER} !important;"'
    )
    body = "".join(
        "<tr>"
        f"<td {cell_token}>{_tok(row.text)}</td>"
        f"<td {cell_num}>{row.token_id}</td>"
        f"<td {cell_num}>{row.count}</td>"
        f'<td {cell} title="count per layer, low to high">{_count_bar(row.count_by_layer, layers)}</td>'
        "</tr>"
        for row in rows
    )
    summary_th = (
        f'style="position:sticky; top:0; z-index:1; text-align:left; '
        f'background:{_BG}; padding:3px 10px; font-weight:600; '
        f'border:{_MUTED_BORDER} !important;"'
    )
    header = "".join(
        f"<th {summary_th}>{name}</th>"
        for name in ("Token", "ID", "Count", "By layer")
    )
    return (
        f"{caption}{_SCROLL_DIV}<table style=\"{_LIST_STYLE}\">"
        f"<tr>{header}</tr>{body}</table></div>"
    )


def distribution_html(
    rows: list[ReadoutRow], layers: list[int], *, limit: int | None = None,
    intervened: dict[int, str] | None = None,
) -> str:
    """Token x layer grid of readout counts (theme-aware accent scale).

    ``intervened`` maps edited layer indices to a hover description, tinting
    those layers' column headers (no glyph — the columns are too narrow).
    """
    rows = rows[:limit] if limit is not None else rows
    if not rows or not layers:
        return ""
    peak = max(max(row.count_by_layer) for row in rows) or 1

    header = "".join(
        (
            f'<th {_COL_TH} title="{html.escape(intervened[layer], quote=True)}">'
            if intervened and layer in intervened
            else f"<th {_COL_TH}>"
        )
        + f"{_fixed(_layer_label(layer, intervened, glyph=False), _COUNT_COL_W)}</th>"
        for layer in layers
    )
    body = []
    for row in rows:
        cells = [f"<th {_ROW_TH}>{_tok(row.text)} ({row.count})</th>"]
        for layer, count in zip(layers, row.count_by_layer, strict=True):
            if count:
                # Alpha capped at 0.8: the theme's own text color then keeps
                # >4.5:1 contrast over the composited cell in BOTH themes (a
                # fixed white/black fg cannot — the effective cell color
                # depends on the theme background under the alpha).
                alpha = 0.15 + 0.65 * count / peak
                style = f"background:rgba({_ACCENT},{alpha:.2f}); padding:5px 0; border:0;"
                label = str(count)
            else:
                style = "padding:5px 0; border:0;"
                label = ""
            cells.append(
                f'<td style="{style}" title="L{layer}: {count} cells">'
                f"{_fixed(label, _COUNT_COL_W)}</td>"
            )
        body.append('<tr style="border:0;">' + "".join(cells) + "</tr>")

    return (
        f"<p {_CAPTION}><b>Readout counts by layer</b> · how often each "
        "token appears in the selected cells' top-k</p>"
        f'{_SCROLL_DIV}<table style="{_GRID_STYLE}">'
        f'<tr style="border:0;"><th {_CORNER_TH}></th>{header}</tr>{"".join(body)}'
        "</table></div>"
    )
