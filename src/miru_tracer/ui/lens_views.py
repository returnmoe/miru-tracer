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

_SCROLL_DIV = (
    '<div style="overflow:auto; max-height:75vh; max-width:100%; '
    'padding-bottom:6px; margin-top:6px;">'
)
# Grids (heatmap, counts): flush borderless cells, uniform column width
# (content wrapped in fixed-width divs — max-width on td is ignored in auto
# table layout).
_GRID_STYLE = (
    "border:0; border-collapse:collapse; font-size:0.8em; line-height:1.5; "
    "font-family:var(--font-mono, monospace); white-space:nowrap;"
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

READOUT_INSPECTOR_JS = """
const renderLayer = (root, layer) => {
  const wanted = layer == null ? 'all' : String(layer);
  root.querySelectorAll('[data-readout-panel]').forEach((panel) => {
    panel.hidden = panel.dataset.readoutPanel !== wanted;
  });
  const all = root.querySelector('[data-readout-all]');
  if (all) {
    all.classList.toggle('miru-readout-all-active', layer == null);
    all.setAttribute('aria-pressed', String(layer == null));
  }
  root.querySelectorAll('.miru-readout-layer-slot').forEach((slot) => {
    const active = String(slot.dataset.readoutLayer) === wanted;
    slot.classList.toggle('miru-readout-layer-active', active);
    slot.setAttribute('aria-pressed', String(active));
  });
  root.querySelectorAll('.miru-readout-mini [data-readout-layer]').forEach((cell) => {
    cell.classList.toggle('miru-readout-layer-focus', String(cell.dataset.readoutLayer) === wanted);
  });
  const label = root.querySelector('[data-readout-active-label]');
  if (label) label.textContent = layer == null ? 'All Layers' : `Layer ${layer}`;
};
element.addEventListener('pointerover', (event) => {
  const slot = event.target.closest?.('[data-readout-layer]');
  const root = slot?.closest?.('[data-readout-inspector]');
  if (!slot || !root) return;
  renderLayer(root, slot.dataset.readoutLayer);
});
element.addEventListener('pointerout', (event) => {
  const slot = event.target.closest?.('[data-readout-layer]');
  const root = slot?.closest?.('[data-readout-inspector]');
  if (!slot || !root) return;
  const next = event.relatedTarget?.closest?.('[data-readout-layer]');
  if (next?.closest?.('[data-readout-inspector]') === root) return;
  renderLayer(root, root.dataset.lockedLayer || null);
});
element.addEventListener('click', (event) => {
  const all = event.target.closest?.('[data-readout-all]');
  const slot = event.target.closest?.('[data-readout-layer]');
  const root = (all || slot)?.closest?.('[data-readout-inspector]');
  if (!root) return;
  if (all) {
    root.dataset.lockedLayer = '';
    renderLayer(root, null);
  } else if (slot) {
    root.dataset.lockedLayer = slot.dataset.readoutLayer;
    renderLayer(root, slot.dataset.readoutLayer);
  }
});
element.addEventListener('keydown', (event) => {
  if (event.key !== 'Enter' && event.key !== ' ') return;
  const control = event.target.closest?.('[data-readout-all], .miru-readout-layer-slot');
  if (!control) return;
  event.preventDefault();
  control.click();
});
"""

_INSPECTOR_STYLE = """
<style>
.miru-readout-inspector { margin-top: 6px; }
.miru-readout-context { margin: 2px 0 8px; font-size: .9em; }
.miru-readout-note { opacity: .78; font-size: .82em; }
.miru-readout-selector { display:flex; gap:12px; align-items:stretch; margin:8px 0 12px; overflow-anchor:none; }
.miru-readout-all { box-sizing:border-box; display:flex; flex:0 0 76px; flex-direction:column; align-items:center; justify-content:center; width:76px; min-width:76px; max-width:76px; height:46px; min-height:46px; padding:4px 6px; overflow:hidden; border:1px solid rgba(71,85,105,.65); border-radius:7px; background:rgba(100,116,139,.34); font-size:.70em; font-weight:700; line-height:1.12; text-align:center; white-space:normal; text-transform:uppercase; cursor:pointer; user-select:none; }
.miru-readout-all:hover, .miru-readout-all-active { background:rgba(79,70,229,.62) !important; border-color:rgba(67,56,202,.85); color:#fff; }
.miru-readout-layer-track { display:flex; flex:1; min-width:0; height:46px; border:1px solid rgba(71,85,105,.62); border-radius:7px; overflow:hidden; background:rgba(100,116,139,.20); }
.miru-readout-layer-slot { box-sizing:border-box; display:block; flex:1; min-width:2px; height:100%; border:0; border-left:1px solid rgba(30,41,59,.18); border-radius:0; background:rgba(100,116,139,.38); padding:0; cursor:pointer; transition:background-color 80ms ease, box-shadow 80ms ease; }
.miru-readout-layer-slot:hover { background:rgba(71,85,105,.58) !important; box-shadow:inset 0 0 0 1px rgba(30,41,59,.72); }
.miru-readout-layer-active { background:rgba(79,70,229,.72) !important; box-shadow:inset 0 0 0 1px rgba(49,46,129,.90) !important; }
.miru-readout-layer-slot.miru-readout-early { background:rgba(217,119,6,.30); }
.miru-readout-layer-slot.miru-readout-output { border-left:2px solid rgba(79,70,229,.75); }
.miru-readout-layer-labels { display:flex; justify-content:space-between; font-size:.7em; opacity:.65; margin-top:2px; }
.miru-readout-columns { display:grid; grid-template-columns:minmax(0,1fr); gap:16px; }
.miru-readout-columns.miru-readout-compare { grid-template-columns:repeat(2,minmax(0,1fr)); }
.miru-readout-column { min-width:0; }
.miru-readout-column h3 { margin:3px 0 7px; font-size:1em; }
.miru-readout-panel { box-sizing:border-box; height:clamp(20rem,54vh,34rem); overflow-y:auto; overflow-x:auto; overscroll-behavior:contain; overflow-anchor:none; scrollbar-gutter:stable; border:1px solid rgba(127,127,127,.16); border-radius:6px; }
.miru-readout-panel[hidden] { display:none; }
.miru-readout-panel-head { position:sticky; top:0; z-index:2; display:grid; grid-template-columns:minmax(9rem,1fr) 5.5rem 5rem minmax(12rem,1.35fr); gap:8px; min-width:38rem; padding:6px 8px; background:var(--body-background-fill,#fff); font-size:.72em; text-transform:uppercase; opacity:.94; border-bottom:1px solid rgba(127,127,127,.24); }
.miru-readout-row { display:grid; grid-template-columns:minmax(9rem,1fr) 5.5rem 5rem minmax(12rem,1.35fr); gap:8px; min-width:38rem; align-items:center; padding:5px 8px; border-bottom:1px solid rgba(127,127,127,.12); font-size:.88em; }
.miru-readout-token { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; font-family:var(--font-mono,monospace); }
.miru-readout-id { text-align:right; opacity:.72; font-family:var(--font-mono,monospace); font-variant-numeric:tabular-nums; }
.miru-readout-number { text-align:right; font-variant-numeric:tabular-nums; }
.miru-readout-mini { display:flex; height:20px; overflow:hidden; border:1px solid rgba(148,163,184,.28); border-radius:3px; background:rgba(241,245,249,.25); }
.miru-readout-mini span { flex:1; min-width:1px; cursor:pointer; }
.miru-readout-mini .miru-readout-layer-focus { box-shadow:inset 2px 0 rgba(255,255,255,.72), inset -2px 0 rgba(255,255,255,.72); filter:brightness(1.12); }
.miru-readout-warning { margin:5px 8px; padding:5px 8px; border:1px solid rgba(245,158,11,.35); border-radius:5px; background:rgba(245,158,11,.08); font-size:.8em; }
.miru-readout-interventions { margin:7px 0; padding:6px 9px; border:1px solid rgba(245,158,11,.30); border-radius:5px; background:rgba(245,158,11,.07); font-size:.82em; }
.miru-readout-empty { padding:14px 8px; opacity:.7; }
@media (max-width:900px) {
  .miru-readout-columns.miru-readout-compare { grid-template-columns:minmax(0,1fr); }
  .miru-readout-panel { height:clamp(18rem,52vh,30rem); }
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


def _position_header(slice_: LensSlice, index: int) -> tuple[str, str]:
    position = slice_.positions[index]
    token = slice_.position_texts[index]
    relation = f"Readout aligned to token {token}, position {position}"
    return f'{position}<br>{_tok(token)}', relation


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

    header_cells = []
    for index in range(len(slice_.positions)):
        label, relation = _position_header(slice_, index)
        header_cells.append(
            f'<th {_COL_TH} title="{html.escape(relation, quote=True)}">'
            f"{_fixed(label, _HEAT_COL_W)}</th>"
        )
    header = "".join(header_cells)

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
        f"<b>Lens readouts — {slice_.mode}</b> · each column is aligned to its "
        "displayed token using the preceding causal state that produced it · "
        "color = top-1 probability · "
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


def _layer_strip_html(
    row: ReadoutRow | None,
    layers: list[int],
    *,
    interactive: bool,
) -> str:
    counts = row.count_by_layer if row is not None else []
    peak = max(counts, default=0)
    cells = []
    for index, layer in enumerate(layers):
        count = counts[index] if index < len(counts) else 0
        best = (
            row.best_rank_by_layer[index]
            if row is not None and index < len(row.best_rank_by_layer)
            else None
        )
        probability = (
            row.peak_prob_by_layer[index]
            if row is not None and index < len(row.peak_prob_by_layer)
            else 0.0
        )
        alpha = 0.0 if count == 0 else 0.12 + 0.78 * count / max(peak, 1)
        details = [f"L{layer}: {count} occurrence{'s' if count != 1 else ''}"]
        if best is not None:
            details.append(f"best displayed rank {best + 1}")
            details.append(f"peak probability {probability:.3%}")
        data = f' data-readout-layer="{layer}"' if interactive else ""
        cells.append(
            f'<span{data} style="background:rgba({_ACCENT},{alpha:.3f});" '
            f'title="{html.escape(" · ".join(details), quote=True)}"></span>'
        )
    return f'<div class="miru-readout-mini">{"".join(cells)}</div>'


def _aggregate_inspector_rows(
    rows: list[ReadoutRow], layers: list[int], *, interactive: bool
) -> str:
    if not rows:
        return '<div class="miru-readout-empty">No readout tokens in this scope.</div>'
    return "".join(
        '<div class="miru-readout-row">'
        f'<div class="miru-readout-token" title="{html.escape(row.text, quote=True)}">'
        f"{_tok(row.text)}</div>"
        f'<div class="miru-readout-id">{row.token_id}</div>'
        f'<div class="miru-readout-number" title="{row.count} appearances · '
        f'reciprocal-rank tie-break {row.relevance_score:.3f}">'
        f"{row.count}</div>"
        f"{_layer_strip_html(row, layers, interactive=interactive)}"
        "</div>"
        for row in rows
    )


def _exact_layer_rows(
    slice_: LensSlice,
    layer_index: int,
    stats_by_id: dict[int, ReadoutRow],
    *,
    interactive: bool,
) -> str:
    if len(slice_.positions) != 1:
        return ""
    tokens = slice_.tokens[layer_index][0]
    texts = slice_.texts[layer_index][0]
    probabilities = slice_.probs[layer_index][0]
    return "".join(
        '<div class="miru-readout-row">'
        f'<div class="miru-readout-token" title="rank {rank + 1} · '
        f'{html.escape(text, quote=True)}">{_tok(text)}</div>'
        f'<div class="miru-readout-id">{token_id}</div>'
        f'<div class="miru-readout-number">{probability:.2%}</div>'
        f"{_layer_strip_html(stats_by_id.get(token_id), slice_.layers, interactive=interactive)}"
        "</div>"
        for rank, (token_id, text, probability) in enumerate(
            zip(tokens, texts, probabilities, strict=True)
        )
    )


def _inspector_column_html(
    slice_: LensSlice,
    rows: list[ReadoutRow],
    all_rows: list[ReadoutRow],
    *,
    recommended_start: int,
    interactive: bool,
) -> str:
    stats_by_id = {row.token_id: row for row in all_rows}
    all_panel = (
        '<div class="miru-readout-panel" data-readout-panel="all">'
        '<div class="miru-readout-panel-head"><span>Readout token</span>'
        '<span style="text-align:right">ID</span>'
        '<span style="text-align:right">Count ↓</span><span>Count by layer</span></div>'
        f"{_aggregate_inspector_rows(rows, slice_.layers, interactive=interactive)}"
        "</div>"
    )
    layer_panels = []
    if interactive:
        final = slice_.layers[-1]
        for index, layer in enumerate(slice_.layers):
            if layer == final:
                label = f"Layer {layer} · final model distribution for selected token"
            elif slice_.mode == "logit":
                label = f"Layer {layer} · Logit readout for selected token"
            else:
                label = f"Layer {layer} · J-lens concepts at selected token"
            warning = ""
            if slice_.mode == "jacobian" and layer < recommended_start:
                warning = (
                    '<div class="miru-readout-warning">This early fitted layer is '
                    "often degenerate; interpret its J-lens tokens cautiously.</div>"
                )
            layer_panels.append(
                f'<div class="miru-readout-panel" data-readout-panel="{layer}" hidden>'
                f'<div class="miru-readout-panel-head"><span>{label}</span>'
                '<span style="text-align:right">ID</span>'
                '<span style="text-align:right">Probability</span>'
                '<span>Count by layer</span></div>'
                f"{warning}{_exact_layer_rows(slice_, index, stats_by_id, interactive=True)}"
                "</div>"
            )
    return (
        '<section class="miru-readout-column">'
        f"<h3>{slice_.mode.title()} Lens</h3>{all_panel}{''.join(layer_panels)}</section>"
    )


def readout_inspector_html(
    *,
    mode: str,
    slices: dict[str, LensSlice],
    rows: dict[str, list[ReadoutRow]],
    all_rows: dict[str, list[ReadoutRow]],
    recommended_start: int,
    intervened: dict[int, str] | None = None,
) -> str:
    """Neuronpedia-style aggregate and exact-layer readout inspector."""
    modes = ["jacobian", "logit"] if mode == "compare" else [mode]
    if not modes or any(name not in slices for name in modes):
        return ""
    representative = slices[modes[0]]
    single_position = len(representative.positions) == 1

    if single_position:
        position = representative.positions[0]
        token = representative.position_texts[0]
        context = (
            '<p class="miru-readout-context"><b>Selected token</b> '
            f'<code>{_tok(token)}</code> · position {position}</p>'
        )
    else:
        context = (
            f'<p class="miru-readout-context"><b>{len(representative.positions)} positions</b> '
            "aggregated. Select exactly one sequence token to inspect exact per-layer "
            "probabilities.</p>"
        )

    selector = ""
    if single_position and representative.layers:
        final = representative.layers[-1]
        slots = []
        for layer in representative.layers:
            classes = ["miru-readout-layer-slot"]
            if mode in ("jacobian", "compare") and layer < recommended_start:
                classes.append("miru-readout-early")
            if layer == final:
                classes.append("miru-readout-output")
            title = f"Layer {layer}"
            if layer == final:
                title += " · final model distribution for the selected token"
            elif layer < recommended_start and mode in ("jacobian", "compare"):
                title += " · early J-lens layer; often degenerate"
            slots.append(
                f'<span role="button" tabindex="0" class="{" ".join(classes)}" '
                f'data-readout-layer="{layer}" title="{title}" '
                f'aria-label="Preview layer {layer}" aria-pressed="false"></span>'
            )
        selector = (
            '<div class="miru-readout-selector">'
            '<span role="button" tabindex="0" '
            'class="miru-readout-all miru-readout-all-active" '
            'data-readout-all aria-label="Show all layers" aria-pressed="true">'
            '<span>All</span><span>Layers</span></span>'
            '<div style="flex:1;min-width:0">'
            f'<div class="miru-readout-layer-track">{"".join(slots)}</div>'
            '<div class="miru-readout-layer-labels">'
            f"<span>Layer {representative.layers[0]}</span>"
            '<span data-readout-active-label>All Layers</span>'
            f"<span>Layer {representative.layers[-1]} · output</span>"
            "</div></div></div>"
        )

    columns = "".join(
        _inspector_column_html(
            slices[name],
            rows[name],
            all_rows[name],
            recommended_start=recommended_start,
            interactive=single_position,
        )
        for name in modes
    )
    compare_class = " miru-readout-compare" if mode == "compare" else ""
    intervention_banner = ""
    if intervened:
        items = "; ".join(
            f"L{layer}: {html.escape(desc)}"
            for layer, desc in sorted(intervened.items())
        )
        intervention_banner = (
            '<p class="miru-readout-interventions">⚡ <b>Interventions</b> — '
            f"{items}</p>"
        )
    return (
        f"{_INSPECTOR_STYLE}"
        '<div class="miru-readout-inspector" data-readout-inspector>'
        f"{context}{intervention_banner}"
        '<p class="miru-readout-note">Readouts are aligned to the selected token: '
        "for token p, Miru decodes the preceding causal state p−1 that produced it. "
        "The final row is therefore the model distribution for the selected token, "
        "not for the token that follows it.</p>"
        f'{selector}<div class="miru-readout-columns{compare_class}">{columns}</div>'
        "</div>"
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
