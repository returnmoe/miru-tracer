"""Shared helpers for the lens UI (Lens tab + Interactive Mode panel)."""

from __future__ import annotations

import html
import json

from miru_tracer.core.interventions import Intervention
from miru_tracer.core.tokenizer_utils import (
    format_token_label,
    safe_decode_token,
    visible_whitespace,
)
from miru_tracer.ui.helpers import static_table_html

LENS_MODE_CHOICES = ("Logit", "Jacobian", "Compare (Jacobian / Logit)")
JACOBIAN_DEFAULT_LAYER_FRACTION = 0.29

_SPARK_BLOCKS = "▁▂▃▄▅▆▇█"

INTERVENTIONS_TABLE_JS = """
element.addEventListener('click', (event) => {
    const target = event.target.closest?.('[data-miru-iv-action]');
    if (!target || target.dataset.miruIvAction !== 'delete') return;
    event.preventDefault();
    event.stopPropagation();
    trigger('click', {
        action: 'delete',
        index: Number(target.dataset.index),
    });
});
element.addEventListener('change', (event) => {
    const target = event.target;
    if (!(target instanceof HTMLElement)) return;
    if (target.dataset?.miruIvAction !== 'toggle') return;
    trigger('click', {
        action: 'toggle',
        index: Number(target.dataset.index),
        enabled: target.checked === true,
    });
});
"""

_IV_BORDER = "1px solid rgba(127,127,127,0.18)"
_IV_TABLE_STYLE = (
    f"width:100%; border:{_IV_BORDER} !important; border-collapse:collapse; "
    "table-layout:fixed; font-size:0.92rem;"
)
_IV_CELL_STYLE = (
    f"border:{_IV_BORDER} !important; "
    "padding:0.35rem 0.45rem; vertical-align:middle; overflow-wrap:anywhere;"
)
_IV_TH_STYLE = f"{_IV_CELL_STYLE} color:var(--body-text-color-subdued); font-weight:600; text-align:left;"


def lens_mode_key(ui_choice: str) -> str:
    choice = (ui_choice or "").lower()
    if choice.startswith("jacobian"):
        return "jacobian"
    if choice.startswith("compare"):
        return "compare"
    return "logit"


def token_mode_key(ui_choice: str) -> str:
    """Map the Text/ID selector label to a ``token_ref_to_id`` mode."""
    return "id" if (ui_choice or "").strip().lower() == "id" else "text"


def token_ref_to_id(ref: str, tokenizer, mode: str) -> int:
    """Resolve a user token reference under an explicit interpretation mode.

    ``mode == "id"``: parse a numeric token id; surrounding whitespace is
    tolerated and non-numeric input is rejected. ``mode == "text"``: encode
    the text verbatim — leading/trailing whitespace is significant (" Paris"
    and "Paris" are different BPE tokens) and digits are NEVER treated as an
    id — returning the first token id.

    Raises:
        ValueError: empty ref, unencodable text, non-numeric id, or an
        out-of-range id.
    """
    if mode == "id":
        stripped = ref.strip()
        if not stripped:
            raise ValueError("Empty token reference")
        if not stripped.lstrip("-").isdigit():
            raise ValueError(f"Not a numeric token id: {ref!r}")
        token_id = int(stripped)
        if not 0 <= token_id < len(tokenizer):
            raise ValueError(
                f"Token id {token_id} out of range (vocab size {len(tokenizer)})"
            )
        return token_id
    if not ref.strip():
        raise ValueError("Empty token reference")
    encoded = tokenizer.encode(ref, add_special_tokens=False)
    if not encoded:
        raise ValueError(f"Could not tokenize {ref!r}")
    return int(encoded[0])


def parse_layer_refs(text: str) -> list[int]:
    """Comma-separated layers and inclusive ranges -> unique sorted layer list.

    E.g. ``"11, 12-15, 18"`` -> ``[11, 12, 13, 14, 15, 18]``.

    Raises:
        ValueError: empty input, malformed entry, or a descending range.
    """
    layers: list[int] = []
    for part in (str(text) if text is not None else "").split(","):
        part = part.strip()
        if not part:
            continue
        lo, sep, hi = part.partition("-")
        lo, hi = lo.strip(), hi.strip()
        if not lo.isdigit() or (sep and not hi.isdigit()):
            raise ValueError(
                f"Bad layer reference {part!r}: use a number or range like 12-15"
            )
        start, end = int(lo), int(hi) if sep else int(lo)
        if end < start:
            raise ValueError(f"Descending layer range {part!r}")
        layers.extend(range(start, end + 1))
    if not layers:
        raise ValueError("Empty layer reference")
    return sorted(set(layers))


def parse_token_refs(text: str, tokenizer, mode: str) -> list[int]:
    """Comma-separated token references -> unique token ids (order kept).

    ``mode`` (``"text"``/``"id"``) applies to every entry — a single list is
    all-text or all-id.
    """
    ids: list[int] = []
    for part in (text or "").split(","):
        part = part.strip()
        if not part:
            continue
        token_id = token_ref_to_id(part, tokenizer, mode)
        if token_id not in ids:
            ids.append(token_id)
    return ids


def add_pinned_token(
    pinned_ids: list[int], token_ref: str, tokenizer, mode: str
) -> list[int]:
    """Append one pinned token id, preserving order and avoiding duplicates."""
    token_id = token_ref_to_id(token_ref, tokenizer, mode)
    updated = list(pinned_ids or [])
    if token_id not in updated:
        updated.append(token_id)
    return updated


def remove_pinned_tokens(pinned_ids: list[int], selected: list[str]) -> list[int]:
    selected_ids = {int(s) for s in (selected or [])}
    return [token_id for token_id in (pinned_ids or []) if token_id not in selected_ids]


def pinned_token_choices(pinned_ids: list[int], tokenizer=None) -> list[tuple[str, str]]:
    choices = []
    for token_id in pinned_ids or []:
        label = str(token_id)
        if tokenizer is not None:
            label = f"{token_id}: {format_token_label(tokenizer, token_id)}"
        choices.append((label, str(token_id)))
    return choices


def pinned_tokens_table_html(pinned_ids: list[int], tokenizer=None) -> str:
    if not pinned_ids:
        return '<p style="margin-top:6px; opacity:0.72;">No pinned tokens.</p>'
    rows = []
    for token_id in pinned_ids:
        raw = decoded = str(token_id)
        if tokenizer is not None:
            decoded_text, raw_text, incomplete = safe_decode_token(tokenizer, token_id)
            raw = raw_text or f"<{token_id}>"
            decoded = decoded_text or incomplete or "[unavailable]"
        rows.append([token_id, raw, decoded])
    return static_table_html(["ID", "Token", "Decoded"], rows)


def intervention_group(
    interventions: list[Intervention], *, enabled: bool = True
) -> dict:
    """Build the UI row created by one Add-intervention action."""
    concrete = list(interventions)
    if not concrete:
        raise ValueError("An intervention group needs at least one layer edit")
    return {"enabled": bool(enabled), "interventions": concrete}


def enabled_interventions(groups: list) -> list[Intervention]:
    """Flatten enabled UI groups in click order, then per-group layer order."""
    return [
        intervention
        for group in groups or []
        if group.get("enabled", True)
        for intervention in group["interventions"]
    ]


def enabled_intervention_group_count(groups: list) -> int:
    """Count enabled Add-action rows rather than concrete layer edits."""
    return sum(group.get("enabled", True) for group in groups or [])


def format_layer_refs(layers: list[int]) -> str:
    """Compact sorted layer numbers into inclusive ranges (``14-18, 33``)."""
    ordered = sorted(set(layers))
    ranges: list[str] = []
    start = end = None
    for layer in ordered:
        if start is None:
            start = end = layer
        elif layer == end + 1:
            end = layer
        else:
            ranges.append(str(start) if start == end else f"{start}-{end}")
            start = end = layer
    if start is not None:
        ranges.append(str(start) if start == end else f"{start}-{end}")
    return ", ".join(ranges)


def intervention_description(iv: Intervention, tokenizer=None) -> str:
    """Describe an edit without its layer, which has a dedicated table column."""
    def token_name(token_id: int) -> str:
        if tokenizer is None:
            return str(token_id)
        return repr(format_token_label(tokenizer, token_id))

    if iv.kind == "steer":
        return f"steer {token_name(iv.token_id)} (α={iv.strength:+g})"
    if iv.kind == "ablate":
        return f"ablate {token_name(iv.token_id)}"
    return f"swap {token_name(iv.token_id)}→{token_name(iv.token_id_to)}"


def apply_intervention_table_action(groups: list, payload: str | dict) -> tuple[list, str]:
    """Apply one action emitted by the custom Active Interventions table."""
    if isinstance(payload, dict):
        data = payload
    else:
        try:
            data = json.loads(payload or "{}")
        except (TypeError, json.JSONDecodeError):
            return list(groups or []), "Ignored invalid intervention action."

    action = data.get("action")
    try:
        index = int(data.get("index"))
    except (TypeError, ValueError):
        return list(groups or []), "Ignored invalid intervention action."

    updated = list(groups or [])

    if index < 0 or index >= len(updated):
        return updated, "Ignored invalid intervention action."

    if action == "toggle":
        updated[index] = {**updated[index], "enabled": bool(data.get("enabled"))}
        state = "enabled" if updated[index]["enabled"] else "disabled"
        return updated, f"Intervention group {index} {state}."
    if action == "delete":
        return [group for i, group in enumerate(updated) if i != index], (
            f"Deleted intervention group {index}."
        )
    return updated, "Ignored invalid intervention action."


def sparkline(counts: list[int]) -> str:
    """Compact unicode bar chart of per-layer counts."""
    peak = max(counts) if counts else 0
    if peak == 0:
        return _SPARK_BLOCKS[0] * len(counts)
    return "".join(
        _SPARK_BLOCKS[min(int(c / peak * (len(_SPARK_BLOCKS) - 1) + 0.5), 7)]
        for c in counts
    )


def interventions_table_html(groups: list, tokenizer=None) -> str:
    if not groups:
        return '<p style="margin-top:6px; opacity:0.72;">No active interventions.</p>'

    rows = [
        '<div class="miru-iv-table-wrap">',
        f'<table class="miru-iv-table" style="{_IV_TABLE_STYLE}">',
        "<thead><tr>",
        f'<th style="{_IV_TH_STYLE}; width:3.2rem; text-align:center;">On</th>'
        f'<th style="{_IV_TH_STYLE}; width:6.5rem;">Layers</th>'
        f'<th style="{_IV_TH_STYLE}; width:5.5rem;">Kind</th>'
        f'<th style="{_IV_TH_STYLE}">Intervention</th>'
        f'<th style="{_IV_TH_STYLE}; width:5.5rem;">Basis</th>'
        f'<th style="{_IV_TH_STYLE}; width:3.6rem;">α</th>'
        f'<th style="{_IV_TH_STYLE}; width:5.6rem; text-align:right;"></th>',
        "</tr></thead><tbody>",
    ]
    for i, group in enumerate(groups):
        iv = group["interventions"][0]
        enabled = group.get("enabled", True)
        layers = format_layer_refs([item.layer for item in group["interventions"]])
        strength = f"{iv.strength:+g}" if iv.kind == "steer" else ""
        checked = " checked" if enabled else ""
        muted = "" if enabled else " miru-iv-row-disabled"
        rows.append(
            f'<tr class="{muted.strip()}" data-miru-iv-row="{i}" '
            f'data-miru-iv-enabled="{str(enabled).lower()}">'
            f'<td class="miru-iv-on" style="{_IV_CELL_STYLE} text-align:center;">'
            f'<input type="checkbox" class="miru-iv-toggle" '
            f'data-miru-iv-action="toggle" data-index="{i}"{checked} '
            f'aria-label="Enable intervention group {i} for layers {html.escape(layers)}">'
            "</td>"
            f'<td style="{_IV_CELL_STYLE}">{html.escape(layers)}</td>'
            f'<td style="{_IV_CELL_STYLE}">{html.escape(iv.kind)}</td>'
            f'<td style="{_IV_CELL_STYLE}">'
            f'{html.escape(intervention_description(iv, tokenizer))}</td>'
            f'<td style="{_IV_CELL_STYLE}">{html.escape(iv.basis)}</td>'
            f'<td style="{_IV_CELL_STYLE}">{html.escape(strength)}</td>'
            f'<td class="miru-iv-actions" style="{_IV_CELL_STYLE} text-align:right;">'
            f'<button type="button" class="miru-iv-delete" '
            f'data-miru-iv-action="delete" data-index="{i}" '
            f'aria-label="Remove intervention group {i} for layers '
            f'{html.escape(layers)}">Remove</button>'
            "</td></tr>"
        )
    rows.append("</tbody></table></div>")
    return "".join(rows)


def describe_with_basis(iv: Intervention, tokenizer=None) -> str:
    """Human description of an intervention with its basis appended."""
    return f"{iv.describe(tokenizer)} ({iv.basis})"


def intervened_layer_titles(
    interventions: list[Intervention], tokenizer=None
) -> dict[int, str]:
    """Map each edited layer to a ``'; '``-joined description of its edits."""
    titles: dict[int, str] = {}
    for iv in interventions:
        desc = describe_with_basis(iv, tokenizer)
        titles[iv.layer] = f"{titles[iv.layer]}; {desc}" if iv.layer in titles else desc
    return titles


def interventions_summary(
    interventions: list[Intervention], tokenizer=None, *, limit: int = 4
) -> str:
    """One-line summary of the active interventions for the status area."""
    parts = [describe_with_basis(iv, tokenizer) for iv in interventions[:limit]]
    if len(interventions) > limit:
        parts.append(f"+{len(interventions) - limit} more")
    return "; ".join(parts)


def intervention_visibility_warning(
    interventions: list[Intervention], mode: str, n_layers: int, tokenizer=None
) -> str | None:
    """Warn when an edit's basis differs from the current lens view mode.

    A jacobian-basis edit moves the residual along pre-transport directions
    (``J_ℓ v = û_t``), visible under the Jacobian lens but nearly invisible
    under the Logit lens — and vice versa. The final layer is basis-independent
    (both bases use ``û_t`` directly), and Compare renders both readouts, so
    neither triggers a warning. For a fixed ``mode`` at most one basis can
    mismatch, so this returns a single line (or None).
    """
    if mode == "compare":
        return None
    final = n_layers - 1
    mismatched = [iv for iv in interventions if iv.layer != final and iv.basis != mode]
    if not mismatched:
        return None
    basis = mismatched[0].basis  # only one basis can mismatch a given mode
    shown = "; ".join(iv.describe(tokenizer) for iv in mismatched[:3])
    if len(mismatched) > 3:
        shown += f" (+{len(mismatched) - 3} more)"
    verb = "uses" if len(mismatched) == 1 else "use"
    pronoun = "its" if len(mismatched) == 1 else "their"
    want = "Jacobian" if basis == "jacobian" else "Logit"
    return (
        f"⚠ {shown} {verb} {basis} basis — switch Lens to {want} (or Compare) "
        f"to see {pronoun} effect in the readouts."
    )


def layer_selection(n_layers: int, start, end, stride) -> list[int]:
    """Resolve UI layer-range inputs into a concrete layer list.

    ``end`` is inclusive; -1 (or blank) means the final layer. The final
    selected layer is always included even if the stride skips it.
    """
    start = int(start) if start is not None else 0
    end = int(end) if end is not None else -1
    stride = max(int(stride) if stride else 1, 1)
    if end < 0:
        end = n_layers - 1
    start = max(0, min(start, n_layers - 1))
    end = max(start, min(end, n_layers - 1))
    layers = list(range(start, end + 1, stride))
    if layers[-1] != end:
        layers.append(end)
    return layers


def lens_layer_selection(
    n_layers: int,
    start,
    end,
    stride,
    mode: str,
    *,
    include_final: bool = True,
) -> list[int]:
    """Mode-aware layer selection for lens views.

    ``start=-1`` selects the recommended default: the first 29% of layers are
    skipped for Jacobian/Compare because their fitted readouts are commonly
    degenerate, while Logit starts at L0. An explicit nonnegative start always
    wins. The final identity/output layer remains available as a reference.
    """
    resolved_start = int(start) if start is not None else -1
    if resolved_start < 0:
        resolved_start = (
            int(n_layers * JACOBIAN_DEFAULT_LAYER_FRACTION)
            if mode in ("jacobian", "compare")
            else 0
        )
    layers = layer_selection(n_layers, resolved_start, end, stride)
    final = n_layers - 1
    if include_final and final not in layers:
        layers.append(final)
    return layers


# Categories for the token-sequence HighlightedText. Every token must carry a
# category: Gradio's frontend only dispatches the select event for spans whose
# category is non-None, so a None (plain-text) span is not clickable at all.
TOKEN_UNSELECTED = "tok"
TOKEN_SELECTED = "sel"
TOKEN_COLOR_MAP = {TOKEN_UNSELECTED: "gray", TOKEN_SELECTED: "orange"}


def _token_label(text: str) -> str:
    shown = visible_whitespace(text)
    return shown if shown.strip() else "·"


def highlighted_tokens(
    position_texts: list[str], selected: list[int]
) -> list[tuple[str, str]]:
    """Value for gr.HighlightedText: one clickable span per position."""
    chosen = set(selected)
    return [
        (_token_label(text), TOKEN_SELECTED if i in chosen else TOKEN_UNSELECTED)
        for i, text in enumerate(position_texts)
    ]


def selection_summary(selected: list[int], seq_len: int | None) -> str:
    """One-line description of the position selection, shown under the sequence."""
    if seq_len is None:
        return ""
    if not selected:
        available = max(seq_len - 1, 0)
        return (
            f"**Selection:** all {available} token-aligned positions "
            "(position 0 has no preceding causal state)."
        )
    shown = ", ".join(str(p) for p in selected[:12])
    if len(selected) > 12:
        shown += ", …"
    return f"**Selection:** {len(selected)} of {seq_len} positions ({shown})."


def toggle_position(selected: list[int], index: int) -> list[int]:
    """Toggle one position in the selection (returns a new list)."""
    updated = list(selected)
    if index in updated:
        updated.remove(index)
    else:
        updated.append(index)
    return sorted(updated)


# --------------------------------------------------------------------------
# Active interventions registry.
#
# Miru Tracer is a single-user tool (see ModelManager); the intervention list
# is app-global so the Lens tab (where it is edited) and Interactive Mode
# (where it can be applied to a session) stay in sync without cross-tab
# component plumbing.

_active_interventions: list[Intervention] = []


def get_active_interventions() -> list[Intervention]:
    return list(_active_interventions)


def set_active_interventions(interventions: list[Intervention]) -> None:
    _active_interventions.clear()
    _active_interventions.extend(interventions)
