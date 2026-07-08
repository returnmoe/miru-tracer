"""Shared helpers for the lens UI (Lens tab + Interactive Mode panel)."""

from __future__ import annotations

import pandas as pd

from miru_tracer.core.interventions import Intervention
from miru_tracer.core.tokenizer_utils import visible_whitespace

LENS_MODE_CHOICES = ("Logit", "Jacobian", "Diff (Jacobian − Logit)")

_SPARK_BLOCKS = "▁▂▃▄▅▆▇█"


def lens_mode_key(ui_choice: str) -> str:
    choice = (ui_choice or "").lower()
    if choice.startswith("jacobian"):
        return "jacobian"
    if choice.startswith("diff"):
        return "diff"
    return "logit"


def token_ref_to_id(ref: str, tokenizer) -> int:
    """Resolve a user token reference: a numeric id, or text (first token).

    Text is encoded verbatim — leading/trailing whitespace is significant
    (" Paris" and "Paris" are different BPE tokens); only numeric ids
    tolerate surrounding spaces.

    Raises:
        ValueError: empty ref, unencodable text, or out-of-range id.
    """
    stripped = ref.strip()
    if not stripped:
        raise ValueError("Empty token reference")
    if stripped.lstrip("-").isdigit():
        token_id = int(stripped)
        if not 0 <= token_id < len(tokenizer):
            raise ValueError(
                f"Token id {token_id} out of range (vocab size {len(tokenizer)})"
            )
        return token_id
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


def parse_token_refs(text: str, tokenizer) -> list[int]:
    """Comma-separated token references -> unique token ids (order kept)."""
    ids: list[int] = []
    for part in (text or "").split(","):
        part = part.strip()
        if not part:
            continue
        token_id = token_ref_to_id(part, tokenizer)
        if token_id not in ids:
            ids.append(token_id)
    return ids


def sparkline(counts: list[int]) -> str:
    """Compact unicode bar chart of per-layer counts."""
    peak = max(counts) if counts else 0
    if peak == 0:
        return _SPARK_BLOCKS[0] * len(counts)
    return "".join(
        _SPARK_BLOCKS[min(int(c / peak * (len(_SPARK_BLOCKS) - 1) + 0.5), 7)]
        for c in counts
    )


def interventions_dataframe(
    interventions: list[Intervention], tokenizer=None
) -> pd.DataFrame:
    return pd.DataFrame(
        [[i, iv.describe(tokenizer), iv.basis] for i, iv in enumerate(interventions)],
        columns=["#", "Intervention", "Basis"],
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
        return f"**Selection:** all {seq_len} positions (readouts cover the whole sequence)."
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
