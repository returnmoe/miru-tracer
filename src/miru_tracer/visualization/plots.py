"""Plotly visualizations for LLM generation analysis.

All functions take a plain ``list[TokenStep]`` (from a live tracer's
``.history`` or a parsed log's ``.history``) — no tracer or tokenizer needed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from miru_tracer.core.lens import LensSlice, ReadoutRow
from miru_tracer.core.schema import TokenStep

_TRUNCATE_AT = 15


def _display_text(text: str) -> str:
    return text[:12] + "..." if len(text) > _TRUNCATE_AT else text


def _step_entropy(step: TokenStep) -> tuple[float, bool]:
    """Entropy in nats for one step.

    Uses the full-vocabulary distribution when it was logged (exact); falls
    back to the renormalized top-k distribution (an underestimate, capped at
    log(k)). Returns (entropy, is_exact).
    """
    if step.full_probs:
        probs = np.asarray(step.full_probs)
        exact = True
    else:
        probs = np.asarray(step.top_k_probs)
        probs = probs / probs.sum()
        exact = False
    positive = probs[probs > 0]
    return float(-np.sum(positive * np.log(positive))), exact


def _probs_for_mode(step: TokenStep, use_raw: bool) -> tuple[float, list[float]]:
    """(selected probability, top-k probabilities) in the requested mode."""
    if use_raw:
        return step.raw_probability, list(step.top_k_raw_probs or step.top_k_probs)
    return step.probability, list(step.top_k_probs)


def plot_probability_visualizations(
    history: list[TokenStep],
    top_k: int = 5,
    probability_mode: str = "adjusted",
    temperature: float = 1.0,
) -> list[go.Figure]:
    """
    Create probability visualizations for generation analysis.

    Args:
        history: Recorded generation steps.
        top_k: Number of top-k ranks to show in the heatmap.
        probability_mode: "adjusted" (post-temperature) or "raw" (pre-temperature).
        temperature: Temperature used during generation (shown in tooltips).

    Returns:
        [heatmap_figure, confidence_figure], or [] for empty history.
    """
    if not history:
        return []

    use_raw = probability_mode == "raw"
    mode_label = "Raw (Pre-Temperature)" if use_raw else f"Adjusted (T={temperature})"
    steps_axis = [f"Step {i}" for i in range(len(history))]
    hover = (
        "%{x}<br>%{y}<br>"
        "Raw: %{text}<br>"
        "Decoded: %{customdata[1]}<br>"
        "Token ID: %{customdata[0]}<br>"
        "Raw Probability: %{customdata[2]:.4f}<br>"
        f"Adjusted (T={temperature}): %{{customdata[3]:.4f}}<extra></extra>"
    )

    # ---- Visualization 1: rank heatmap with a separate "Selected" row
    max_ranks = min(top_k, len(history[0].top_k_tokens))

    # rank rows: [rank][step]
    rank_probs = [[0.0] * len(history) for _ in range(max_ranks)]
    rank_texts = [[""] * len(history) for _ in range(max_ranks)]
    rank_custom = [[[None, "", 0, 0]] * len(history) for _ in range(max_ranks)]
    selected_probs, selected_texts, selected_custom = [], [], []

    for step_index, step in enumerate(history):
        _, top_probs_mode = _probs_for_mode(step, use_raw)
        raw_list = step.top_k_raw_probs or step.top_k_probs
        texts_raw = step.top_k_texts_raw or step.top_k_texts
        for rank in range(min(max_ranks, len(step.top_k_probs))):
            rank_probs[rank][step_index] = top_probs_mode[rank]
            rank_texts[rank][step_index] = _display_text(texts_raw[rank])
            rank_custom[rank][step_index] = [
                step.top_k_tokens[rank],
                step.top_k_texts[rank],
                raw_list[rank],
                step.top_k_probs[rank],
            ]

        display_prob, _ = _probs_for_mode(step, use_raw)
        selected_probs.append(display_prob)
        selected_texts.append(_display_text(step.token_text_raw or step.token_text))
        selected_custom.append(
            [step.token_id, step.token_text, step.raw_probability, step.probability]
        )

    fig1 = make_subplots(
        rows=2,
        cols=1,
        row_heights=[1 / (max_ranks + 1), max_ranks / (max_ranks + 1)],
        vertical_spacing=0.02,
        shared_xaxes=True,
    )
    fig1.add_trace(
        go.Heatmap(
            z=[selected_probs],
            text=[selected_texts],
            customdata=[selected_custom],
            texttemplate="%{text}<br>%{z:.3f}",
            textfont={"size": 12},
            x=steps_axis,
            y=["Selected"],
            colorscale=[[0, "lightgray"], [1, "darkgray"]],
            showscale=False,
            hovertemplate=hover,
        ),
        row=1,
        col=1,
    )
    fig1.add_trace(
        go.Heatmap(
            z=rank_probs,
            text=rank_texts,
            customdata=rank_custom,
            texttemplate="%{text}<br>%{z:.3f}",
            textfont={"size": 12},
            x=steps_axis,
            y=[f"Rank {i + 1}" for i in range(max_ranks)],
            colorscale="YlOrRd",
            colorbar=dict(title="Probability"),
            hovertemplate=hover,
        ),
        row=2,
        col=1,
    )
    fig1.update_xaxes(title_text="Generation Step (Time →)", row=2, col=1)
    fig1.update_yaxes(title_text="Token Ranks", autorange="reversed", row=2, col=1)
    fig1.update_layout(
        title=(
            f"Rank-Based Probability Heatmap - {mode_label}"
            "<br><sub>Selected row (gray) shows chosen token | Ranks show "
            "alternatives | Hover for both values</sub>"
        ),
        height=max(550, (max_ranks + 1) * 50),
        dragmode=False,
        hovermode="closest",
        autosize=True,
    )

    # ---- Visualization 2: top-1 probability and entropy over time
    top1_probs = [_probs_for_mode(step, use_raw)[1][0] for step in history]
    entropy_pairs = [_step_entropy(step) for step in history]
    entropies = [e for e, _ in entropy_pairs]
    all_exact = all(exact for _, exact in entropy_pairs)
    entropy_label = "Entropy (nats)" if all_exact else "Top-k entropy (nats, renormalized)"

    fig2 = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(
            "Top-1 Probability (Higher = More Confident)",
            f"{entropy_label} (Lower = More Certain)",
        ),
        vertical_spacing=0.15,
    )
    fig2.add_trace(
        go.Scatter(
            x=list(range(len(top1_probs))),
            y=top1_probs,
            mode="lines+markers",
            name="Top-1 Probability",
            line=dict(color="blue", width=2),
            marker=dict(size=6),
            fill="tozeroy",
            fillcolor="rgba(0,0,255,0.1)",
            hovertemplate="Step %{x}<br>Top-1 Prob: %{y:.4f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig2.add_trace(
        go.Scatter(
            x=list(range(len(entropies))),
            y=entropies,
            mode="lines+markers",
            name="Entropy",
            line=dict(color="red", width=2),
            marker=dict(size=6),
            fill="tozeroy",
            fillcolor="rgba(255,0,0,0.1)",
            hovertemplate="Step %{x}<br>Entropy: %{y:.3f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig2.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1)
    fig2.update_xaxes(title_text="Generation Step", row=2, col=1)
    fig2.update_yaxes(title_text="Probability", row=1, col=1)
    fig2.update_yaxes(title_text=entropy_label, row=2, col=1)
    fig2.update_layout(
        title=(
            f"Model Confidence Analysis - {mode_label}"
            "<br><sub>Top: Higher is more confident | Bottom: Lower is more certain</sub>"
        ),
        height=600,
        showlegend=False,
        dragmode=False,
        hovermode="closest",
    )

    return [fig1, fig2]


def get_generation_stats(
    history: list[TokenStep], probability_mode: str = "adjusted"
) -> dict[str, Any]:
    """
    Summary statistics for a generation.

    Args:
        history: Recorded generation steps.
        probability_mode: "adjusted" (post-temperature) or "raw" (pre-temperature).
    """
    if not history:
        return {}

    use_raw = probability_mode == "raw"
    selected_probs, top1_probs = [], []
    for step in history:
        selected, top_probs = _probs_for_mode(step, use_raw)
        selected_probs.append(selected)
        top1_probs.append(top_probs[0])

    entropy_pairs = [_step_entropy(step) for step in history]
    entropies = [e for e, _ in entropy_pairs]
    entropy_key = (
        "avg_entropy" if all(exact for _, exact in entropy_pairs) else "avg_topk_entropy"
    )

    return {
        "total_steps": len(history),
        "avg_top1_prob": float(np.mean(top1_probs)),
        "avg_selected_prob": float(np.mean(selected_probs)),
        entropy_key: float(np.mean(entropies)),
        "min_confidence": float(np.min(top1_probs)),
        "min_confidence_step": int(np.argmin(top1_probs)),
        "max_confidence": float(np.max(top1_probs)),
        "max_confidence_step": int(np.argmax(top1_probs)),
    }


# --------------------------------------------------------------------- lenses


def plot_lens_heatmap(slice_: LensSlice) -> go.Figure | None:
    """Position x layer heatmap of top-1 lens readouts (paper Figure-5 style).

    Cell color is the top-1 readout probability (score for "diff" mode); cell
    text is the top-1 token; hover lists the full top-k of the cell.
    """
    if not slice_.layers or not slice_.positions:
        return None

    x_labels = [
        f"{p}: {_display_text(text)}"
        for p, text in zip(slice_.positions, slice_.position_texts, strict=True)
    ]
    y_labels = [f"L{layer}" for layer in slice_.layers]

    z, text, hover = [], [], []
    for i in range(len(slice_.layers)):
        z_row, text_row, hover_row = [], [], []
        for j in range(len(slice_.positions)):
            probs = slice_.probs[i][j]
            texts = slice_.texts[i][j]
            z_row.append(probs[0] if probs else 0.0)
            text_row.append(_display_text(texts[0]) if texts else "")
            hover_row.append(
                "<br>".join(
                    f"{rank + 1}. {_display_text(t)} ({p:.3f})"
                    for rank, (t, p) in enumerate(zip(texts, probs, strict=True))
                )
                or "(empty)"
            )
        z.append(z_row)
        text.append(text_row)
        hover.append(hover_row)

    value_name = "Δprob (J-lens − logit)" if slice_.mode == "diff" else "Probability"
    fig = go.Figure(
        go.Heatmap(
            z=z,
            text=text,
            customdata=hover,
            texttemplate="%{text}",
            textfont={"size": 11},
            x=x_labels,
            y=y_labels,
            colorscale="YlOrRd",
            colorbar=dict(title=value_name),
            hovertemplate="%{x}<br>%{y}<br>%{customdata}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Lens readouts — {slice_.mode}<br>"
        "<sub>Top-1 readout per (position, layer) | Hover for the full top-k</sub>",
        xaxis_title="Position (input token)",
        yaxis_title="Layer",
        height=max(400, 30 * len(slice_.layers) + 150),
        dragmode=False,
        hovermode="closest",
        autosize=True,
    )
    return fig


def plot_readout_distribution(
    rows: list[ReadoutRow], layers: list[int], *, limit: int = 20
) -> go.Figure | None:
    """Token x layer heatmap of readout counts (Neuronpedia's per-token bars)."""
    rows = rows[:limit]
    if not rows or not layers:
        return None
    fig = go.Figure(
        go.Heatmap(
            z=[row.count_by_layer for row in rows],
            x=[f"L{layer}" for layer in layers],
            y=[f"{_display_text(row.text)} ({row.count})" for row in rows],
            colorscale="Greys",
            colorbar=dict(title="Count"),
            hovertemplate="%{y}<br>%{x}: %{z} cells<extra></extra>",
        )
    )
    fig.update_layout(
        title="Readout counts by layer<br>"
        "<sub>How often each token appears in the selected cells' top-k</sub>",
        xaxis_title="Layer",
        yaxis_title="Readout token (total count)",
        height=max(350, 25 * len(rows) + 150),
        yaxis=dict(autorange="reversed"),
        dragmode=False,
        autosize=True,
    )
    return fig


def plot_pinned_token_ranks(slice_: LensSlice, tokenizer=None) -> go.Figure | None:
    """Median rank across selected positions vs layer, one line per pinned token."""
    if not slice_.pinned_ranks:
        return None
    fig = go.Figure()
    for token_id, grid in slice_.pinned_ranks.items():
        medians = [float(np.median(row)) for row in grid]  # per layer
        label = str(token_id)
        if tokenizer is not None:
            label = tokenizer.convert_ids_to_tokens([token_id])[0]
        fig.add_trace(
            go.Scatter(
                x=[f"L{layer}" for layer in slice_.layers],
                y=[m + 1 for m in medians],  # 1-indexed for log axis
                mode="lines+markers",
                name=_display_text(label),
                hovertemplate="%{x}<br>median rank %{y:.0f}<extra>"
                + _display_text(label)
                + "</extra>",
            )
        )
    fig.update_layout(
        title="Pinned token ranks across layers<br>"
        "<sub>Median rank over the selected positions (lower = closer to top-1)</sub>",
        xaxis_title="Layer",
        yaxis_title="Rank (log scale)",
        yaxis=dict(type="log", autorange="reversed"),
        height=450,
        dragmode=False,
        hovermode="x unified",
    )
    return fig
