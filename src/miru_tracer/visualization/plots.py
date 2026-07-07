"""Visualization functions for LLM generation analysis."""

from typing import Optional, List, Dict, Any
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from miru_tracer.core.tracer import LLMTracer
from miru_tracer.core.tokenizer_utils import safe_decode_token


def plot_probability_visualizations(
    tracer: LLMTracer,
    top_k: int = 5,
    track_tokens: Optional[List] = None,
    probability_mode: str = "adjusted",
    temperature: float = 1.0,
) -> List[go.Figure]:
    """
    Create comprehensive probability visualizations for generation analysis.

    Args:
        tracer: LLMTracer instance with generation history
        top_k: Number of top-k tokens to show in heatmap (default 5)
        track_tokens: List of tokens to track in Visualization 3 (optional)
                     Can be token IDs (List[int]) or token texts (List[str])
        probability_mode: "adjusted" (post-temperature, default) or "raw" (pre-temperature)
        temperature: Temperature value used during generation (for display in tooltips)

    Returns:
        List of plotly figures [heatmap_fig, confidence_fig, evolution_fig (optional)]
    """
    if len(tracer.history) == 0:
        return []

    figures = []

    # Visualization 1: Rank-Based Heatmap with Selected Token Row
    max_ranks = min(top_k, len(tracer.history[0].top_k_tokens))
    heatmap_data = []
    heatmap_text = []
    heatmap_customdata = []

    # Select probability source based on mode
    use_raw = probability_mode == "raw"

    # Build data in step-by-rank format first (before transpose)
    for step_data in tracer.history:
        row_probs = []
        row_texts = []
        row_customdata = []
        for i in range(max_ranks):
            if i < len(step_data.top_k_probs):
                # Select display probability based on mode
                display_prob = (
                    step_data.top_k_raw_probs[i]
                    if use_raw
                    and hasattr(step_data, "top_k_raw_probs")
                    and step_data.top_k_raw_probs
                    else step_data.top_k_probs[i]
                )
                row_probs.append(display_prob)

                # Use raw token text for display (fallback to decoded for old JSON files)
                token_text_raw = (
                    step_data.top_k_texts_raw[i]
                    if hasattr(step_data, "top_k_texts_raw")
                    and step_data.top_k_texts_raw
                    else step_data.top_k_texts[i]
                )
                # Truncate for display
                if len(token_text_raw) > 15:
                    token_text_raw = token_text_raw[:12] + "..."
                row_texts.append(token_text_raw)

                # Store token ID, decoded text, AND both probabilities in customdata
                token_text_decoded = step_data.top_k_texts[i]
                raw_prob = (
                    step_data.top_k_raw_probs[i]
                    if hasattr(step_data, "top_k_raw_probs")
                    and step_data.top_k_raw_probs
                    else step_data.top_k_probs[i]  # Fallback if raw not available
                )
                adj_prob = step_data.top_k_probs[i]
                row_customdata.append(
                    [step_data.top_k_tokens[i], token_text_decoded, raw_prob, adj_prob]
                )
            else:
                row_probs.append(0)
                row_texts.append("")
                row_customdata.append([None, "", 0, 0])
        heatmap_data.append(row_probs)
        heatmap_text.append(row_texts)
        heatmap_customdata.append(row_customdata)

    # Transpose data for better temporal flow (steps on X-axis)
    heatmap_data_T = list(map(list, zip(*heatmap_data)))
    heatmap_text_T = list(map(list, zip(*heatmap_text)))
    heatmap_customdata_T = list(map(list, zip(*heatmap_customdata)))

    # Add SELECTED row
    selected_row_probs = []
    selected_row_texts = []
    selected_row_customdata = []
    for step_data in tracer.history:
        # Select display probability based on mode
        display_prob = (
            step_data.raw_probability
            if use_raw and hasattr(step_data, "raw_probability")
            else step_data.probability
        )
        selected_row_probs.append(display_prob)

        # Use raw token text for display (fallback to decoded for old JSON files)
        token_text_raw = (
            step_data.token_text_raw
            if hasattr(step_data, "token_text_raw") and step_data.token_text_raw
            else step_data.token_text
        )
        # Truncate for display
        if len(token_text_raw) > 15:
            token_text_raw = token_text_raw[:12] + "..."
        selected_row_texts.append(token_text_raw)

        # Store token ID, decoded text, AND both probabilities in customdata
        token_text_decoded = step_data.token_text
        raw_prob = (
            step_data.raw_probability
            if hasattr(step_data, "raw_probability")
            else step_data.probability  # Fallback if raw not available
        )
        adj_prob = step_data.probability
        selected_row_customdata.append(
            [step_data.token_id, token_text_decoded, raw_prob, adj_prob]
        )

    # Create heatmap figure
    fig1 = make_subplots(
        rows=2,
        cols=1,
        row_heights=[1, max_ranks],
        vertical_spacing=0.02,
        shared_xaxes=True,
        subplot_titles=("", ""),
    )

    # Add SELECTED row (gray background)
    fig1.add_trace(
        go.Heatmap(
            z=[selected_row_probs],
            text=[selected_row_texts],
            customdata=[selected_row_customdata],
            texttemplate="%{text}<br>%{z:.3f}",
            textfont={"size": 12},
            x=[f"Step {i}" for i in range(len(tracer.history))],
            y=["Selected"],
            colorscale=[[0, "lightgray"], [1, "darkgray"]],
            showscale=False,
            hovertemplate=(
                "%{x}<br>Selected<br>"
                "Raw: %{text}<br>"
                "Decoded: %{customdata[1]}<br>"
                "Token ID: %{customdata[0]}<br>"
                "Raw Probability: %{customdata[2]:.4f}<br>"
                f"Adjusted (T={temperature}): %{{customdata[3]:.4f}}<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

    # Add Rank rows (probability colorscale)
    fig1.add_trace(
        go.Heatmap(
            z=heatmap_data_T,
            text=heatmap_text_T,
            customdata=heatmap_customdata_T,
            texttemplate="%{text}<br>%{z:.3f}",
            textfont={"size": 12},
            x=[f"Step {i}" for i in range(len(tracer.history))],
            y=[f"Rank {i+1}" for i in range(max_ranks)],
            colorscale="YlOrRd",
            colorbar=dict(title="Probability"),
            hovertemplate=(
                "%{x}<br>%{y}<br>"
                "Raw: %{text}<br>"
                "Decoded: %{customdata[1]}<br>"
                "Token ID: %{customdata[0]}<br>"
                "Raw Probability: %{customdata[2]:.4f}<br>"
                f"Adjusted (T={temperature}): %{{customdata[3]:.4f}}<extra></extra>"
            ),
        ),
        row=2,
        col=1,
    )

    fig1.update_xaxes(title_text="Generation Step (Time →)", row=2, col=1)
    fig1.update_yaxes(title_text="", row=1, col=1)
    fig1.update_yaxes(title_text="Token Ranks", autorange="reversed", row=2, col=1)

    mode_label = "Raw (Pre-Temperature)" if use_raw else f"Adjusted (T={temperature})"
    fig1.update_layout(
        title=f"Rank-Based Probability heatmap - {mode_label}<br><sub>Selected row (gray) shows chosen token | Ranks show alternatives | Hover for both values</sub>",
        height=max(550, (max_ranks + 1) * 50),
        dragmode=False,
        hovermode="closest",
        autosize=True,
    )

    figures.append(fig1)

    # Visualization 2: Entropy/Confidence Chart
    # Use selected probability mode
    if use_raw:
        top1_probs = [
            (
                step.top_k_raw_probs[0]
                if hasattr(step, "top_k_raw_probs") and step.top_k_raw_probs
                else step.top_k_probs[0]
            )
            for step in tracer.history
        ]
        selected_probs = [
            (
                step.raw_probability
                if hasattr(step, "raw_probability")
                else step.probability
            )
            for step in tracer.history
        ]
    else:
        top1_probs = [step.top_k_probs[0] for step in tracer.history]
        selected_probs = [step.probability for step in tracer.history]

    # Calculate entropy for each step (using selected mode)
    entropies = []
    for step_data in tracer.history:
        if (
            use_raw
            and hasattr(step_data, "top_k_raw_probs")
            and step_data.top_k_raw_probs
        ):
            probs = np.array(step_data.top_k_raw_probs)
        else:
            probs = np.array(step_data.top_k_probs)
        probs = probs / probs.sum()
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        entropies.append(entropy)

    fig2 = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(
            "Top-1 Probability (Higher = More Confident)",
            "Entropy (Lower = More Certain)",
        ),
        vertical_spacing=0.15,
    )

    # Top-1 probability
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

    # Entropy
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

    fig2.add_hline(
        y=0.5, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1
    )

    fig2.update_xaxes(title_text="Generation Step", row=2, col=1)
    fig2.update_yaxes(title_text="Probability", row=1, col=1)
    fig2.update_yaxes(title_text="Entropy (nats)", row=2, col=1)

    fig2.update_layout(
        title=f"Model Confidence analysis - {mode_label}<br><sub>Top: Higher is more confident | Bottom: Lower is more certain</sub>",
        height=600,
        showlegend=False,
        dragmode=False,
        hovermode="closest",
    )

    figures.append(fig2)

    # Visualization 3: Token Probability Evolution
    if track_tokens is not None:
        tokens_to_track = []
        for token in track_tokens:
            if isinstance(token, str):
                encoded = tracer.tokenizer.encode(token, add_special_tokens=False)
                if encoded:
                    tokens_to_track.append(encoded[0])
            elif isinstance(token, int):
                tokens_to_track.append(token)

        if tokens_to_track:
            fig3 = go.Figure()
            colors = [
                "red",
                "blue",
                "green",
                "orange",
                "purple",
                "brown",
                "pink",
                "gray",
                "cyan",
                "magenta",
            ]

            for idx, token_id in enumerate(tokens_to_track):
                token_text = safe_decode_token(tracer.tokenizer, token_id)[0] or str(
                    token_id
                )
                probs_over_time = []
                for step_data in tracer.history:
                    if token_id in step_data.top_k_tokens:
                        token_idx = step_data.top_k_tokens.index(token_id)
                        # Use selected probability mode
                        if (
                            use_raw
                            and hasattr(step_data, "top_k_raw_probs")
                            and step_data.top_k_raw_probs
                        ):
                            probs_over_time.append(step_data.top_k_raw_probs[token_idx])
                        else:
                            probs_over_time.append(step_data.top_k_probs[token_idx])
                    else:
                        probs_over_time.append(0)

                fig3.add_trace(
                    go.Scatter(
                        x=list(range(len(probs_over_time))),
                        y=probs_over_time,
                        mode="lines+markers",
                        name=f"{token_text}",
                        line=dict(color=colors[idx % len(colors)], width=2),
                        marker=dict(size=6),
                        hovertemplate=f"Token: {token_text}<br>Step: %{{x}}<br>Probability: %{{y:.4f}}<extra></extra>",
                    )
                )

            fig3.update_layout(
                title=f"Token Probability Evolution - {mode_label} ({len(tokens_to_track)} tokens tracked)",
                xaxis_title="Generation Step",
                yaxis_title="Probability",
                height=500,
                dragmode=False,
                hovermode="x unified",
                autosize=True,
            )

            figures.append(fig3)

    return figures


def plot_token_distribution(tracer: LLMTracer, step: int = 0) -> Optional[go.Figure]:
    """
    Plot the full token probability distribution for a specific step.

    Args:
        tracer: LLMTracer instance with generation history
        step: Step number to visualize

    Returns:
        Plotly figure or None if step is invalid
    """
    if step >= len(tracer.history):
        return None

    step_data = tracer.history[step]

    fig = go.Figure()

    # Bar chart of top tokens
    fig.add_trace(
        go.Bar(
            x=step_data.top_k_texts,
            y=step_data.top_k_probs,
            marker_color=[
                "red" if t == step_data.token_id else "lightblue"
                for t in step_data.top_k_tokens
            ],
            text=[f"{p:.4f}" for p in step_data.top_k_probs],
            textposition="auto",
        )
    )

    fig.update_layout(
        title=f"Token Distribution at Step {step} (Selected: {step_data.token_text})",
        xaxis_title="Token",
        yaxis_title="Probability",
        height=500,
        dragmode=False,
        hovermode="closest",
    )

    return fig


def get_generation_stats(
    tracer: LLMTracer, probability_mode: str = "adjusted"
) -> Dict[str, Any]:
    """
    Calculate summary statistics for a generation.

    Args:
        tracer: LLMTracer instance with generation history
        probability_mode: "adjusted" (post-temperature, default) or "raw" (pre-temperature)

    Returns:
        Dictionary with statistics
    """
    if not tracer.history:
        return {}

    use_raw = probability_mode == "raw"

    # Use selected probability mode
    if use_raw:
        top1_probs = [
            (
                step.top_k_raw_probs[0]
                if hasattr(step, "top_k_raw_probs") and step.top_k_raw_probs
                else step.top_k_probs[0]
            )
            for step in tracer.history
        ]
        selected_probs = [
            (
                step.raw_probability
                if hasattr(step, "raw_probability")
                else step.probability
            )
            for step in tracer.history
        ]
    else:
        top1_probs = [step.top_k_probs[0] for step in tracer.history]
        selected_probs = [step.probability for step in tracer.history]

    entropies = []
    for step_data in tracer.history:
        if (
            use_raw
            and hasattr(step_data, "top_k_raw_probs")
            and step_data.top_k_raw_probs
        ):
            probs = np.array(step_data.top_k_raw_probs)
        else:
            probs = np.array(step_data.top_k_probs)
        probs = probs / probs.sum()
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        entropies.append(entropy)

    return {
        "total_steps": len(tracer.history),
        "avg_top1_prob": float(np.mean(top1_probs)),
        "avg_selected_prob": float(np.mean(selected_probs)),
        "avg_entropy": float(np.mean(entropies)),
        "min_confidence": float(np.min(top1_probs)),
        "min_confidence_step": int(np.argmin(top1_probs)),
        "max_confidence": float(np.max(top1_probs)),
        "max_confidence_step": int(np.argmax(top1_probs)),
    }
