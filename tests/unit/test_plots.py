"""Plot construction and stats from plain TokenStep histories."""

import math

import pytest

from miru_tracer.core.schema import TokenStep
from miru_tracer.visualization.plots import (
    _display_text,
    get_generation_stats,
    plot_probability_visualizations,
)


class TestDisplayText:
    def test_whitespace_made_visible(self):
        assert _display_text(" Paris") == "␣Paris"
        assert _display_text("\n") == "⏎"

    def test_long_text_truncated(self):
        assert _display_text("x" * 30) == "x" * 12 + "..."


def make_step(step, full_probs=None):
    return TokenStep(
        step=step,
        token_id=1,
        token_text="a",
        probability=0.6,
        top_k_tokens=[1, 2, 3],
        top_k_probs=[0.6, 0.3, 0.1],
        top_k_texts=["a", "b", "c"],
        raw_probability=0.5,
        top_k_raw_probs=[0.5, 0.35, 0.15],
        full_probs=full_probs,
        token_text_raw="a",
        top_k_texts_raw=["a", "b", "c"],
    )


HISTORY = [make_step(i) for i in range(4)]


class TestVisualizations:
    def test_empty_history_returns_no_figures(self):
        assert plot_probability_visualizations([]) == []

    def test_two_figures_with_expected_traces(self):
        figures = plot_probability_visualizations(HISTORY, top_k=3)
        assert len(figures) == 2
        heatmap, confidence = figures
        # heatmap: selected row + rank rows
        assert len(heatmap.data) == 2
        assert list(heatmap.data[0].y) == ["Selected"]
        assert list(heatmap.data[1].y) == ["Rank 1", "Rank 2", "Rank 3"]
        assert len(heatmap.data[1].z[0]) == len(HISTORY)
        # confidence: top-1 + entropy lines
        assert len(confidence.data) == 2

    def test_topk_entropy_labeled_honestly(self):
        """Regression: top-k-only entropy was labeled plain 'Entropy (nats)'."""
        figures = plot_probability_visualizations(HISTORY, top_k=3)
        confidence = figures[1]
        labels = [a.text for a in confidence.layout.annotations]
        assert any("Top-k entropy" in label for label in labels)

    def test_full_probs_entropy_is_exact_and_plainly_labeled(self):
        history = [make_step(i, full_probs=[0.25, 0.25, 0.25, 0.25]) for i in range(3)]
        figures = plot_probability_visualizations(history, top_k=3)
        confidence = figures[1]
        labels = [a.text for a in confidence.layout.annotations]
        assert any(
            "Entropy (nats)" in label and "Top-k" not in label for label in labels
        )
        # uniform over 4 => log(4)
        assert confidence.data[1].y[0] == pytest.approx(math.log(4))

    def test_raw_mode_uses_raw_probabilities(self):
        figures = plot_probability_visualizations(HISTORY, top_k=3, probability_mode="raw")
        heatmap = figures[0]
        assert heatmap.data[0].z[0][0] == pytest.approx(0.5)  # raw_probability

    def test_top_k_capped_at_logged(self):
        figures = plot_probability_visualizations(HISTORY, top_k=99)
        assert len(figures[0].data[1].z) == 3  # only 3 ranks were logged


class TestStats:
    def test_empty(self):
        assert get_generation_stats([]) == {}

    def test_keys_and_values(self):
        stats = get_generation_stats(HISTORY)
        assert stats["total_steps"] == 4
        assert stats["avg_top1_prob"] == pytest.approx(0.6)
        assert stats["avg_selected_prob"] == pytest.approx(0.6)
        assert "avg_topk_entropy" in stats  # top-k only => honest key name

    def test_full_probs_uses_exact_entropy_key(self):
        history = [make_step(i, full_probs=[0.5, 0.5]) for i in range(2)]
        stats = get_generation_stats(history)
        assert stats["avg_entropy"] == pytest.approx(math.log(2))

    def test_raw_mode(self):
        stats = get_generation_stats(HISTORY, probability_mode="raw")
        assert stats["avg_selected_prob"] == pytest.approx(0.5)
