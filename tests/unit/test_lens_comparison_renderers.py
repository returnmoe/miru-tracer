import re
from dataclasses import replace

import pytest

from miru_tracer.core.lens import LensSlice
from miru_tracer.ui.lens_views import comparison_heatmap_html, comparison_html
from miru_tracer.visualization.plots import (
    plot_lens_heatmap_comparison,
    plot_pinned_token_ranks_comparison,
)

JACOBIAN = LensSlice(
    mode="jacobian",
    layers=[0, 2],
    positions=[0, 1],
    position_texts=["Hel", "lo"],
    tokens=[[[11, 12], [13, 14]], [[15, 16], [17, 18]]],
    probs=[[[0.2, 0.1], [0.8, 0.05]], [[0.6, 0.2], [0.7, 0.1]]],
    texts=[
        [["j-one", "j-two"], ["j-three", "j-four"]],
        [["j-five", "j-six"], ["j-seven", "j-eight"]],
    ],
    pinned_ranks={7: [[9, 3], [1, 1]], 8: [[20, 10], [8, 4]]},
)

LOGIT = LensSlice(
    mode="logit",
    layers=[0, 2],
    positions=[0, 1],
    position_texts=["Hel", "lo"],
    tokens=[[[21, 22], [23, 24]], [[25, 26], [27, 28]]],
    probs=[[[0.2, 0.15], [0.4, 0.3]], [[0.3, 0.2], [0.6, 0.2]]],
    texts=[
        [["l-one", "l-two"], ["l-three", "l-four"]],
        [["l-five", "l-six"], ["l-seven", "l-eight"]],
    ],
    pinned_ranks={7: [[99, 49], [19, 9]], 8: [[7, 3], [2, 0]]},
)


def test_comparison_html_is_ordered_and_responsive():
    rendered = comparison_html("<p>jacobian body</p>", "<p>logit body</p>")

    assert rendered.index('data-lens-mode="jacobian"') < rendered.index('data-lens-mode="logit"')
    assert rendered.index("jacobian body") < rendered.index("logit body")
    assert "grid-template-columns: minmax(0, 1fr) minmax(0, 1fr)" in rendered
    assert "@media (max-width: 900px)" in rendered
    assert "grid-template-columns: minmax(0, 1fr)" in rendered
    assert comparison_html("", "") == ""


def test_static_comparison_heatmap_keeps_independent_results():
    rendered = comparison_heatmap_html(JACOBIAN, LOGIT)

    assert rendered.index("j-one") < rendered.index("l-one")
    assert "j-seven" in rendered and "l-seven" in rendered
    assert "Jacobian Lens" in rendered and "Logit Lens" in rendered
    assert "Δprob" not in rendered
    # Both cells have probability 0.6. Although that is an interior value on
    # the Jacobian side and the maximum on the Logit side, the shared range
    # must give them exactly the same color.
    jacobian_color = re.search(
        r'<td style="background:(rgb\([^)]+\));[^>]+title="1\. j-five', rendered
    ).group(1)
    logit_color = re.search(
        r'<td style="background:(rgb\([^)]+\));[^>]+title="1\. l-seven', rendered
    ).group(1)
    assert jacobian_color == logit_color


def test_comparison_heatmap_has_shared_probability_coloraxis():
    fig = plot_lens_heatmap_comparison(JACOBIAN, LOGIT)

    assert len(fig.data) == 2
    assert fig.data[0].z[0][1] == pytest.approx(0.8)
    assert fig.data[1].z[0][1] == pytest.approx(0.4)
    assert fig.data[0].text[0][0] == "j-one"
    assert fig.data[1].text[0][0] == "l-one"
    assert fig.data[0].coloraxis == "coloraxis"
    assert fig.data[1].coloraxis == "coloraxis"
    assert fig.layout.coloraxis.cmin == pytest.approx(0.0)
    assert fig.layout.coloraxis.cmax == pytest.approx(0.8)
    assert fig.layout.yaxis2.matches == "y"
    assert "Independent" in fig.layout.title.text
    assert "Δ" not in fig.layout.title.text


def test_comparison_heatmap_rejects_diff_and_handles_empty_pair():
    with pytest.raises(ValueError, match="jacobian slice followed by a logit slice"):
        plot_lens_heatmap_comparison(replace(JACOBIAN, mode="diff"), LOGIT)

    empty_jacobian = replace(
        JACOBIAN,
        layers=[],
        positions=[],
        position_texts=[],
        tokens=[],
        probs=[],
        texts=[],
        pinned_ranks={},
    )
    empty_logit = replace(empty_jacobian, mode="logit")
    assert plot_lens_heatmap_comparison(empty_jacobian, empty_logit) is None


def test_pinned_rank_comparison_shares_scale_and_token_colors():
    fig = plot_pinned_token_ranks_comparison(JACOBIAN, LOGIT)

    # Traces are grouped by panel: Jacobian token 7/8, then Logit token 7/8.
    assert len(fig.data) == 4
    assert fig.data[0].legendgroup == fig.data[2].legendgroup == "7"
    assert fig.data[0].line.color == fig.data[2].line.color
    assert fig.data[1].line.color == fig.data[3].line.color
    assert fig.data[0].y[0] == pytest.approx(7.0)  # median(9, 3) + 1
    assert fig.data[2].y[0] == pytest.approx(75.0)  # median(99, 49) + 1
    assert fig.layout.yaxis.type == "log"
    assert fig.layout.yaxis.autorange == "reversed"
    assert fig.layout.yaxis2.matches == "y"
    assert "Independent" in fig.layout.title.text


def test_pinned_rank_comparison_returns_none_without_tokens():
    jacobian = replace(JACOBIAN, pinned_ranks={})
    logit = replace(LOGIT, pinned_ranks={})
    assert plot_pinned_token_ranks_comparison(jacobian, logit) is None
