"""Lens UI helpers and the new lens plot builders."""

import pytest

from miru_tracer.core.interventions import Intervention
from miru_tracer.core.lens import LensSlice, ReadoutRow
from miru_tracer.ui.lens_common import (
    get_active_interventions,
    highlighted_tokens,
    layer_selection,
    lens_mode_key,
    parse_token_refs,
    readouts_dataframe,
    set_active_interventions,
    sparkline,
    toggle_position,
    token_ref_to_id,
)
from miru_tracer.visualization.plots import (
    plot_lens_heatmap,
    plot_pinned_token_ranks,
    plot_readout_distribution,
)


class TestLensModeKey:
    @pytest.mark.parametrize("ui,key", [
        ("Logit", "logit"),
        ("Jacobian", "jacobian"),
        ("Diff (Jacobian − Logit)", "diff"),
        (None, "logit"),
    ])
    def test_mapping(self, ui, key):
        assert lens_mode_key(ui) == key


class TestTokenRefs:
    def test_numeric_id(self, tiny_tokenizer):
        assert token_ref_to_id("42", tiny_tokenizer) == 42

    def test_text_encodes_first_token(self, tiny_tokenizer):
        expected = tiny_tokenizer.encode("a", add_special_tokens=False)[0]
        assert token_ref_to_id("a", tiny_tokenizer) == expected

    def test_out_of_range_id_rejected(self, tiny_tokenizer):
        with pytest.raises(ValueError, match="out of range"):
            token_ref_to_id("99999", tiny_tokenizer)

    def test_empty_rejected(self, tiny_tokenizer):
        with pytest.raises(ValueError, match="Empty"):
            token_ref_to_id("  ", tiny_tokenizer)

    def test_parse_list_deduplicates(self, tiny_tokenizer):
        ids = parse_token_refs("42, 42, 7", tiny_tokenizer)
        assert ids == [42, 7]

    def test_parse_empty(self, tiny_tokenizer):
        assert parse_token_refs("", tiny_tokenizer) == []
        assert parse_token_refs(None, tiny_tokenizer) == []


class TestLayerSelection:
    def test_full_range_default(self):
        assert layer_selection(6, 0, -1, 1) == [0, 1, 2, 3, 4, 5]

    def test_stride_always_includes_end(self):
        assert layer_selection(10, 0, 9, 4) == [0, 4, 8, 9]

    def test_clamping(self):
        assert layer_selection(4, -5, 99, 1) == [0, 1, 2, 3]

    def test_none_inputs(self):
        assert layer_selection(3, None, None, None) == [0, 1, 2]


class TestSparkline:
    def test_shapes(self):
        assert sparkline([0, 0]) == "▁▁"
        line = sparkline([1, 4, 8])
        assert len(line) == 3
        assert line[-1] == "█"

    def test_empty(self):
        assert sparkline([]) == ""


class TestPositions:
    def test_toggle(self):
        assert toggle_position([], 3) == [3]
        assert toggle_position([3], 3) == []
        assert toggle_position([5], 3) == [3, 5]

    def test_highlight_marks_selection(self):
        value = highlighted_tokens(["a", " ", "b"], [2])
        assert value[0] == ("a", None)
        assert value[1][0] == "·"  # whitespace made visible
        assert value[2] == ("b", "sel")


class TestActiveInterventionsRegistry:
    def test_roundtrip_and_isolation(self):
        set_active_interventions([])
        assert get_active_interventions() == []
        iv = Intervention(kind="steer", layer=0, token_id=1, basis="logit")
        set_active_interventions([iv])
        got = get_active_interventions()
        assert got == [iv]
        got.append(iv)  # mutating the copy must not affect the registry
        assert len(get_active_interventions()) == 1
        set_active_interventions([])


SLICE = LensSlice(
    mode="logit",
    layers=[0, 2],
    positions=[0, 1],
    position_texts=["Hel", "lo"],
    tokens=[[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
    probs=[[[0.5, 0.1], [0.4, 0.2]], [[0.6, 0.3], [0.9, 0.05]]],
    texts=[[["a", "b"], ["c", "d"]], [["e", "f"], ["g", "h"]]],
    pinned_ranks={7: [[10, 3], [1, 0]]},
)


class TestLensPlots:
    def test_heatmap_structure(self):
        fig = plot_lens_heatmap(SLICE)
        heat = fig.data[0]
        assert list(heat.y) == ["L0", "L2"]
        assert len(heat.z) == 2 and len(heat.z[0]) == 2
        assert heat.z[1][1] == pytest.approx(0.9)
        assert heat.text[1][1] == "g"

    def test_heatmap_empty(self):
        empty = LensSlice(
            mode="logit", layers=[], positions=[], position_texts=[],
            tokens=[], probs=[], texts=[],
        )
        assert plot_lens_heatmap(empty) is None

    def test_distribution_plot(self):
        rows = [
            ReadoutRow(token_id=1, text="a", count=3, count_by_layer=[2, 1]),
            ReadoutRow(token_id=2, text="b", count=1, count_by_layer=[0, 1]),
        ]
        fig = plot_readout_distribution(rows, [0, 2])
        assert list(fig.data[0].x) == ["L0", "L2"]
        assert list(fig.data[0].z[0]) == [2, 1]

    def test_pinned_ranks_plot(self):
        fig = plot_pinned_token_ranks(SLICE)
        assert len(fig.data) == 1
        # median of [10,3] is 6.5 -> +1 for the 1-indexed log axis
        assert fig.data[0].y[0] == pytest.approx(7.5)

    def test_pinned_ranks_none_when_empty(self):
        empty = LensSlice(
            mode="logit", layers=[0], positions=[0], position_texts=["x"],
            tokens=[[[1]]], probs=[[[1.0]]], texts=[[["a"]]],
        )
        assert plot_pinned_token_ranks(empty) is None


class TestReadoutsDataframe:
    def test_columns_and_sparkline(self):
        rows = [ReadoutRow(token_id=9, text=" tok", count=5, count_by_layer=[5, 0])]
        df = readouts_dataframe(rows)
        assert list(df.columns) == ["Token", "ID", "Count", "By layer"]
        assert df.iloc[0]["Count"] == 5
        assert df.iloc[0]["By layer"][0] == "█"
