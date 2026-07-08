"""Lens UI helpers and the new lens plot builders."""

import pytest

from miru_tracer.core.interventions import Intervention
from miru_tracer.core.lens import LensSlice, ReadoutRow
from miru_tracer.ui.lens_common import (
    get_active_interventions,
    highlighted_tokens,
    intervened_layer_titles,
    intervention_visibility_warning,
    interventions_summary,
    layer_selection,
    lens_mode_key,
    parse_layer_refs,
    parse_token_refs,
    selection_summary,
    set_active_interventions,
    sparkline,
    toggle_position,
    token_mode_key,
    token_ref_to_id,
)
from miru_tracer.ui.lens_views import (
    distribution_html,
    heatmap_html,
    readouts_table_html,
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


class TestTokenModeKey:
    @pytest.mark.parametrize("ui,key", [
        ("Text", "text"),
        ("ID", "id"),
        ("id", "id"),
        (" ID ", "id"),
        ("", "text"),
        (None, "text"),
    ])
    def test_mapping(self, ui, key):
        assert token_mode_key(ui) == key


class TestTokenRefs:
    def test_id_mode_numeric(self, tiny_tokenizer):
        assert token_ref_to_id("42", tiny_tokenizer, "id") == 42

    def test_id_mode_tolerates_spaces(self, tiny_tokenizer):
        assert token_ref_to_id(" 42 ", tiny_tokenizer, "id") == 42

    def test_id_mode_rejects_non_numeric(self, tiny_tokenizer):
        with pytest.raises(ValueError, match="Not a numeric token id"):
            token_ref_to_id("a", tiny_tokenizer, "id")

    def test_id_mode_out_of_range_rejected(self, tiny_tokenizer):
        with pytest.raises(ValueError, match="out of range"):
            token_ref_to_id("99999", tiny_tokenizer, "id")

    def test_text_mode_encodes_first_token(self, tiny_tokenizer):
        expected = tiny_tokenizer.encode("a", add_special_tokens=False)[0]
        assert token_ref_to_id("a", tiny_tokenizer, "text") == expected

    def test_text_mode_digits_are_literal(self, tiny_tokenizer):
        # Regression: "6" in text mode encodes the string "6", NOT token id 6.
        expected = tiny_tokenizer.encode("6", add_special_tokens=False)[0]
        assert token_ref_to_id("6", tiny_tokenizer, "text") == expected

    def test_text_mode_leading_whitespace_preserved(self, tiny_tokenizer):
        expected = tiny_tokenizer.encode(" a", add_special_tokens=False)[0]
        assert token_ref_to_id(" a", tiny_tokenizer, "text") == expected
        assert token_ref_to_id(" a", tiny_tokenizer, "text") != token_ref_to_id(
            "a", tiny_tokenizer, "text"
        )

    def test_empty_rejected(self, tiny_tokenizer):
        with pytest.raises(ValueError, match="Empty"):
            token_ref_to_id("  ", tiny_tokenizer, "text")
        with pytest.raises(ValueError, match="Empty"):
            token_ref_to_id("  ", tiny_tokenizer, "id")

    def test_parse_list_id_deduplicates(self, tiny_tokenizer):
        ids = parse_token_refs("42, 42, 7", tiny_tokenizer, "id")
        assert ids == [42, 7]

    def test_parse_list_text(self, tiny_tokenizer):
        expected = [
            tiny_tokenizer.encode(t, add_special_tokens=False)[0] for t in ("a", "b")
        ]
        assert parse_token_refs("a, b, a", tiny_tokenizer, "text") == expected

    def test_parse_empty(self, tiny_tokenizer):
        assert parse_token_refs("", tiny_tokenizer, "text") == []
        assert parse_token_refs(None, tiny_tokenizer, "id") == []


class TestParseLayerRefs:
    def test_lists_and_ranges(self):
        assert parse_layer_refs("11,12-15,18") == [11, 12, 13, 14, 15, 18]

    def test_single_layer_and_spaces(self):
        assert parse_layer_refs(" 3 ") == [3]
        assert parse_layer_refs("5, 2-3") == [2, 3, 5]

    def test_deduplicates(self):
        assert parse_layer_refs("2,1-3,2") == [1, 2, 3]

    def test_empty_rejected(self):
        with pytest.raises(ValueError, match="Empty"):
            parse_layer_refs("  ,  ")
        with pytest.raises(ValueError, match="Empty"):
            parse_layer_refs(None)

    def test_malformed_rejected(self):
        with pytest.raises(ValueError, match="Bad layer"):
            parse_layer_refs("1,foo")
        with pytest.raises(ValueError, match="Bad layer"):
            parse_layer_refs("-2")
        with pytest.raises(ValueError, match="Descending"):
            parse_layer_refs("5-3")


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
        assert value[0] == ("a", "tok")
        assert value[1] == ("␣", "tok")  # whitespace made visible
        assert value[2] == ("b", "sel")

    def test_highlight_every_token_clickable(self):
        # Gradio only dispatches the select event for spans whose category is
        # non-None; every position must carry one to stay clickable.
        for _text, category in highlighted_tokens(["x", " ", "", "y"], [1]):
            assert category in ("tok", "sel")

    def test_highlight_leading_space_visible(self):
        value = highlighted_tokens([" Paris"], [])
        assert value[0][0] == "␣Paris"

    def test_selection_summary(self):
        assert selection_summary([], None) == ""
        assert "all 5 positions" in selection_summary([], 5)
        summary = selection_summary([1, 3], 5)
        assert "2 of 5 positions" in summary and "1, 3" in summary
        many = selection_summary(list(range(20)), 30)
        assert "20 of 30 positions" in many and "…" in many


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


class TestLensViews:
    def test_readouts_table(self):
        rows = [ReadoutRow(token_id=9, text=" tok", count=5, count_by_layer=[5, 0])]
        out = readouts_table_html(rows)
        assert ">␣tok<" in out  # leading space made visible
        assert ">9<" in out and ">5<" in out
        assert "█" in out  # sparkline
        assert readouts_table_html([]) == ""
        assert "⚡" not in out  # no intervention caption without the arg

    def test_readouts_table_intervention_caption(self):
        rows = [ReadoutRow(token_id=9, text="tok", count=5, count_by_layer=[5, 0])]
        out = readouts_table_html(rows, {0: "ablate 'x' @L0 (logit)"})
        assert "⚡" in out and "L0:" in out
        assert "ablate" in out and "@L0" in out and "(logit)" in out

    def test_readouts_table_caption_escapes(self):
        rows = [ReadoutRow(token_id=9, text="tok", count=5, count_by_layer=[5, 0])]
        out = readouts_table_html(rows, {2: 'swap "<b>"→"x" @L2 (logit)'})
        # the user-supplied description is escaped; no raw HTML injection
        assert "&lt;b&gt;" in out and '"<b>"' not in out

    def test_heatmap_grid(self):
        out = heatmap_html(SLICE)
        # final layer on top: L2's row markup precedes L0's
        assert out.index(">L2<") < out.index(">L0<")
        assert ">g<" in out  # top-1 cell text
        assert "2. h (0.050)" in out  # hover lists the top-k
        assert "<table" in out and "overflow:auto" in out  # scrolls both ways
        assert "⚡" not in out  # no marker without the arg
        empty = LensSlice(
            mode="logit", layers=[], positions=[], position_texts=[],
            tokens=[], probs=[], texts=[],
        )
        assert heatmap_html(empty) == ""

    def test_heatmap_marks_intervened_layers(self):
        out = heatmap_html(SLICE, {2: "steer 'a' @L2 (α=+3) (logit)"})
        assert "⚡L2" in out and "⚡L0" not in out
        assert 'title="steer' in out
        assert "intervened layer" in out  # caption note

    def test_heatmap_intervened_title_escaped(self):
        out = heatmap_html(SLICE, {2: 'swap "<b>"→"x" @L2 (logit)'})
        assert "&lt;b&gt;" in out
        assert 'title="swap "<b>"' not in out  # no raw HTML in the title attr

    def test_heatmap_escapes_tokens(self):
        s = LensSlice(
            mode="logit", layers=[0], positions=[0],
            position_texts=["<|im_start|>"],
            tokens=[[[1]]], probs=[[[1.0]]], texts=[[["<b>"]]],
        )
        out = heatmap_html(s)
        assert "&lt;b&gt;" in out and "&lt;|im_start|&gt;" in out
        assert "><b><" not in out  # token never lands unescaped in a cell

    def test_distribution_grid(self):
        rows = [
            ReadoutRow(token_id=1, text="a", count=3, count_by_layer=[2, 1]),
            ReadoutRow(token_id=2, text="b", count=1, count_by_layer=[0, 1]),
        ]
        out = distribution_html(rows, [0, 2])
        assert ">L0<" in out and ">L2<" in out
        assert ">a (3)<" in out
        assert 'title="L0: 2 cells"' in out
        assert distribution_html([], [0]) == ""

    def test_distribution_marks_intervened_layer(self):
        rows = [ReadoutRow(token_id=1, text="a", count=3, count_by_layer=[2, 1])]
        out = distribution_html(rows, [0, 2], intervened={2: "steer 'a' @L2 (jacobian)"})
        assert 'title="steer &#x27;a&#x27; @L2 (jacobian)"' in out
        assert ">L0<" in out  # unmarked layer label intact


class TestInterventionVisibility:
    def _iv(self, **kw):
        return Intervention(
            kind=kw.get("kind", "steer"), layer=kw["layer"],
            token_id=kw.get("token_id", 1), basis=kw["basis"],
        )

    def test_intervened_layer_titles_joins_same_layer(self):
        ivs = [
            self._iv(layer=5, basis="jacobian"),
            self._iv(layer=5, basis="logit", kind="ablate"),
        ]
        titles = intervened_layer_titles(ivs)
        assert set(titles) == {5}
        assert "; " in titles[5]
        assert "(jacobian)" in titles[5] and "(logit)" in titles[5]

    def test_summary_caps_with_more(self):
        ivs = [self._iv(layer=i, basis="logit") for i in range(6)]
        summary = interventions_summary(ivs, limit=4)
        assert summary.count(";") == 4  # 4 edits + the "+2 more" tail
        assert "+2 more" in summary

    @pytest.mark.parametrize("basis,mode,layer,warns", [
        ("jacobian", "logit", 5, True),
        ("jacobian", "jacobian", 5, False),
        ("jacobian", "diff", 5, False),
        ("logit", "jacobian", 5, True),
        ("logit", "logit", 5, False),
        ("logit", "diff", 5, False),
        ("jacobian", "logit", 31, False),  # final layer (n_layers=32) exempt
    ])
    def test_warning_truth_table(self, basis, mode, layer, warns):
        ivs = [self._iv(layer=layer, basis=basis)]
        result = intervention_visibility_warning(ivs, mode, n_layers=32)
        if not warns:
            assert result is None
        else:
            assert result is not None and "⚠" in result
            assert f"{basis} basis" in result
            assert ("Jacobian" if basis == "jacobian" else "Logit") in result

    def test_warning_names_only_mismatched(self):
        ivs = [
            self._iv(layer=5, basis="jacobian"),  # mismatched in logit view
            self._iv(layer=6, basis="logit"),     # matches logit view
        ]
        result = intervention_visibility_warning(ivs, "logit", n_layers=32)
        assert result is not None
        assert "@L5" in result and "@L6" not in result
