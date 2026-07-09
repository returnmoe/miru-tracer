"""Lens UI helpers and the new lens plot builders."""

import pytest

from miru_tracer.core.interventions import Intervention
from miru_tracer.core.lens import LensSlice, ReadoutRow
from miru_tracer.ui.lens_common import (
    INTERVENTIONS_TABLE_JS,
    add_pinned_token,
    apply_intervention_table_action,
    enabled_intervention_group_count,
    enabled_interventions,
    format_layer_refs,
    get_active_interventions,
    highlighted_tokens,
    intervened_layer_titles,
    intervention_group,
    intervention_visibility_warning,
    interventions_summary,
    interventions_table_html,
    layer_selection,
    lens_layer_selection,
    lens_mode_key,
    parse_layer_refs,
    parse_token_refs,
    pinned_token_choices,
    pinned_tokens_table_html,
    remove_pinned_tokens,
    selection_summary,
    set_active_interventions,
    sparkline,
    toggle_position,
    token_mode_key,
    token_ref_to_id,
)
from miru_tracer.ui.lens_views import (
    READOUT_INSPECTOR_JS,
    distribution_html,
    heatmap_html,
    readout_inspector_html,
)
from miru_tracer.visualization.plots import (
    plot_lens_heatmap,
    plot_pinned_token_ranks,
    plot_readout_distribution,
)


class MultilingualTokenizer:
    def convert_ids_to_tokens(self, ids):
        return ["æ³ķåĽ½" if token_id == 7 else str(token_id) for token_id in ids]

    def decode(self, ids, **_kwargs):
        return "法国" if ids[0] == 7 else str(ids[0])


class TestLensModeKey:
    @pytest.mark.parametrize("ui,key", [
        ("Logit", "logit"),
        ("Jacobian", "jacobian"),
        ("Compare (Jacobian / Logit)", "compare"),
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

    def test_jacobian_auto_start_skips_first_29_percent(self):
        layers = lens_layer_selection(64, -1, -1, 1, "jacobian")
        assert layers[0] == 18
        assert layers[-1] == 63

    def test_compare_auto_start_and_explicit_zero(self):
        assert lens_layer_selection(36, -1, -1, 1, "compare")[0] == 10
        assert lens_layer_selection(36, 0, -1, 1, "compare")[0] == 0

    def test_logit_auto_starts_at_zero(self):
        assert lens_layer_selection(8, -1, -1, 2, "logit") == [0, 2, 4, 6, 7]

    def test_final_output_is_appended_outside_requested_range(self):
        assert lens_layer_selection(16, 5, 9, 2, "jacobian") == [5, 7, 9, 15]


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
        summary_all = selection_summary([], 5)
        assert "all 4 token-aligned positions" in summary_all
        assert "position 0" in summary_all
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


class TestActiveInterventionsTable:
    def test_empty_message(self):
        out = interventions_table_html([])
        assert "No active interventions" in out
        assert "<table" not in out

    def test_rows_rendered_and_escaped(self, tiny_tokenizer):
        token_id = tiny_tokenizer.encode("<b>", add_special_tokens=False)[0]
        group = intervention_group(
            [Intervention(kind="steer", layer=layer, token_id=token_id, basis="logit")
             for layer in [14, 15, 16, 17, 18, 33]]
        )
        out = interventions_table_html([group], tiny_tokenizer)
        assert "<table" in out
        assert ">14-18, 33<" in out and ">logit<" in out
        assert "@L14" not in out
        assert "steer" in out
        assert "&lt;" in out and "<b>" not in out
        assert 'data-miru-iv-action="toggle"' in out
        assert 'data-miru-iv-action="delete"' in out
        assert 'aria-label="Enable intervention group 0 for layers 14-18, 33"' in out
        assert 'aria-label="Remove intervention group 0 for layers 14-18, 33"' in out
        assert 'style="width:100%; border:1px solid rgba(127,127,127,0.18) !important;' in out
        assert "border:1px solid rgba(127,127,127,0.18) !important" in out

    def test_multilingual_token_label(self):
        iv = Intervention(kind="ablate", layer=0, token_id=7, basis="logit")
        out = interventions_table_html([intervention_group([iv])], MultilingualTokenizer())
        assert "æ³ķåĽ½ (法国)" in out
        assert "æ³ķåĽ½ (法国)" in iv.describe(MultilingualTokenizer())

    def test_table_js_triggers_gradio_html_click(self):
        assert "trigger('click'" in INTERVENTIONS_TABLE_JS
        assert "miru-iv-action-payload" not in INTERVENTIONS_TABLE_JS

    def test_enabled_groups_flatten_in_group_and_layer_order(self):
        first_edits = [
            Intervention(kind="steer", layer=0, token_id=1),
            Intervention(kind="steer", layer=1, token_id=1),
        ]
        second_edits = [
            Intervention(kind="ablate", layer=1, token_id=2),
            Intervention(kind="ablate", layer=2, token_id=2),
        ]
        first = intervention_group(first_edits)
        second = intervention_group(second_edits, enabled=False)
        assert enabled_interventions([first, second]) == first_edits
        assert enabled_intervention_group_count([first, second]) == 1

        enabled = {**second, "enabled": True}
        assert enabled_interventions([first, enabled]) == first_edits + second_edits
        assert enabled_intervention_group_count([first, enabled]) == 2

    def test_separate_identical_groups_are_preserved(self):
        edit = Intervention(kind="steer", layer=0, token_id=1, strength=0.5)
        groups = [intervention_group([edit]), intervention_group([edit])]
        assert len(groups) == 2
        assert enabled_interventions(groups) == [edit, edit]

    @pytest.mark.parametrize(
        "layers, expected",
        [
            ([14, 15, 16, 17, 18, 33], "14-18, 33"),
            ([3], "3"),
            ([5, 2, 3, 2], "2-3, 5"),
            ([], ""),
        ],
    )
    def test_layer_labels_are_sorted_and_compacted(self, layers, expected):
        assert format_layer_refs(layers) == expected

    def test_group_action_toggle_and_delete(self):
        rows = [
            intervention_group(
                [
                    Intervention(kind="steer", layer=0, token_id=1),
                    Intervention(kind="steer", layer=1, token_id=1),
                ]
            ),
            intervention_group(
                [Intervention(kind="ablate", layer=1, token_id=2)]
            ),
        ]
        updated, status = apply_intervention_table_action(
            rows, '{"action":"toggle","index":0,"enabled":false}'
        )
        assert "group 0 disabled" in status
        assert enabled_interventions(updated) == rows[1]["interventions"]

        updated, status = apply_intervention_table_action(
            updated, '{"action":"delete","index":1}'
        )
        assert "Deleted intervention group 1" in status
        assert len(updated) == 1
        assert updated[0]["interventions"] == rows[0]["interventions"]

    def test_table_action_rejects_bad_payload(self):
        rows = [intervention_group([Intervention(kind="steer", layer=0, token_id=1)])]
        updated, status = apply_intervention_table_action(rows, "not-json")
        assert updated == rows
        assert "Ignored invalid" in status

        updated, status = apply_intervention_table_action(
            rows, {"action": "delete", "index": 99}
        )
        assert updated == rows
        assert "Ignored invalid" in status


class TestPinnedTokens:
    def test_add_remove_and_choices(self, tiny_tokenizer):
        token_id = tiny_tokenizer.encode("A", add_special_tokens=False)[0]
        ids = add_pinned_token([], "A", tiny_tokenizer, "text")
        assert ids == [token_id]
        assert add_pinned_token(ids, str(token_id), tiny_tokenizer, "id") == ids
        assert pinned_token_choices(ids, tiny_tokenizer)[0][1] == str(token_id)
        assert remove_pinned_tokens(ids, [str(token_id)]) == []

    def test_table(self, tiny_tokenizer):
        assert "No pinned tokens" in pinned_tokens_table_html([])
        token_id = tiny_tokenizer.encode("A", add_special_tokens=False)[0]
        out = pinned_tokens_table_html([token_id], tiny_tokenizer)
        assert "<table" in out and str(token_id) in out

    def test_multilingual_choices_and_table(self):
        tokenizer = MultilingualTokenizer()
        assert pinned_token_choices([7], tokenizer) == [
            ("7: æ³ķåĽ½ (法国)", "7")
        ]
        table = pinned_tokens_table_html([7], tokenizer)
        assert "æ³ķåĽ½" in table and "法国" in table


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

    def test_multilingual_pinned_rank_label(self):
        fig = plot_pinned_token_ranks(SLICE, MultilingualTokenizer())
        assert fig.data[0].name == "æ³ķåĽ½ (法国)"

    def test_multilingual_hover_is_complete_and_escaped(self):
        multilingual = LensSlice(
            mode="logit",
            layers=[0],
            positions=[0],
            position_texts=["ctx"],
            tokens=[[[7, 8]]],
            probs=[[[0.8, 0.1]]],
            texts=[[["æ³ķåĽ½ (法国)", "raw<script> (日本)"]]],
        )
        fig = plot_lens_heatmap(multilingual)
        hover = fig.data[0].customdata[0][0]
        assert "æ³ķåĽ½ (法国)" in hover
        assert "raw&lt;script&gt; (日本)" in hover
        assert "..." not in hover

    def test_pinned_ranks_none_when_empty(self):
        empty = LensSlice(
            mode="logit", layers=[0], positions=[0], position_texts=["x"],
            tokens=[[[1]]], probs=[[[1.0]]], texts=[[["a"]]],
        )
        assert plot_pinned_token_ranks(empty) is None


class TestLensViews:
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


class TestReadoutInspector:
    @staticmethod
    def _slice(mode="jacobian"):
        return LensSlice(
            mode=mode,
            layers=[0, 2],
            positions=[4],
            position_texts=["planning"],
            tokens=[[[7, 8]], [[10, 7]]],
            probs=[[[0.2, 0.1]], [[0.7, 0.05]]],
            texts=[[['plans', 'plan']], [['planning', 'plans']]],
        )

    @staticmethod
    def _rows():
        return [
            ReadoutRow(
                token_id=7,
                text="plans",
                count=2,
                count_by_layer=[1, 1],
                relevance_score=1.5,
                relevance_by_layer=[1.0, 0.5],
                best_rank_by_layer=[0, 1],
                peak_prob_by_layer=[0.2, 0.05],
            )
        ]

    def test_single_position_has_layer_preview_and_exact_probabilities(self):
        slice_ = self._slice()
        rows = self._rows()
        out = readout_inspector_html(
            mode="jacobian",
            slices={"jacobian": slice_},
            rows={"jacobian": rows},
            all_rows={"jacobian": rows},
            recommended_start=1,
        )
        assert "Selected token" in out and "planning" in out and "position 4" in out
        assert "Count ↓" in out
        assert ">ID</span>" in out
        assert '<div class="miru-readout-id">7</div>' in out
        assert '<div class="miru-readout-id">10</div>' in out
        assert "<span>All</span><span>Layers</span>" in out
        assert 'data-readout-layer="0"' in out
        assert 'data-readout-panel="0"' in out
        assert "20.00%" in out and "final model distribution for selected token" in out
        assert "preceding causal state p−1" in out
        assert "often degenerate" in out
        assert "best displayed rank 1" in out
        assert "miru-readout-all-active" in out
        assert "height:clamp(20rem,54vh,34rem)" in out
        assert "overflow-y:auto" in out
        assert "scrollbar-gutter:stable" in out
        assert "miru-readout-layer-focus" in out

    def test_interventions_move_into_readouts_and_are_escaped(self):
        slice_ = self._slice()
        rows = self._rows()
        out = readout_inspector_html(
            mode="jacobian",
            slices={"jacobian": slice_},
            rows={"jacobian": rows},
            all_rows={"jacobian": rows},
            recommended_start=1,
            intervened={2: 'swap "<b>"→"x" @L2 (jacobian)'},
        )
        assert "⚡" in out and "Interventions" in out and "L2:" in out
        assert "&lt;b&gt;" in out and '"<b>"' not in out

    def test_multilingual_aggregate_label_is_preserved_and_escaped(self):
        slice_ = self._slice()
        rows = [
            ReadoutRow(
                token_id=7,
                text="raw<script> (法国)",
                count=1,
                count_by_layer=[1, 0],
            )
        ]
        out = readout_inspector_html(
            mode="jacobian",
            slices={"jacobian": slice_},
            rows={"jacobian": rows},
            all_rows={"jacobian": rows},
            recommended_start=1,
        )
        assert "raw&lt;script&gt; (法国)" in out
        assert "raw<script>" not in out

    def test_compare_uses_one_shared_selector_and_two_columns(self):
        jacobian = self._slice("jacobian")
        logit = self._slice("logit")
        rows = self._rows()
        out = readout_inspector_html(
            mode="compare",
            slices={"jacobian": jacobian, "logit": logit},
            rows={"jacobian": rows, "logit": rows},
            all_rows={"jacobian": rows, "logit": rows},
            recommended_start=1,
        )
        assert out.count('<div class="miru-readout-selector">') == 1
        assert "Jacobian Lens" in out and "Logit Lens" in out
        assert "miru-readout-compare" in out

    def test_multiple_positions_stays_aggregate(self):
        rows = self._rows()
        out = readout_inspector_html(
            mode="logit",
            slices={"logit": SLICE},
            rows={"logit": rows},
            all_rows={"logit": rows},
            recommended_start=0,
        )
        assert "2 positions" in out
        assert "Select exactly one" in out
        assert '<div class="miru-readout-selector">' not in out

    def test_js_supports_hover_preview_click_lock_and_all_layers(self):
        assert "pointerover" in READOUT_INSPECTOR_JS
        assert "pointerout" in READOUT_INSPECTOR_JS
        assert "lockedLayer" in READOUT_INSPECTOR_JS
        assert "data-readout-all" in READOUT_INSPECTOR_JS
        assert "miru-readout-all-active" in READOUT_INSPECTOR_JS
        assert "root.querySelectorAll('.miru-readout-layer-slot')" in READOUT_INSPECTOR_JS
        assert "root.querySelectorAll('[data-readout-layer]').forEach" not in READOUT_INSPECTOR_JS
        assert "miru-readout-mini [data-readout-layer]" in READOUT_INSPECTOR_JS
        assert "miru-readout-layer-focus" in READOUT_INSPECTOR_JS
        assert "keydown" in READOUT_INSPECTOR_JS

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
        ("jacobian", "compare", 5, False),
        ("logit", "jacobian", 5, True),
        ("logit", "logit", 5, False),
        ("logit", "compare", 5, False),
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
