"""Shared UI helper behavior (no Gradio server needed)."""

import os

import pytest

from miru_tracer.core.sampling import MIN_TEMPERATURE
from miru_tracer.ui.helpers import (
    ChatValidationError,
    ExportManager,
    build_prob_table,
    build_radio_choices,
    parse_chat_messages,
    prob_mode_key,
    static_table_html,
    thinking_key,
    toggle_mode_visibility,
    toggle_temperature,
    toggle_think_prefill,
    ui_sampling_params,
)
from miru_tracer.ui.theme import footer_js


class TestParseChatMessages:
    def test_valid(self):
        messages = parse_chat_messages('[{"role": "user", "content": "hi"}]')
        assert messages == [{"role": "user", "content": "hi"}]

    @pytest.mark.parametrize(
        "text",
        [
            "not json",
            '{"role": "user"}',  # not a list
            "[]",  # empty
            '[{"role": "user"}]',  # missing content
            '["just a string"]',  # not a dict
        ],
    )
    def test_invalid_raises(self, text):
        with pytest.raises(ChatValidationError):
            parse_chat_messages(text)


class TestModeToggles:
    @pytest.mark.parametrize(
        "mode,expected",
        [
            ("Completion", (True, False, False)),
            ("Chat", (False, True, False)),
            ("Raw", (False, False, True)),
        ],
    )
    def test_mode_visibility(self, mode, expected):
        updates = toggle_mode_visibility(mode)
        assert tuple(u["visible"] for u in updates) == expected

    def test_temperature_disabled_for_greedy(self):
        update = toggle_temperature("greedy")
        assert update["interactive"] is False
        assert update["info"]  # explains why it has no effect

    def test_temperature_enabled_for_sampling(self):
        update = toggle_temperature("sampling")
        assert update["interactive"] is True

    @pytest.mark.parametrize(
        "choice,key",
        [
            ("Template default", "auto"),
            ("Off (no thinking)", "off"),
            ("Prefill thought…", "prefill"),
            (None, "auto"),
        ],
    )
    def test_thinking_key(self, choice, key):
        assert thinking_key(choice) == key

    def test_think_prefill_visibility(self):
        assert toggle_think_prefill("Prefill thought…")["visible"] is True
        assert toggle_think_prefill("Off (no thinking)")["visible"] is False


class TestExportManager:
    def test_reuses_one_file(self):
        """Regression: exports used to leak one temp file per call."""
        manager = ExportManager("test")
        paths = set()
        for i in range(20):
            update = manager.prepare({"n": i})
            paths.add(update["value"])
        assert len(paths) == 1
        files = os.listdir(os.path.dirname(paths.pop()))
        assert len(files) == 1

    def test_content_is_latest(self):
        import json

        manager = ExportManager("test")
        manager.prepare({"n": 1})
        update = manager.prepare({"n": 2})
        with open(update["value"]) as f:
            assert json.load(f)["n"] == 2


class TestProbTable:
    def test_probability_column_is_numeric(self, tracer):
        """Regression: the Dataframe declared 'number' but got strings."""
        tracer.reset("Hello")
        dist = tracer.peek(top_k=5)
        df = build_prob_table(dist)
        assert df["Probability"].dtype.kind == "f"
        assert df["Token ID"].dtype.kind in "iu"
        assert len(df) == 5

    def test_radio_choices_values_are_token_ids(self, tracer):
        tracer.reset("Hello")
        dist = tracer.peek(top_k=5)
        choices = build_radio_choices(dist)
        assert [int(value) for _, value in choices] == dist.top_k_tokens[:10]


class TestUiSamplingParams:
    def test_clamps_out_of_range_widget_values(self):
        params = ui_sampling_params("sampling", 0.0, None, 0.0)
        assert params.temperature == MIN_TEMPERATURE
        assert params.top_k == 0
        assert 0 < params.top_p <= 1.0

    def test_passthrough(self):
        params = ui_sampling_params("greedy", 0.7, 50, 0.9)
        assert (params.strategy, params.temperature, params.top_k, params.top_p) == (
            "greedy",
            0.7,
            50,
            0.9,
        )


class TestProbModeKey:
    def test_mapping(self):
        assert prob_mode_key("Raw (pre-temperature)") == "raw"
        assert prob_mode_key("Adjusted (post-temperature)") == "adjusted"
        assert prob_mode_key(None) == "adjusted"


class TestStaticTableHtml:
    def test_rows_and_headers_rendered(self):
        out = static_table_html(["ID", "Token"], [[65, "A"], [66, "B"]])
        assert out.count("<tr>") == 3  # header + 2 rows
        assert "<th" in out and "ID" in out and "Token" in out
        assert "65" in out and "B" in out

    def test_cells_escaped(self):
        out = static_table_html(["T"], [["<script>alert(1)</script>"]])
        assert "<script>" not in out
        assert "&lt;script&gt;" in out

    def test_empty_rows(self):
        assert static_table_html(["A"], []) == ""


class TestThemeJs:
    def test_footer_js_is_function_expression(self):
        script = footer_js("test")
        assert script.lstrip().startswith("() => {")
        assert not script.lstrip().startswith("function()")
        assert "miruIvBridgeInstalled" not in script
        assert "miru-iv-action-apply" not in script
        assert "footer-version" in script
