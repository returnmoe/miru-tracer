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
    ui_sampling_params,
)


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
