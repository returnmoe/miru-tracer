"""Export schema v3 and backward-compatible parsing of older logs."""

import json
from pathlib import Path

import pytest

from miru_tracer.core.sampling import SamplingParams
from miru_tracer.core.schema import SCHEMA_VERSION, TokenStep, parse_log

LEGACY_LOG = Path(__file__).parent.parent / "data" / "legacy_log_v1.json"


class TestExportV3:
    def test_export_shape(self, tracer):
        tracer.reset("Hello")
        params = SamplingParams(strategy="sampling", temperature=0.7, top_k=40, top_p=0.9)
        tracer.step(params, log_full_probs=True)
        data = tracer.export_to_dict(params)

        assert data["schema_version"] == SCHEMA_VERSION
        assert data["mode"] == "completion"
        assert data["num_steps"] == 1
        assert data["sampling_params"]["temperature"] == 0.7
        assert data["sampling_params"]["strategy"] == "sampling"

        step = data["history"][0]
        # Regression: the field was misnamed all_logits while storing probabilities.
        assert "all_logits" not in step
        assert step["full_probs"] is not None
        assert step["full_raw_probs"] is not None
        assert sum(step["full_probs"]) == pytest.approx(1.0, abs=1e-4)
        assert sum(step["full_raw_probs"]) == pytest.approx(1.0, abs=1e-4)
        assert "raw_probability" in step
        assert "top_k_raw_probs" in step
        assert step["sampling_params"] == params.to_dict()
        assert step["selection_source"] == "sampled"

    def test_mixed_history_is_not_mislabeled(self, tracer):
        tracer.reset("Hello")
        tracer.step(SamplingParams(strategy="greedy", temperature=1.0))
        tracer.step(SamplingParams(strategy="sampling", temperature=0.5), token_id=1)
        data = tracer.export_to_dict()

        assert data["sampling_params"] == {"mixed": True}
        assert data["history"][0]["selection_source"] == "greedy"
        assert data["history"][1]["selection_source"] == "manual"
        assert data["history"][0]["sampling_params"]["temperature"] == 1.0
        assert data["history"][1]["sampling_params"]["temperature"] == 0.5

    def test_export_is_json_serializable(self, tracer):
        tracer.reset("Hello")
        tracer.step()
        json.dumps(tracer.export_to_dict())

    def test_roundtrip_through_parse_log(self, tracer):
        tracer.reset("Hello")
        for _ in range(3):
            tracer.step()
        log = parse_log(json.loads(json.dumps(tracer.export_to_dict())))
        assert log.schema_version == SCHEMA_VERSION
        assert log.num_steps == 3
        assert [s.token_id for s in log.history] == tracer.generated_tokens


class TestParseLegacyV1:
    def test_parses_committed_v1_fixture(self):
        data = json.loads(LEGACY_LOG.read_text())
        log = parse_log(data)

        assert log.schema_version == 1
        assert log.num_steps == 2
        assert log.mode == "completion"
        # bare top-level temperature is surfaced through sampling_params
        assert log.temperature == pytest.approx(0.8)

        first = log.history[0]
        # v1 'all_logits' (which held probabilities) maps to full_probs
        assert first.full_probs == [0.1, 0.2, 0.7]
        # missing raw fields fall back to adjusted values
        assert first.raw_probability == first.probability
        assert first.top_k_raw_probs == first.top_k_probs
        assert first.token_text_raw == first.token_text
        assert first.selection_source == "unknown"
        assert first.sampling_params["temperature"] == pytest.approx(0.8)

        assert log.history[1].full_probs is None

    def test_prompt_null_tolerated(self):
        data = json.loads(LEGACY_LOG.read_text())
        data["prompt"] = None
        assert parse_log(data).prompt == ""


class TestParseErrors:
    def test_non_dict_rejected(self):
        with pytest.raises(ValueError, match="JSON object"):
            parse_log([1, 2, 3])

    def test_missing_history_rejected(self):
        with pytest.raises(ValueError, match="history"):
            parse_log({"mode": "completion"})

    def test_malformed_step_rejected(self):
        with pytest.raises(ValueError, match="Malformed"):
            parse_log({"history": [{"step": 0}]})

    def test_future_schema_rejected(self):
        with pytest.raises(ValueError, match="Unsupported schema_version"):
            parse_log({"schema_version": SCHEMA_VERSION + 1, "history": []})

    def test_invalid_selection_source_rejected(self):
        data = json.loads(LEGACY_LOG.read_text())
        data["schema_version"] = SCHEMA_VERSION
        data["history"][0]["selection_source"] = "roulette"
        with pytest.raises(ValueError, match="selection_source"):
            parse_log(data)


class TestTokenStep:
    def test_to_from_dict_roundtrip(self):
        step = TokenStep(
            step=0,
            token_id=5,
            token_text="a",
            probability=0.5,
            top_k_tokens=[5],
            top_k_probs=[0.5],
            top_k_texts=["a"],
            raw_probability=0.6,
            top_k_raw_probs=[0.6],
            full_probs=None,
            full_raw_probs=None,
            token_text_raw="a",
            top_k_texts_raw=["a"],
            sampling_params={"strategy": "greedy", "temperature": 1.0},
            selection_source="greedy",
        )
        assert TokenStep.from_dict(step.to_dict()) == step
