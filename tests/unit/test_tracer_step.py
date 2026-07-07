"""Step / undo / goto / dual-probability behavior of LLMTracer."""

import pytest
import torch

from miru_tracer.core.sampling import SamplingParams


class TestStep:
    def test_step_records_history_and_extends_sequence(self, tracer):
        tracer.reset("Hello")
        prompt_len = tracer.seq_len
        step = tracer.step()
        assert step.step == 0
        assert len(tracer.history) == 1
        assert tracer.seq_len == prompt_len + 1
        assert tracer.input_ids[0, -1].item() == step.token_id

    def test_greedy_picks_top_token(self, tracer):
        tracer.reset("Hello")
        dist = tracer.peek(top_k=1)
        step = tracer.step()  # default greedy
        assert step.token_id == dist.top_k_tokens[0]

    def test_token_override(self, tracer):
        tracer.reset("Hello")
        step = tracer.step(token_id=42)
        assert step.token_id == 42
        assert tracer.input_ids[0, -1].item() == 42

    def test_override_token_outside_top_k_has_probability(self, tracer):
        """Old design ran a full re-forward for this; must still be correct."""
        tracer.reset("Hello")
        dist = tracer.peek(top_k=3)
        outside = next(i for i in range(200) if i not in dist.top_k_tokens)
        step = tracer.step(token_id=outside, log_top_k=3)
        assert 0.0 <= step.probability <= 1.0
        assert 0.0 <= step.raw_probability <= 1.0

    def test_dual_probabilities_diverge_under_temperature(self, tracer):
        """Regression: greedy probability used to be temperature-scaled only."""
        tracer.reset("Hello")
        step = tracer.step(SamplingParams(temperature=2.0))
        assert step.probability != pytest.approx(step.raw_probability)

        # raw_probability must equal the temperature-1 softmax of raw logits
        tracer.undo()
        raw_probs = torch.softmax(tracer._next_raw_logits(), dim=-1)
        assert step.raw_probability == pytest.approx(
            raw_probs[step.token_id].item(), rel=1e-5
        )

    def test_full_probs_logged_and_sum_to_one(self, tracer):
        tracer.reset("Hello")
        step = tracer.step(log_full_probs=True)
        assert step.full_probs is not None
        assert sum(step.full_probs) == pytest.approx(1.0, abs=1e-4)

    def test_full_probs_not_logged_by_default(self, tracer):
        tracer.reset("Hello")
        assert tracer.step().full_probs is None

    def test_sampling_deterministic_with_same_seed(self, tiny_model, tiny_tokenizer):
        from miru_tracer.core.tracer import LLMTracer

        params = SamplingParams(strategy="sampling", temperature=1.0, top_k=0)
        tokens = []
        for _ in range(2):
            t = LLMTracer(tiny_model, tiny_tokenizer, device="cpu", seed=7)
            t.reset("Hello")
            tokens.append([t.step(params).token_id for _ in range(5)])
        assert tokens[0] == tokens[1]


class TestUndo:
    def test_undo_removes_last_step(self, tracer):
        tracer.reset("Hello")
        tracer.step()
        first = tracer.step()
        assert tracer.undo() is True
        assert len(tracer.history) == 1
        assert tracer.input_ids[0, -1].item() != first.token_id or tracer.seq_len == (
            tracer._prompt_len + 1
        )

    def test_undo_empty_history_returns_false(self, tracer):
        tracer.reset("Hello")
        assert tracer.undo() is False

    def test_undo_more_than_available_returns_false_without_change(self, tracer):
        tracer.reset("Hello")
        tracer.step()
        assert tracer.undo(2) is False
        assert len(tracer.history) == 1

    def test_undo_multiple(self, tracer):
        tracer.reset("Hello")
        for _ in range(4):
            tracer.step()
        assert tracer.undo(3) is True
        assert len(tracer.history) == 1

    def test_generation_identical_after_undo(self, tracer):
        """Undo must be perfectly invisible to subsequent greedy generation."""
        tracer.reset("Hello")
        first_run = [tracer.step().token_id for _ in range(3)]
        tracer.undo(3)
        second_run = [tracer.step().token_id for _ in range(3)]
        assert first_run == second_run

    def test_undo_does_not_retokenize_prompt(self, tracer, monkeypatch):
        tracer.reset("Hello world, this is a test")
        tracer.step()

        def boom(*args, **kwargs):  # pragma: no cover
            raise AssertionError("undo must not re-encode the prompt")

        monkeypatch.setattr(tracer.tokenizer, "encode", boom)
        assert tracer.undo() is True


class TestGoto:
    def test_goto_earlier_step(self, tracer):
        tracer.reset("Hello")
        for _ in range(5):
            tracer.step()
        assert tracer.goto_step(2) is True
        assert len(tracer.history) == 2

    def test_goto_current_step_is_noop(self, tracer):
        tracer.reset("Hello")
        tracer.step()
        assert tracer.goto_step(1) is True
        assert len(tracer.history) == 1

    def test_goto_out_of_range(self, tracer):
        tracer.reset("Hello")
        tracer.step()
        assert tracer.goto_step(5) is False
        assert tracer.goto_step(-1) is False


class TestChatMode:
    def test_chat_mode_applies_template(self, tracer):
        messages = [{"role": "user", "content": "Hi"}]
        tracer.reset(messages=messages)
        assert tracer.mode == "chat"
        assert tracer.messages == messages
        assert "user: Hi" in tracer.prompt
        assert tracer.prompt.endswith("assistant: ")

    def test_chat_mode_without_template_falls_back(self, tiny_model, tiny_tokenizer):
        from miru_tracer.core.tracer import LLMTracer

        t = LLMTracer(tiny_model, tiny_tokenizer, device="cpu")
        t.has_chat_template = False
        messages = [{"role": "user", "content": "Hi"}]
        t.reset(messages=messages)
        # Regression: the fallback path used to leave self.messages unset.
        assert t.messages == messages
        assert t.export_to_dict()["messages"] == messages
        t.step()
        assert t.undo() is True

    def test_completion_mode(self, tracer):
        tracer.reset("plain text")
        assert tracer.mode == "completion"
        assert tracer.messages is None


class TestText:
    def test_generated_and_full_text(self, tracer):
        tracer.reset("Hello")
        tracer.step()
        generated = tracer.get_generated_text()
        full = tracer.get_full_text()
        assert full.startswith("Hello")
        assert full == "Hello" + generated
