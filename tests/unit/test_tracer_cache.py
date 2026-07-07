"""KV-cache invariant tests — the regression suite for the cache-desync bug.

The old design re-fed the last token into an already-advanced in-place
DynamicCache whenever a peek and the following step used different display
parameters, silently corrupting all subsequent logits. These tests pin the
new invariants:

1. Cache length == sequence length right after any peek/step; never greater.
2. Peeking is pure: any number of peeks at one position = exactly one forward.
3. Incremental (cached) logits match a fresh full no-cache forward.
4. A manufactured desync is recovered from, not silently accepted.
"""

import pytest
import torch

from miru_tracer.core.sampling import SamplingParams


def count_forwards(monkeypatch, model):
    calls = {"n": 0}
    original = model.forward

    def counting_forward(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "forward", counting_forward)
    return calls


def fresh_full_logits(tracer):
    """Ground truth: full-sequence forward with no cache."""
    with torch.inference_mode():
        out = tracer.model(tracer.input_ids, use_cache=False)
    return out.logits[0, -1, :].float()


def assert_invariants(tracer):
    ok, error = tracer.validate_state()
    assert ok, error
    assert tracer._cache_len() <= tracer.seq_len


class TestCacheInvariants:
    def test_cache_matches_seq_len_after_peek_and_step(self, tracer):
        tracer.reset("Hello world")
        tracer.peek()
        assert tracer._cache_len() == tracer.seq_len
        tracer.step()
        assert_invariants(tracer)
        tracer.peek()
        assert tracer._cache_len() == tracer.seq_len

    def test_peek_is_pure_across_parameter_changes(self, tracer, monkeypatch):
        """Regression: peek+step with mismatched params must not re-forward."""
        tracer.reset("Hello world")
        calls = count_forwards(monkeypatch, tracer.model)

        tracer.peek(top_k=5, temperature=1.0)
        tracer.peek(top_k=20, temperature=0.3)
        tracer.peek(top_k=3, temperature=2.0)
        assert calls["n"] == 1, "peeks with different params must reuse cached logits"

        #

        # The old bug trigger: step with different logging params than the peek.
        tracer.step(SamplingParams(temperature=0.7), log_top_k=10, log_full_probs=True)
        assert calls["n"] == 1, "step after peek must not run another forward"
        assert_invariants(tracer)

        # Next step needs exactly one incremental forward.
        tracer.step()
        assert calls["n"] == 2
        assert_invariants(tracer)

    def test_incremental_logits_match_full_forward(self, tracer):
        tracer.reset("The quick brown fox")
        for _ in range(4):
            tracer.step()
        cached = tracer._next_raw_logits()
        assert torch.allclose(cached, fresh_full_logits(tracer), atol=1e-5)

    def test_logits_correct_after_peek_step_interleave(self, tracer):
        """Corruption test for the exact old-bug sequence: peek, step, peek..."""
        tracer.reset("abc")
        for i in range(3):
            tracer.peek(top_k=5, temperature=1.0 + i)  # peek with varying params
            tracer.step(SamplingParams(temperature=0.5))
        assert torch.allclose(
            tracer._next_raw_logits(), fresh_full_logits(tracer), atol=1e-5
        )

    def test_manufactured_desync_recovers(self, tracer):
        tracer.reset("Hello world")
        tracer.step()
        # Advance the cache past input_ids behind the tracer's back,
        # simulating an interrupted operation.
        with torch.inference_mode():
            tracer.model(
                tracer.input_ids[:, -1:],
                past_key_values=tracer._kv,
                use_cache=True,
            )
        tracer._logits_slot = None
        assert tracer._cache_len() > tracer.seq_len - 1

        # The next peek must recover and produce correct logits.
        tracer.peek()
        assert tracer._cache_len() == tracer.seq_len
        assert torch.allclose(
            tracer._next_raw_logits(), fresh_full_logits(tracer), atol=1e-5
        )

    def test_undo_then_step_logits_match_full_forward(self, tracer):
        tracer.reset("Hello world")
        for _ in range(5):
            tracer.step()
        tracer.undo(3)
        assert_invariants(tracer)
        assert torch.allclose(
            tracer._next_raw_logits(), fresh_full_logits(tracer), atol=1e-5
        )
        tracer.step()
        assert_invariants(tracer)

    def test_undo_to_previous_position_reuses_logits_slot(self, tracer, monkeypatch):
        tracer.reset("Hello world")
        tracer.step()  # forward 1 (prompt)
        tracer.step()  # forward 2 (incremental)
        calls = count_forwards(monkeypatch, tracer.model)
        tracer.undo()
        # The memo slot for this position is still valid: no forward needed.
        tracer.peek()
        assert calls["n"] == 0

    def test_randomized_operation_sequence_holds_invariants(self, tracer):
        import random

        rng = random.Random(1234)
        tracer.reset("Once upon a time")
        for _ in range(40):
            op = rng.choice(["peek", "step", "step_sample", "undo", "goto"])
            if op == "peek":
                tracer.peek(top_k=rng.randint(1, 20), temperature=rng.uniform(0.1, 3.0))
            elif op == "step":
                tracer.step(log_full_probs=rng.random() < 0.3)
            elif op == "step_sample":
                tracer.step(
                    SamplingParams(
                        strategy="sampling",
                        temperature=rng.uniform(0.1, 2.0),
                        top_k=rng.randint(0, 30),
                        top_p=rng.uniform(0.5, 1.0),
                    )
                )
            elif op == "undo":
                tracer.undo(rng.randint(1, 3))
            elif op == "goto":
                tracer.goto_step(rng.randint(0, len(tracer.history)))
            assert_invariants(tracer)

        # After it all: logits still exactly match ground truth.
        assert torch.allclose(
            tracer._next_raw_logits(), fresh_full_logits(tracer), atol=1e-5
        )


class TestPeekWithoutPrompt:
    def test_peek_without_reset_raises(self, tracer):
        tracer.reset("")
        with pytest.raises(ValueError, match="No prompt"):
            tracer.peek()
