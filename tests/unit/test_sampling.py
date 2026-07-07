"""Unit tests for the pure sampling/post-processing layer."""

import math

import pytest
import torch

from miru_tracer.core.sampling import (
    MIN_TEMPERATURE,
    SamplingParams,
    apply_temperature,
    entropy,
    filter_top_k,
    filter_top_p,
    select_token,
)


class TestSamplingParams:
    def test_defaults(self):
        params = SamplingParams()
        assert params.strategy == "greedy"
        assert params.temperature == 1.0

    def test_unknown_strategy_rejected(self):
        with pytest.raises(ValueError, match="strategy"):
            SamplingParams(strategy="beam")

    def test_temperature_clamped_not_zero(self):
        params = SamplingParams(temperature=0.0)
        assert params.temperature == MIN_TEMPERATURE
        params = SamplingParams(temperature=-5.0)
        assert params.temperature == MIN_TEMPERATURE

    def test_negative_top_k_rejected(self):
        with pytest.raises(ValueError, match="top_k"):
            SamplingParams(top_k=-1)

    def test_top_p_bounds(self):
        with pytest.raises(ValueError, match="top_p"):
            SamplingParams(top_p=0.0)
        with pytest.raises(ValueError, match="top_p"):
            SamplingParams(top_p=1.5)
        SamplingParams(top_p=1.0)  # boundary is valid


class TestFilters:
    def test_apply_temperature_scales(self):
        logits = torch.tensor([2.0, 4.0])
        assert torch.allclose(apply_temperature(logits, 2.0), torch.tensor([1.0, 2.0]))

    def test_apply_temperature_zero_does_not_produce_nan(self):
        logits = torch.tensor([2.0, 4.0, -1.0])
        probs = torch.softmax(apply_temperature(logits, 0.0), dim=-1)
        assert not torch.isnan(probs).any()
        assert probs.argmax().item() == 1

    def test_top_k_keeps_k_tokens(self):
        logits = torch.tensor([1.0, 5.0, 3.0, 4.0, 2.0])
        filtered = filter_top_k(logits, 2)
        kept = torch.isfinite(filtered)
        assert kept.sum().item() == 2
        assert kept[1] and kept[3]

    def test_top_k_disabled_or_oversized(self):
        logits = torch.tensor([1.0, 2.0, 3.0])
        assert torch.isfinite(filter_top_k(logits, 0)).all()
        # k > vocab must not crash (regression: unguarded torch.topk)
        assert torch.isfinite(filter_top_k(logits, 100)).all()

    def test_top_k_does_not_mutate_input(self):
        logits = torch.tensor([1.0, 5.0, 3.0])
        filter_top_k(logits, 1)
        assert torch.isfinite(logits).all()

    def test_top_p_keeps_nucleus(self):
        # probs ~ [0.643, 0.237, 0.087, 0.032] for logits [4,3,2,1]
        logits = torch.tensor([4.0, 3.0, 2.0, 1.0])
        filtered = filter_top_p(logits, 0.7)
        kept = torch.isfinite(filtered)
        # 0.643 alone < 0.7, so token 1 is also kept (first past threshold)
        assert kept[0] and kept[1]
        assert not kept[2] and not kept[3]

    def test_top_p_always_keeps_top_token(self):
        logits = torch.tensor([10.0, 1.0, 0.0])
        filtered = filter_top_p(logits, 0.01)
        assert torch.isfinite(filtered[0])
        assert torch.isfinite(filtered).sum().item() == 1

    def test_top_p_disabled(self):
        logits = torch.tensor([1.0, 2.0])
        assert torch.isfinite(filter_top_p(logits, 1.0)).all()


class TestSelectToken:
    def test_greedy_picks_argmax(self):
        logits = torch.tensor([1.0, 9.0, 3.0])
        assert select_token(logits, SamplingParams()) == 1

    def test_sampling_deterministic_with_seed(self):
        logits = torch.randn(100)
        params = SamplingParams(strategy="sampling", temperature=1.0, top_k=50)
        picks_a = [
            select_token(logits, params, torch.Generator().manual_seed(42))
            for _ in range(5)
        ]
        picks_b = [
            select_token(logits, params, torch.Generator().manual_seed(42))
            for _ in range(5)
        ]
        assert picks_a == picks_b

    def test_sampling_respects_top_k_1(self):
        logits = torch.tensor([1.0, 9.0, 3.0])
        params = SamplingParams(strategy="sampling", top_k=1)
        gen = torch.Generator().manual_seed(0)
        assert all(select_token(logits, params, gen) == 1 for _ in range(10))


class TestEntropy:
    def test_uniform(self):
        probs = torch.full((8,), 1 / 8)
        assert entropy(probs) == pytest.approx(math.log(8))

    def test_one_hot_is_zero(self):
        probs = torch.tensor([0.0, 1.0, 0.0])
        assert entropy(probs) == pytest.approx(0.0)
