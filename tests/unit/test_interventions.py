"""Interventions: steer/ablate/swap behavior, composition, cache safety."""

import copy

import pytest
import torch

from miru_tracer.core._jlens import fit, from_hf
from miru_tracer.core._jlens.hooks import ActivationRecorder
from miru_tracer.core.interventions import (
    Intervention,
    InterventionSet,
    apply_interventions,
    lens_vector,
    unembed_direction,
)
from miru_tracer.core.lens import compute_lens_slice
from miru_tracer.core.tracer import LLMTracer

FINAL = 1  # tiny model has 2 blocks: indices 0 (fitted) and 1 (final)
TOKEN_A, TOKEN_B = 65, 66  # arbitrary in-vocab byte tokens


@pytest.fixture(scope="module")
def tiny_lens(tiny_model, tiny_tokenizer):
    wrapper = from_hf(tiny_model, tiny_tokenizer, force_bos=False)
    return fit(
        wrapper,
        [
            "Hello world, this is a much longer test prompt for fitting the lens today.",
            "The quick brown fox jumps over the lazy dog again and again without stop.",
        ],
        dim_batch=8,
    )


def next_prob(tracer, token_id):
    return torch.softmax(tracer._next_raw_logits(), dim=-1)[token_id].item()


class TestInterventionValidation:
    def test_unknown_kind(self):
        with pytest.raises(ValueError, match="kind"):
            Intervention(kind="boost", layer=0, token_id=1)

    def test_swap_needs_target(self):
        with pytest.raises(ValueError, match="token_id_to"):
            Intervention(kind="swap", layer=0, token_id=1)

    def test_out_of_range_layer(self, tiny_model):
        with pytest.raises(ValueError, match="out of range"):
            InterventionSet(
                [Intervention(kind="steer", layer=99, token_id=1, basis="logit")],
                tiny_model,
            )

    def test_jacobian_basis_needs_lens(self, tiny_model):
        with pytest.raises(ValueError, match="fitted lens"):
            InterventionSet(
                [Intervention(kind="steer", layer=0, token_id=1, basis="jacobian")],
                tiny_model,
            )

    def test_describe(self, tiny_model, tiny_tokenizer):
        iv = Intervention(kind="steer", layer=0, token_id=TOKEN_A, strength=2.0, basis="logit")
        assert "@L0" in iv.describe(tiny_tokenizer)
        assert "+2" in iv.describe(tiny_tokenizer)


class TestLensVectors:
    def test_logit_basis_is_unembed_direction(self, tiny_model):
        v = lens_vector(tiny_model, TOKEN_A, 0, basis="logit")
        assert torch.allclose(v, unembed_direction(tiny_model, TOKEN_A))
        assert v.norm().item() == pytest.approx(1.0)

    def test_jacobian_basis_transports_to_token(self, tiny_model, tiny_lens):
        """J @ v must point toward the token's unembedding direction."""
        v = lens_vector(tiny_model, TOKEN_A, 0, basis="jacobian", jlens=tiny_lens, n_layers=2)
        transported = tiny_lens.jacobians[0].float() @ v
        cosine = torch.nn.functional.cosine_similarity(
            transported, unembed_direction(tiny_model, TOKEN_A), dim=0
        )
        assert cosine.item() > 0.9

    def test_final_layer_uses_unembed_direction(self, tiny_model, tiny_lens):
        v = lens_vector(tiny_model, TOKEN_A, FINAL, basis="jacobian", jlens=tiny_lens, n_layers=2)
        assert torch.allclose(v, unembed_direction(tiny_model, TOKEN_A))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_jacobian_basis_with_cpu_lens_and_cuda_model(self, tiny_model, tiny_lens):
        """JacobianLens.load puts jacobians on CPU; the model may be on CUDA."""
        model = copy.deepcopy(tiny_model).to("cuda")
        v = lens_vector(model, TOKEN_A, 0, basis="jacobian", jlens=tiny_lens, n_layers=2)
        assert v.device.type == "cuda"
        assert torch.isfinite(v).all()
        assert v.norm().item() == pytest.approx(1.0)


class TestSteer:
    def test_steer_raises_token_probability(self, tiny_model, tiny_tokenizer):
        tracer = LLMTracer(tiny_model, tiny_tokenizer, device="cpu")
        tracer.reset("Hello world")
        baseline = next_prob(tracer, TOKEN_A)

        tracer.set_interventions(
            [Intervention(kind="steer", layer=FINAL, token_id=TOKEN_A, strength=4.0, basis="logit")]
        )
        steered = next_prob(tracer, TOKEN_A)
        assert steered > baseline

        tracer.set_interventions(None)
        assert next_prob(tracer, TOKEN_A) == pytest.approx(baseline, rel=1e-4)

    def test_negative_steer_lowers_probability(self, tiny_model, tiny_tokenizer):
        tracer = LLMTracer(tiny_model, tiny_tokenizer, device="cpu")
        tracer.reset("Hello world")
        baseline = next_prob(tracer, TOKEN_A)
        tracer.set_interventions(
            [Intervention(kind="steer", layer=FINAL, token_id=TOKEN_A, strength=-4.0, basis="logit")]
        )
        assert next_prob(tracer, TOKEN_A) < baseline
        tracer.set_interventions(None)

    def test_steer_in_jacobian_basis_raises_readout(
        self, tiny_model, tiny_tokenizer, tiny_lens
    ):
        """Steering at the fitted early layer must raise the token's J-lens
        readout probability at that layer."""
        ids = tiny_tokenizer.encode("Hello world test", return_tensors="pt")

        def readout_prob(interventions):
            slice_ = compute_lens_slice(
                tiny_model, tiny_tokenizer, ids,
                layers=[0], mode="jacobian", jlens=tiny_lens,
                pinned_token_ids=[TOKEN_A], interventions=interventions,
            )
            return slice_.pinned_ranks[TOKEN_A][0][-1]  # rank at last position

        baseline_rank = readout_prob(None)
        iset = InterventionSet(
            [Intervention(kind="steer", layer=0, token_id=TOKEN_A, strength=4.0)],
            tiny_model,
            jlens=tiny_lens,
        )
        steered_rank = readout_prob(iset)
        assert steered_rank < baseline_rank  # rank 0 = top


class TestAblate:
    def test_ablate_zeroes_projection(self, tiny_model, tiny_tokenizer):
        v = lens_vector(tiny_model, TOKEN_A, 0, basis="logit")
        iset = InterventionSet(
            [Intervention(kind="ablate", layer=0, token_id=TOKEN_A, basis="logit")],
            tiny_model,
        )
        ids = tiny_tokenizer.encode("Hello world", return_tensors="pt")
        blocks = tiny_model.model.layers
        with (
            torch.no_grad(),
            apply_interventions(tiny_model, iset),
            ActivationRecorder(blocks, at=[0]) as recorder,
        ):
            tiny_model.model(input_ids=ids, use_cache=False)
            edited = recorder.activations[0].detach()
        projections = (edited[0].float() @ v).abs()
        assert projections.max().item() < 1e-4


class TestSwap:
    def test_swap_transfers_coefficient(self, tiny_model, tiny_tokenizer):
        v_a = lens_vector(tiny_model, TOKEN_A, 0, basis="logit")
        v_b = lens_vector(tiny_model, TOKEN_B, 0, basis="logit")
        iset = InterventionSet(
            [Intervention(kind="swap", layer=0, token_id=TOKEN_A, token_id_to=TOKEN_B, basis="logit")],
            tiny_model,
        )
        ids = tiny_tokenizer.encode("Hello world", return_tensors="pt")
        blocks = tiny_model.model.layers

        def record(interventions):
            with (
                torch.no_grad(),
                apply_interventions(tiny_model, interventions),
                ActivationRecorder(blocks, at=[0]) as recorder,
            ):
                tiny_model.model(input_ids=ids, use_cache=False)
                return recorder.activations[0].detach()[0].float()

        before = record(None)
        after = record(iset)
        coef_before = before @ v_a
        # a-component removed...
        assert (after @ v_a).abs().max().item() < (
            0.2 * coef_before.abs().max().item() + 1e-4
        )
        # ...and moved onto b
        expected_b = before @ v_b + coef_before * (v_b @ v_b) - coef_before * (v_a @ v_b)
        assert torch.allclose(after @ v_b, expected_b, atol=1e-3)


class TestComposition:
    def test_two_steers_both_take_effect(self, tiny_model, tiny_tokenizer):
        tracer = LLMTracer(tiny_model, tiny_tokenizer, device="cpu")
        tracer.reset("Hello world")
        base_a = next_prob(tracer, TOKEN_A)
        base_b = next_prob(tracer, TOKEN_B)

        tracer.set_interventions(
            [
                Intervention(kind="steer", layer=FINAL, token_id=TOKEN_A, strength=3.0, basis="logit"),
                Intervention(kind="steer", layer=FINAL, token_id=TOKEN_B, strength=3.0, basis="logit"),
            ]
        )
        assert next_prob(tracer, TOKEN_A) > base_a
        assert next_prob(tracer, TOKEN_B) > base_b
        tracer.set_interventions(None)

    def test_interventions_across_layers(self, tiny_model, tiny_tokenizer, tiny_lens):
        tracer = LLMTracer(tiny_model, tiny_tokenizer, device="cpu")
        tracer.reset("Hello world")
        base = next_prob(tracer, TOKEN_A)
        tracer.set_interventions(
            [
                Intervention(kind="steer", layer=0, token_id=TOKEN_A, strength=2.0, basis="jacobian"),
                Intervention(kind="steer", layer=FINAL, token_id=TOKEN_A, strength=2.0, basis="logit"),
            ],
            jlens=tiny_lens,
        )
        assert next_prob(tracer, TOKEN_A) > base
        tracer.set_interventions(None)


class TestCacheSafetyUnderInterventions:
    def test_set_interventions_invalidates_cache(self, tiny_model, tiny_tokenizer):
        tracer = LLMTracer(tiny_model, tiny_tokenizer, device="cpu")
        tracer.reset("Hello world")
        tracer.step()
        assert tracer._cache_len() > 0
        tracer.set_interventions(
            [Intervention(kind="steer", layer=FINAL, token_id=TOKEN_A, strength=1.0, basis="logit")]
        )
        assert tracer._cache_len() == 0
        assert tracer._logits_slot is None

    def test_incremental_matches_full_forward_under_interventions(
        self, tiny_model, tiny_tokenizer
    ):
        """The KV-cache correctness invariant must hold with edits active."""
        tracer = LLMTracer(tiny_model, tiny_tokenizer, device="cpu")
        tracer.reset("The quick brown fox")
        tracer.set_interventions(
            [
                Intervention(kind="steer", layer=0, token_id=TOKEN_A, strength=1.5, basis="logit"),
                Intervention(kind="ablate", layer=FINAL, token_id=TOKEN_B, basis="logit"),
            ]
        )
        for _ in range(4):
            tracer.step()
        incremental = tracer._next_raw_logits()

        iset = tracer._intervention_set
        with torch.inference_mode(), apply_interventions(tiny_model, iset):
            full = tiny_model(tracer.input_ids, use_cache=False).logits[0, -1].float()
        assert torch.allclose(incremental, full, atol=1e-5)
        tracer.set_interventions(None)

    def test_generation_deterministic_and_restorable(self, tiny_model, tiny_tokenizer):
        tracer = LLMTracer(tiny_model, tiny_tokenizer, device="cpu")
        tracer.reset("Hello world")
        baseline_tokens = [tracer.step().token_id for _ in range(3)]

        tracer.reset("Hello world")
        tracer.set_interventions(
            [Intervention(kind="steer", layer=FINAL, token_id=TOKEN_A, strength=8.0, basis="logit")]
        )
        steered_tokens = [tracer.step().token_id for _ in range(3)]
        assert TOKEN_A in steered_tokens  # strong steer dominates greedy decoding

        tracer.set_interventions(None)
        tracer.reset("Hello world")
        restored_tokens = [tracer.step().token_id for _ in range(3)]
        assert restored_tokens == baseline_tokens
