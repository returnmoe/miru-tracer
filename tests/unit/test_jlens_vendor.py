"""Vendored jlens: fitting, persistence, and the final-layer invariant.

Fitting on the tiny model takes seconds; prompts must exceed the fitter's
SKIP_FIRST_N_POSITIONS=16 tokens to contribute positions.
"""

import pytest
import torch

from miru_tracer.core._jlens import JacobianLens, fit, from_hf

FIT_PROMPTS = [
    "Hello world, this is a much longer test prompt for fitting the lens properly today.",
    "The quick brown fox jumps over the lazy dog again and again without stopping at all.",
    "Numbers like 12345 and 67890 mixed with words make for varied byte sequences here.",
]


@pytest.fixture(scope="module")
def fitted(tiny_model, tiny_tokenizer):
    wrapper = from_hf(tiny_model, tiny_tokenizer, force_bos=False)
    lens = fit(wrapper, FIT_PROMPTS, dim_batch=8)
    return wrapper, lens


class TestFit:
    def test_fit_produces_layer_matrices(self, fitted):
        wrapper, lens = fitted
        assert lens.d_model == wrapper.d_model
        assert lens.n_prompts == len(FIT_PROMPTS)
        for layer in lens.source_layers:
            assert lens.jacobians[layer].shape == (wrapper.d_model, wrapper.d_model)
            assert torch.isfinite(lens.jacobians[layer]).all()

    def test_final_layer_readout_equals_model_logits(self, fitted, tiny_tokenizer):
        """The jlens invariant — also guards the pre/post-norm hook pitfall."""
        wrapper, lens = fitted
        lens_logits, model_logits, input_ids = lens.apply(
            wrapper,
            "Hello world test",
            layers=lens.source_layers,
            positions=[-1],
        )
        with torch.no_grad():
            real = wrapper._hf_model(input_ids).logits[0, -1].float()
        assert torch.allclose(model_logits[0], real, atol=1e-4)

    def test_logit_lens_baseline_path(self, fitted):
        wrapper, lens = fitted
        lens_logits, _, _ = lens.apply(
            wrapper, "Hello world", layers=[0], positions=[-1], use_jacobian=False
        )
        assert lens_logits[0].shape[-1] == wrapper._hf_model.config.vocab_size


class TestPersistence:
    def test_save_load_roundtrip(self, fitted, tmp_path):
        _, lens = fitted
        path = tmp_path / "lens.pt"
        lens.save(str(path))
        loaded = JacobianLens.load(str(path))
        assert loaded.source_layers == lens.source_layers
        assert loaded.n_prompts == lens.n_prompts
        for layer in lens.source_layers:
            # saved as fp16 — compare loosely
            assert torch.allclose(
                loaded.jacobians[layer], lens.jacobians[layer], atol=1e-2
            )

    def test_load_rejects_non_lens_file(self, tmp_path):
        path = tmp_path / "other.pt"
        torch.save({"something": 1}, str(path))
        with pytest.raises(ValueError, match="not a JacobianLens"):
            JacobianLens.load(str(path))

    def test_merge_weighted_mean(self, fitted, tiny_model, tiny_tokenizer):
        wrapper, lens = fitted
        other = fit(wrapper, FIT_PROMPTS[:1], dim_batch=8)
        merged = JacobianLens.merge([lens, other])
        assert merged.n_prompts == lens.n_prompts + other.n_prompts
        layer = lens.source_layers[0]
        expected = (
            lens.jacobians[layer] * lens.n_prompts
            + other.jacobians[layer] * other.n_prompts
        ) / merged.n_prompts
        assert torch.allclose(merged.jacobians[layer], expected, atol=1e-6)
