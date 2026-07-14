"""Cross-architecture gate: tracer, lens, and interventions on tiny models
of every architecture family Miru claims to support.

- llama: the Llama/Qwen3 family (GQA, standard cache) — also the default
  fixture used by the rest of the suite.
- gemma4: Gemma 4 (hybrid attention, logit softcapping, per-layer input
  embeddings, unified K/V) — tiny version of google/gemma-4-31B's text stack.
- glm_dsa: GLM MoE-DSA (MoE MLPs, MLA-style sparse attention with indexer)
  — tiny version of zai-org/GLM-5.2's architecture.

Everything here runs offline on randomly initialized models; it proves the
code paths, not model quality.
"""

import pytest
import torch

from miru_tracer.core._jlens import fit, from_hf
from miru_tracer.core.interventions import Intervention
from miru_tracer.core.lens import compute_lens_slice
from miru_tracer.core.tracer import LLMTracer

ARCHS = ["llama", "gemma4", "glm_dsa"]

# MLA/DSA-style attention (glm_dsa) uses different compute paths for prefill
# vs incremental decode, so cached logits differ from a fresh full forward by
# ~1e-3 (measured: plateaus, does not grow; top-5 ranks stay identical).
# Standard attention architectures reproduce to float32 noise.
CACHE_EQUIVALENCE_ATOL = {"llama": 1e-4, "gemma4": 1e-4, "glm_dsa": 5e-3}

FIT_PROMPTS = [
    "Hello world, this is a much longer test prompt for fitting the lens today.",
    "The quick brown fox jumps over the lazy dog again and again without stop.",
]


@pytest.fixture(scope="module")
def models(tiny_model, tiny_gemma4, tiny_glm_dsa):
    return {"llama": tiny_model, "gemma4": tiny_gemma4, "glm_dsa": tiny_glm_dsa}


@pytest.fixture(params=ARCHS)
def arch_model(request, models):
    return request.param, models[request.param]


def count_forwards(monkeypatch, model):
    calls = {"n": 0}
    original = model.forward

    def counting(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "forward", counting)
    return calls


class TestTracerAcrossArchitectures:
    def test_step_peek_undo_and_cache_equivalence(
        self, arch_model, tiny_tokenizer, monkeypatch
    ):
        arch, model = arch_model
        tracer = LLMTracer(model, tiny_tokenizer, device="cpu")
        tracer.reset("Hello world test")
        for _ in range(3):
            tracer.step()
            ok, error = tracer.validate_state()
            assert ok, error

        # peek purity: repeated peeks with different params = one forward
        calls = count_forwards(monkeypatch, model)
        tracer.peek(top_k=5, temperature=0.5)
        tracer.peek(top_k=12, temperature=2.0)
        assert calls["n"] <= 1

        # undo then replay must reproduce the same tokens
        replay_expected = [step.token_id for step in tracer.history[-2:]]
        assert tracer.undo(2)
        assert [tracer.step().token_id for _ in range(2)] == replay_expected

        # incremental (cached) logits vs fresh full no-cache forward:
        # numerically close (per-arch tolerance) AND rank-identical
        incremental = tracer._next_raw_logits()
        with torch.inference_mode():
            full = model(tracer.input_ids, use_cache=False).logits[0, -1].float()
        assert torch.allclose(incremental, full, atol=CACHE_EQUIVALENCE_ATOL[arch])
        assert (
            torch.topk(incremental, 5).indices.tolist()
            == torch.topk(full, 5).indices.tolist()
        )


class TestLensAcrossArchitectures:
    def test_layout_autodetection_and_fit(self, arch_model, tiny_tokenizer):
        _arch, model = arch_model
        wrapper = from_hf(model, tiny_tokenizer, force_bos=False)
        assert wrapper.n_layers == len(wrapper.layers)
        lens = fit(wrapper, FIT_PROMPTS, dim_batch=8)
        assert lens.source_layers  # at least one layer fitted
        d_model = model.config.get_text_config().hidden_size
        assert lens.d_model == d_model

    def test_final_layer_lens_matches_model_logits(self, arch_model, tiny_tokenizer):
        """Also validates the softcapping path for gemma4 end-to-end."""
        _arch, model = arch_model
        ids = tiny_tokenizer.encode("Hello world test", return_tensors="pt")
        final = model.config.get_text_config().num_hidden_layers - 1
        slice_ = compute_lens_slice(
            model, tiny_tokenizer, ids, layers=[final], mode="logit", top_k=5
        )
        with torch.no_grad():
            real = model(ids).logits[0, -1].float()
        expected = torch.topk(torch.softmax(real, -1), 5)
        assert slice_.tokens[0][-1] == expected.indices.tolist()
        assert slice_.probs[0][-1] == pytest.approx(
            expected.values.tolist(), rel=1e-3
        )


class TestInterventionsAcrossArchitectures:
    def test_steer_raises_probability_and_cache_invalidates(
        self, arch_model, tiny_tokenizer
    ):
        _arch, model = arch_model
        token = 65
        final = model.config.get_text_config().num_hidden_layers - 1
        tracer = LLMTracer(model, tiny_tokenizer, device="cpu")
        tracer.reset("Hello world")
        baseline = torch.softmax(tracer._next_raw_logits(), -1)[token].item()
        assert tracer._cache_len() > 0

        tracer.set_interventions(
            [Intervention(kind="steer", layer=final, token_id=token, strength=4.0, basis="logit")]
        )
        assert tracer._cache_len() == 0  # mandatory invalidation
        steered = torch.softmax(tracer._next_raw_logits(), -1)[token].item()
        assert steered > baseline

        tracer.set_interventions(None)
        restored = torch.softmax(tracer._next_raw_logits(), -1)[token].item()
        assert restored == pytest.approx(baseline, rel=1e-3)
