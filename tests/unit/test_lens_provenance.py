"""Fitted-lens provenance and runtime compatibility checks."""

from __future__ import annotations

import pytest
import torch

from miru_tracer.core._jlens import JacobianLens
from miru_tracer.core.interventions import Intervention
from miru_tracer.core.lens import compute_lens_slice
from miru_tracer.core.lens_provenance import (
    MODEL_ARCHITECTURE_HASH_KIND,
    MODEL_CONFIG_HASH_KIND,
    check_lens_compatibility,
    clear_provenance_caches,
    model_architecture_sha256,
    model_config_sha256,
    require_lens_compatible,
    resolve_hub_model_commit,
    tokenizer_sha256,
)
from miru_tracer.core.tracer import LLMTracer


@pytest.fixture(autouse=True)
def _fresh_fingerprint_caches():
    clear_provenance_caches()
    yield
    clear_provenance_caches()


@pytest.fixture()
def identified_runtime(tiny_model, tiny_tokenizer, monkeypatch):
    monkeypatch.setattr(tiny_model.config, "_name_or_path", "example/model")
    monkeypatch.setattr(tiny_model.config, "_commit_hash", "a" * 40, raising=False)
    monkeypatch.setattr(
        tiny_tokenizer,
        "name_or_path",
        "example/model",
        raising=False,
    )
    return tiny_model, tiny_tokenizer


def _lens(
    provenance: dict[str, object] | None,
    *,
    d_model: int = 32,
    source_layer: int = 0,
) -> JacobianLens:
    metadata = None if provenance is None else {"schema_version": 1, "provenance": provenance}
    return JacobianLens(
        jacobians={source_layer: torch.eye(d_model)},
        n_prompts=1,
        d_model=d_model,
        fit_metadata=metadata,
    )


def _runtime_provenance(model, tokenizer) -> dict[str, object]:
    return {
        "model_name_or_path": "example/model",
        "model_commit_hash": "a" * 40,
        "model_config_sha256": model_config_sha256(model),
        "model_config_sha256_kind": MODEL_CONFIG_HASH_KIND,
        "model_architecture_sha256": model_architecture_sha256(model),
        "model_architecture_sha256_kind": MODEL_ARCHITECTURE_HASH_KIND,
        "tokenizer_name_or_path": "example/model",
        "tokenizer_sha256": tokenizer_sha256(tokenizer),
    }


def test_matching_strong_provenance_is_accepted(identified_runtime):
    model, tokenizer = identified_runtime
    result = check_lens_compatibility(
        _lens(_runtime_provenance(model, tokenizer)),
        model,
        tokenizer,
        model_name_or_path="example/model",
    )
    assert result.compatible
    assert result.errors == ()
    assert result.warnings == ()
    assert {
        "d_model",
        "layer_count",
        "model_name_or_path",
        "model_commit_hash",
        "model_config_sha256",
        "model_architecture_sha256",
        "tokenizer_sha256",
    }.issubset(result.compared_fields)


def test_same_shape_from_another_hub_model_is_rejected(identified_runtime):
    model, tokenizer = identified_runtime
    provenance = _runtime_provenance(model, tokenizer)
    provenance["model_name_or_path"] = "other/model"
    result = check_lens_compatibility(
        _lens(provenance),
        model,
        tokenizer,
        model_name_or_path="example/model",
    )
    assert not result.compatible
    assert "different model" in " ".join(result.errors)


def test_different_resolved_revision_is_rejected(identified_runtime):
    model, tokenizer = identified_runtime
    provenance = _runtime_provenance(model, tokenizer)
    provenance["model_commit_hash"] = "b" * 40
    result = check_lens_compatibility(
        _lens(provenance),
        model,
        tokenizer,
        model_name_or_path="example/model",
    )
    assert not result.compatible
    assert "revision" in " ".join(result.errors)


def test_miru_retained_revision_is_used_when_transformers_omits_private_field(
    identified_runtime,
):
    model, tokenizer = identified_runtime
    provenance = _runtime_provenance(model, tokenizer)
    model.config._commit_hash = None
    model._miru_model_commit = "a" * 40

    result = check_lens_compatibility(
        _lens(provenance),
        model,
        tokenizer,
        model_name_or_path="example/model",
    )

    assert result.compatible
    assert "model_commit_hash" in result.compared_fields


def test_different_config_or_tokenizer_is_rejected(identified_runtime):
    model, tokenizer = identified_runtime

    wrong_config = _runtime_provenance(model, tokenizer)
    wrong_config["model_architecture_sha256"] = "0" * 64
    config_result = check_lens_compatibility(
        _lens(wrong_config),
        model,
        tokenizer,
        model_name_or_path="example/model",
    )
    assert "model-architecture fingerprint" in " ".join(config_result.errors)

    wrong_tokenizer = _runtime_provenance(model, tokenizer)
    wrong_tokenizer["tokenizer_sha256"] = "0" * 64
    tokenizer_result = check_lens_compatibility(
        _lens(wrong_tokenizer),
        model,
        tokenizer,
        model_name_or_path="example/model",
    )
    assert "tokenizer fingerprint" in " ".join(tokenizer_result.errors)


def test_legacy_lens_remains_usable_with_an_explicit_warning(identified_runtime):
    model, tokenizer = identified_runtime
    lens = _lens(None)
    result = require_lens_compatible(
        lens,
        model,
        tokenizer,
        model_name_or_path="example/model",
    )
    assert result.compatible
    assert "legacy/upstream" in " ".join(result.warnings)


def test_local_alias_mismatch_is_advisory_not_fatal(identified_runtime):
    model, tokenizer = identified_runtime
    result = check_lens_compatibility(
        _lens({"model_name_or_path": "renamed-local-checkpoint"}),
        model,
        tokenizer,
        model_name_or_path="/models/example-checkpoint",
    )
    assert result.compatible
    assert "aliases" in " ".join(result.warnings)


def test_normalized_architecture_hash_ignores_loader_identity(identified_runtime):
    model, _tokenizer = identified_runtime
    first = model_architecture_sha256(model)
    model.config._name_or_path = "/a/moved/local/checkpoint"
    model.config._commit_hash = "b" * 40
    clear_provenance_caches()
    assert model_architecture_sha256(model) == first


def test_hub_revision_is_resolved_from_commit_addressed_cache_path(
    tmp_path,
    monkeypatch,
):
    import huggingface_hub

    commit = "c" * 40
    config_path = tmp_path / "models--example--model" / "snapshots" / commit / "config.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("{}")
    recorded = {}

    def fake_download(**kwargs):
        recorded.update(kwargs)
        return str(config_path)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)

    assert (
        resolve_hub_model_commit(
            "example/model",
            revision="main",
            cache_dir=tmp_path / "hub",
        )
        == commit
    )
    assert recorded == {
        "repo_id": "example/model",
        "filename": "config.json",
        "revision": "main",
        "cache_dir": tmp_path / "hub",
    }


def test_full_hub_commit_and_local_model_need_no_resolution(tmp_path, monkeypatch):
    import huggingface_hub

    def unexpected_download(**_kwargs):
        raise AssertionError("Hub download should not be called")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", unexpected_download)
    commit = "d" * 40

    assert resolve_hub_model_commit("example/model", revision=commit) == commit
    assert resolve_hub_model_commit(str(tmp_path), revision=None) is None


def test_hub_resolution_rejects_a_non_snapshot_cache_path(tmp_path, monkeypatch):
    import huggingface_hub

    config_path = tmp_path / "config.json"
    config_path.write_text("{}")
    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_download",
        lambda **_kwargs: str(config_path),
    )

    with pytest.raises(RuntimeError, match="commit-addressed snapshot"):
        resolve_hub_model_commit("example/model")


def test_matching_architecture_allows_runtime_dtype_change_without_commit(
    identified_runtime,
    monkeypatch,
):
    model, tokenizer = identified_runtime
    monkeypatch.setattr(model.config, "dtype", torch.bfloat16)
    clear_provenance_caches()
    provenance = _runtime_provenance(model, tokenizer)
    provenance["model_name_or_path"] = "checkpoint"
    provenance.pop("model_commit_hash")
    model.config._commit_hash = None

    original_config_hash = provenance["model_config_sha256"]
    original_architecture_hash = provenance["model_architecture_sha256"]
    model.config.dtype = torch.float16
    clear_provenance_caches()

    assert model_config_sha256(model) != original_config_hash
    assert model_architecture_sha256(model) == original_architecture_hash

    result = check_lens_compatibility(
        _lens(provenance),
        model,
        tokenizer,
        model_name_or_path="/models/checkpoint",
    )

    assert result.compatible
    assert result.errors == ()
    assert "runtime/load-configuration difference" in " ".join(result.warnings)


def test_legacy_full_config_hash_allows_runtime_dtype_change_with_warning(
    identified_runtime,
    monkeypatch,
):
    """Pre-v0.3.0 artifacts recorded a load-sensitive hash and no semantic hash."""
    model, tokenizer = identified_runtime
    monkeypatch.setattr(model.config, "dtype", torch.bfloat16)
    clear_provenance_caches()
    provenance = _runtime_provenance(model, tokenizer)
    provenance.pop("model_commit_hash")
    provenance.pop("model_config_sha256_kind")
    provenance.pop("model_architecture_sha256")
    provenance.pop("model_architecture_sha256_kind")
    model.config._commit_hash = None
    model.config.dtype = torch.float16
    clear_provenance_caches()

    result = check_lens_compatibility(
        _lens(provenance),
        model,
        tokenizer,
        model_name_or_path="example/model",
    )

    assert result.compatible
    assert result.errors == ()
    warning = " ".join(result.warnings)
    assert "legacy artifact" in warning
    assert "runtime/load settings" in warning
    assert "remain unverified" in warning


def test_legacy_full_config_warning_does_not_mask_tokenizer_conflict(
    identified_runtime,
    monkeypatch,
):
    model, tokenizer = identified_runtime
    monkeypatch.setattr(model.config, "dtype", torch.bfloat16)
    clear_provenance_caches()
    provenance = _runtime_provenance(model, tokenizer)
    provenance.pop("model_commit_hash")
    provenance.pop("model_config_sha256_kind")
    provenance.pop("model_architecture_sha256")
    provenance.pop("model_architecture_sha256_kind")
    provenance["tokenizer_sha256"] = "0" * 64
    model.config._commit_hash = None
    model.config.dtype = torch.float16
    clear_provenance_caches()

    result = check_lens_compatibility(
        _lens(provenance),
        model,
        tokenizer,
        model_name_or_path="example/model",
    )

    assert not result.compatible
    assert "tokenizer fingerprint" in " ".join(result.errors)


def test_explicit_force_bypasses_provenance_conflict(identified_runtime):
    model, tokenizer = identified_runtime
    provenance = _runtime_provenance(model, tokenizer)
    provenance["model_commit_hash"] = "b" * 40
    lens = _lens(provenance)

    result = require_lens_compatible(
        lens,
        model,
        tokenizer,
        model_name_or_path="example/model",
        force_provenance=True,
    )

    assert not result.compatible
    assert result.structural_errors == ()
    assert "revision" in " ".join(result.provenance_errors)


def test_explicit_force_does_not_bypass_structural_conflict(identified_runtime):
    model, tokenizer = identified_runtime
    lens = _lens(_runtime_provenance(model, tokenizer), d_model=16)

    with pytest.raises(ValueError, match="d_model"):
        require_lens_compatible(
            lens,
            model,
            tokenizer,
            model_name_or_path="example/model",
            force_provenance=True,
        )


def test_jacobian_readout_and_intervention_reject_incompatible_lens(
    identified_runtime,
):
    model, tokenizer = identified_runtime
    provenance = _runtime_provenance(model, tokenizer)
    provenance["model_name_or_path"] = "other/model"
    wrong_lens = _lens(provenance)
    input_ids = tokenizer.encode("Hello", return_tensors="pt")

    with pytest.raises(ValueError, match="incompatible fitted lens"):
        compute_lens_slice(
            model,
            tokenizer,
            input_ids,
            layers=[0],
            mode="jacobian",
            jlens=wrong_lens,
        )

    tracer = LLMTracer(model, tokenizer, device="cpu")
    with pytest.raises(ValueError, match="incompatible fitted lens"):
        tracer.set_interventions(
            [Intervention("steer", layer=0, token_id=1, basis="jacobian")],
            jlens=wrong_lens,
        )

    forced_slice = compute_lens_slice(
        model,
        tokenizer,
        input_ids,
        layers=[0],
        mode="jacobian",
        jlens=wrong_lens,
        force_lens_provenance=True,
    )
    assert forced_slice.layers == [0]

    tracer.set_interventions(
        [Intervention("steer", layer=0, token_id=1, basis="jacobian")],
        jlens=wrong_lens,
        force_lens_provenance=True,
    )
    assert len(tracer.interventions) == 1
