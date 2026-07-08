"""Lens artifact I/O: safetensors default, legacy .pt fallback, format checks."""

import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from miru_tracer.core._jlens import JacobianLens, fit, from_hf
from miru_tracer.core.lens_io import (
    FORMAT_MARKER,
    FORMAT_VERSION,
    convert_main,
    load_lens,
    save_lens,
)


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


def assert_same_lens(loaded: JacobianLens, original: JacobianLens) -> None:
    assert loaded.source_layers == original.source_layers
    assert loaded.n_prompts == original.n_prompts
    assert loaded.d_model == original.d_model
    for layer in original.source_layers:
        # fp16 on disk in both formats
        assert torch.allclose(loaded.jacobians[layer], original.jacobians[layer], atol=1e-2)


class TestSafetensorsRoundtrip:
    def test_roundtrip(self, tiny_lens, tmp_path):
        path = tmp_path / "lens.safetensors"
        save_lens(tiny_lens, path)
        assert_same_lens(load_lens(path), tiny_lens)

    def test_file_is_genuinely_safetensors(self, tiny_lens, tmp_path):
        path = tmp_path / "lens.safetensors"
        save_lens(tiny_lens, path)
        # torch.save writes a zip ("PK"); safetensors starts with a little-endian
        # header length, so this doubles as a not-a-pickle check
        with safe_open(str(path), framework="pt", device="cpu") as f:
            assert set(f.keys()) == {f"J.{layer}" for layer in tiny_lens.source_layers}

    def test_metadata_schema(self, tiny_lens, tmp_path):
        path = tmp_path / "lens.safetensors"
        save_lens(tiny_lens, path)
        with safe_open(str(path), framework="pt", device="cpu") as f:
            metadata = f.metadata()
        assert metadata["format"] == FORMAT_MARKER
        assert metadata["version"] == FORMAT_VERSION
        assert metadata["n_prompts"] == str(tiny_lens.n_prompts)
        assert metadata["d_model"] == str(tiny_lens.d_model)
        assert json.loads(metadata["source_layers"]) == tiny_lens.source_layers


class TestLegacyPt:
    def test_pt_extension_writes_legacy_format(self, tiny_lens, tmp_path):
        path = tmp_path / "lens.pt"
        save_lens(tiny_lens, path)
        # readable by the vendored codec alone
        assert_same_lens(JacobianLens.load(str(path)), tiny_lens)

    def test_load_lens_reads_legacy_pt(self, tiny_lens, tmp_path):
        path = tmp_path / "lens.pt"
        tiny_lens.save(str(path))
        assert_same_lens(load_lens(path), tiny_lens)


class TestConvertCli:
    def test_pt_to_safetensors_default_dst(self, tiny_lens, tmp_path):
        src = tmp_path / "lens.pt"
        tiny_lens.save(str(src))
        assert convert_main([str(src)]) == 0
        assert_same_lens(load_lens(tmp_path / "lens.safetensors"), tiny_lens)

    def test_safetensors_to_pt(self, tiny_lens, tmp_path):
        src = tmp_path / "lens.safetensors"
        save_lens(tiny_lens, src)
        dst = tmp_path / "legacy.pt"
        assert convert_main([str(src), str(dst)]) == 0
        assert_same_lens(JacobianLens.load(str(dst)), tiny_lens)

    def test_same_file_is_an_error(self, tiny_lens, tmp_path):
        src = tmp_path / "lens.safetensors"
        save_lens(tiny_lens, src)
        with pytest.raises(SystemExit):
            convert_main([str(src), str(src)])


class TestRejectsNonLensFiles:
    def test_foreign_safetensors(self, tmp_path):
        path = tmp_path / "weights.safetensors"
        save_file({"weight": torch.zeros(2, 2)}, str(path))
        with pytest.raises(ValueError, match="not a JacobianLens"):
            load_lens(path)

    def test_non_lens_pt(self, tmp_path):
        path = tmp_path / "other.pt"
        torch.save({"something": 1}, str(path))
        with pytest.raises(ValueError, match="not a JacobianLens"):
            load_lens(path)
