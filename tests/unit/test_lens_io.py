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
        # Both codecs store fp16 by default, and loading promotes back to fp32.
        expected = original.jacobians[layer].to(torch.float16).float()
        assert torch.equal(loaded.jacobians[layer], expected)


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

    def test_pt_to_safetensors_preserves_fp16_values_exactly(self, tiny_lens, tmp_path):
        src = tmp_path / "lens.pt"
        dst = tmp_path / "lens.safetensors"
        tiny_lens.save(str(src))
        pt_lens = JacobianLens.load(str(src))

        assert convert_main([str(src), str(dst)]) == 0

        converted = load_lens(dst)
        for layer in tiny_lens.source_layers:
            assert torch.equal(converted.jacobians[layer], pt_lens.jacobians[layer])

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


def _valid_metadata(**overrides: str) -> dict[str, str]:
    metadata = {
        "format": FORMAT_MARKER,
        "version": FORMAT_VERSION,
        "n_prompts": "2",
        "d_model": "2",
        "source_layers": "[0]",
    }
    metadata.update(overrides)
    return metadata


class TestSafetensorsValidation:
    def test_rejects_unsupported_version(self, tmp_path):
        path = tmp_path / "future.safetensors"
        save_file(
            {"J.0": torch.eye(2)},
            str(path),
            metadata=_valid_metadata(version="2"),
        )
        with pytest.raises(ValueError, match="unsupported format version '2'"):
            load_lens(path)

    @pytest.mark.parametrize("field", ["version", "n_prompts", "d_model", "source_layers"])
    def test_rejects_missing_required_metadata(self, tmp_path, field):
        path = tmp_path / f"missing-{field}.safetensors"
        metadata = _valid_metadata()
        del metadata[field]
        save_file({"J.0": torch.eye(2)}, str(path), metadata=metadata)
        with pytest.raises(ValueError, match=rf"missing required '{field}' metadata"):
            load_lens(path)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("n_prompts", "many"),
            ("n_prompts", "0"),
            ("d_model", "2.5"),
            ("d_model", "-2"),
        ],
    )
    def test_rejects_invalid_positive_integer_metadata(self, tmp_path, field, value):
        path = tmp_path / f"invalid-{field}.safetensors"
        save_file(
            {"J.0": torch.eye(2)},
            str(path),
            metadata=_valid_metadata(**{field: value}),
        )
        with pytest.raises(ValueError, match=rf"'{field}'.*positive integer"):
            load_lens(path)

    @pytest.mark.parametrize(
        ("source_layers", "message"),
        [
            ("not-json", "not valid JSON"),
            ('{"layer": 0}', "must be a JSON list"),
            ("[true]", "only nonnegative integers"),
            ("[-1]", "only nonnegative integers"),
            ("[0, 0]", "sorted and contain no duplicates"),
            ("[1, 0]", "sorted and contain no duplicates"),
        ],
    )
    def test_rejects_invalid_source_layers_metadata(self, tmp_path, source_layers, message):
        path = tmp_path / "invalid-source-layers.safetensors"
        save_file(
            {"J.0": torch.eye(2)},
            str(path),
            metadata=_valid_metadata(source_layers=source_layers),
        )
        with pytest.raises(ValueError, match=message):
            load_lens(path)

    @pytest.mark.parametrize("key", ["weight", "J.foo", "J.-1", "J.01"])
    def test_rejects_invalid_tensor_keys(self, tmp_path, key):
        path = tmp_path / "invalid-key.safetensors"
        save_file(
            {key: torch.eye(2)},
            str(path),
            metadata=_valid_metadata(),
        )
        with pytest.raises(ValueError, match="tensor keys.*'J.<layer>'"):
            load_lens(path)

    def test_rejects_empty_tensor_mapping(self, tmp_path):
        path = tmp_path / "empty.safetensors"
        save_file({}, str(path), metadata=_valid_metadata(source_layers="[]"))
        with pytest.raises(ValueError, match="contains no Jacobian matrices"):
            load_lens(path)

    def test_rejects_source_layer_key_mismatch(self, tmp_path):
        path = tmp_path / "mismatched-layers.safetensors"
        save_file(
            {"J.1": torch.eye(2)},
            str(path),
            metadata=_valid_metadata(source_layers="[0]"),
        )
        with pytest.raises(ValueError, match="does not match tensor layers"):
            load_lens(path)

    def test_rejects_non_floating_jacobian(self, tmp_path):
        path = tmp_path / "integer.safetensors"
        save_file(
            {"J.0": torch.eye(2, dtype=torch.int64)},
            str(path),
            metadata=_valid_metadata(),
        )
        with pytest.raises(ValueError, match="must have a floating dtype"):
            load_lens(path)

    @pytest.mark.parametrize("shape", [(2,), (2, 3), (2, 2, 1)])
    def test_rejects_wrong_jacobian_shape(self, tmp_path, shape):
        path = tmp_path / "wrong-shape.safetensors"
        save_file(
            {"J.0": torch.zeros(shape)},
            str(path),
            metadata=_valid_metadata(),
        )
        with pytest.raises(ValueError, match=r"must have shape \(2, 2\)"):
            load_lens(path)
