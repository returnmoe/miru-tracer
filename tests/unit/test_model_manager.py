"""ModelManager loading behavior (from_pretrained mocked — no network)."""

import warnings

import pytest

import miru_tracer.core.model_manager as mm


@pytest.fixture()
def manager(monkeypatch, tiny_model, tiny_tokenizer):
    """A ModelManager whose HF loaders return the tiny fixtures."""
    recorded = {}

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(name, **kwargs):
            recorded["model_kwargs"] = kwargs
            return tiny_model

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(name, **kwargs):
            recorded["tokenizer_kwargs"] = kwargs
            return tiny_tokenizer

    monkeypatch.setattr(mm, "AutoModelForCausalLM", FakeAutoModel)
    monkeypatch.setattr(mm, "AutoTokenizer", FakeAutoTokenizer)
    # Isolate the singleton's class-level state between tests
    monkeypatch.setattr(mm.ModelManager, "_instance", None)
    monkeypatch.setattr(mm.ModelManager, "_model", None)
    monkeypatch.setattr(mm.ModelManager, "_tokenizer", None)
    monkeypatch.setattr(mm.ModelManager, "_is_loading", False)

    return mm.ModelManager(), recorded


class TestLoadModel:
    def test_cpu_load_uses_dtype_kwarg(self, manager):
        """transformers 5 renamed torch_dtype to dtype; make sure we send it."""
        instance, recorded = manager
        model, tokenizer, device, info = instance.load_model("fake/model")
        assert device == "cpu"
        assert "dtype" in recorded["model_kwargs"]
        assert "torch_dtype" not in recorded["model_kwargs"]
        assert info["quantization"] == "none"

    def test_quantization_on_cpu_reports_note(self, manager):
        """Regression: 4bit/8bit on CPU used to be silently ignored."""
        instance, recorded = manager
        *_, info = instance.load_model("fake/model", quantization="4bit")
        assert info["quantization"] == "none"
        assert info["quantization_note"] is not None
        assert "CUDA" in info["quantization_note"]
        assert "quantization_config" not in recorded["model_kwargs"]

    def test_is_loaded_reflects_state(self, manager):
        instance, _ = manager
        assert instance.is_loaded() is False
        instance.load_model("fake/model")
        assert instance.is_loaded() is True
        assert instance.get_model_name() == "fake/model"

    def test_unload_clears_state(self, manager):
        instance, _ = manager
        instance.load_model("fake/model")
        result = instance.unload_model()
        assert result["status"] == "success"
        assert instance.is_loaded() is False
        assert instance.unload_model()["status"] == "warning"


class TestModuleHygiene:
    def test_import_does_not_suppress_warnings(self):
        """Regression: models.py used to call warnings.filterwarnings('ignore')
        at import time, silencing every warning in the process."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            import importlib

            importlib.reload(mm)
            warnings.warn("canary", UserWarning, stacklevel=1)
        assert any("canary" in str(w.message) for w in caught)
