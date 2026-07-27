"""ModelManager loading behavior (from_pretrained mocked — no network)."""

import warnings

import pytest

import miru_tracer.core.model_manager as mm


@pytest.fixture()
def manager(monkeypatch, tiny_model, tiny_tokenizer):
    """A ModelManager whose HF loaders return the tiny fixtures."""
    recorded = {}
    resolved_commit = "a" * 40

    def resolve_commit(name, *, revision):
        recorded["resolve"] = (name, revision)
        return resolved_commit

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
    monkeypatch.setattr(mm, "resolve_hub_model_commit", resolve_commit)
    monkeypatch.setattr(tiny_model.config, "_commit_hash", None, raising=False)
    monkeypatch.delattr(tiny_model, "_miru_model_commit", raising=False)
    # Isolate the singleton's class-level state between tests
    monkeypatch.setattr(mm.ModelManager, "_instance", None)
    monkeypatch.setattr(mm.ModelManager, "_model", None)
    monkeypatch.setattr(mm.ModelManager, "_tokenizer", None)
    monkeypatch.setattr(mm.ModelManager, "_model_revision", None)
    monkeypatch.setattr(mm.ModelManager, "_model_commit", None)
    monkeypatch.setattr(mm.ModelManager, "_is_loading", False)
    monkeypatch.setattr(mm.ModelManager, "_generation", 0)

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

    def test_revision_is_shared_by_model_and_tokenizer(
        self,
        manager,
        tiny_model,
        monkeypatch,
        caplog,
    ):
        instance, recorded = manager
        monkeypatch.setattr(tiny_model.config, "_commit_hash", "a" * 40)
        caplog.set_level("INFO", logger=mm.__name__)

        *_, info = instance.load_model("fake/model", revision="  immutable-revision  ")

        assert recorded["resolve"] == ("fake/model", "immutable-revision")
        assert recorded["model_kwargs"]["revision"] == "a" * 40
        assert recorded["tokenizer_kwargs"]["revision"] == "a" * 40
        assert instance.get_model_revision() == "immutable-revision"
        assert instance.get_model_commit() == "a" * 40
        assert info["requested_revision"] == "immutable-revision"
        assert info["resolved_revision"] == "a" * 40
        assert (
            "Model provenance: component=miru-tracer model=fake/model "
            f"requested_revision=immutable-revision commit_sha={'a' * 40}" in caplog.messages
        )

    def test_default_revision_is_resolved_once_for_model_and_tokenizer(self, manager):
        instance, recorded = manager

        instance.load_model("fake/model", revision="  ")

        assert recorded["resolve"] == ("fake/model", None)
        assert recorded["model_kwargs"]["revision"] == "a" * 40
        assert recorded["tokenizer_kwargs"]["revision"] == "a" * 40
        assert instance.get_model_revision() is None
        assert instance.get_model_commit() == "a" * 40

    def test_contradictory_loaded_revision_is_rejected(
        self,
        manager,
        tiny_model,
        monkeypatch,
    ):
        instance, _recorded = manager
        monkeypatch.setattr(tiny_model.config, "_commit_hash", "b" * 40)

        with pytest.raises(RuntimeError, match="Hub resolved"):
            instance.load_model("fake/model", revision="main")

        assert instance.is_loaded() is False
        assert instance.get_model_commit() is None

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
        assert instance.get_model_revision() is None
        assert instance.get_model_commit() is None
        assert instance.unload_model()["status"] == "warning"

    def test_failed_reload_leaves_atomic_unloaded_state(self, manager, monkeypatch):
        instance, _ = manager
        instance.load_model("fake/model")
        generation = instance.get_generation()

        class BrokenTokenizer:
            @staticmethod
            def from_pretrained(name, **kwargs):
                raise OSError("broken checkpoint")

        monkeypatch.setattr(mm, "AutoTokenizer", BrokenTokenizer)
        with pytest.raises(OSError, match="broken checkpoint"):
            instance.load_model("broken/model")

        assert instance.snapshot() is None
        assert instance.get_model() is None
        assert instance.get_tokenizer() is None
        assert instance.get_generation() > generation


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
