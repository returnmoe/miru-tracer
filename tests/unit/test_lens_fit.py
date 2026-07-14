"""Chunked lens fitting: progress, resume, cancellation, artifact validity."""

import os

import pytest
import torch

from miru_tracer.core._jlens import JacobianLens
from miru_tracer.core._jlens.fitting import _convergence_state, fit
from miru_tracer.core.lens_fit import (
    DEFAULT_MIN_PROMPTS,
    DEFAULT_NUM_PROMPTS,
    DEFAULT_STOP_AT_DELTA,
    DEFAULT_STOP_WINDOW,
    _chunk_text_records,
    _configure_hf_home,
    iter_fit_lens,
    prompts_from_file,
    wikitext_prompts,
)
from miru_tracer.core.lens_io import load_lens

PROMPTS = [
    "Hello world, this is a much longer test prompt for fitting the lens today ok.",
    "The quick brown fox jumps over the lazy dog again and again without stopping.",
    "Numbers like 12345 and 67890 mixed with words make for varied byte sequences.",
    "A fourth prompt exists so that chunked fitting has more than one full chunk.",
]


class TestIterFitLens:
    def test_progress_and_artifact(self, tiny_model, tiny_tokenizer, tmp_path):
        out = tmp_path / "lens.safetensors"
        updates = list(
            iter_fit_lens(
                tiny_model,
                tiny_tokenizer,
                PROMPTS,
                out_path=out,
                chunk_size=2,
                dim_batch=8,
            )
        )
        assert [u.prompts_done for u in updates] == [2, 4]
        assert out.exists()
        lens = load_lens(out)
        assert lens.n_prompts == 4

    def test_intermediate_artifact_is_valid(self, tiny_model, tiny_tokenizer, tmp_path):
        out = tmp_path / "lens.safetensors"
        first = next(
            iter(
                iter_fit_lens(
                    tiny_model,
                    tiny_tokenizer,
                    PROMPTS,
                    out_path=out,
                    chunk_size=2,
                    dim_batch=8,
                )
            )
        )
        assert first.prompts_done == 2
        # partial artifact on disk is loadable and averaged over 2 prompts
        assert load_lens(out).n_prompts == 2

    def test_should_stop_cancels_between_chunks(self, tiny_model, tiny_tokenizer, tmp_path):
        out = tmp_path / "lens.safetensors"
        updates = list(
            iter_fit_lens(
                tiny_model,
                tiny_tokenizer,
                PROMPTS,
                out_path=out,
                chunk_size=2,
                dim_batch=8,
                should_stop=lambda: True,
            )
        )
        assert len(updates) == 1  # stopped after the first chunk

    def test_resume_from_checkpoint(self, tiny_model, tiny_tokenizer, tmp_path):
        """A second run picks up where a cancelled one stopped."""
        out = tmp_path / "lens.safetensors"
        list(
            iter_fit_lens(
                tiny_model,
                tiny_tokenizer,
                PROMPTS,
                out_path=out,
                chunk_size=2,
                dim_batch=8,
                should_stop=lambda: True,  # stops after 2 prompts
            )
        )
        assert (tmp_path / "lens.checkpoint.pt").exists()
        updates = list(
            iter_fit_lens(
                tiny_model,
                tiny_tokenizer,
                PROMPTS,
                out_path=out,
                chunk_size=2,
                dim_batch=8,
            )
        )
        # Resume jumps directly to the next unfinished chunk.
        assert len(updates) == 1
        assert updates[0].prompts_processed_this_run == 2
        assert updates[-1].prompts_done == 4
        assert load_lens(out).n_prompts == 4

    def test_convergence_can_stop_inside_a_chunk(self, tiny_model, tiny_tokenizer, tmp_path):
        out = tmp_path / "lens.safetensors"
        updates = list(
            iter_fit_lens(
                tiny_model,
                tiny_tokenizer,
                PROMPTS,
                out_path=out,
                chunk_size=4,
                dim_batch=8,
                min_prompts=2,
                stop_window=1,
                stop_at_delta=1e9,
            )
        )

        assert len(updates) == 1
        assert updates[0].converged is True
        assert updates[0].prompts_done == 2
        lens = load_lens(out)
        assert lens.n_prompts == 2
        assert lens.fit_metadata["convergence"]["converged"] is True

    def test_resume_preserves_convergence_window(self, tiny_model, tiny_tokenizer, tmp_path):
        out = tmp_path / "lens.safetensors"
        list(
            iter_fit_lens(
                tiny_model,
                tiny_tokenizer,
                PROMPTS,
                out_path=out,
                chunk_size=2,
                dim_batch=8,
                stop_at_delta=0,
                should_stop=lambda: True,
            )
        )

        updates = list(
            iter_fit_lens(
                tiny_model,
                tiny_tokenizer,
                PROMPTS,
                out_path=out,
                chunk_size=4,
                dim_batch=8,
                min_prompts=3,
                stop_window=2,
                stop_at_delta=1e9,
            )
        )

        # Prompt 2's delta came from the first process; retaining it lets the
        # two-value window stop immediately after prompt 3.
        assert updates[-1].prompts_done == 3
        assert updates[-1].converged is True
        history = load_lens(out).fit_metadata["convergence"]["history"]
        assert [point["n_prompts"] for point in history] == [1, 2, 3]

    def test_resume_rejects_a_different_prompt_prefix(self, tiny_model, tiny_tokenizer, tmp_path):
        out = tmp_path / "lens.safetensors"
        list(
            iter_fit_lens(
                tiny_model,
                tiny_tokenizer,
                PROMPTS,
                out_path=out,
                chunk_size=2,
                dim_batch=8,
                stop_at_delta=0,
                should_stop=lambda: True,
            )
        )
        changed = ["A different first prompt with enough bytes to fit safely.", *PROMPTS[1:]]

        with pytest.raises(ValueError, match="different ordered prompt prefix"):
            list(
                iter_fit_lens(
                    tiny_model,
                    tiny_tokenizer,
                    changed,
                    out_path=out,
                    chunk_size=2,
                    dim_batch=8,
                    stop_at_delta=0,
                )
            )

    def test_resume_rejects_different_model_provenance(self, tiny_model, tiny_tokenizer, tmp_path):
        out = tmp_path / "lens.safetensors"
        list(
            iter_fit_lens(
                tiny_model,
                tiny_tokenizer,
                PROMPTS,
                out_path=out,
                chunk_size=2,
                dim_batch=8,
                stop_at_delta=0,
                fit_provenance={"model_name_or_path": "model-a"},
                should_stop=lambda: True,
            )
        )

        with pytest.raises(ValueError, match="model_name_or_path.*model-a.*model-b"):
            list(
                iter_fit_lens(
                    tiny_model,
                    tiny_tokenizer,
                    PROMPTS,
                    out_path=out,
                    chunk_size=2,
                    dim_batch=8,
                    stop_at_delta=0,
                    fit_provenance={"model_name_or_path": "model-b"},
                )
            )

    def test_resume_rejects_changed_tokenizer_fingerprint(
        self, tiny_model, tiny_tokenizer, tmp_path, monkeypatch
    ):
        out = tmp_path / "lens.safetensors"
        list(
            iter_fit_lens(
                tiny_model,
                tiny_tokenizer,
                PROMPTS,
                out_path=out,
                chunk_size=2,
                dim_batch=8,
                should_stop=lambda: True,
            )
        )
        monkeypatch.setattr(
            tiny_tokenizer,
            "chat_template",
            f"{tiny_tokenizer.chat_template}\nchanged",
        )

        with pytest.raises(ValueError, match="tokenizer_sha256"):
            list(
                iter_fit_lens(
                    tiny_model,
                    tiny_tokenizer,
                    PROMPTS,
                    out_path=out,
                    chunk_size=2,
                    dim_batch=8,
                )
            )

    def test_resume_rejects_malformed_convergence_history(
        self, tiny_model, tiny_tokenizer, tmp_path
    ):
        out = tmp_path / "lens.safetensors"
        list(
            iter_fit_lens(
                tiny_model,
                tiny_tokenizer,
                PROMPTS,
                out_path=out,
                chunk_size=2,
                dim_batch=8,
                should_stop=lambda: True,
            )
        )
        checkpoint = tmp_path / "lens.checkpoint.pt"
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state["fit_history"][-1]["n_prompts"] = 1
        torch.save(state, checkpoint)

        with pytest.raises(ValueError, match="invalid fit_history"):
            list(
                iter_fit_lens(
                    tiny_model,
                    tiny_tokenizer,
                    PROMPTS,
                    out_path=out,
                    chunk_size=2,
                    dim_batch=8,
                )
            )

    def test_resume_rejects_legacy_checkpoint_without_prompt_digest(
        self, tiny_model, tiny_tokenizer, tmp_path
    ):
        out = tmp_path / "lens.safetensors"
        list(
            iter_fit_lens(
                tiny_model,
                tiny_tokenizer,
                PROMPTS,
                out_path=out,
                chunk_size=2,
                dim_batch=8,
                should_stop=lambda: True,
            )
        )
        checkpoint = tmp_path / "lens.checkpoint.pt"
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        del state["prompt_prefix_sha256"]
        torch.save(state, checkpoint)

        with pytest.raises(ValueError, match="cannot be resumed safely.*--fresh"):
            list(
                iter_fit_lens(
                    tiny_model,
                    tiny_tokenizer,
                    PROMPTS,
                    out_path=out,
                    chunk_size=2,
                    dim_batch=8,
                )
            )


class TestConvergenceMetric:
    def test_uses_strict_rolling_threshold(self):
        history = [
            {"mean_relative_change": 0.3},
            {"mean_relative_change": 0.1},
        ]

        converged, rolling = _convergence_state(
            history,
            n_done=2,
            min_prompts=2,
            stop_window=2,
            stop_at_delta=0.2,
        )

        assert rolling == pytest.approx(0.2)
        assert converged is False  # equality is not enough

    def test_missing_latest_update_does_not_backfill_from_old_history(self):
        history = [{"mean_relative_change": 0.001} for _ in range(9)]
        history.extend(
            [
                {"mean_relative_change": None},
                {"mean_relative_change": 0.001},
            ]
        )

        converged, rolling = _convergence_state(
            history,
            n_done=11,
            min_prompts=1,
            stop_window=10,
            stop_at_delta=0.002,
        )

        assert rolling is None
        assert converged is False

    def test_matches_neuronpedia_layer_mean_formula(self, monkeypatch):
        import miru_tracer.core._jlens.fitting as fitting_module

        class DummyModel:
            n_layers = 3
            d_model = 1

        values = {
            "first": {0: 1.0, 1: 2.0},
            "second": {0: 3.0, 1: 10.0},
        }

        def fake_jacobian(_model, prompt, source_layers, **_kwargs):
            return (
                {layer: torch.tensor([[values[prompt][layer]]]) for layer in source_layers},
                128,
                111,
            )

        monkeypatch.setattr(fitting_module, "jacobian_for_prompt", fake_jacobian)
        lens = fit(DummyModel(), ["first", "second"], stop_at_delta=None)

        # Layer 0 moves 1 -> 2 (1/2); layer 1 moves 2 -> 6 (4/6).
        expected = (0.5 + 4 / 6) / 2
        history = lens.fit_metadata["convergence"]["history"]
        assert history[0]["mean_relative_change"] is None
        assert history[1]["mean_relative_change"] == pytest.approx(expected)

    def test_nonfinite_prompt_jacobian_is_skipped(self, monkeypatch):
        import miru_tracer.core._jlens.fitting as fitting_module

        class DummyModel:
            n_layers = 2
            d_model = 1

        values = {"first": 1.0, "bad": float("nan"), "second": 3.0}

        def fake_jacobian(_model, prompt, source_layers, **_kwargs):
            return ({source_layers[0]: torch.tensor([[values[prompt]]])}, 128, 111)

        monkeypatch.setattr(fitting_module, "jacobian_for_prompt", fake_jacobian)
        lens = fit(DummyModel(), ["first", "bad", "second"], stop_at_delta=None)

        assert lens.n_prompts == 2
        assert lens.fit_metadata["fit"]["processed_prompts"] == 3
        assert lens.fit_metadata["fit"]["skipped_prompts"] == 1
        assert [point["prompt_index"] for point in lens.fit_metadata["convergence"]["history"]] == [
            0,
            2,
        ]

    def test_rejects_zero_checkpoint_frequency(self):
        class DummyModel:
            n_layers = 2
            d_model = 1

        with pytest.raises(ValueError, match="checkpoint_every"):
            fit(DummyModel(), ["prompt"], checkpoint_every=0)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"min_prompts": True},
            {"min_prompts": 1.5},
            {"stop_window": False},
            {"stop_window": 2.5},
            {"stop_at_delta": True},
        ],
    )
    def test_rejects_invalid_library_convergence_parameters(self, kwargs):
        class DummyModel:
            n_layers = 2
            d_model = 1

        with pytest.raises(ValueError):
            fit(DummyModel(), ["prompt"], **kwargs)

    def test_embedded_history_is_bounded(self, monkeypatch):
        import miru_tracer.core._jlens.fitting as fitting_module

        class DummyModel:
            n_layers = 2
            d_model = 1

        def fake_jacobian(_model, _prompt, source_layers, **_kwargs):
            return ({source_layers[0]: torch.ones(1, 1)}, 128, 111)

        monkeypatch.setattr(fitting_module, "jacobian_for_prompt", fake_jacobian)
        lens = fit(DummyModel(), ["prompt"] * 1_001, stop_at_delta=None)
        convergence = lens.fit_metadata["convergence"]

        assert convergence["history_total_points"] == 1_001
        assert convergence["history_truncated"] is True
        assert len(convergence["history"]) == 1_000
        assert convergence["history"][0]["n_prompts"] == 2


def test_convergence_defaults_match_documentation():
    assert DEFAULT_NUM_PROMPTS == 1_000
    assert DEFAULT_MIN_PROMPTS == 100
    assert DEFAULT_STOP_WINDOW == 10
    assert DEFAULT_STOP_AT_DELTA == 0.002


class TestPromptSources:
    def test_prompts_from_file(self, tmp_path):
        f = tmp_path / "prompts.txt"
        f.write_text("first prompt\n\n  second prompt  \n")
        assert prompts_from_file(f) == ["first prompt", "second prompt"]

    def test_wikitext_rows_are_concatenated_and_rechunked(self):
        records = ["= heading =", "abc", "defgh", "", "ijklmnop"]

        prompts = _chunk_text_records(iter(records), 2, max_chars=10, min_chars=3)

        assert prompts == ["abc defgh", "ijklmnop"]

    def test_rechunking_honors_prompt_budget(self):
        records = ["abc", "defgh", "ijklmnop"]
        assert _chunk_text_records(iter(records), 1, max_chars=10) == ["abc defgh"]

    def test_wikitext_downloads_use_explicit_cache_dir(self, tmp_path, monkeypatch):
        import huggingface_hub
        import pandas as pd

        downloads = []

        def fake_download(repo_id, filename, **kwargs):
            downloads.append((repo_id, filename, kwargs))
            return tmp_path / filename.replace("/", "-")

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
        monkeypatch.setattr(
            pd,
            "read_parquet",
            lambda _path, columns: pd.DataFrame({"text": ["abcdefghij"]}),
        )
        cache_dir = tmp_path / "hf-home" / "hub"

        prompts = wikitext_prompts(2, max_chars=5, min_chars=1, cache_dir=cache_dir)

        assert prompts == ["abcd", "efghi"]
        assert len(downloads) == 1
        assert downloads[0][0] == "Salesforce/wikitext"
        assert downloads[0][2] == {"repo_type": "dataset", "cache_dir": cache_dir}

    def test_hf_home_configures_all_cache_families(self, tmp_path, monkeypatch):
        names = [
            "HF_HOME",
            "HF_HUB_CACHE",
            "HUGGINGFACE_HUB_CACHE",
            "TRANSFORMERS_CACHE",
            "HF_XET_CACHE",
            "HF_ASSETS_CACHE",
            "HF_MODULES_CACHE",
        ]
        for name in names:
            monkeypatch.setenv(name, "before-test")

        root, hub = _configure_hf_home(tmp_path / "hf-home")

        assert root == (tmp_path / "hf-home").resolve()
        assert hub == root / "hub"
        assert os.environ["HF_HOME"] == str(root)
        assert os.environ["HF_HUB_CACHE"] == str(hub)
        assert os.environ["HF_XET_CACHE"] == str(root / "xet")
        assert os.environ["HF_ASSETS_CACHE"] == str(root / "assets")
        assert os.environ["HF_MODULES_CACHE"] == str(root / "modules")


class TestCliMain:
    # lens.pt pins the explicit legacy-format escape hatch (--out foo.pt)
    @pytest.mark.parametrize(
        "out_name,loader",
        [
            ("lens.safetensors", load_lens),
            ("lens.pt", lambda p: JacobianLens.load(str(p))),
        ],
    )
    def test_device_map_and_prompts_file_flow(
        self, tiny_model, tiny_tokenizer, tmp_path, monkeypatch, out_name, loader
    ):
        """main() end-to-end with mocked HF loaders: --device-map must reach
        from_pretrained, and the fit must produce a loadable artifact."""
        import transformers

        recorded = {}

        class FakeModel:
            @staticmethod
            def from_pretrained(name, **kwargs):
                recorded.update(kwargs)
                return tiny_model

        class FakeTokenizer:
            @staticmethod
            def from_pretrained(name, **kwargs):
                return tiny_tokenizer

        monkeypatch.setattr(transformers, "AutoModelForCausalLM", FakeModel)
        monkeypatch.setattr(transformers, "AutoTokenizer", FakeTokenizer)

        prompts_file = tmp_path / "prompts.txt"
        prompts_file.write_text("\n".join(PROMPTS))
        out = tmp_path / out_name

        from miru_tracer.core.lens_fit import main

        code = main(
            [
                "tiny/test-model",
                "--prompts-file",
                str(prompts_file),
                "--out",
                str(out),
                "--device-map",
                "auto",
                "--dim-batch",
                "8",
                "--num-prompts",
                "3",
                "--min-prompts",
                "2",
                "--stop-window",
                "1",
                "--stop-at-delta",
                "0",
            ]
        )
        assert code == 0
        assert recorded.get("device_map") == "auto"
        lens = loader(out)
        assert lens.n_prompts == 3
        assert lens.fit_metadata["convergence"]["enabled"] is False

    def test_out_and_hf_home_are_independent(
        self, tiny_model, tiny_tokenizer, tmp_path, monkeypatch
    ):
        import transformers

        recorded = {}

        class FakeModel:
            @staticmethod
            def from_pretrained(_name, **kwargs):
                recorded["model"] = kwargs
                return tiny_model

        class FakeTokenizer:
            @staticmethod
            def from_pretrained(_name, **kwargs):
                recorded["tokenizer"] = kwargs
                return tiny_tokenizer

        monkeypatch.setattr(transformers, "AutoModelForCausalLM", FakeModel)
        monkeypatch.setattr(transformers, "AutoTokenizer", FakeTokenizer)
        for name in (
            "HF_HOME",
            "HF_HUB_CACHE",
            "HUGGINGFACE_HUB_CACHE",
            "TRANSFORMERS_CACHE",
            "HF_XET_CACHE",
            "HF_ASSETS_CACHE",
            "HF_MODULES_CACHE",
        ):
            monkeypatch.setenv(name, "before-test")

        prompts_file = tmp_path / "prompts.txt"
        prompts_file.write_text("\n".join(PROMPTS))
        output_dir = tmp_path / "network-output"
        out = output_dir / "lens.safetensors"
        hf_home = tmp_path / "local-hf"

        from miru_tracer.core.lens_fit import main

        code = main(
            [
                "tiny/test-model",
                "--prompts-file",
                str(prompts_file),
                "--out",
                str(out),
                "--hf-home",
                str(hf_home),
                "--device-map",
                "auto",
                "--dim-batch",
                "8",
                "--num-prompts",
                "2",
                "--min-prompts",
                "2",
                "--stop-window",
                "1",
                "--stop-at-delta",
                "0",
            ]
        )

        expected_hub = hf_home.resolve() / "hub"
        assert code == 0
        assert recorded["tokenizer"]["cache_dir"] == expected_hub
        assert recorded["model"]["cache_dir"] == expected_hub
        assert set(output_dir.iterdir()) == {
            out,
            output_dir / "lens.checkpoint.pt",
        }
        assert hf_home.is_dir()
