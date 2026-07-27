"""Chunked lens fitting: progress, resume, cancellation, artifact validity."""

import logging
import os
import threading
import weakref

import pytest
import torch

from miru_tracer.core._jlens import JacobianLens
from miru_tracer.core._jlens.fitting import (
    NoFittedPromptsError,
    PromptTooShortError,
    _convergence_state,
    fit,
    valid_position_mask,
)
from miru_tracer.core.fit_diagnostics import (
    PromptTelemetrySampler,
    prompt_slowdown,
)
from miru_tracer.core.lens_fit import (
    DEFAULT_MIN_PROMPTS,
    DEFAULT_NUM_PROMPTS,
    DEFAULT_STOP_AT_DELTA,
    DEFAULT_STOP_WINDOW,
    WIKITEXT_REVISION,
    _chunk_text_records,
    _configure_hf_home,
    _linear_attention_backend_status,
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

    def test_diagnostic_baseline_spans_real_chunk_calls(
        self, tiny_model, tiny_tokenizer, tmp_path, monkeypatch
    ):
        import miru_tracer.core.lens_fit as lens_fit_module

        original_fit = lens_fit_module.fit
        duration_history_ids = []

        def recording_fit(*args, diagnostic_prompt_durations, **kwargs):
            duration_history_ids.append(id(diagnostic_prompt_durations))
            return original_fit(
                *args,
                diagnostic_prompt_durations=diagnostic_prompt_durations,
                **kwargs,
            )

        monkeypatch.setattr(lens_fit_module, "fit", recording_fit)
        list(
            iter_fit_lens(
                tiny_model,
                tiny_tokenizer,
                PROMPTS,
                out_path=tmp_path / "lens.safetensors",
                chunk_size=2,
                dim_batch=8,
                telemetry_interval_s=None,
            )
        )

        assert len(duration_history_ids) == 2
        assert len(set(duration_history_ids)) == 1

    def test_all_short_first_chunk_continues_to_later_valid_prompt(
        self, tiny_model, tiny_tokenizer, tmp_path
    ):
        out = tmp_path / "lens.safetensors"
        updates = list(
            iter_fit_lens(
                tiny_model,
                tiny_tokenizer,
                ["x", "y", PROMPTS[0]],
                out_path=out,
                chunk_size=2,
                dim_batch=8,
            )
        )

        assert len(updates) == 1
        assert updates[0].prompts_processed == 3
        assert updates[0].prompts_done == 1
        assert load_lens(out).n_prompts == 1

    def test_all_short_corpus_keeps_clear_error(self, tiny_model, tiny_tokenizer, tmp_path):
        with pytest.raises(NoFittedPromptsError, match="no prompts were long enough"):
            list(
                iter_fit_lens(
                    tiny_model,
                    tiny_tokenizer,
                    ["x", "y"],
                    out_path=tmp_path / "lens.safetensors",
                    chunk_size=1,
                    dim_batch=8,
                )
            )

    def test_writes_one_checkpoint_per_chunk(
        self, tiny_model, tiny_tokenizer, tmp_path, monkeypatch
    ):
        import miru_tracer.core._jlens.fitting as fitting_module

        writes = []
        original_atomic_save = fitting_module._atomic_save

        def counted_atomic_save(obj, path):
            writes.append(path)
            original_atomic_save(obj, path)

        monkeypatch.setattr(fitting_module, "_atomic_save", counted_atomic_save)
        list(
            iter_fit_lens(
                tiny_model,
                tiny_tokenizer,
                PROMPTS,
                out_path=tmp_path / "lens.safetensors",
                chunk_size=2,
                dim_batch=8,
            )
        )

        assert len(writes) == 2

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

    def test_logs_stale_atomic_temp_and_remaining_disk(
        self, tiny_model, tiny_tokenizer, tmp_path, caplog
    ):
        out = tmp_path / "lens.safetensors"
        stale = tmp_path / "lens.safetensors.tmp.123"
        stale.write_bytes(b"partial")
        caplog.set_level(logging.INFO)

        list(
            iter_fit_lens(
                tiny_model,
                tiny_tokenizer,
                PROMPTS[:1],
                out_path=out,
                chunk_size=1,
                dim_batch=8,
            )
        )

        messages = [record.message for record in caplog.records]
        storage = next(message for message in messages if message.startswith("fit_storage_plan"))
        chunk = next(
            message for message in messages if message.startswith("fit_io event=chunk_saved")
        )
        assert "stale_temp_files=1" in storage
        assert "stale_temp_bytes=7" in storage
        assert "output_disk_free_bytes=" in chunk
        assert any("kind=stale_atomic_temp_files files=1 bytes=7" in item for item in messages)
        assert stale.exists()

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

    def test_resume_from_checkpoint(self, tiny_model, tiny_tokenizer, tmp_path, monkeypatch):
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
        checkpoint_loads = 0
        original_torch_load = torch.load

        def counted_torch_load(path, *args, **kwargs):
            nonlocal checkpoint_loads
            if str(path).endswith(".checkpoint.pt"):
                checkpoint_loads += 1
            return original_torch_load(path, *args, **kwargs)

        monkeypatch.setattr(torch, "load", counted_torch_load)
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
        assert checkpoint_loads == 1
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

    def test_only_prompt_too_short_errors_are_skipped(self, monkeypatch):
        import miru_tracer.core._jlens.fitting as fitting_module

        class DummyModel:
            n_layers = 2
            d_model = 1

        def fake_jacobian(_model, _prompt, _source_layers, **_kwargs):
            raise ValueError("simulated model backend failure")

        monkeypatch.setattr(fitting_module, "jacobian_for_prompt", fake_jacobian)

        with pytest.raises(ValueError, match="simulated model backend failure"):
            fit(DummyModel(), ["prompt"], checkpoint_every=None)

    def test_short_prompt_uses_dedicated_error(self):
        with pytest.raises(PromptTooShortError, match="prompt too short"):
            valid_position_mask(17)

    def test_rejects_zero_checkpoint_frequency(self):
        class DummyModel:
            n_layers = 2
            d_model = 1

        with pytest.raises(ValueError, match="checkpoint_every"):
            fit(DummyModel(), ["prompt"], checkpoint_every=0)

    def test_final_checkpoint_does_not_duplicate_periodic_write(self, monkeypatch, tmp_path):
        import miru_tracer.core._jlens.fitting as fitting_module

        class DummyModel:
            n_layers = 2
            d_model = 1

        def fake_jacobian(_model, _prompt, source_layers, **_kwargs):
            return ({source_layers[0]: torch.ones(1, 1)}, 128, 111)

        writes = []
        original_atomic_save = fitting_module._atomic_save

        def counted_atomic_save(obj, path):
            writes.append(path)
            original_atomic_save(obj, path)

        monkeypatch.setattr(fitting_module, "jacobian_for_prompt", fake_jacobian)
        monkeypatch.setattr(fitting_module, "_atomic_save", counted_atomic_save)
        fit(
            DummyModel(),
            ["first", "second"],
            checkpoint_path=str(tmp_path / "lens.checkpoint.pt"),
            checkpoint_every=1,
        )

        assert len(writes) == 2

    def test_failed_checkpoint_write_removes_atomic_temp(self, monkeypatch, tmp_path):
        import miru_tracer.core._jlens.fitting as fitting_module

        checkpoint = tmp_path / "lens.checkpoint.pt"

        def failed_save(_obj, path):
            with open(path, "wb") as handle:
                handle.write(b"partial")
            raise OSError("simulated full disk")

        monkeypatch.setattr(fitting_module.torch, "save", failed_save)

        with pytest.raises(OSError, match="simulated full disk"):
            fitting_module._atomic_save({}, str(checkpoint))

        assert not checkpoint.exists()
        assert list(tmp_path.glob("lens.checkpoint.pt.tmp.*")) == []

    def test_mid_prompt_failure_checkpoints_last_completed_prompt(self, monkeypatch, tmp_path):
        import miru_tracer.core._jlens.fitting as fitting_module

        class DummyModel:
            n_layers = 2
            d_model = 1

        calls = 0

        def fake_jacobian(_model, _prompt, source_layers, **_kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("simulated backend failure")
            return ({source_layers[0]: torch.ones(1, 1)}, 128, 111)

        checkpoint = tmp_path / "lens.checkpoint.pt"
        monkeypatch.setattr(fitting_module, "jacobian_for_prompt", fake_jacobian)

        with pytest.raises(RuntimeError, match="simulated backend failure"):
            fit(
                DummyModel(),
                ["first", "second"],
                checkpoint_path=str(checkpoint),
                checkpoint_every=None,
            )

        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        assert state["next_idx"] == 1
        assert state["n_done"] == 1

    def test_releases_previous_prompt_jacobian_before_next_prompt(self, monkeypatch):
        import miru_tracer.core._jlens.fitting as fitting_module

        class DummyModel:
            n_layers = 2
            d_model = 1

        previous: list[weakref.ReferenceType[torch.Tensor]] = []

        def fake_jacobian(_model, _prompt, source_layers, **_kwargs):
            if previous:
                assert previous[-1]() is None
            matrix = torch.ones(1, 1)
            previous.append(weakref.ref(matrix))
            return ({source_layers[0]: matrix}, 128, 111)

        monkeypatch.setattr(fitting_module, "jacobian_for_prompt", fake_jacobian)
        fit(DummyModel(), ["first", "second"], checkpoint_every=None)

    def test_diagnostic_log_has_prompt_and_resource_fields(self, monkeypatch, caplog):
        import miru_tracer.core._jlens.fitting as fitting_module

        class DummyModel:
            n_layers = 2
            d_model = 1

        def fake_jacobian(_model, _prompt, source_layers, *, timings, **_kwargs):
            timings["backward_and_copy_s"] = 1.25
            return ({source_layers[0]: torch.ones(1, 1)}, 128, 111)

        monkeypatch.setattr(fitting_module, "jacobian_for_prompt", fake_jacobian)
        caplog.set_level(logging.INFO)
        fit(
            DummyModel(),
            ["first"],
            checkpoint_every=None,
            log_diagnostics=True,
        )

        record = next(
            item.message
            for item in caplog.records
            if "fit_telemetry event=prompt_complete" in item.message
        )
        assert "global_prompt=1" in record
        assert "process_prompt=1" in record
        assert "phase_backward_and_copy_s=1.25" in record

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
        assert downloads[0][2] == {
            "revision": WIKITEXT_REVISION,
            "repo_type": "dataset",
            "cache_dir": cache_dir,
        }

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


class TestFitDiagnostics:
    def test_intra_prompt_sampler_records_phase_and_progress(self, monkeypatch):
        import miru_tracer.core.fit_diagnostics as diagnostics

        records = []
        sampled = threading.Event()

        def capture(_logger, event, **fields):
            records.append((event, fields))
            if event == "prompt_sample":
                sampled.set()

        monkeypatch.setattr(diagnostics, "log_fit_telemetry", capture)
        with PromptTelemetrySampler(
            logging.getLogger("test"),
            interval_s=0.01,
            global_prompt=138,
            process_prompt=1,
        ) as sampler:
            sampler.update_phase(
                "backward_and_copy",
                backward_pass=17,
                backward_passes=160,
            )
            assert sampled.wait(timeout=1)

        assert records[0][0] == "prompt_start"
        sample = next(fields for event, fields in records if event == "prompt_sample")
        assert sample["global_prompt"] == 138
        assert sample["process_prompt"] == 1
        assert sample["phase"] == "backward_and_copy"
        assert sample["backward_pass"] == 17
        assert sample["backward_passes"] == 160
        assert sample["elapsed_s"] > 0

    def test_slowdown_detector_requires_ratio_and_material_excess(self):
        assert prompt_slowdown(1_883, [400, 405, 395, 410, 390]) == pytest.approx(
            (400, 1_883 / 400)
        )
        assert prompt_slowdown(20, [10, 10, 10, 10, 10]) is None
        assert prompt_slowdown(700, [400, 405, 395, 410, 390]) is None

    def test_resource_snapshot_includes_host_io_and_each_cuda_device(self, monkeypatch):
        import miru_tracer.core.fit_diagnostics as diagnostics

        def fake_proc_file(path):
            return {
                "/proc/self/status": {
                    "VmRSS": "2048 kB",
                    "VmHWM": "4096 kB",
                    "Threads": "7",
                },
                "/proc/self/smaps_rollup": {
                    "Pss": "1024 kB",
                    "Private_Dirty": "512 kB",
                },
                "/proc/meminfo": {
                    "MemAvailable": "8192 kB",
                    "Dirty": "256 kB",
                    "Writeback": "128 kB",
                },
                "/proc/self/io": {
                    "read_bytes": "11",
                    "write_bytes": "22",
                    "cancelled_write_bytes": "3",
                },
            }.get(path, {})

        monkeypatch.setattr(diagnostics, "_read_key_value_file", fake_proc_file)
        monkeypatch.setattr(diagnostics.torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(diagnostics.torch.cuda, "device_count", lambda: 2)
        monkeypatch.setattr(
            diagnostics.torch.cuda,
            "memory_allocated",
            lambda device: (device + 1) * 1024 * 1024,
        )
        monkeypatch.setattr(
            diagnostics.torch.cuda,
            "memory_reserved",
            lambda device: (device + 2) * 1024 * 1024,
        )
        monkeypatch.setattr(diagnostics.torch.cuda, "max_memory_allocated", lambda _device: 0)
        monkeypatch.setattr(diagnostics.torch.cuda, "max_memory_reserved", lambda _device: 0)
        monkeypatch.setattr(
            diagnostics.torch.cuda,
            "memory_stats",
            lambda _device: {
                "num_alloc_retries": 4,
                "num_ooms": 1,
            },
        )
        monkeypatch.setattr(
            diagnostics.torch.cuda,
            "mem_get_info",
            lambda _device: (3 * 1024 * 1024, 4 * 1024 * 1024),
        )

        snapshot = diagnostics.resource_snapshot()

        assert snapshot["rss_mib"] == 2.0
        assert snapshot["proc_write_bytes"] == 22
        assert snapshot["host_dirty_mib"] == 0.2
        assert snapshot["cuda0_allocated_mib"] == 1.0
        assert snapshot["cuda1_reserved_mib"] == 3.0
        assert snapshot["cuda1_alloc_retries"] == 4
        assert snapshot["cuda1_ooms"] == 1

    def test_nvidia_smi_snapshot_captures_device_level_signals(self, monkeypatch):
        import miru_tracer.core.fit_diagnostics as diagnostics

        row = "1, GPU-abc, 580.65.06, P2, 87, 42, 45678, 95830, 71, 612.5, 1740, 1593"
        completed = diagnostics.subprocess.CompletedProcess(
            args=["nvidia-smi"],
            returncode=0,
            stdout=row,
            stderr="",
        )
        monkeypatch.setattr(diagnostics.shutil, "which", lambda _name: "/usr/bin/nvidia-smi")
        monkeypatch.setattr(
            diagnostics.subprocess,
            "run",
            lambda *_args, **_kwargs: completed,
        )

        snapshot = diagnostics._nvidia_smi_snapshot()

        assert snapshot["nvidia1_uuid"] == "GPU-abc"
        assert snapshot["nvidia1_driver_version"] == "580.65.06"
        assert snapshot["nvidia1_utilization_pct"] == 87
        assert snapshot["nvidia1_device_used_mib"] == 45678
        assert snapshot["nvidia1_power_w"] == 612.5
        assert snapshot["nvidia1_sm_clock_mhz"] == 1740

    def test_cuda_cache_emptying_is_opt_in(self, monkeypatch):
        import miru_tracer.core.fit_diagnostics as diagnostics

        synchronized = []
        emptied = []
        monkeypatch.setattr(diagnostics, "log_fit_telemetry", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(diagnostics.gc, "collect", lambda: 0)
        monkeypatch.setattr(diagnostics.torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(diagnostics.torch.cuda, "device_count", lambda: 2)
        monkeypatch.setattr(
            diagnostics.torch.cuda,
            "synchronize",
            lambda device: synchronized.append(device),
        )

        class DeviceContext:
            def __enter__(self):
                return None

            def __exit__(self, *_args):
                return False

        monkeypatch.setattr(
            diagnostics.torch.cuda,
            "device",
            lambda _device: DeviceContext(),
        )
        monkeypatch.setattr(
            diagnostics.torch.cuda,
            "empty_cache",
            lambda: emptied.append(True),
        )

        diagnostics.cleanup_runtime(logging.getLogger("test"), chunk_end=5)
        assert synchronized == []
        assert emptied == []

        diagnostics.cleanup_runtime(
            logging.getLogger("test"),
            chunk_end=10,
            empty_cuda_cache=True,
        )
        assert synchronized == [0, 1]
        assert emptied == [True, True]

    def test_qwen35_fallback_backend_is_detected(self, tiny_qwen35):
        has_gated_delta, implementations, fallback_components = _linear_attention_backend_status(
            tiny_qwen35
        )

        assert has_gated_delta is True
        assert any(
            component == "chunk_gated_delta_rule"
            and implementation.endswith(".torch_chunk_gated_delta_rule")
            for component, implementation in implementations
        )
        assert "chunk_gated_delta_rule" in fallback_components
        assert "causal_conv1d_fn" in fallback_components


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
        self,
        tiny_model,
        tiny_tokenizer,
        tmp_path,
        monkeypatch,
        caplog,
        out_name,
        loader,
    ):
        """main() end-to-end with mocked HF loaders: --device-map must reach
        from_pretrained, and the fit must produce a loadable artifact."""
        import transformers

        import miru_tracer.core.lens_fit as lens_fit_module
        import miru_tracer.core.logging_config as logging_config

        recorded = {}
        resolved_commit = "a" * 40
        monkeypatch.setattr(tiny_model.config, "_commit_hash", None, raising=False)
        monkeypatch.setattr(logging_config, "setup_logging", lambda: None)
        caplog.set_level(logging.INFO, logger=lens_fit_module.__name__)

        def resolve_commit(name, *, revision, cache_dir):
            recorded["resolve"] = (name, revision, cache_dir)
            return resolved_commit

        class FakeModel:
            @staticmethod
            def from_pretrained(name, **kwargs):
                recorded["model"] = kwargs
                return tiny_model

        class FakeTokenizer:
            @staticmethod
            def from_pretrained(name, **kwargs):
                recorded["tokenizer"] = kwargs
                return tiny_tokenizer

        monkeypatch.setattr(transformers, "AutoModelForCausalLM", FakeModel)
        monkeypatch.setattr(transformers, "AutoTokenizer", FakeTokenizer)
        monkeypatch.setattr(lens_fit_module, "_resolve_hub_model_commit", resolve_commit)

        prompts_file = tmp_path / "prompts.txt"
        prompts_file.write_text("\n".join(PROMPTS))
        out = tmp_path / out_name

        from miru_tracer.core.lens_fit import main

        code = main(
            [
                "tiny/test-model",
                "--revision",
                "0123456789abcdef",
                "--prompts-file",
                str(prompts_file),
                "--out",
                str(out),
                "--device-map",
                "auto",
                "--dim-batch",
                "8",
                "--chunk-size",
                "2",
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
        assert recorded["resolve"] == ("tiny/test-model", "0123456789abcdef", None)
        assert recorded["model"]["device_map"] == "auto"
        assert recorded["model"]["revision"] == resolved_commit
        assert recorded["tokenizer"]["revision"] == resolved_commit
        lens = loader(out)
        assert lens.n_prompts == 3
        assert lens.fit_metadata["convergence"]["enabled"] is False
        assert lens.fit_metadata["provenance"]["model_commit_hash"] == resolved_commit
        assert (
            "Model provenance: component=miru-tracer-fit-lens "
            "model=tiny/test-model requested_revision=0123456789abcdef "
            f"commit_sha={resolved_commit}" in caplog.messages
        )

    def test_out_and_hf_home_are_independent(
        self, tiny_model, tiny_tokenizer, tmp_path, monkeypatch
    ):
        import transformers

        import miru_tracer.core.lens_fit as lens_fit_module

        recorded = {}
        resolved_commit = "b" * 40

        def resolve_commit(name, *, revision, cache_dir):
            recorded["resolve"] = (name, revision, cache_dir)
            return resolved_commit

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
        monkeypatch.setattr(lens_fit_module, "_resolve_hub_model_commit", resolve_commit)
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
        assert recorded["resolve"] == ("tiny/test-model", None, expected_hub)
        assert recorded["tokenizer"]["cache_dir"] == expected_hub
        assert recorded["model"]["cache_dir"] == expected_hub
        assert recorded["tokenizer"]["revision"] == resolved_commit
        assert recorded["model"]["revision"] == resolved_commit
        assert load_lens(out).fit_metadata["provenance"]["model_commit_hash"] == resolved_commit
        assert set(output_dir.iterdir()) == {
            out,
            output_dir / "lens.checkpoint.pt",
        }
        assert hf_home.is_dir()
