"""Chunked lens fitting: progress, resume, cancellation, artifact validity."""


import pytest

from miru_tracer.core._jlens import JacobianLens
from miru_tracer.core.lens_fit import iter_fit_lens, prompts_from_file
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
                tiny_model, tiny_tokenizer, PROMPTS,
                out_path=out, chunk_size=2, dim_batch=8,
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
                    tiny_model, tiny_tokenizer, PROMPTS,
                    out_path=out, chunk_size=2, dim_batch=8,
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
                tiny_model, tiny_tokenizer, PROMPTS,
                out_path=out, chunk_size=2, dim_batch=8,
                should_stop=lambda: True,
            )
        )
        assert len(updates) == 1  # stopped after the first chunk

    def test_resume_from_checkpoint(self, tiny_model, tiny_tokenizer, tmp_path):
        """A second run picks up where a cancelled one stopped."""
        out = tmp_path / "lens.safetensors"
        list(
            iter_fit_lens(
                tiny_model, tiny_tokenizer, PROMPTS,
                out_path=out, chunk_size=2, dim_batch=8,
                should_stop=lambda: True,  # stops after 2 prompts
            )
        )
        assert (tmp_path / "lens.checkpoint.pt").exists()
        updates = list(
            iter_fit_lens(
                tiny_model, tiny_tokenizer, PROMPTS,
                out_path=out, chunk_size=2, dim_batch=8,
            )
        )
        # first yield re-covers the checkpointed prefix without recomputing,
        # final artifact covers all prompts
        assert updates[-1].prompts_done == 4
        assert load_lens(out).n_prompts == 4


class TestPromptSources:
    def test_prompts_from_file(self, tmp_path):
        f = tmp_path / "prompts.txt"
        f.write_text("first prompt\n\n  second prompt  \n")
        assert prompts_from_file(f) == ["first prompt", "second prompt"]


class TestCliMain:
    # lens.pt pins the explicit legacy-format escape hatch (--out foo.pt)
    @pytest.mark.parametrize("out_name,loader", [
        ("lens.safetensors", load_lens),
        ("lens.pt", lambda p: JacobianLens.load(str(p))),
    ])
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

        code = main([
            "tiny/test-model",
            "--prompts-file", str(prompts_file),
            "--out", str(out),
            "--device-map", "auto",
            "--dim-batch", "8",
        ])
        assert code == 0
        assert recorded.get("device_map") == "auto"
        assert loader(out).n_prompts == len(PROMPTS)
