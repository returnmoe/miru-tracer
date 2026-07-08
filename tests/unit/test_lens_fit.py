"""Chunked lens fitting: progress, resume, cancellation, artifact validity."""


from miru_tracer.core._jlens import JacobianLens
from miru_tracer.core.lens_fit import iter_fit_lens, prompts_from_file

PROMPTS = [
    "Hello world, this is a much longer test prompt for fitting the lens today ok.",
    "The quick brown fox jumps over the lazy dog again and again without stopping.",
    "Numbers like 12345 and 67890 mixed with words make for varied byte sequences.",
    "A fourth prompt exists so that chunked fitting has more than one full chunk.",
]


class TestIterFitLens:
    def test_progress_and_artifact(self, tiny_model, tiny_tokenizer, tmp_path):
        out = tmp_path / "lens.pt"
        updates = list(
            iter_fit_lens(
                tiny_model, tiny_tokenizer, PROMPTS,
                out_path=out, chunk_size=2, dim_batch=8,
            )
        )
        assert [u.prompts_done for u in updates] == [2, 4]
        assert out.exists()
        lens = JacobianLens.load(str(out))
        assert lens.n_prompts == 4

    def test_intermediate_artifact_is_valid(self, tiny_model, tiny_tokenizer, tmp_path):
        out = tmp_path / "lens.pt"
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
        assert JacobianLens.load(str(out)).n_prompts == 2

    def test_should_stop_cancels_between_chunks(self, tiny_model, tiny_tokenizer, tmp_path):
        out = tmp_path / "lens.pt"
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
        out = tmp_path / "lens.pt"
        list(
            iter_fit_lens(
                tiny_model, tiny_tokenizer, PROMPTS,
                out_path=out, chunk_size=2, dim_batch=8,
                should_stop=lambda: True,  # stops after 2 prompts
            )
        )
        updates = list(
            iter_fit_lens(
                tiny_model, tiny_tokenizer, PROMPTS,
                out_path=out, chunk_size=2, dim_batch=8,
            )
        )
        # first yield re-covers the checkpointed prefix without recomputing,
        # final artifact covers all prompts
        assert updates[-1].prompts_done == 4
        assert JacobianLens.load(str(out)).n_prompts == 4


class TestPromptSources:
    def test_prompts_from_file(self, tmp_path):
        f = tmp_path / "prompts.txt"
        f.write_text("first prompt\n\n  second prompt  \n")
        assert prompts_from_file(f) == ["first prompt", "second prompt"]
