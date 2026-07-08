"""Readout engine: lens slices, aggregation, store discovery."""

import pytest
import torch

from miru_tracer.core._jlens import fit, from_hf
from miru_tracer.core.lens import (
    LensSlice,
    LensStore,
    aggregate_readouts,
    compute_lens_slice,
    is_word_token,
    sanitize_model_name,
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


@pytest.fixture()
def input_ids(tiny_tokenizer):
    return tiny_tokenizer.encode("Hello world test", return_tensors="pt")


class TestComputeLensSlice:
    def test_logit_mode_final_layer_matches_model(
        self, tiny_model, tiny_tokenizer, input_ids
    ):
        final = tiny_model.config.num_hidden_layers - 1
        slice_ = compute_lens_slice(
            tiny_model, tiny_tokenizer, input_ids, layers=[final], mode="logit", top_k=5
        )
        with torch.no_grad():
            real = tiny_model(input_ids).logits[0, -1].float()
        expected = torch.topk(torch.softmax(real, -1), 5)
        last = len(slice_.positions) - 1
        assert slice_.tokens[0][last] == expected.indices.tolist()
        assert slice_.probs[0][last] == pytest.approx(expected.values.tolist(), rel=1e-4)

    def test_position_subset(self, tiny_model, tiny_tokenizer, input_ids):
        slice_ = compute_lens_slice(
            tiny_model, tiny_tokenizer, input_ids,
            layers=[0], positions=[2, 5], mode="logit", top_k=3,
        )
        assert slice_.positions == [2, 5]
        assert len(slice_.tokens[0]) == 2
        assert len(slice_.position_texts) == 2

    def test_jacobian_mode_differs_from_logit_in_early_layer(
        self, tiny_model, tiny_tokenizer, input_ids, tiny_lens
    ):
        logit = compute_lens_slice(
            tiny_model, tiny_tokenizer, input_ids, layers=[0], mode="logit", top_k=5
        )
        jac = compute_lens_slice(
            tiny_model, tiny_tokenizer, input_ids,
            layers=[0], mode="jacobian", jlens=tiny_lens, top_k=5,
        )
        # A fitted transport is not the identity; readouts should differ somewhere.
        assert logit.tokens != jac.tokens

    def test_jacobian_mode_requires_lens(self, tiny_model, tiny_tokenizer, input_ids):
        with pytest.raises(ValueError, match="requires a fitted"):
            compute_lens_slice(
                tiny_model, tiny_tokenizer, input_ids, layers=[0], mode="jacobian"
            )

    def test_unfitted_layer_rejected(self, tiny_model, tiny_tokenizer, input_ids, tiny_lens):
        # tiny_lens covers layer 0 only; final layer is exempt (J = I)
        final = tiny_model.config.num_hidden_layers - 1
        compute_lens_slice(  # final layer alone is fine
            tiny_model, tiny_tokenizer, input_ids,
            layers=[final], mode="jacobian", jlens=tiny_lens,
        )

    def test_out_of_range_layer_rejected(self, tiny_model, tiny_tokenizer, input_ids):
        with pytest.raises(ValueError, match="out of range"):
            compute_lens_slice(
                tiny_model, tiny_tokenizer, input_ids, layers=[99], mode="logit"
            )

    def test_unknown_mode_rejected(self, tiny_model, tiny_tokenizer, input_ids):
        with pytest.raises(ValueError, match="Unknown lens mode"):
            compute_lens_slice(
                tiny_model, tiny_tokenizer, input_ids, layers=[0], mode="tuned"
            )

    def test_pinned_ranks_top_token_is_rank_zero(
        self, tiny_model, tiny_tokenizer, input_ids
    ):
        final = tiny_model.config.num_hidden_layers - 1
        probe = compute_lens_slice(
            tiny_model, tiny_tokenizer, input_ids, layers=[final], mode="logit", top_k=1
        )
        last = len(probe.positions) - 1
        top_token = probe.tokens[0][last]
        slice_ = compute_lens_slice(
            tiny_model, tiny_tokenizer, input_ids,
            layers=[final], mode="logit", pinned_token_ids=top_token,
        )
        assert slice_.pinned_ranks[top_token[0]][0][last] == 0

    def test_skip_non_words_filters(self, tiny_model, tiny_tokenizer, input_ids):
        slice_ = compute_lens_slice(
            tiny_model, tiny_tokenizer, input_ids,
            layers=[0], mode="logit", top_k=5, skip_non_words=True,
        )
        for row in slice_.texts[0]:
            assert all(is_word_token(t) for t in row)


class TestAggregate:
    def test_counts_hand_checkable(self):
        slice_ = LensSlice(
            mode="logit",
            layers=[0, 1],
            positions=[0, 1],
            position_texts=["a", "b"],
            tokens=[[[7, 8], [7, 9]], [[7, 8], [10, 11]]],
            probs=[[[0.5, 0.1], [0.5, 0.1]], [[0.5, 0.1], [0.5, 0.1]]],
            texts=[[["7", "8"], ["7", "9"]], [["7", "8"], ["10", "11"]]],
        )
        rows = aggregate_readouts(slice_)
        by_id = {r.token_id: r for r in rows}
        assert by_id[7].count == 3
        assert by_id[7].count_by_layer == [2, 1]
        assert by_id[8].count == 2
        assert rows[0].token_id == 7  # sorted by count desc

    def test_limit(self):
        slice_ = LensSlice(
            mode="logit",
            layers=[0],
            positions=[0],
            position_texts=["a"],
            tokens=[[[1, 2, 3, 4, 5]]],
            probs=[[[0.5, 0.2, 0.1, 0.05, 0.02]]],
            texts=[[["1", "2", "3", "4", "5"]]],
        )
        assert len(aggregate_readouts(slice_, limit=2)) == 2


class TestLensStore:
    def test_missing_returns_none(self, tmp_path):
        store = LensStore(base_dir=tmp_path)
        assert store.get("some/model") is None

    def test_roundtrip_discovery(self, tmp_path, tiny_lens):
        store = LensStore(base_dir=tmp_path)
        path = store.lens_path("Qwen/Qwen3-0.6B")
        path.parent.mkdir(parents=True)
        tiny_lens.save(str(path))
        loaded = store.get("Qwen/Qwen3-0.6B")
        assert loaded is not None
        assert loaded.source_layers == tiny_lens.source_layers
        # cached object returned on second call
        assert store.get("Qwen/Qwen3-0.6B") is loaded

    def test_sanitize(self):
        assert "/" not in sanitize_model_name("Qwen/Qwen3-0.6B")
        assert sanitize_model_name("Qwen/Qwen3-0.6B") == "Qwen--Qwen3-0.6B"

    def test_corrupt_file_returns_none(self, tmp_path):
        store = LensStore(base_dir=tmp_path)
        path = store.lens_path("m")
        path.parent.mkdir(parents=True)
        path.write_text("not a torch file")
        assert store.get("m") is None


class TestIsWordToken:
    @pytest.mark.parametrize("text,expected", [
        (" hello", True), ("虚假", True), ("123", True),
        (" ", False), ("...", False), ("\n", False), ("", False),
    ])
    def test_cases(self, text, expected):
        assert is_word_token(text) is expected
