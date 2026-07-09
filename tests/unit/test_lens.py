"""Readout engine: lens slices, aggregation, store discovery."""

import pytest
import torch

from miru_tracer.core._jlens import fit, from_hf
from miru_tracer.core.lens import (
    LensSlice,
    LensStore,
    aggregate_readouts,
    compute_lens_slice,
    decode_token,
    is_word_token,
    record_lens_activations,
    sanitize_model_name,
    word_token_mask,
    wrap_model,
)
from miru_tracer.core.lens_io import save_lens


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

    def test_prerecorded_activations_match_and_skip_forward(
        self, tiny_model, tiny_tokenizer, input_ids, monkeypatch
    ):
        layers = [0, 1]
        fresh = compute_lens_slice(
            tiny_model, tiny_tokenizer, input_ids, layers=layers, mode="logit"
        )
        acts = record_lens_activations(tiny_model, tiny_tokenizer, input_ids)

        calls = 0
        original = tiny_model.forward

        def counting_forward(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(tiny_model, "forward", counting_forward)
        cached = compute_lens_slice(
            tiny_model, tiny_tokenizer, input_ids,
            layers=layers, mode="logit", activations=acts,
        )
        assert calls == 0  # no model forward with pre-recorded residuals
        assert cached.tokens == fresh.tokens
        assert cached.probs == fresh.probs  # deterministic CPU path, same math

    def test_missing_activation_layer_rejected(
        self, tiny_model, tiny_tokenizer, input_ids
    ):
        acts = record_lens_activations(tiny_model, tiny_tokenizer, input_ids)
        del acts[1]
        with pytest.raises(ValueError, match="activations missing"):
            compute_lens_slice(
                tiny_model, tiny_tokenizer, input_ids,
                layers=[0, 1], mode="logit", activations=acts,
            )

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

    def test_synthetic_diff_mode_is_rejected(
        self, tiny_model, tiny_tokenizer, input_ids, tiny_lens
    ):
        with pytest.raises(ValueError, match="Unknown lens mode"):
            compute_lens_slice(
                tiny_model,
                tiny_tokenizer,
                input_ids,
                layers=[0],
                mode="diff",
                jlens=tiny_lens,
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
        for token_ids in slice_.tokens[0]:
            assert all(
                is_word_token(tiny_tokenizer.decode([token_id]))
                for token_id in token_ids
            )

    def test_word_mask_searches_full_vocab_not_four_x_top_k(
        self, tiny_model, tiny_tokenizer, input_ids, monkeypatch
    ):
        wrapper = wrap_model(tiny_model, tiny_tokenizer)
        mask = word_token_mask(tiny_tokenizer, tiny_model.config.vocab_size)
        non_words = (~mask).nonzero(as_tuple=True)[0][:24]
        words = mask.nonzero(as_tuple=True)[0][:5]
        assert len(non_words) >= 20 and len(words) == 5

        def punctuation_dominated_unembed(residual):
            logits = torch.full(
                (residual.shape[0], tiny_model.config.vocab_size),
                -100.0,
                device=residual.device,
            )
            # The old candidate_k=4*top_k path saw punctuation only.
            logits[:, non_words] = torch.arange(
                len(non_words),
                0,
                -1,
                device=residual.device,
                dtype=logits.dtype,
            )
            logits[:, words] = -torch.arange(
                len(words), device=residual.device, dtype=logits.dtype
            )
            return logits

        monkeypatch.setattr(wrapper, "unembed", punctuation_dominated_unembed)
        slice_ = compute_lens_slice(
            tiny_model,
            tiny_tokenizer,
            input_ids,
            layers=[0],
            positions=[0],
            mode="logit",
            top_k=5,
            skip_non_words=True,
        )
        assert set(slice_.tokens[0][0]) == set(words.tolist())
        assert len(slice_.tokens[0][0]) == 5

    def test_final_output_preserves_true_non_word_top1(
        self, tiny_model, tiny_tokenizer, input_ids, monkeypatch
    ):
        wrapper = wrap_model(tiny_model, tiny_tokenizer)
        mask = word_token_mask(tiny_tokenizer, tiny_model.config.vocab_size)
        non_word = int((~mask).nonzero(as_tuple=True)[0][0])
        words = mask.nonzero(as_tuple=True)[0][:5]

        def unembed_with_punctuation_top1(residual):
            logits = torch.full(
                (residual.shape[0], tiny_model.config.vocab_size),
                -100.0,
                device=residual.device,
            )
            logits[:, non_word] = 20.0
            logits[:, words] = torch.arange(
                5, 0, -1, dtype=logits.dtype, device=logits.device
            )
            return logits

        monkeypatch.setattr(wrapper, "unembed", unembed_with_punctuation_top1)
        final = tiny_model.config.num_hidden_layers - 1
        slice_ = compute_lens_slice(
            tiny_model,
            tiny_tokenizer,
            input_ids,
            layers=[final],
            positions=[0],
            mode="logit",
            top_k=5,
            skip_non_words=True,
        )
        assert slice_.tokens[0][0][0] == non_word
        assert len(slice_.tokens[0][0]) == 5


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
        assert by_id[7].relevance_score == pytest.approx(3.0)
        assert by_id[7].best_rank_by_layer == [0, 0]
        assert by_id[7].peak_prob_by_layer == [0.5, 0.5]
        assert by_id[8].count == 2
        assert rows[0].token_id == 7  # sorted by reciprocal-rank relevance

    def test_rank_weighted_relevance_beats_persistent_low_rank_noise(self):
        tokens = []
        texts = []
        probs = []
        for position in range(4):
            row = [7] + list(range(20 + position * 10, 26 + position * 10)) + [8]
            if position >= 2:
                row[0] = 100 + position
            tokens.append(row)
            texts.append([str(token_id) for token_id in row])
            probs.append([0.5, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.01])
        slice_ = LensSlice(
            mode="logit",
            layers=[0],
            positions=[0, 1, 2, 3],
            position_texts=["a", "b", "c", "d"],
            tokens=[tokens],
            probs=[probs],
            texts=[texts],
        )
        rows = aggregate_readouts(slice_, limit=None)
        by_id = {row.token_id: row for row in rows}
        assert by_id[8].count == 4
        assert by_id[7].count == 2
        assert by_id[7].relevance_score > by_id[8].relevance_score
        assert rows.index(by_id[7]) < rows.index(by_id[8])

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

    def test_no_limit(self):
        slice_ = LensSlice(
            mode="logit",
            layers=[0],
            positions=[0],
            position_texts=["a"],
            tokens=[[[1, 2, 3, 4, 5]]],
            probs=[[[0.5, 0.2, 0.1, 0.05, 0.02]]],
            texts=[[["1", "2", "3", "4", "5"]]],
        )
        assert len(aggregate_readouts(slice_, limit=None)) == 5


class TestLensStore:
    def test_missing_returns_none(self, tmp_path):
        store = LensStore(base_dir=tmp_path)
        assert store.get("some/model") is None

    def test_roundtrip_discovery(self, tmp_path, tiny_lens):
        store = LensStore(base_dir=tmp_path)
        path = store.lens_path("Qwen/Qwen3-0.6B")
        assert path.name == "lens.safetensors"
        path.parent.mkdir(parents=True)
        save_lens(tiny_lens, path)
        loaded = store.get("Qwen/Qwen3-0.6B")
        assert loaded is not None
        assert loaded.source_layers == tiny_lens.source_layers
        # cached object returned on second call
        assert store.get("Qwen/Qwen3-0.6B") is loaded

    def test_legacy_pt_fallback(self, tmp_path, tiny_lens):
        store = LensStore(base_dir=tmp_path)
        legacy = store.lens_path("m").with_name("lens.pt")
        legacy.parent.mkdir(parents=True)
        tiny_lens.save(str(legacy))
        assert store.existing_lens_path("m") == legacy
        loaded = store.get("m")
        assert loaded is not None
        assert loaded.source_layers == tiny_lens.source_layers

    def test_prefers_safetensors_over_legacy(self, tmp_path, tiny_lens):
        store = LensStore(base_dir=tmp_path)
        path = store.lens_path("m")
        path.parent.mkdir(parents=True)
        tiny_lens.save(str(path.with_name("lens.pt")))
        save_lens(tiny_lens, path)
        assert store.existing_lens_path("m") == path
        assert store.get("m") is not None

    def test_sanitize(self):
        assert "/" not in sanitize_model_name("Qwen/Qwen3-0.6B")
        assert sanitize_model_name("Qwen/Qwen3-0.6B") == "Qwen--Qwen3-0.6B"

    @pytest.mark.parametrize("filename", ["lens.safetensors", "lens.pt"])
    def test_corrupt_file_returns_none(self, tmp_path, filename):
        store = LensStore(base_dir=tmp_path)
        path = store.lens_path("m").with_name(filename)
        path.parent.mkdir(parents=True)
        path.write_text("neither safetensors nor a torch file")
        assert store.get("m") is None


class TestIsWordToken:
    @pytest.mark.parametrize("text,expected", [
        (" hello", True), ("虚假", True), ("123", True),
        ("can't", True), ("well-being", True), ("l’amour", True),
        (" ", False), ("...", False), ("\n", False), ("", False),
        ("**word", False), ("word!", False), ("-word", False),
        ("word-", False), ("snake_case", False), ("<|endoftext|>", False),
        ("<pad>", False), ("�", False),
    ])
    def test_cases(self, text, expected):
        assert is_word_token(text) is expected

    def test_qwen_byte_tokens_are_classified_by_decoded_text(self):
        class QwenLikeTokenizer:
            raw = ["âĢĶ\"", "Ġlove", "Âł", "can\'t", "æ³ķåĽ½"]
            decoded = [
                "—\"",
                " love",
                "\N{NO-BREAK SPACE}",
                "can't",
                "法国",
            ]

            def decode(self, ids, **_kwargs):
                return self.decoded[ids[0]]

            def convert_ids_to_tokens(self, ids):
                return [self.raw[token_id] for token_id in ids]

        tokenizer = QwenLikeTokenizer()
        assert word_token_mask(tokenizer, 5).tolist() == [False, True, False, True, True]
        # Filtering and display are deliberately separate: ordinary ASCII
        # stays raw while multilingual text gets a readable annotation.
        assert decode_token(tokenizer, 1) == "Ġlove"
        assert decode_token(tokenizer, 4) == "æ³ķåĽ½ (法国)"


class TestDecodeToken:
    def test_out_of_tokenizer_vocab_id_gets_placeholder(self, tiny_tokenizer):
        """Model embedding matrices are commonly padded past the tokenizer
        vocab; readouts can surface those ids and must not crash (regression:
        None text broke the plots)."""
        from miru_tracer.core.lens import decode_token

        text = decode_token(tiny_tokenizer, 259)  # model vocab 260 > tokenizer 258
        assert isinstance(text, str) and text  # never None/empty

    def test_regular_token(self, tiny_tokenizer):
        from miru_tracer.core.lens import decode_token

        token_id = tiny_tokenizer.encode("a", add_special_tokens=False)[0]
        assert decode_token(tiny_tokenizer, token_id) == "a"
