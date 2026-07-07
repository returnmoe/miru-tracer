"""Token decoding utilities."""

from miru_tracer.core.tokenizer_utils import (
    detect_byte_level_bpe,
    extract_token_bytes,
    safe_decode_token,
)


class TestSafeDecodeToken:
    def test_decodes_regular_token(self, tiny_tokenizer):
        token_id = tiny_tokenizer.encode("a", add_special_tokens=False)[0]
        decoded, raw, replacement = safe_decode_token(tiny_tokenizer, token_id)
        assert decoded == "a"
        assert raw  # raw vocabulary string is always present
        assert replacement is None

    def test_incomplete_utf8_flagged(self, tiny_tokenizer):
        # A lone continuation byte cannot decode to text on its own.
        token_id = tiny_tokenizer.encode("é", add_special_tokens=False)[0]
        decoded, raw, replacement = safe_decode_token(tiny_tokenizer, token_id)
        assert decoded is None
        assert replacement is not None and "�" in replacement


class TestExtractTokenBytes:
    def test_extracts_bytes(self, tiny_tokenizer):
        raw = tiny_tokenizer.convert_ids_to_tokens(
            tiny_tokenizer.encode(" hi", add_special_tokens=False)
        )[0]
        extracted = extract_token_bytes(tiny_tokenizer, raw)
        assert extracted is not None
        assert extracted.startswith(b" ")

    def test_unknown_chars_return_none(self, tiny_tokenizer):
        assert extract_token_bytes(tiny_tokenizer, "世") is None

    def test_non_fast_tokenizer_returns_none(self):
        class NotFast:
            pass

        assert extract_token_bytes(NotFast(), "abc") is None


class TestDetectByteLevelBpe:
    def test_detects_byte_level(self, tiny_tokenizer):
        assert detect_byte_level_bpe(tiny_tokenizer) is True

    def test_non_fast_is_false(self):
        class NotFast:
            pass

        assert detect_byte_level_bpe(NotFast()) is False
