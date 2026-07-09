"""Token decoding utilities."""

from miru_tracer.core.tokenizer_utils import (
    detect_byte_level_bpe,
    extract_token_bytes,
    format_token_label,
    safe_decode_token,
    visible_whitespace,
)


class TestVisibleWhitespace:
    def test_leading_space(self):
        assert visible_whitespace(" Paris") == "␣Paris"

    def test_trailing_space(self):
        assert visible_whitespace("Paris ") == "Paris␣"

    def test_newline_and_tab(self):
        assert visible_whitespace("\n") == "⏎"
        assert visible_whitespace("a\tb") == "a⇥b"

    def test_all_spaces(self):
        assert visible_whitespace("  ") == "␣␣"

    def test_interior_space_untouched(self):
        assert visible_whitespace("a b") == "a b"

    def test_empty(self):
        assert visible_whitespace("") == ""


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


class TestFormatTokenLabel:
    class MultilingualTokenizer:
        raw = [
            "æ³ķåĽ½",
            "Ġlove",
            "ÐłÐ¾ÑģÑģÐ¸Ñı",
            "æĹ¥æľ¬",
            "cafÃ©",
            "日本",
            "Ã",
        ]
        decoded = [" 法国 ", " love", "Россия", "日本", "café", "日本", "�"]

        def convert_ids_to_tokens(self, ids):
            return [self.raw[token_id] for token_id in ids]

        def decode(self, ids, **_kwargs):
            return self.decoded[ids[0]]

    def test_qwen_chinese_keeps_raw_and_adds_readable_text(self):
        tokenizer = self.MultilingualTokenizer()
        assert format_token_label(tokenizer, 0) == "æ³ķåĽ½ (法国)"

    def test_russian_japanese_and_accented_latin(self):
        tokenizer = self.MultilingualTokenizer()
        assert format_token_label(tokenizer, 2).endswith("(Россия)")
        assert format_token_label(tokenizer, 3).endswith("(日本)")
        assert format_token_label(tokenizer, 4).endswith("(café)")

    def test_ascii_token_stays_compact(self):
        assert format_token_label(self.MultilingualTokenizer(), 1) == "Ġlove"

    def test_already_readable_unicode_is_not_repeated(self):
        assert format_token_label(self.MultilingualTokenizer(), 5) == "日本"

    def test_incomplete_utf8_keeps_raw_form(self):
        assert format_token_label(self.MultilingualTokenizer(), 6) == "Ã"

    def test_out_of_tokenizer_vocab_uses_id_placeholder(self):
        assert format_token_label(self.MultilingualTokenizer(), 99) == "<99>"


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
