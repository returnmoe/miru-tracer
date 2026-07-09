"""Tokenize Text UI table helpers."""

from miru_tracer.ui.tokenize_text import TOKENIZE_HEADERS, tokenize_rows


class TestTokenizeRows:
    def test_headers_match_dataframe_shape(self):
        assert TOKENIZE_HEADERS == ["Position", "Type", "ID", "Representation"]

    def test_regular_token_rows(self, tiny_tokenizer):
        rows = tokenize_rows(tiny_tokenizer, "AB")

        assert rows
        assert len(rows[0]) == len(TOKENIZE_HEADERS)
        assert [row[0] for row in rows] == list(range(len(rows)))
        assert all(row[1] in {"Regular", "Special 🟡"} for row in rows)
        assert all(isinstance(row[2], int) for row in rows)
        assert all(isinstance(row[3], str) for row in rows)

    def test_empty_text_has_no_rows(self, tiny_tokenizer):
        assert tokenize_rows(tiny_tokenizer, "") == []
