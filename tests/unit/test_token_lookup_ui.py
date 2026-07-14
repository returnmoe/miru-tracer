"""Token Lookup UI table helpers."""

from miru_tracer.ui.token_lookup import token_lookup_rows


def rows_dict(rows):
    return {field: value for field, value in rows}


class TestTokenLookupRows:
    def test_valid_token_rows(self, tiny_tokenizer):
        token_id = tiny_tokenizer.encode("A", add_special_tokens=False)[0]
        rows, status = token_lookup_rows(tiny_tokenizer, token_id)
        data = rows_dict(rows)

        assert status == "Token decoded."
        assert data["ID"] == str(token_id)
        assert "Token representation" in data
        assert "Decoded text" in data
        assert data["Raw bytes"]
        assert data["UTF-8 status"] in {"Complete/valid", "Incomplete sequence"}
        assert data["Special token"] == "No"

    def test_special_token_row(self, tiny_tokenizer):
        rows, status = token_lookup_rows(tiny_tokenizer, tiny_tokenizer.eos_token_id)
        data = rows_dict(rows)

        assert status == "Token decoded."
        assert data["Special token"] == "Yes"

    def test_out_of_range(self, tiny_tokenizer):
        rows, status = token_lookup_rows(tiny_tokenizer, len(tiny_tokenizer))

        assert rows == []
        assert "Invalid token ID" in status
        assert "Valid range" in status
