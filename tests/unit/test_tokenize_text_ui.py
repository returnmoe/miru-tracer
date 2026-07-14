"""Tokenize Text UI table helpers."""

import gradio as gr

from miru_tracer.ui.tokenize_text import (
    TOKENIZE_HEADERS,
    create_tokenize_text_tab,
    tokenize_rows,
)


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


def test_tokenize_refresh_unmounts_and_remounts_dataframe():
    """Regression: Gradio's virtualizer otherwise retains the old row count."""
    with gr.Blocks() as app:
        create_tokenize_text_tab(object())

    dependencies = app.config["dependencies"]
    tokenized = next(d for d in dependencies if d["api_name"] == "tokenize_handler")
    hidden = next(d for d in dependencies if d["id"] == tokenized["trigger_after"])
    shown = [d for d in dependencies if d["trigger_after"] == tokenized["id"]]
    table_id = tokenized["outputs"][0]
    table = next(c for c in app.config["components"] if c["id"] == table_id)

    assert table["type"] == "dataframe"
    assert hidden["outputs"] == [table_id]
    assert len(shown) == 2
    assert {
        (d["trigger_only_on_success"], d["trigger_only_on_failure"])
        for d in shown
    } == {(True, False), (False, True)}
    assert all(d["outputs"] == [table_id] for d in shown)
    assert hidden["api_visibility"] == "private"
    assert all(d["api_visibility"] == "private" for d in shown)
    assert app.fns[hidden["id"]].fn()["visible"] is False
    assert all(app.fns[d["id"]].fn()["visible"] is True for d in shown)
