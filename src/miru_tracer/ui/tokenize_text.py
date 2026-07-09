"""Tokenize Text tab for Gradio UI."""

import gradio as gr

from miru_tracer.core.model_manager import ModelManager
from miru_tracer.core.tokenizer_utils import (
    detect_byte_level_bpe,
    safe_decode_token,
)

TOKENIZE_HEADERS = ["Position", "Type", "ID", "Representation"]


def tokenize_rows(tokenizer, text: str) -> list[list]:
    """Build rows for the Tokenize Text results table."""
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    special_token_ids = set(getattr(tokenizer, "all_special_ids", []))

    rows = []
    for i, token_id in enumerate(token_ids):
        raw_token = safe_decode_token(tokenizer, token_id)[1]
        token_type = "Special 🟡" if token_id in special_token_ids else "Regular"
        rows.append([i, token_type, token_id, raw_token])
    return rows


def create_tokenize_text_tab(model_manager: ModelManager) -> gr.Tab:
    """
    Create the tokenize text tab interface.

    Args:
        model_manager: Singleton ModelManager instance

    Returns:
        Gradio Tab component
    """
    with gr.Tab("Tokenize Text") as tab, gr.Column(elem_classes="miru-narrow"):
        gr.Markdown("See how text is tokenized by the loaded model.")

        text_input = gr.Textbox(
            label="Input",
            placeholder="Enter text to tokenize...",
            lines=3,
            value="The future of artificial intelligence is",
        )

        gr.Markdown("### Results")

        tokens_output = gr.Dataframe(
            headers=TOKENIZE_HEADERS,
            datatype=["number", "str", "number", "str"],
            label="Tokens",
            interactive=False,
        )

        additional_info = gr.Textbox(
            elem_classes="miru-textbox-mono",
            label="Additional information",
            interactive=False,
            lines=2,
        )

        tokenize_button = gr.Button("Tokenize", variant="primary")

        def tokenize_handler(text):
            """Handle text tokenization."""
            tokenizer = model_manager.get_tokenizer()

            if tokenizer is None:
                return [], "Error: No model loaded. Please load a model first."

            if not text:
                return [], "Error: Please enter text to tokenize"

            try:
                rows = tokenize_rows(tokenizer, text)
                count_msg = f"Total tokens: {len(rows)}"

                # Check for byte-level BPE
                if detect_byte_level_bpe(tokenizer):
                    count_msg += "\nThis is a byte-level BPE tokenizer. Some tokens may show as incomplete UTF-8 sequences."

                return rows, count_msg

            except Exception as e:
                return [], f"Error: {str(e)}"

        tokenize_button.click(
            fn=tokenize_handler,
            inputs=[text_input],
            outputs=[tokens_output, additional_info],
        )

    return tab
