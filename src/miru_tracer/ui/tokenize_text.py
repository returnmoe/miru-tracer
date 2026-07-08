"""Tokenize Text tab for Gradio UI."""

import gradio as gr

from miru_tracer.core.model_manager import ModelManager
from miru_tracer.core.tokenizer_utils import (
    detect_byte_level_bpe,
    safe_decode_token,
)
from miru_tracer.ui.helpers import static_table_html


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

        # Server-rendered HTML rather than gr.Dataframe: Gradio 6's
        # virtualized dataframe keeps the previous row count when a new
        # value changes the table's shape (see helpers.static_table_html).
        tokens_output = gr.HTML(elem_classes="token-table")

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
                return "", "Error: No model loaded. Please load a model first."

            if not text:
                return "", "Error: Please enter text to tokenize"

            try:
                # Tokenize with add_special_tokens=False (don't auto-add BOS/EOS)
                token_ids = tokenizer.encode(text, add_special_tokens=False)

                special_token_ids = set(getattr(tokenizer, "all_special_ids", []))

                # Build table rows
                rows = []
                for i, token_id in enumerate(token_ids):
                    # Get only the raw token (second return value from safe_decode_token)
                    result = safe_decode_token(tokenizer, token_id)
                    raw_token = result[1]

                    # Determine token type with visual indicator
                    token_type = (
                        "Special 🟡" if token_id in special_token_ids else "Regular"
                    )

                    rows.append([i, token_type, token_id, raw_token])

                table = static_table_html(
                    ["Position", "Type", "ID", "Representation"], rows
                )
                count_msg = f"Total tokens: {len(token_ids)}"

                # Check for byte-level BPE
                if detect_byte_level_bpe(tokenizer):
                    count_msg += "\nThis is a byte-level BPE tokenizer. Some tokens may show as incomplete UTF-8 sequences."

                return table, count_msg

            except Exception as e:
                return "", f"Error: {str(e)}"

        tokenize_button.click(
            fn=tokenize_handler,
            inputs=[text_input],
            outputs=[tokens_output, additional_info],
        )

    return tab
