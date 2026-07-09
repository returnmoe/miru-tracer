"""Token Lookup tab for Gradio UI."""

import gradio as gr

from miru_tracer.core.model_manager import ModelManager
from miru_tracer.core.tokenizer_utils import (
    extract_token_bytes,
    safe_decode_token,
)

TOKEN_LOOKUP_HEADERS = ["Field", "Value"]


def token_lookup_rows(tokenizer, token_id) -> tuple[list[list[str]], str]:
    """Build the Token Lookup table rows and status message."""
    token_id = int(token_id)

    vocab_size = len(tokenizer)
    if token_id < 0 or token_id >= vocab_size:
        return [], f"Invalid token ID: {token_id}. Valid range: 0 to {vocab_size - 1}."

    decoded, token_str, incomplete_decoded = safe_decode_token(tokenizer, token_id)
    token_bytes = extract_token_bytes(tokenizer, token_str)
    byte_repr = " ".join(f"{b:02x}" for b in token_bytes) if token_bytes else ""

    special = False
    if hasattr(tokenizer, "all_special_tokens"):
        special_tokens = tokenizer.all_special_tokens
        special = (decoded and decoded in special_tokens) or token_str in special_tokens

    rows = [
        ["ID", str(token_id)],
        ["Token representation", repr(token_str)],
        ["Decoded text", repr(decoded) if decoded else "[error]"],
        ["Raw bytes", byte_repr or "[unavailable]"],
        [
            "UTF-8 status",
            "Incomplete sequence" if incomplete_decoded else "Complete/valid",
        ],
        ["Special token", "Yes" if special else "No"],
    ]
    return rows, "Token decoded."


def create_token_lookup_tab(model_manager: ModelManager) -> gr.Tab:
    """
    Create the token lookup tab interface.

    Args:
        model_manager: Singleton ModelManager instance

    Returns:
        Gradio Tab component
    """
    with gr.Tab("Token Lookup") as tab, gr.Column(elem_classes="miru-narrow"):
        gr.Markdown("Decode individual token IDs and inspect their properties.")

        token_id_input = gr.Number(
            label="Token ID",
            value=0,
            precision=0,
            minimum=0,
            info="Enter a token ID to decode",
        )

        lookup_output = gr.Dataframe(
            headers=TOKEN_LOOKUP_HEADERS,
            datatype=["str", "str"],
            label="Result",
            interactive=False,
        )
        status_output = gr.Textbox(label="Status", interactive=False, lines=1)

        lookup_button = gr.Button("Lookup Token", variant="primary")

        def lookup_token_handler(token_id):
            """Handle token ID lookup."""
            tokenizer = model_manager.get_tokenizer()

            if tokenizer is None:
                return [], "Error: No model loaded. Please load a model first."

            try:
                return token_lookup_rows(tokenizer, token_id)

            except Exception as e:
                return [], f"Error: {str(e)}"

        lookup_button.click(
            fn=lookup_token_handler,
            inputs=[token_id_input],
            outputs=[lookup_output, status_output],
        )

    return tab
