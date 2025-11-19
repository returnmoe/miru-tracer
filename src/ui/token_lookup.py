"""Token Lookup tab for Gradio UI."""

import gradio as gr
from ..core.models import ModelManager
from ..core.tokenizer_utils import (
    safe_decode_token,
    extract_token_bytes,
)


def create_token_lookup_tab(model_manager: ModelManager) -> gr.Tab:
    """
    Create the token lookup tab interface.

    Args:
        model_manager: Singleton ModelManager instance

    Returns:
        Gradio Tab component
    """
    with gr.Tab("Token Lookup") as tab:
        gr.Markdown("Decode individual token IDs and inspect their properties.")

        token_id_input = gr.Number(
            label="Token ID",
            value=0,
            precision=0,
            minimum=0,
            info="Enter a token ID to decode",
        )

        lookup_output = gr.Textbox(
            label="Result", interactive=False, elem_classes="miru-textbox-mono", lines=9
        )

        lookup_button = gr.Button("Lookup Token", variant="primary")

        def lookup_token_handler(token_id):
            """Handle token ID lookup."""
            tokenizer = model_manager.get_tokenizer()

            if tokenizer is None:
                return "Error: No model loaded. Please load a model first."

            try:
                token_id = int(token_id)

                vocab_size = len(tokenizer)
                if token_id < 0 or token_id >= vocab_size:
                    return f"Invalid token ID: {token_id}\nValid range: 0 to {vocab_size - 1}"

                # Get token information
                decoded, token_str, incomplete_decoded = safe_decode_token(
                    tokenizer, token_id
                )

                result = f"ID {token_id} → {token_str!r}\n"

                if decoded:
                    result += f"Decoded: {decoded!r}\n\n"
                else:
                    result += f"Decoded: [error]\n\n"

                # Try to show the raw bytes
                token_bytes = extract_token_bytes(tokenizer, token_str)
                if token_bytes:
                    byte_repr = " ".join(f"{b:02x}" for b in token_bytes)
                    result += f"Raw bytes:\n{byte_repr}\n"

                if incomplete_decoded:
                    # Incomplete UTF-8 sequence
                    result += "\nThis token contains an INCOMPLETE UTF-8 sequence.\n"
                    result += (
                        "Individual tokens in byte-level BPE may not decode properly.\n"
                    )
                    result += "They need to be combined with adjacent tokens to form valid UTF-8."

                # Check if it's a special token
                if hasattr(tokenizer, "all_special_tokens"):
                    special_tokens = tokenizer.all_special_tokens
                    if (
                        decoded and decoded in special_tokens
                    ) or token_str in special_tokens:
                        result += f"\nThis is a special token."

                return result

            except Exception as e:
                return f"Error: {str(e)}"

        lookup_button.click(
            fn=lookup_token_handler, inputs=[token_id_input], outputs=[lookup_output]
        )

    return tab
