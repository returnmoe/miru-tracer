"""Tokenizer utility functions for handling byte-level BPE tokens."""

from functools import lru_cache


def visible_whitespace(text: str) -> str:
    """Make a single token's whitespace visible for display.

    Leading/trailing spaces become ``␣``, newlines ``⏎``, tabs ``⇥`` — so
    " Paris" and "Paris" are distinguishable in token sequences, heatmap
    cells, and readout tables. Interior spaces are left alone.
    """
    if not text:
        return text
    text = text.replace("\n", "⏎").replace("\t", "⇥")
    lead = len(text) - len(text.lstrip(" "))
    if lead == len(text):  # all spaces
        return "␣" * lead
    trail = len(text) - len(text.rstrip(" "))
    return "␣" * lead + text[lead : len(text) - trail] + "␣" * trail


@lru_cache(maxsize=1)
def _gpt2_byte_decoder() -> dict[str, int]:
    """The standard GPT-2 byte-level BPE unicode-to-byte mapping."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs, strict=True)}


def extract_token_bytes(tokenizer, token_str: str) -> bytes | None:
    """
    Extract the actual byte values from a byte-level BPE token string.

    The token_str contains special unicode characters (like Ġ, ð, Ł) that represent
    bytes in the byte-level BPE encoding.

    Args:
        tokenizer: The tokenizer instance
        token_str: The token string to extract bytes from

    Returns:
        Bytes if extraction successful, None otherwise
    """
    if not hasattr(tokenizer, "backend_tokenizer"):
        return None

    byte_decoder = _gpt2_byte_decoder()
    byte_values = []
    for char in token_str:
        if char not in byte_decoder:
            # Character not in the standard byte-level mapping
            return None
        byte_values.append(byte_decoder[char])

    return bytes(byte_values) if byte_values else None


def safe_decode_token(
    tokenizer, token_id: int
) -> tuple[str | None, str, str | None]:
    """
    Safely decode a single token ID for display.

    For byte-level BPE tokenizers (like Qwen2, GPT-2, RoBERTa), individual tokens
    may contain incomplete UTF-8 byte sequences that cannot be decoded in isolation.

    Args:
        tokenizer: The tokenizer instance
        token_id: Token ID to decode

    Returns:
        Tuple of (decoded_text, token_str, replacement_char_decoded)
        - decoded_text: Successfully decoded text or None if incomplete
        - token_str: Raw token string from vocabulary
        - replacement_char_decoded: Decoded text with replacement chars if incomplete
    """
    # Get the raw token string from vocabulary
    token_str = tokenizer.convert_ids_to_tokens([token_id])[0]

    # Try to decode using the tokenizer
    try:
        decoded = tokenizer.decode([token_id], skip_special_tokens=False)
    except Exception:
        decoded = None

    # Check if we got replacement characters (incomplete UTF-8)
    if decoded and "�" in decoded:
        # This is an incomplete UTF-8 sequence - return a special marker
        return None, token_str, decoded
    elif decoded:
        # Successfully decoded
        return decoded, token_str, None
    else:
        # Decoding failed entirely
        return None, token_str, None


def detect_byte_level_bpe(tokenizer) -> bool:
    """
    Detect if this is a byte-level BPE tokenizer.

    Args:
        tokenizer: The tokenizer instance

    Returns:
        True if byte-level BPE, False otherwise
    """
    # Check for common indicators
    if hasattr(tokenizer, "backend_tokenizer") and hasattr(
        tokenizer.backend_tokenizer, "decoder"
    ):
        decoder_type = type(tokenizer.backend_tokenizer.decoder).__name__
        if "ByteLevel" in decoder_type:
            return True
    return False
