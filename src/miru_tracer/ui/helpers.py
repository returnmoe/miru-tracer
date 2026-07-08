"""Shared helpers for the Gradio UI tabs.

Everything that used to be duplicated (or triplicated) across
interactive_mode and logging_mode lives here once: chat JSON validation,
export-file management, probability tables, and visibility toggles.
"""

from __future__ import annotations

import html
import json
import tempfile
from pathlib import Path

import gradio as gr
import pandas as pd

from miru_tracer.core.logging_config import get_logger
from miru_tracer.core.sampling import SamplingParams
from miru_tracer.core.tracer import NextTokenDistribution

logger = get_logger(__name__)


def ui_sampling_params(strategy, temperature, top_k, top_p) -> SamplingParams:
    """Build SamplingParams from raw UI widget values (clamped, not raising)."""
    return SamplingParams(
        strategy=strategy,
        temperature=float(temperature),
        top_k=int(top_k or 0),
        top_p=min(max(float(top_p), 1e-3), 1.0),
    )

CHAT_ROLES = ("system", "user", "assistant")

DEFAULT_CHAT_JSON = json.dumps(
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me about the future of AI."},
    ],
    indent=2,
)

CHAT_MODE_HELP = (
    "Edit the JSON to add/remove/modify messages. "
    "Supported roles: system, user, assistant. "
    "End with an assistant message to prefill it — generation continues "
    "that message instead of starting a new turn. Reasoning templates "
    "(e.g. Qwen3) insert an empty `<think>` block before a plain-text "
    "prefill (so it already skips thinking). A COMPLETE think block in "
    "canonical form (`<think>\\n...\\n</think>\\n\\n` + answer) passes "
    "through; to continue mid-thought (unclosed `<think>`) use Raw mode — "
    "the template would prepend its own empty block."
)

RAW_MODE_PLACEHOLDER = (
    "<|im_start|>user\nTell me about Paris.<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\nParis is"
)

RAW_MODE_HELP = (
    "Tokenized as-is: no BOS or other special tokens are added, so template "
    "markers you type (e.g. `<|im_start|>`) are the only specials. Useful "
    "for hand-written chat templates and prefills."
)

THINKING_CHOICES = ("Template default", "Off (no thinking)", "Prefill thought…")

THINK_PREFILL_INFO = (
    "Inserted after the generation prompt as an UNCLOSED `<think>` + this "
    "text — the model continues the thought and closes it itself."
)


def thinking_key(ui_choice: str) -> str:
    """Map the Thinking radio label to the tracer's thinking mode."""
    choice = ui_choice or ""
    if choice.startswith("Off"):
        return "off"
    if choice.startswith("Prefill"):
        return "prefill"
    return "auto"


def toggle_think_prefill(ui_choice: str):
    """Show the thought textbox only for the prefill choice."""
    return gr.update(visible=thinking_key(ui_choice) == "prefill")


class ChatValidationError(ValueError):
    """The chat JSON the user entered is not a valid message list."""


def parse_chat_messages(text: str) -> list[dict[str, str]]:
    """Parse and validate the chat-messages JSON from the UI.

    Raises:
        ChatValidationError: with a user-presentable message.
    """
    try:
        messages = json.loads(text)
    except json.JSONDecodeError as e:
        raise ChatValidationError(f"Invalid JSON: {e}") from e
    if not isinstance(messages, list) or not messages:
        raise ChatValidationError("Chat messages must be a non-empty JSON array")
    for message in messages:
        if (
            not isinstance(message, dict)
            or "role" not in message
            or "content" not in message
        ):
            raise ChatValidationError(
                "Each message must have 'role' and 'content' fields"
            )
    return messages


GENERATION_MODES = ("Completion", "Chat", "Raw")


def toggle_mode_visibility(mode: str) -> tuple:
    """Show the completion, chat, or raw input group based on the mode radio."""
    return (
        gr.update(visible=mode == "Completion"),
        gr.update(visible=mode == "Chat"),
        gr.update(visible=mode == "Raw"),
    )


TEMPERATURE_GREEDY_INFO = "No effect with greedy strategy: it always picks the argmax."


def toggle_temperature(strategy: str):
    """Enable the temperature slider only when it has an effect (sampling)."""
    sampling = strategy == "sampling"
    return gr.update(
        interactive=sampling, info=None if sampling else TEMPERATURE_GREEDY_INFO
    )


def prob_mode_key(ui_choice: str) -> str:
    """Map the probability-mode radio label to the internal key."""
    return "raw" if "Raw" in (ui_choice or "") else "adjusted"


class ExportManager:
    """One stable export file per key, overwritten in place.

    Replaces the old per-call NamedTemporaryFile(delete=False) pattern, which
    leaked one file per prepared download — in the worst case one full-history
    JSON per generated token.
    """

    def __init__(self, prefix: str):
        self._prefix = prefix
        self._dir = Path(tempfile.mkdtemp(prefix=f"miru_{prefix}_"))

    def prepare(self, export_dict: dict, key: str = "current"):
        """Write the export JSON and return a DownloadButton update."""
        try:
            path = self._dir / f"{self._prefix}_{key}.json"
            with open(path, "w") as f:
                json.dump(export_dict, f, indent=2)
            return gr.update(value=str(path), interactive=True)
        except Exception as e:
            logger.error(f"Export error: {e}")
            return gr.update(interactive=False)

    def disabled(self):
        return gr.update(interactive=False)


def static_table_html(headers: list[str], rows: list[list]) -> str:
    """Server-rendered read-only table (see lens_views.py for why: Gradio 6's
    virtualized gr.Dataframe keeps the previous row count when its value
    changes shape, so growing results stay hidden until the user interacts
    with the table). Cells are escaped; the header sticks while the scroll
    container caps the height."""
    if not rows:
        return ""
    table_style = (
        "border:0; border-collapse:collapse; font-size:0.9em; line-height:1.5; "
        "font-family:var(--font-mono, monospace); white-space:pre;"
    )
    th = (
        'style="position:sticky; top:0; z-index:1; text-align:left; '
        "background:var(--body-background-fill, #fff); padding:5px 14px; "
        'font-weight:600; border:0;"'
    )
    td = (
        'style="padding:5px 14px; border:0; '
        'border-bottom:1px solid rgba(127,127,127,0.18);"'
    )
    header = "".join(f"<th {th}>{html.escape(str(h))}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td {td}>{html.escape(str(c))}</td>" for c in row) + "</tr>"
        for row in rows
    )
    return (
        '<div style="overflow:auto; max-height:75vh; max-width:100%; '
        'padding-bottom:6px; margin-top:6px;">'
        f'<table style="{table_style}"><tr>{header}</tr>{body}</table></div>'
    )


def build_prob_table(dist: NextTokenDistribution) -> pd.DataFrame:
    """Next-token candidates as a dataframe (probability stays numeric)."""
    rows = [
        [rank, token_id, text_raw, round(prob, 4)]
        for rank, (token_id, prob, text_raw) in enumerate(
            zip(dist.top_k_tokens, dist.top_k_probs, dist.top_k_texts_raw, strict=True)
        )
    ]
    return pd.DataFrame(rows, columns=["Rank", "Token ID", "Token", "Probability"])


def build_radio_choices(dist: NextTokenDistribution, limit: int = 10) -> list[tuple[str, str]]:
    """(label, token_id) choices for the next-token preview radio."""
    return [
        (f"Rank {rank}: {text_raw} (p={prob:.4f})", str(token_id))
        for rank, (token_id, prob, text_raw) in enumerate(
            zip(
                dist.top_k_tokens[:limit],
                dist.top_k_probs[:limit],
                dist.top_k_texts_raw[:limit],
                strict=True,
            )
        )
    ]
