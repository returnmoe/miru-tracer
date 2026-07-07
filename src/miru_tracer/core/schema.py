"""Data model and JSON schema for generation logs.

Export schema history:

- **v1** (no ``schema_version`` field): written by earlier releases. The full
  probability vector, when logged, was stored under the misleading name
  ``all_logits`` (the values were softmax probabilities, not logits). Early v1
  logs also lack the raw/adjusted dual-probability fields.
- **v2** (``schema_version: 2``): ``all_logits`` renamed to ``full_probs``,
  plus a top-level ``sampling_params`` object recording how the text was
  generated.

``parse_log`` accepts both, so logs exported by any prior version of Miru
Tracer remain loadable in the Log Analysis tab.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

SCHEMA_VERSION = 2


@dataclass
class TokenStep:
    """Records information about a single token generation step."""

    step: int
    token_id: int
    token_text: str
    probability: float  # Post-temperature (adjusted) probability
    top_k_tokens: list[int]
    top_k_probs: list[float]  # Post-temperature (adjusted) probabilities
    top_k_texts: list[str]
    raw_probability: float = 0.0  # Pre-temperature (raw model) probability
    top_k_raw_probs: Optional[list[float]] = None  # Pre-temperature probabilities
    full_probs: Optional[list[float]] = None  # Full-vocabulary probabilities (optional)
    token_text_raw: Optional[str] = None  # Raw token representation (visible \n, \t, ...)
    top_k_texts_raw: Optional[list[str]] = None  # Raw representations for top-k tokens

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TokenStep":
        """Build a TokenStep from a v1 or v2 log entry."""
        return cls(
            step=data["step"],
            token_id=data["token_id"],
            token_text=data["token_text"],
            probability=data["probability"],
            top_k_tokens=data["top_k_tokens"],
            top_k_probs=data["top_k_probs"],
            top_k_texts=data["top_k_texts"],
            raw_probability=data.get("raw_probability", data["probability"]),
            top_k_raw_probs=data.get("top_k_raw_probs", data["top_k_probs"]),
            # v2 name first, then the misnamed v1 field (values were always
            # probabilities in both versions).
            full_probs=data.get("full_probs", data.get("all_logits")),
            token_text_raw=data.get("token_text_raw", data["token_text"]),
            top_k_texts_raw=data.get("top_k_texts_raw", data["top_k_texts"]),
        )


@dataclass
class GenerationLog:
    """A parsed generation log (any schema version)."""

    mode: str = "unknown"
    prompt: str = ""
    messages: Optional[list[dict[str, str]]] = None
    generated_text: str = ""
    full_text: str = ""
    timestamp: str = "unknown"
    history: list[TokenStep] = field(default_factory=list)
    sampling_params: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    @property
    def num_steps(self) -> int:
        return len(self.history)

    @property
    def temperature(self) -> float:
        return float(self.sampling_params.get("temperature", 1.0))


def parse_log(data: dict[str, Any]) -> GenerationLog:
    """Parse an exported generation log dict (v1 or v2) into a GenerationLog.

    Raises:
        ValueError: if the data is not a recognizable Miru Tracer log.
    """
    if not isinstance(data, dict):
        raise ValueError("Log file must contain a JSON object.")
    history_raw = data.get("history")
    if not isinstance(history_raw, list):
        raise ValueError("Log file has no 'history' list — not a Miru Tracer log.")

    try:
        history = [TokenStep.from_dict(step) for step in history_raw]
    except (KeyError, TypeError) as e:
        raise ValueError(f"Malformed history entry in log file: {e}") from e

    sampling_params = data.get("sampling_params") or {}
    # v1 logs may carry a bare top-level temperature.
    if "temperature" not in sampling_params and "temperature" in data:
        sampling_params["temperature"] = data["temperature"]

    return GenerationLog(
        mode=data.get("mode") or "unknown",
        prompt=data.get("prompt") or "",
        messages=data.get("messages"),
        generated_text=data.get("generated_text") or "",
        full_text=data.get("full_text") or "",
        timestamp=data.get("timestamp") or "unknown",
        history=history,
        sampling_params=sampling_params,
        schema_version=int(data.get("schema_version", 1)),
    )
