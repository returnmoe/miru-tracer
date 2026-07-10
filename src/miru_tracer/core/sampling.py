"""Pure tensor post-processing for next-token distributions.

Everything in this module operates on raw (pre-temperature) logits and has no
model or cache dependencies, which keeps it trivially unit-testable. The
LLMTracer produces one raw logits tensor per sequence position; temperature,
top-k and top-p are cheap tensor transforms applied on demand, so changing any
of them never requires another forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

# Floor for temperature: low enough that softmax is effectively an argmax,
# high enough that logits / temperature cannot overflow float32.
MIN_TEMPERATURE = 1e-6

STRATEGIES = ("greedy", "sampling")


@dataclass(frozen=True)
class SamplingParams:
    """Validated, immutable sampling configuration."""

    strategy: str = "greedy"
    temperature: float = 1.0
    top_k: int = 50  # 0 disables top-k filtering
    top_p: float = 1.0  # 1.0 disables nucleus filtering

    def __post_init__(self):
        if self.strategy not in STRATEGIES:
            raise ValueError(
                f"Unknown strategy: {self.strategy!r}. Use one of {STRATEGIES}."
            )
        object.__setattr__(
            self, "temperature", max(float(self.temperature), MIN_TEMPERATURE)
        )
        if self.top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {self.top_k}")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON representation stored in generation logs."""
        return {
            "strategy": self.strategy,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
        }


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Scale logits by temperature (clamped away from zero)."""
    return logits / max(float(temperature), MIN_TEMPERATURE)


def filter_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Mask everything outside the k highest logits to -inf.

    k <= 0 or k >= vocab size disables filtering. Returns a new tensor.
    """
    vocab_size = logits.shape[-1]
    if k <= 0 or k >= vocab_size:
        return logits.clone()
    threshold = torch.topk(logits, k).values[..., -1, None]
    return logits.masked_fill(logits < threshold, float("-inf"))


def filter_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus filtering: keep the smallest set of tokens with cumulative
    probability >= p (always keeping at least the top token). Returns a new tensor.
    """
    if p >= 1.0:
        return logits.clone()
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    # Tokens whose cumulative probability (excluding themselves) already
    # exceeds p are removed; shifting right keeps the first token past the
    # threshold in the nucleus.
    remove = cumulative > p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    remove_original = remove.scatter(-1, sorted_indices, remove)
    return logits.masked_fill(remove_original, float("-inf"))


def select_token(
    raw_logits: torch.Tensor,
    params: SamplingParams,
    generator: torch.Generator | None = None,
) -> int:
    """Pick the next token id from raw logits according to params."""
    if params.strategy == "greedy":
        return int(torch.argmax(raw_logits).item())

    logits = apply_temperature(raw_logits, params.temperature)
    logits = filter_top_k(logits, params.top_k)
    logits = filter_top_p(logits, params.top_p)
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1, generator=generator).item())


def entropy(probs: torch.Tensor) -> float:
    """Shannon entropy in nats of a probability vector (zeros ignored)."""
    positive = probs[probs > 0]
    return float(-(positive * positive.log()).sum().item())
