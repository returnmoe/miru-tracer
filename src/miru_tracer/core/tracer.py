"""LLMTracer: step-through LLM generation with a single source of truth.

Design
------
The tracer separates the expensive part (model forward passes) from the cheap
part (turning logits into distributions):

- ``_next_raw_logits()`` is the only place the model is called. It maintains
  the KV cache with an explicit length invariant — ``cache length == seq_len``
  immediately after any forward — and *recovers* (crop or recompute) when the
  cache disagrees with ``input_ids`` instead of silently corrupting state.
  The raw (pre-temperature) logits for the current position are memoized in a
  single slot keyed by sequence length.
- ``peek()`` and ``step()`` derive everything else (temperature scaling,
  top-k/top-p, probabilities) from those raw logits via
  :mod:`miru_tracer.core.sampling`. Peeking repeatedly with different display
  parameters therefore never triggers another forward pass.

Undo crops the KV cache (``DynamicCache.crop``) and truncates ``input_ids``;
the prompt is never re-tokenized after ``reset()``.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from typing import Any

import torch

from miru_tracer.core.interventions import (
    Intervention,
    InterventionSet,
    apply_interventions,
)
from miru_tracer.core.logging_config import get_logger
from miru_tracer.core.sampling import (
    SamplingParams,
    apply_temperature,
    select_token,
)
from miru_tracer.core.schema import SCHEMA_VERSION, TokenStep
from miru_tracer.core.tokenizer_utils import safe_decode_token

logger = get_logger(__name__)


class CacheDesyncError(RuntimeError):
    """The KV cache length disagrees with input_ids after a forward pass."""


def _collect_eos_ids(tokenizer, model) -> frozenset[int]:
    """Union of EOS token ids from the tokenizer and the model's generation config.

    Either source may expose ``eos_token_id`` as None, an int, or a list of ints
    (e.g. Qwen chat models declare both ``<|im_end|>`` and ``<|endoftext|>``).
    """
    ids: set[int] = set()
    sources = [
        getattr(tokenizer, "eos_token_id", None),
        getattr(getattr(model, "generation_config", None), "eos_token_id", None),
    ]
    for source in sources:
        if source is None:
            continue
        if isinstance(source, int):
            ids.add(source)
        else:
            ids.update(int(token_id) for token_id in source)
    return frozenset(ids)


class NextTokenDistribution:
    """The distribution over the next token at a fixed position.

    Wraps the raw logits so probabilities under any temperature can be derived
    without touching the model.
    """

    def __init__(self, raw_logits: torch.Tensor, temperature: float, top_k: int, tokenizer):
        self.raw_logits = raw_logits
        self.temperature = temperature

        raw_probs = torch.softmax(raw_logits, dim=-1)
        adjusted_probs = torch.softmax(apply_temperature(raw_logits, temperature), dim=-1)
        k = min(top_k, raw_logits.shape[-1])
        top_probs, top_indices = torch.topk(adjusted_probs, k)

        self.top_k_tokens: list[int] = top_indices.cpu().tolist()
        self.top_k_probs: list[float] = top_probs.cpu().tolist()
        self.top_k_raw_probs: list[float] = raw_probs[top_indices].cpu().tolist()
        decoded = [safe_decode_token(tokenizer, t) for t in self.top_k_tokens]
        self.top_k_texts: list[str] = [d[0] if d[0] else d[1] for d in decoded]
        self.top_k_texts_raw: list[str] = [d[1] for d in decoded]

    def prob_of(self, token_id: int) -> tuple[float, float]:
        """Return (adjusted, raw) probability of a token — no model call."""
        raw = torch.softmax(self.raw_logits, dim=-1)[token_id].item()
        adjusted = torch.softmax(
            apply_temperature(self.raw_logits, self.temperature), dim=-1
        )[token_id].item()
        return float(adjusted), float(raw)

    def full_probs(self) -> list[float]:
        """Full-vocabulary adjusted probabilities (large: ~vocab_size floats)."""
        return torch.softmax(
            apply_temperature(self.raw_logits, self.temperature), dim=-1
        ).cpu().tolist()


class LLMTracer:
    """Interactive LLM tracer with step-through and logging capabilities."""

    def __init__(self, model, tokenizer, device: str = "cpu", seed: int | None = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.eos_ids = _collect_eos_ids(tokenizer, model)
        self._generator: torch.Generator | None = None
        if seed is not None:
            self._generator = torch.Generator(device=device).manual_seed(seed)

        self.has_chat_template = (
            hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None
        )
        # Activation interventions (steer/ablate/swap). Persist across reset():
        # they are properties of how the model runs, not of the prompt.
        self._intervention_set: InterventionSet | None = None

        logger.info(
            f"LLMTracer initialized (device={device}, eos_ids={sorted(self.eos_ids)}, "
            f"chat_template={'available' if self.has_chat_template else 'not available'})"
        )

        self.reset()

    # ------------------------------------------------------------------ state

    def reset(
        self,
        prompt: str = "",
        messages: list[dict[str, str]] | None = None,
        mode: str = "auto",
        thinking: str = "auto",
        think_prefill: str = "",
    ) -> None:
        """Reset generation state with a new prompt or chat messages.

        Args:
            prompt: Direct text prompt (completion and raw modes).
            messages: Chat messages [{"role": ..., "content": ...}, ...].
                If the final message is from the assistant, it is treated as a
                prefill and generation continues it (no generation prompt is
                appended).
            mode: "completion", "chat", "raw", or "auto" (detect from inputs).
                Raw mode tokenizes the prompt without auto-added special
                tokens, so template markers typed in the text (e.g.
                ``<|im_start|>``) are the only specials in the sequence.
            thinking: Chat-mode reasoning control. "auto" leaves the template
                alone; "off" renders with ``enable_thinking=False`` (Qwen3
                emits an empty ``<think>`` block; a no-op on templates without
                the switch); "prefill" opens an UNCLOSED ``<think>`` seeded
                with ``think_prefill`` after the generation prompt — the model
                continues that thought. Incompatible with an assistant-prefill
                final message (raises ValueError).
            think_prefill: The thought text for ``thinking="prefill"``.
        """
        self.history: list[TokenStep] = []
        self._kv = None
        self._logits_slot: tuple[int, torch.Tensor] | None = None
        self._stop_requested = False
        self.warnings: list[str] = []

        if mode == "auto":
            mode = "chat" if messages is not None else "completion"
        self.mode = mode
        self.messages = messages if mode == "chat" else None

        if mode == "chat" and messages is not None:
            if self.has_chat_template:
                # A trailing assistant message is a prefill: continue it
                # instead of opening a fresh assistant turn.
                prefill = messages[-1].get("role") == "assistant"
                if thinking != "auto" and prefill:
                    raise ValueError(
                        "Thinking control conflicts with an assistant-prefill "
                        "final message — put the thoughts in the prefill "
                        "instead, or drop the assistant message."
                    )
                if thinking == "off":
                    text = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                elif thinking == "prefill":
                    # Templates cannot express an unclosed <think>; append it
                    # to the rendered chat so the model continues the thought.
                    text = (
                        self.tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                        + "<think>\n"
                        + think_prefill
                    )
                else:
                    text = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=not prefill,
                        continue_final_message=prefill,
                    )
            else:
                text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
            self.prompt = text
        else:
            self.prompt = prompt

        if self.prompt:
            self.input_ids = self.tokenizer.encode(
                self.prompt,
                return_tensors="pt",
                add_special_tokens=(mode != "raw"),
            ).to(self.device)
            logger.info(
                f"Reset in {mode} mode: prompt_tokens={self.input_ids.shape[1]}"
            )
        else:
            self.input_ids = None
            logger.debug("Reset with empty prompt")

        self._prompt_len = self.input_ids.shape[1] if self.input_ids is not None else 0

    @property
    def seq_len(self) -> int:
        """Current total sequence length (prompt + generated tokens)."""
        return 0 if self.input_ids is None else int(self.input_ids.shape[1])

    @property
    def current_step(self) -> int:
        """Number of generated tokens (derived, cannot desync)."""
        return len(self.history)

    @property
    def generated_tokens(self) -> list[int]:
        return [step.token_id for step in self.history]

    def is_eos(self, token_id: int) -> bool:
        return token_id in self.eos_ids

    def request_stop(self) -> None:
        """Request that ongoing generation stop after the current token."""
        self._stop_requested = True
        logger.info("Stop requested for ongoing generation")

    def clear_stop_flag(self) -> None:
        self._stop_requested = False

    # ---------------------------------------------------------- interventions

    @property
    def interventions(self) -> list[Intervention]:
        if self._intervention_set is None:
            return []
        return list(self._intervention_set.interventions)

    def set_interventions(
        self, interventions: list[Intervention] | None, jlens=None
    ) -> None:
        """Replace the active intervention set.

        Invalidates the KV cache and logits memo — cached values were computed
        under the previous intervention state and would silently poison
        subsequent forwards otherwise.
        """
        if interventions:
            self._intervention_set = InterventionSet(
                interventions, self.model, jlens=jlens
            )
        else:
            self._intervention_set = None
        self._invalidate_kv()
        logger.info(
            f"Interventions set: {[iv.describe(self.tokenizer) for iv in (interventions or [])]}"
        )

    # ---------------------------------------------------------------- forward

    def _cache_len(self) -> int:
        return int(self._kv.get_seq_length()) if self._kv is not None else 0

    def _invalidate_kv(self) -> None:
        self._kv = None
        self._logits_slot = None

    def _next_raw_logits(self) -> torch.Tensor:
        """Raw (pre-temperature) logits for the next token.

        The single enforcement point for the cache invariant: on return, the
        KV cache covers exactly ``seq_len`` positions and the memo slot holds
        the logits for this position. Never call the model anywhere else.
        """
        if self.input_ids is None:
            raise ValueError("No prompt set. Call reset(prompt) first.")

        seq_len = self.seq_len
        if self._logits_slot is not None and self._logits_slot[0] == seq_len:
            return self._logits_slot[1]

        cache_len = self._cache_len()
        if cache_len >= seq_len:
            # The cache is ahead of input_ids (interrupted operation, undo
            # without crop support, external mutation). Recover instead of
            # assuming: trim to just before the last token, or start over.
            logger.warning(
                f"KV cache length {cache_len} >= sequence length {seq_len}; recovering"
            )
            if self._kv is not None and hasattr(self._kv, "crop"):
                self._kv.crop(seq_len - 1)
                cache_len = self._cache_len()
            else:
                self._kv = None
                cache_len = 0

        with torch.inference_mode(), apply_interventions(
            self.model, self._intervention_set
        ):
            if self._kv is None:
                outputs = self.model(self.input_ids, use_cache=True)
            else:
                outputs = self.model(
                    self.input_ids[:, cache_len:],
                    past_key_values=self._kv,
                    use_cache=True,
                    cache_position=torch.arange(
                        cache_len, seq_len, device=self.input_ids.device
                    ),
                )
            self._kv = outputs.past_key_values

        actual = self._cache_len()
        if actual != seq_len:
            self._invalidate_kv()
            raise CacheDesyncError(
                f"KV cache covers {actual} positions after forward, expected {seq_len}"
            )

        raw_logits = outputs.logits[0, -1, :].float().clone()
        self._logits_slot = (seq_len, raw_logits)
        return raw_logits

    # ------------------------------------------------------------- inspection

    def peek(
        self,
        top_k: int = 10,
        temperature: float = 1.0,
    ) -> NextTokenDistribution:
        """Inspect the next-token distribution without generating.

        Free to call repeatedly with different ``top_k``/``temperature`` —
        only the first call at a given position runs the model.
        """
        raw_logits = self._next_raw_logits()
        return NextTokenDistribution(raw_logits, temperature, top_k, self.tokenizer)

    # ------------------------------------------------------------- generation

    def step(
        self,
        params: SamplingParams | None = None,
        token_id: int | None = None,
        log_top_k: int = 10,
        log_full_probs: bool = False,
    ) -> TokenStep:
        """Generate (or force) one token and record the step.

        Args:
            params: Sampling configuration (defaults to greedy).
            token_id: If provided, use this token instead of sampling.
            log_top_k: How many top candidates to record in the step.
            log_full_probs: Record the full-vocabulary distribution
                (~4 bytes x vocab_size per step; large).
        """
        params = params or SamplingParams()
        dist = self.peek(top_k=log_top_k, temperature=params.temperature)

        if token_id is None:
            token_id = select_token(dist.raw_logits, params, self._generator)
        adjusted_prob, raw_prob = dist.prob_of(token_id)

        decoded, raw_text, _ = safe_decode_token(self.tokenizer, token_id)
        step_data = TokenStep(
            step=len(self.history),
            token_id=token_id,
            token_text=decoded if decoded else raw_text,
            probability=adjusted_prob,
            top_k_tokens=dist.top_k_tokens,
            top_k_probs=dist.top_k_probs,
            top_k_texts=dist.top_k_texts,
            raw_probability=raw_prob,
            top_k_raw_probs=dist.top_k_raw_probs,
            full_probs=dist.full_probs() if log_full_probs else None,
            token_text_raw=raw_text,
            top_k_texts_raw=dist.top_k_texts_raw,
        )

        self.history.append(step_data)
        self.input_ids = torch.cat(
            [
                self.input_ids,
                torch.tensor([[token_id]], device=self.input_ids.device),
            ],
            dim=1,
        )
        # The memo slot intentionally survives: it is keyed by sequence length,
        # so it is simply stale now — and becomes valid again after undo(1).

        logger.debug(
            f"Step {len(self.history)}: token_id={token_id}, "
            f"text={step_data.token_text!r}, prob={adjusted_prob:.4f}"
        )
        return step_data

    def generate_stream(
        self,
        max_new_tokens: int = 50,
        params: SamplingParams | None = None,
        log_top_k: int = 10,
        log_full_probs: bool = False,
        stop_at_eos: bool = True,
    ) -> Iterator[TokenStep]:
        """Generate tokens one at a time, yielding each recorded step.

        Stops at ``max_new_tokens``, on EOS (when ``stop_at_eos``), or when
        ``request_stop()`` was called.
        """
        if max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be at least 1, got {max_new_tokens}")
        self.clear_stop_flag()
        for _ in range(max_new_tokens):
            step_data = self.step(
                params=params, log_top_k=log_top_k, log_full_probs=log_full_probs
            )
            yield step_data
            if stop_at_eos and self.is_eos(step_data.token_id):
                logger.debug(f"EOS token reached at step {len(self.history)}")
                break
            if self._stop_requested:
                logger.info(f"Generation stopped by request at step {len(self.history)}")
                break

    def generate(
        self,
        max_new_tokens: int = 50,
        params: SamplingParams | None = None,
        log_top_k: int = 10,
        log_full_probs: bool = False,
        stop_at_eos: bool = True,
    ) -> str:
        """Generate multiple tokens and return the generated text."""
        for _ in self.generate_stream(
            max_new_tokens=max_new_tokens,
            params=params,
            log_top_k=log_top_k,
            log_full_probs=log_full_probs,
            stop_at_eos=stop_at_eos,
        ):
            pass
        logger.info(f"Generation complete: {len(self.history)} tokens generated")
        return self.get_generated_text()

    # ------------------------------------------------------------------- undo

    def undo(self, n: int = 1) -> bool:
        """Undo the last ``n`` generated tokens.

        Returns False (without changing state) if fewer than ``n`` steps exist.
        """
        if n < 1 or len(self.history) < n:
            logger.debug(f"Undo failed: requested {n}, have {len(self.history)}")
            return False

        del self.history[-n:]
        self.input_ids = self.input_ids[:, :-n]

        if self._kv is not None:
            if hasattr(self._kv, "crop") and self.seq_len > 0:
                # Trim to just before the (new) last position so the next
                # forward recomputes exactly one token.
                self._kv.crop(self.seq_len - 1)
            else:
                self._kv = None
        # The memo slot stays: if it matches the new seq_len it is valid again.

        logger.debug(f"Undo({n}) successful: steps={len(self.history)}")
        return True

    def goto_step(self, step: int) -> bool:
        """Rewind so that exactly ``step`` generated tokens remain."""
        if step < 0 or step > len(self.history):
            return False
        if step == len(self.history):
            return True
        return self.undo(len(self.history) - step)

    # ------------------------------------------------------------------- text

    def get_generated_text(self) -> str:
        return self.tokenizer.decode(self.generated_tokens)

    def get_full_text(self) -> str:
        if self.input_ids is None:
            return ""
        return self.tokenizer.decode(self.input_ids[0])

    def get_warnings(self) -> list[str]:
        return self.warnings.copy()

    # ------------------------------------------------------------ diagnostics

    def validate_state(self) -> tuple[bool, str | None]:
        """Cheap internal consistency checks (no re-tokenization)."""
        expected_len = self._prompt_len + len(self.history)
        if self.seq_len != expected_len:
            return False, (
                f"input_ids length ({self.seq_len}) != prompt + steps ({expected_len})"
            )
        if self._cache_len() > self.seq_len:
            return False, (
                f"KV cache ({self._cache_len()}) longer than sequence ({self.seq_len})"
            )
        return True, None

    def get_state_info(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "current_step": self.current_step,
            "history_length": len(self.history),
            "generated_tokens_count": len(self.history),
            "input_ids_length": self.seq_len if self.input_ids is not None else None,
            "kv_cache_length": self._cache_len(),
            "has_cached_logits": self._logits_slot is not None
            and self._logits_slot[0] == self.seq_len,
            "warnings_count": len(self.warnings),
        }

    # ----------------------------------------------------------------- export

    def export_to_dict(self, params: SamplingParams | None = None) -> dict:
        """Export generation history (schema v2)."""
        return {
            "schema_version": SCHEMA_VERSION,
            "mode": self.mode,
            "prompt": self.prompt,
            "messages": self.messages,
            "generated_text": self.get_generated_text(),
            "full_text": self.get_full_text(),
            "timestamp": datetime.now().isoformat(),
            "num_steps": len(self.history),
            "sampling_params": (
                {
                    "strategy": params.strategy,
                    "temperature": params.temperature,
                    "top_k": params.top_k,
                    "top_p": params.top_p,
                }
                if params is not None
                else {}
            ),
            "history": [step.to_dict() for step in self.history],
        }

    def export_history(self, filename: str) -> str:
        """Export generation history to a JSON file."""
        import json

        with open(filename, "w") as f:
            json.dump(self.export_to_dict(), f, indent=2)
        return f"Exported history to {filename}"
