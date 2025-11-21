"""LLMTracer class for interactive LLM debugging and generation tracing."""

from typing import Optional, List, Dict, Any
import torch
from datetime import datetime
from core.models import TokenStep
from core.tokenizer_utils import safe_decode_token
from core.logging_config import get_logger

logger = get_logger(__name__)


class LLMTracer:
    """Interactive LLM tracer with step-through and logging capabilities."""

    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self._warned_about_logits = False  # Track if we've warned about memory usage
        self._warned_about_deterministic_sampling = False  # Track sampling warnings
        self.warnings = []  # Collect warnings to display later
        self._cached_next_probs = (
            None  # Cache for next token probabilities (optimization)
        )
        self._stop_requested = False  # Flag to request early termination of generation

        # Detect if model supports chat templates
        self.has_chat_template = (
            hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None
        )

        logger.info(
            f"LLMTracer initialized (device={device}, chat_template={'available' if self.has_chat_template else 'not available'})"
        )

        self.reset()

    def reset(
        self,
        prompt: str = "",
        messages: Optional[List[Dict[str, str]]] = None,
        mode: str = "auto",
    ):
        """Reset the generation state.

        Args:
            prompt: Direct text prompt (for completion mode)
            messages: Chat messages in format [{"role": "system/user/assistant", "content": "..."}]
            mode: "completion", "chat", or "auto" (auto-detect based on inputs)
        """
        self.generated_tokens = []
        self.history: List[TokenStep] = []
        self.current_step = 0
        self.past_key_values = None  # KV cache for efficient generation
        self.warnings = []  # Reset warnings
        self._cached_next_probs = None  # Clear probability cache

        # Determine mode
        if mode == "auto":
            if messages is not None:
                mode = "chat"
            else:
                mode = "completion"

        self.mode = mode

        # Process input based on mode
        if mode == "chat" and messages is not None:
            if not self.has_chat_template:
                # Fallback: concatenate messages
                prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
                self.prompt = prompt
                self.input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(
                    self.device
                )
                logger.info(
                    f"Reset in chat mode (fallback, no template): {len(messages)} messages, prompt_length={len(prompt)} chars"
                )
            else:
                # Use chat template
                self.messages = messages
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                self.prompt = formatted_prompt
                self.input_ids = self.tokenizer.encode(
                    formatted_prompt, return_tensors="pt"
                ).to(self.device)
                logger.info(
                    f"Reset in chat mode: {len(messages)} messages, prompt_tokens={self.input_ids.shape[1]}"
                )
        elif prompt:
            self.prompt = prompt
            self.messages = None
            self.input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(
                self.device
            )
            logger.info(
                f"Reset in completion mode: prompt_length={len(prompt)} chars, prompt_tokens={self.input_ids.shape[1]}"
            )
        else:
            self.prompt = ""
            self.messages = None
            self.input_ids = None
            logger.debug("Reset with empty prompt")

    def get_next_token_probabilities(
        self,
        top_k: int = 10,
        temperature: float = 1.0,
        return_all_logits: bool = False,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """Get probabilities for the next token without generating.

        Uses KV cache for efficient computation - only processes the last token
        when past_key_values is available.

        Args:
            top_k: Number of top tokens to return
            temperature: Sampling temperature
            return_all_logits: Whether to return full vocabulary probabilities
            use_cache: If True, use cached results if available (optimization)

        Returns:
            Dictionary containing top_k_tokens, top_k_probs, top_k_texts, etc.
        """
        if self.input_ids is None:
            raise ValueError("No prompt set. Call reset(prompt) first.")

        # Use cached results if available and requested
        if use_cache and self._cached_next_probs is not None:
            cached = self._cached_next_probs
            # Verify cache parameters match
            if (
                cached.get("top_k") == top_k
                and cached.get("temperature") == temperature
                and cached.get("return_all_logits") == return_all_logits
            ):
                return cached["data"]

        # Guard against temperature <= 0
        if temperature <= 0:
            temperature = 1e-10  # Use very small value instead of 0

        with torch.inference_mode():
            # Use KV cache: only pass last token if we have past_key_values
            if self.past_key_values is not None:
                # Only process the last token
                model_input = self.input_ids[:, -1:]
                outputs = self.model(
                    model_input, past_key_values=self.past_key_values, use_cache=True
                )
            else:
                # First call: process full sequence and initialize cache
                outputs = self.model(self.input_ids, use_cache=True)

            # Get raw logits (pre-temperature)
            raw_logits = outputs.logits[:, -1, :]

            # Compute RAW probabilities (pre-temperature, model's true distribution)
            raw_probs = torch.softmax(raw_logits, dim=-1)

            # Compute ADJUSTED probabilities (post-temperature, for sampling)
            adjusted_logits = raw_logits / temperature
            adjusted_probs = torch.softmax(adjusted_logits, dim=-1)

        # Get top-k from ADJUSTED distribution (used for sampling/display by default)
        top_k_probs, top_k_indices = torch.topk(adjusted_probs[0], k=min(top_k, len(adjusted_probs[0])))

        top_k_tokens = top_k_indices.cpu().tolist()
        top_k_probs_list = top_k_probs.cpu().tolist()
        top_k_texts = [safe_decode_token(self.tokenizer, t) for t in top_k_tokens]

        # Extract raw probabilities for the same top-K tokens (for comparison)
        top_k_raw_probs_list = [raw_probs[0, token_id].item() for token_id in top_k_tokens]

        result = {
            "top_k_tokens": top_k_tokens,
            "top_k_probs": top_k_probs_list,  # Adjusted (post-temperature)
            "top_k_raw_probs": top_k_raw_probs_list,  # Raw (pre-temperature)
            "top_k_texts": top_k_texts,
            "all_probs": adjusted_probs[0].cpu() if return_all_logits else None,
            "past_key_values": outputs.past_key_values,  # Return updated cache
            "logits": adjusted_logits.clone(),  # Return logits for sampling reuse
        }

        return result

    def cache_next_probabilities(
        self, top_k: int = 10, temperature: float = 1.0, return_all_logits: bool = False
    ) -> Dict[str, Any]:
        """
        Compute and cache next token probabilities for efficient reuse.

        This is useful in interactive mode where we peek at next tokens at the end
        of one step and then need them again at the start of the next step.

        Args:
            top_k: Number of top tokens to return
            temperature: Sampling temperature
            return_all_logits: Whether to return full vocabulary probabilities

        Returns:
            Dictionary containing top_k_tokens, top_k_probs, top_k_texts, etc.
        """
        result = self.get_next_token_probabilities(
            top_k=top_k,
            temperature=temperature,
            return_all_logits=return_all_logits,
            use_cache=False,  # Force recomputation
        )

        # Store in cache with parameters
        self._cached_next_probs = {
            "top_k": top_k,
            "temperature": temperature,
            "return_all_logits": return_all_logits,
            "data": result,
        }

        return result

    def invalidate_probability_cache(self):
        """Clear the cached next token probabilities."""
        self._cached_next_probs = None

    def request_stop(self):
        """Request that ongoing generation stop at the next token."""
        self._stop_requested = True
        logger.info("Stop requested for ongoing generation")

    def clear_stop_flag(self):
        """Clear the stop request flag (call before starting new generation)."""
        self._stop_requested = False

    def step(
        self,
        token_id: Optional[int] = None,
        strategy: str = "greedy",
        top_k: int = 50,
        top_p: float = 0.9,
        temperature: float = 1.0,
        log_top_k: int = 10,
        log_all_logits: bool = False,
        suppress_warnings: bool = False,
    ) -> TokenStep:
        """Generate one token and record the step.

        Args:
            token_id: If provided, use this token instead of generating
            strategy: Generation strategy - "greedy" or "sampling"
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            temperature: Sampling temperature
            log_top_k: How many top tokens to log
            log_all_logits: Whether to log all vocabulary logits
                           (WARNING: uses ~600KB per step for 150k vocab)
            suppress_warnings: If True, collect warnings instead of printing them immediately
        """
        # Guard against temperature <= 0
        if temperature <= 0:
            temperature = 1e-10

        # Warn about memory usage for log_all_logits (once per session)
        if log_all_logits and not self._warned_about_logits:
            vocab_size = len(self.tokenizer)
            mem_per_step_mb = (vocab_size * 4) / 1024 / 1024  # 4 bytes per float32
            warning_msg = f"log_all_logits=True will use ~{mem_per_step_mb:.1f}MB per step (~{mem_per_step_mb * 100:.0f}MB for 100 steps)"
            if suppress_warnings:
                self.warnings.append(warning_msg)
            else:
                logger.warning(warning_msg)
            self._warned_about_logits = True

        # Get probabilities first
        prob_data = self.get_next_token_probabilities(
            top_k=log_top_k, temperature=temperature, return_all_logits=log_all_logits
        )

        # Update KV cache
        self.past_key_values = prob_data["past_key_values"]

        # Determine next token
        if token_id is not None:
            # User override
            next_token = token_id
            # Try to get adjusted probability from returned data
            if prob_data["all_probs"] is not None:
                token_prob = prob_data["all_probs"][token_id].item()
            elif token_id in prob_data["top_k_tokens"]:
                token_idx = prob_data["top_k_tokens"].index(token_id)
                token_prob = prob_data["top_k_probs"][token_idx]
            else:
                # Token not in top-k and we don't have all probs, need to compute it
                # Process full sequence from scratch for this rare case
                with torch.inference_mode():
                    outputs = self.model(self.input_ids, use_cache=False)
                    next_token_logits = outputs.logits[:, -1, :] / temperature
                    probs = torch.softmax(next_token_logits, dim=-1)
                    token_prob = probs[0, token_id].item()

            # Get raw probability for the same token
            if token_id in prob_data["top_k_tokens"]:
                token_idx = prob_data["top_k_tokens"].index(token_id)
                token_raw_prob = prob_data["top_k_raw_probs"][token_idx]
            else:
                # Need to compute raw probability
                with torch.inference_mode():
                    outputs = self.model(self.input_ids, use_cache=False)
                    raw_logits = outputs.logits[:, -1, :]
                    raw_probs = torch.softmax(raw_logits, dim=-1)
                    token_raw_prob = raw_probs[0, token_id].item()
        else:
            # Generate based on strategy
            if strategy == "greedy":
                next_token = prob_data["top_k_tokens"][0]
                token_prob = prob_data["top_k_probs"][0]
                token_raw_prob = prob_data["top_k_raw_probs"][0]
            elif strategy == "sampling":
                # Sample from distribution using the logits from get_next_token_probabilities()
                next_token_logits = prob_data["logits"].clone()

                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = (
                        next_token_logits
                        < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    )
                    next_token_logits[indices_to_remove] = float("-inf")

                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(
                        next_token_logits, descending=True
                    )
                    cumulative_probs = torch.cumsum(
                        torch.softmax(sorted_logits, dim=-1), dim=-1
                    )

                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the mask to the right to keep the first token above threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                        ..., :-1
                    ].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    # Scatter back to original indexing
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    next_token_logits[indices_to_remove] = float("-inf")

                # Compute final probabilities for sampling
                probs = torch.softmax(next_token_logits, dim=-1)

                # Check for deterministic sampling (warn once)
                if not self._warned_about_deterministic_sampling:
                    valid_probs = probs[probs > 0]
                    if len(valid_probs) == 1:
                        warning_msg = "Only 1 token has non-zero probability after filtering (deterministic sampling)"
                        if suppress_warnings:
                            self.warnings.append(warning_msg)
                        else:
                            logger.warning(warning_msg)
                        self._warned_about_deterministic_sampling = True

                # Sample from the distribution
                next_token = torch.multinomial(probs[0], num_samples=1).item()

                # Get adjusted probability from ORIGINAL distribution (before filtering)
                # to match what's displayed in the heatmap ranks
                if prob_data["all_probs"] is not None:
                    token_prob = prob_data["all_probs"][next_token].item()
                elif next_token in prob_data["top_k_tokens"]:
                    token_idx = prob_data["top_k_tokens"].index(next_token)
                    token_prob = prob_data["top_k_probs"][token_idx]
                else:
                    # Fallback: recompute from original logits if not in logged top-k
                    # Note: prob_data["logits"] already has temperature applied
                    original_probs = torch.softmax(prob_data["logits"], dim=-1)
                    token_prob = original_probs[0, next_token].item()

                # Get raw probability for the same token
                if next_token in prob_data["top_k_tokens"]:
                    token_idx = prob_data["top_k_tokens"].index(next_token)
                    token_raw_prob = prob_data["top_k_raw_probs"][token_idx]
                else:
                    # Fallback: recompute raw probability
                    with torch.inference_mode():
                        outputs = self.model(self.input_ids, use_cache=False)
                        raw_logits = outputs.logits[:, -1, :]
                        raw_probs = torch.softmax(raw_logits, dim=-1)
                        token_raw_prob = raw_probs[0, next_token].item()
            else:
                raise ValueError(
                    f"Unknown strategy: {strategy}. Use 'greedy' or 'sampling'."
                )

        # Record step
        token_text = safe_decode_token(self.tokenizer, next_token)
        step_data = TokenStep(
            step=self.current_step,
            token_id=next_token,
            token_text=(
                token_text[0] if token_text[0] else token_text[1]
            ),  # Use decoded or raw token
            probability=token_prob,  # Adjusted (post-temperature) probability
            top_k_tokens=prob_data["top_k_tokens"],
            top_k_probs=prob_data["top_k_probs"],  # Adjusted (post-temperature) probabilities
            top_k_texts=[t[0] if t[0] else t[1] for t in prob_data["top_k_texts"]],
            raw_probability=token_raw_prob,  # Raw (pre-temperature) probability
            top_k_raw_probs=prob_data["top_k_raw_probs"],  # Raw (pre-temperature) probabilities
            all_logits=(
                prob_data["all_probs"].tolist()
                if prob_data["all_probs"] is not None
                else None
            ),
            token_text_raw=token_text[1],  # Raw token representation (visible \n, \t, etc.)
            top_k_texts_raw=[t[1] for t in prob_data["top_k_texts"]],  # Raw representations for top-k
        )

        # Update state
        self.generated_tokens.append(next_token)
        self.history.append(step_data)
        self.input_ids = torch.cat(
            [self.input_ids, torch.tensor([[next_token]]).to(self.device)], dim=1
        )
        self.current_step += 1

        # Invalidate probability cache since state changed
        self.invalidate_probability_cache()

        logger.debug(
            f"Step {self.current_step}: token_id={next_token}, text={repr(token_text)}, prob={token_prob:.4f}"
        )

        return step_data

    def generate(
        self,
        max_new_tokens: int = 50,
        strategy: str = "greedy",
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        log_top_k: int = 10,
        log_all_logits: bool = False,
        show_progress: bool = True,
        stop_token_id: Optional[int] = None,
    ) -> str:
        """Generate multiple tokens automatically.

        Args:
            stop_token_id: Optional token ID to stop generation at (ignores max_new_tokens)
        """
        for i in range(max_new_tokens):
            step_data = self.step(
                strategy=strategy,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                log_top_k=log_top_k,
                log_all_logits=log_all_logits,
                suppress_warnings=True,
            )

            # Stop on custom stop token (higher priority than EOS)
            if stop_token_id is not None and step_data.token_id == stop_token_id:
                break

            # Stop on EOS
            if step_data.token_id == self.tokenizer.eos_token_id:
                logger.debug(f"EOS token reached at step {i+1}")
                break

        logger.info(f"Generation complete: {len(self.history)} tokens generated")
        return self.get_generated_text()

    def generate_stream(
        self,
        max_new_tokens: int = 50,
        strategy: str = "greedy",
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        log_top_k: int = 10,
        log_all_logits: bool = False,
        stop_token_id: Optional[int] = None,
    ):
        """Generate tokens one at a time, yielding after each step.

        Yields tuples of (current_text, step_number, is_complete)

        Args:
            stop_token_id: Optional token ID to stop generation at (ignores max_new_tokens)
        """
        for i in range(max_new_tokens):
            # Check if stop was requested (highest priority)
            if self._stop_requested:
                current_text = self.get_full_text()
                yield (current_text, i, True)
                logger.info(f"Generation stopped by user request after {i} tokens")
                break

            step_data = self.step(
                strategy=strategy,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                log_top_k=log_top_k,
                log_all_logits=log_all_logits,
                suppress_warnings=True,
            )

            # Get current full text
            current_text = self.get_full_text()

            # Check if we hit custom stop token (higher priority than EOS)
            if stop_token_id is not None and step_data.token_id == stop_token_id:
                yield (current_text, i + 1, True)
                break

            # Check if we hit EOS
            if step_data.token_id == self.tokenizer.eos_token_id:
                yield (current_text, i + 1, True)
                break
            else:
                yield (current_text, i + 1, False)

    def get_generated_text(self) -> str:
        """Get the currently generated text."""
        return self.tokenizer.decode(self.generated_tokens)

    def get_full_text(self) -> str:
        """Get prompt + generated text."""
        return self.tokenizer.decode(self.input_ids[0])

    def get_warnings(self) -> List[str]:
        """Get collected warnings."""
        return self.warnings.copy()

    def undo_step(self) -> bool:
        """Undo the last generated token.

        Returns:
            True if successful, False if no steps to undo
        """
        if not self.history or not self.generated_tokens:
            logger.debug("Undo failed: no steps to undo")
            return False

        # Remove last token from history and generated tokens
        self.history.pop()
        self.generated_tokens.pop()

        # Rebuild input_ids from prompt + remaining generated tokens
        if self.messages is not None:
            # Chat mode - reapply template
            if self.has_chat_template:
                formatted_prompt = self.tokenizer.apply_chat_template(
                    self.messages, tokenize=False, add_generation_prompt=True
                )
                prompt_ids = self.tokenizer.encode(
                    formatted_prompt, return_tensors="pt"
                ).to(self.device)
            else:
                prompt = "\n".join(
                    [f"{m['role']}: {m['content']}" for m in self.messages]
                )
                prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(
                    self.device
                )
        else:
            # Completion mode
            prompt_ids = self.tokenizer.encode(self.prompt, return_tensors="pt").to(
                self.device
            )

        # Rebuild input_ids with remaining tokens
        if self.generated_tokens:
            generated_ids = torch.tensor([self.generated_tokens]).to(self.device)
            self.input_ids = torch.cat([prompt_ids, generated_ids], dim=1)
        else:
            self.input_ids = prompt_ids

        # Reset KV cache (will be rebuilt on next step)
        self.past_key_values = None

        # Invalidate probability cache since state changed
        self.invalidate_probability_cache()

        # Decrement step counter
        self.current_step -= 1

        logger.debug(f"Undo successful: current_step={self.current_step}")

        return True

    def validate_state(self) -> tuple[bool, Optional[str]]:
        """
        Validate internal state consistency.

        Returns:
            (is_valid, error_message): True if state is valid, False with error message otherwise
        """
        try:
            # Check history length matches generated tokens
            if len(self.history) != len(self.generated_tokens):
                error_msg = f"History length ({len(self.history)}) doesn't match generated tokens ({len(self.generated_tokens)})"
                logger.warning(f"State validation failed: {error_msg}")
                return False, error_msg

            # Check current step matches history
            if self.current_step != len(self.history):
                error_msg = f"Current step ({self.current_step}) doesn't match history length ({len(self.history)})"
                logger.warning(f"State validation failed: {error_msg}")
                return False, error_msg

            # Check input_ids consistency
            if self.input_ids is not None:
                # Recompute expected length
                if self.messages is not None and self.has_chat_template:
                    formatted_prompt = self.tokenizer.apply_chat_template(
                        self.messages, tokenize=False, add_generation_prompt=True
                    )
                    prompt_ids = self.tokenizer.encode(
                        formatted_prompt, return_tensors="pt"
                    )
                elif self.messages is not None:
                    prompt = "\n".join(
                        [f"{m['role']}: {m['content']}" for m in self.messages]
                    )
                    prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt")
                else:
                    prompt_ids = self.tokenizer.encode(self.prompt, return_tensors="pt")

                expected_length = prompt_ids.shape[1] + len(self.generated_tokens)
                actual_length = self.input_ids.shape[1]

                if expected_length != actual_length:
                    error_msg = f"input_ids length ({actual_length}) doesn't match expected ({expected_length})"
                    logger.warning(f"State validation failed: {error_msg}")
                    return False, error_msg

            logger.debug("State validation passed")
            return True, None

        except Exception as e:
            error_msg = f"Validation error: {str(e)}"
            logger.error(error_msg)
            return False, error_msg

    def get_state_info(self) -> Dict[str, Any]:
        """
        Get detailed state information for debugging.

        Returns:
            Dictionary with state details
        """
        return {
            "mode": self.mode,
            "current_step": self.current_step,
            "history_length": len(self.history),
            "generated_tokens_count": len(self.generated_tokens),
            "input_ids_length": (
                self.input_ids.shape[1] if self.input_ids is not None else None
            ),
            "has_kv_cache": self.past_key_values is not None,
            "has_cached_probs": self._cached_next_probs is not None,
            "warnings_count": len(self.warnings),
        }

    def export_to_dict(self) -> dict:
        """
        Export generation history as a dictionary.

        Returns:
            Dictionary containing generation data
        """
        return {
            "mode": self.mode,
            "prompt": self.prompt,
            "messages": self.messages if self.mode == "chat" else None,
            "generated_text": self.get_generated_text(),
            "full_text": self.get_full_text(),
            "timestamp": datetime.now().isoformat(),
            "num_steps": len(self.history),
            "history": [step.to_dict() for step in self.history],
        }

    def export_history(self, filename: str):
        """Export generation history to JSON file."""
        import json

        data = self.export_to_dict()

        with open(filename, "w") as f:
            json.dump(data, f, indent=2)

        return f"Exported history to {filename}"
