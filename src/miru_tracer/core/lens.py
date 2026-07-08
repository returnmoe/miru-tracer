"""Lens readout engine: logit lens and Jacobian lens over a token sequence.

Built on the vendored jlens reference implementation
(:mod:`miru_tracer.core._jlens`, Apache-2.0). The Jacobian lens reads out
``unembed(J_l @ h_l)`` where ``J_l`` is a fitted per-layer transport matrix;
the logit lens is the ``J_l = I`` special case; "diff" surfaces the tokens the
Jacobian lens boosts most relative to the logit lens.

Design notes:
- Readouts run their own single ``no_grad`` forward with block-output hooks.
  Block outputs are the *pre-final-norm* residuals jlens expects — this is why
  we hook rather than use ``output_hidden_states`` (whose last entry is
  post-norm).
- Everything is chunked per layer so ``layers x positions x vocab`` is never
  materialized at once.
- The generation tracer is not involved; lens views are read-only side
  computations on the same frozen model.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import torch

from miru_tracer.core._jlens import HFLensModel, JacobianLens, from_hf
from miru_tracer.core._jlens.hooks import ActivationRecorder
from miru_tracer.core.interventions import InterventionSet, apply_interventions
from miru_tracer.core.logging_config import get_logger
from miru_tracer.core.tokenizer_utils import safe_decode_token

logger = get_logger(__name__)

LENS_MODES = ("logit", "jacobian", "diff")

_WORD_RE = re.compile(r"\w", re.UNICODE)


def is_word_token(text: str) -> bool:
    """True if the decoded token contains at least one word character."""
    return bool(_WORD_RE.search(text))


def default_lens_dir() -> Path:
    return Path(
        os.getenv("MIRU_LENS_DIR", str(Path.home() / ".cache" / "miru-tracer" / "lenses"))
    )


def sanitize_model_name(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "--", model_name)


class LensStore:
    """Discovers and caches fitted JacobianLens artifacts per model name."""

    def __init__(self, base_dir: Path | None = None):
        self._base_dir = base_dir
        self._cache: dict[str, tuple[float, JacobianLens]] = {}

    @property
    def base_dir(self) -> Path:
        return self._base_dir if self._base_dir is not None else default_lens_dir()

    def lens_path(self, model_name: str) -> Path:
        """Where the fitted lens for a model lives (whether or not it exists)."""
        return self.base_dir / sanitize_model_name(model_name) / "lens.pt"

    def get(self, model_name: str) -> JacobianLens | None:
        """Load the fitted lens for a model, or None if not fitted yet."""
        path = self.lens_path(model_name)
        if not path.is_file():
            return None
        mtime = path.stat().st_mtime
        cached = self._cache.get(model_name)
        if cached is not None and cached[0] == mtime:
            return cached[1]
        try:
            lens = JacobianLens.load(str(path))
        except Exception as e:
            logger.error(f"Failed to load lens at {path}: {e}")
            return None
        self._cache[model_name] = (mtime, lens)
        logger.info(f"Loaded Jacobian lens for {model_name}: {lens}")
        return lens


_lens_store: LensStore | None = None


def get_lens_store() -> LensStore:
    global _lens_store
    if _lens_store is None:
        _lens_store = LensStore()
    return _lens_store


_wrapper_cache: dict[int, HFLensModel] = {}


def wrap_model(model, tokenizer) -> HFLensModel:
    """Wrap (and cache) an HF model as a LensModel. Does not mutate the tokenizer."""
    key = id(model)
    wrapper = _wrapper_cache.get(key)
    if wrapper is None or wrapper.tokenizer is not tokenizer:
        wrapper = from_hf(model, tokenizer, force_bos=False)
        _wrapper_cache.clear()  # one loaded model at a time in Miru
        _wrapper_cache[key] = wrapper
    return wrapper


@dataclass
class LensSlice:
    """Top-k lens readouts over a (layers x positions) grid.

    Indexing: ``tokens[i][j]`` is the top-k token-id list for
    ``layers[i]`` at ``positions[j]``; likewise probs/texts.
    ``pinned_ranks[token_id][i][j]`` is that token's rank (0 = top).
    """

    mode: str
    layers: list[int]
    positions: list[int]
    position_texts: list[str]  # decoded input token at each selected position
    tokens: list[list[list[int]]]
    probs: list[list[list[float]]]
    texts: list[list[list[str]]]
    pinned_ranks: dict[int, list[list[int]]] = field(default_factory=dict)


def compute_lens_slice(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    *,
    layers: list[int],
    positions: list[int] | None = None,
    mode: str = "logit",
    jlens: JacobianLens | None = None,
    top_k: int = 8,
    skip_non_words: bool = False,
    pinned_token_ids: list[int] | None = None,
    interventions: InterventionSet | None = None,
) -> LensSlice:
    """Compute per-(layer, position) top-k lens readouts for a sequence.

    Args:
        input_ids: ``[1, seq_len]`` token ids (as produced by the tracer).
        layers: Block indices to read out (final block = model logits row).
        positions: Sequence positions to read out; None = all.
        mode: "logit", "jacobian", or "diff" (jacobian-vs-logit probability gain).
        jlens: Fitted lens; required for "jacobian" and "diff".
        top_k: Readouts per (layer, position) cell.
        skip_non_words: Drop tokens with no word characters from the top-k.
        pinned_token_ids: Tokens whose rank is tracked in every cell.
        interventions: Active activation edits; entered *before* the recorder
            so the recorded residuals are the edited ones.
    """
    if mode not in LENS_MODES:
        raise ValueError(f"Unknown lens mode: {mode!r}. Use one of {LENS_MODES}.")
    if mode in ("jacobian", "diff") and jlens is None:
        raise ValueError(f"mode={mode!r} requires a fitted Jacobian lens")

    wrapper = wrap_model(model, tokenizer)
    seq_len = int(input_ids.shape[1])
    if positions is None:
        positions = list(range(seq_len))
    layers = sorted(set(layers))
    bad = [layer for layer in layers if not 0 <= layer < wrapper.n_layers]
    if bad:
        raise ValueError(f"layers {bad} out of range (n_layers={wrapper.n_layers})")
    if mode in ("jacobian", "diff"):
        missing = set(layers) - set(jlens.source_layers)
        # The final layer works without a J matrix (J = I there by definition).
        missing -= {wrapper.n_layers - 1}
        if missing:
            raise ValueError(
                f"layers {sorted(missing)} not fitted; lens covers {jlens.source_layers}"
            )

    with (
        torch.no_grad(),
        apply_interventions(model, interventions),
        ActivationRecorder(wrapper.layers, at=layers) as recorder,
    ):
        wrapper.forward(input_ids.to(wrapper.input_device))
        activations = {i: recorder.activations[i].detach() for i in layers}

    pinned_token_ids = pinned_token_ids or []
    final_layer = wrapper.n_layers - 1
    all_tokens: list[list[list[int]]] = []
    all_probs: list[list[list[float]]] = []
    all_texts: list[list[list[str]]] = []
    pinned: dict[int, list[list[int]]] = {t: [] for t in pinned_token_ids}

    candidate_k = top_k * 4 if skip_non_words else top_k

    with torch.no_grad():
        for layer in layers:
            residual = activations[layer][0, positions, :].float()  # [P, d]
            use_transport = (
                mode in ("jacobian", "diff")
                and layer != final_layer
                and layer in jlens.jacobians
            )
            if mode == "diff" and layer != final_layer:
                logit_probs = torch.softmax(wrapper.unembed(residual).float(), dim=-1)
                jac_probs = torch.softmax(
                    wrapper.unembed(jlens.transport(residual, layer)).float(), dim=-1
                )
                scores = jac_probs - logit_probs  # tokens the J-lens boosts
                probs_for_pin = jac_probs
            else:
                transported = jlens.transport(residual, layer) if use_transport else residual
                scores = torch.softmax(wrapper.unembed(transported).float(), dim=-1)
                probs_for_pin = scores

            k = min(candidate_k, scores.shape[-1])
            top_scores, top_ids = torch.topk(scores, k, dim=-1)  # [P, k]

            layer_tokens, layer_probs, layer_texts = [], [], []
            for p in range(len(positions)):
                ids = top_ids[p].tolist()
                vals = top_scores[p].tolist()
                row_t, row_p, row_s = [], [], []
                for token_id, value in zip(ids, vals, strict=True):
                    text = decode_token(tokenizer, token_id)
                    if skip_non_words and not is_word_token(text):
                        continue
                    row_t.append(token_id)
                    row_p.append(float(value))
                    row_s.append(text)
                    if len(row_t) == top_k:
                        break
                layer_tokens.append(row_t)
                layer_probs.append(row_p)
                layer_texts.append(row_s)
            all_tokens.append(layer_tokens)
            all_probs.append(layer_probs)
            all_texts.append(layer_texts)

            if pinned_token_ids:
                # rank = number of tokens with strictly higher probability
                ranks = (
                    probs_for_pin > probs_for_pin[:, pinned_token_ids].T.unsqueeze(-1)
                ).sum(dim=-1)  # [n_pinned, P]
                for i, token_id in enumerate(pinned_token_ids):
                    pinned[token_id].append(ranks[i].tolist())

    position_texts = [decode_token(tokenizer, int(input_ids[0, p])) for p in positions]
    return LensSlice(
        mode=mode,
        layers=layers,
        positions=list(positions),
        position_texts=position_texts,
        tokens=all_tokens,
        probs=all_probs,
        texts=all_texts,
        pinned_ranks=pinned,
    )


def decode_token(tokenizer, token_id: int) -> str:
    """Best-effort display text for a token.

    Falls back to ``<id>`` for ids outside the tokenizer vocabulary — models
    commonly pad the embedding matrix past the tokenizer size (Qwen3 included),
    and those ids can legitimately appear in lens readouts.
    """
    decoded, raw, _ = safe_decode_token(tokenizer, token_id)
    text = decoded if decoded else raw
    return text if text is not None else f"<{token_id}>"


@dataclass
class ReadoutRow:
    """One aggregated readout: a token and where it appears in the slice."""

    token_id: int
    text: str
    count: int  # cells (layer, position) whose top-k contains the token
    count_by_layer: list[int]  # aligned with slice.layers


def aggregate_readouts(slice_: LensSlice, *, limit: int = 50) -> list[ReadoutRow]:
    """Neuronpedia-style aggregation: which tokens appear across the selected
    cells, how often, and at which layers."""
    counts: dict[int, ReadoutRow] = {}
    for i, _layer in enumerate(slice_.layers):
        for j in range(len(slice_.positions)):
            for token_id, text in zip(
                slice_.tokens[i][j], slice_.texts[i][j], strict=True
            ):
                row = counts.get(token_id)
                if row is None:
                    row = ReadoutRow(
                        token_id=token_id,
                        text=text,
                        count=0,
                        count_by_layer=[0] * len(slice_.layers),
                    )
                    counts[token_id] = row
                row.count += 1
                row.count_by_layer[i] += 1
    ranked = sorted(counts.values(), key=lambda r: (-r.count, r.token_id))
    return ranked[:limit]
