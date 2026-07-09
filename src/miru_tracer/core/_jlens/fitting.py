# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
# Modified by Miru Tracer: resumable convergence stopping and fit metadata.
"""Fitting the Jacobian lens.

The lens reads out an early-layer residual ``h_l`` by linearly transporting it
into the final-layer basis with the average input-output Jacobian, then
decoding with the model's own unembedding::

    lens_l(h) = unembed( J_l @ h )

Estimator (:func:`jacobian_for_prompt`): for each output dimension, inject a
one-hot cotangent at *every valid target position at once* and backprop. The
gradient at source position ``p`` is then ``sum_{p' >= p} dh_final[p'] / dh_l[p]``,
the sum over later target positions; we take the mean over source positions
``p``. This is the reduction used in the paper. A per-position estimator
(``dh_final[p] / dh_l[p]`` averaged over ``p``) gives a slightly different
``J_l``; both work as a lens.

Cost: one forward pass and ``ceil(d_model / dim_batch)`` backward passes per
prompt. Shard across machines by running :func:`fit` on disjoint prompt
slices and merging with :meth:`jlens.lens.JacobianLens.merge`.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import time
from collections.abc import Sequence
from typing import Any

import torch

from miru_tracer.core._jlens.fit_metadata import normalize_fit_metadata
from miru_tracer.core._jlens.hooks import ActivationRecorder
from miru_tracer.core._jlens.lens import JacobianLens
from miru_tracer.core._jlens.protocol import LensModel

logger = logging.getLogger(__name__)

#: Positions before this index are excluded from the Jacobian average; early
#: positions act as attention sinks and have atypical residual statistics.
SKIP_FIRST_N_POSITIONS = 16

CONVERGENCE_METRIC = "mean_layer_relative_running_mean_change_v1"
ESTIMATOR_VERSION = "summed_current_and_future_targets_v1"
MAX_EMBEDDED_CONVERGENCE_POINTS = 1_000
_RESUME_PROVENANCE_KEYS = (
    "model_name_or_path",
    "model_location_sha256",
    "model_manifest_sha256",
    "model_commit_hash",
    "model_config_sha256",
    "tokenizer_name_or_path",
    "tokenizer_sha256",
    "compute_dtype",
)


def _update_prompt_digest(hasher: Any, prompt: str) -> None:
    """Add one prompt to an ordered, unambiguous SHA-256 stream."""
    encoded = prompt.encode("utf-8")
    hasher.update(len(encoded).to_bytes(8, "big"))
    hasher.update(encoded)


def _convergence_state(
    history: list[dict[str, Any]],
    *,
    n_done: int,
    min_prompts: int,
    stop_window: int,
    stop_at_delta: float | None,
) -> tuple[bool, float | None]:
    recent_points = history[-stop_window:]
    recent: list[float] = []
    for point in recent_points:
        value = point.get("mean_relative_change")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            recent = []
            break
        recent.append(float(value))
    rolling = sum(recent) / stop_window if len(recent) == stop_window else None
    converged = (
        stop_at_delta is not None
        and n_done >= min_prompts
        and rolling is not None
        and rolling < stop_at_delta
    )
    return converged, rolling


def _check_resume_provenance(
    stored: object, current: dict[str, Any] | None, checkpoint_path: str
) -> None:
    if not isinstance(stored, dict) or current is None:
        return
    for key in _RESUME_PROVENANCE_KEYS:
        old_value = stored.get(key)
        new_value = current.get(key)
        # Missing values in legacy/incomplete metadata are not evidence of a
        # mismatch. When both sides identify a value, mixing is unsafe.
        if old_value is not None and new_value is not None and old_value != new_value:
            raise ValueError(
                f"checkpoint at {checkpoint_path} was fitted with {key}="
                f"{old_value!r}, not {new_value!r}; pass resume=False "
                "(CLI: --fresh) to discard it"
            )


def _validate_checkpoint_history(
    history: object,
    *,
    n_done: int,
    next_idx: int,
    checkpoint_path: str,
) -> list[dict[str, Any]]:
    if not isinstance(history, list):
        raise ValueError(f"checkpoint at {checkpoint_path} has invalid fit_history")
    validated: list[dict[str, Any]] = []
    previous_count: int | None = None
    previous_prompt_index = -1
    for point in history:
        if not isinstance(point, dict):
            raise ValueError(f"checkpoint at {checkpoint_path} has invalid fit_history")
        count = point.get("n_prompts")
        prompt_index = point.get("prompt_index")
        change = point.get("mean_relative_change")
        if type(count) is not int or count <= 0 or count > n_done:
            raise ValueError(f"checkpoint at {checkpoint_path} has invalid fit_history")
        if previous_count is not None and count != previous_count + 1:
            raise ValueError(f"checkpoint at {checkpoint_path} has invalid fit_history")
        if (
            type(prompt_index) is not int
            or prompt_index <= previous_prompt_index
            or prompt_index >= next_idx
        ):
            raise ValueError(f"checkpoint at {checkpoint_path} has invalid fit_history")
        if change is not None and (
            isinstance(change, bool)
            or not isinstance(change, (int, float))
            or not math.isfinite(change)
            or change < 0
        ):
            raise ValueError(f"checkpoint at {checkpoint_path} has invalid fit_history")
        previous_count = count
        previous_prompt_index = prompt_index
        validated.append(point)
    if validated and validated[-1]["n_prompts"] != n_done:
        raise ValueError(f"checkpoint at {checkpoint_path} has invalid fit_history")
    return validated


def _validate_checkpoint_accumulator(
    value: object,
    *,
    source_layers: list[int],
    d_model: int,
    checkpoint_path: str,
) -> dict[int, torch.Tensor]:
    if not isinstance(value, dict) or set(value) != set(source_layers):
        raise ValueError(f"checkpoint at {checkpoint_path} has invalid jacobian_sum layers")
    expected_shape = (d_model, d_model)
    for layer, matrix in value.items():
        if (
            not isinstance(matrix, torch.Tensor)
            or not matrix.is_floating_point()
            or tuple(matrix.shape) != expected_shape
        ):
            raise ValueError(
                f"checkpoint at {checkpoint_path} has invalid jacobian_sum matrix at layer {layer}"
            )
    return value


def valid_position_mask(seq_len: int, *, skip_first: int = SKIP_FIRST_N_POSITIONS) -> torch.Tensor:
    """Boolean mask over sequence positions to include in the Jacobian average.

    Early positions are dominated by attention-sink behaviour and the final
    position has no next-token target, so both are excluded.

    Args:
        seq_len: Length of the tokenized prompt.
        skip_first: Number of leading positions to exclude.

    Returns:
        Boolean tensor of shape ``[seq_len]``.

    Raises:
        ValueError: If ``skip_first`` is negative or the prompt is too short to
            leave any valid positions.
    """
    if skip_first < 0:
        raise ValueError(f"skip_first must be >= 0, got {skip_first}")
    mask = torch.zeros(seq_len, dtype=torch.bool)
    mask[skip_first : seq_len - 1] = True
    if mask.sum() == 0:
        raise ValueError(f"prompt too short: seq_len={seq_len}, need > {skip_first + 1} tokens")
    return mask


def _check_layer_indices(
    source_layers: Sequence[int] | None, target_layer: int | None, n_layers: int
) -> tuple[list[int], int]:
    """Resolve None/negative layer indices, bounds-check, enforce source < target."""
    target = n_layers - 1 if target_layer is None else target_layer
    if target < 0:
        target += n_layers
    if not 0 <= target < n_layers:
        raise ValueError(f"target_layer={target_layer} out of range for {n_layers} layers")
    if source_layers is None:
        return list(range(target)), target
    sources = sorted({l + n_layers if l < 0 else l for l in source_layers})
    if not sources or sources[0] < 0 or sources[-1] >= n_layers:
        raise ValueError(
            f"source_layers {sorted(source_layers)} out of range for {n_layers} layers"
        )
    if sources[-1] >= target:
        raise ValueError(
            f"source_layers must all be < target_layer={target}; got max={sources[-1]}"
        )
    return sources, target


def jacobian_for_prompt(
    model: LensModel,
    prompt: str,
    source_layers: Sequence[int],
    *,
    target_layer: int | None = None,
    dim_batch: int = 8,
    max_seq_len: int = 128,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
) -> tuple[dict[int, torch.Tensor], int, int]:
    """Compute the per-layer Jacobian estimator ``J_l`` for one prompt.

    Runs one forward pass on the prompt replicated ``dim_batch`` times along
    the batch axis, retains the graph, then runs ``ceil(d_model / dim_batch)``
    backward passes against it. Each backward computes ``dim_batch`` rows of
    ``J_l`` at once: batch element ``b`` carries a one-hot cotangent at output
    dimension ``dim_start + b``, set at every valid target position. See the
    module docstring for the resulting estimator and how it relates to
    a strict per-position Jacobian.

    Args:
        model: The model to compute Jacobians for.
        prompt: Input text.
        source_layers: Layer indices ``l`` to compute ``J_l`` at.
        target_layer: Layer to take gradients with respect to. Defaults to the
            final layer; negative indices count from the end. In some cases,
            targeting the penultimate layer can give a better-conditioned
            ``J_l``.
        dim_batch: Output dimensions computed per backward pass. Higher uses
            more GPU memory (the prompt is replicated this many times); total
            backward FLOPs are unchanged.
        max_seq_len: Truncate the prompt to this many tokens.
        skip_first: Leading positions to exclude; see :func:`valid_position_mask`.

    Returns:
        ``(jacobians, seq_len, n_valid_positions)``. ``jacobians`` maps each
        source layer to a ``[d_model, d_model]`` fp32 CPU tensor.
    """
    n_layers, d_model = model.n_layers, model.d_model
    source_layers, target_layer = _check_layer_indices(source_layers, target_layer, n_layers)

    input_ids = model.encode(prompt, max_length=max_seq_len)
    seq_len = input_ids.shape[1]
    position_mask = valid_position_mask(seq_len, skip_first=skip_first)
    n_valid_positions = int(position_mask.sum())

    jacobians = {
        layer: torch.zeros(d_model, d_model, dtype=torch.float32) for layer in source_layers
    }
    n_passes = math.ceil(d_model / dim_batch)

    with (
        ActivationRecorder(
            model.layers,
            at=[*source_layers, target_layer],
            start_graph_at=min(source_layers),
        ) as recorder,
        torch.enable_grad(),
    ):
        # One forward on the prompt replicated dim_batch times. The retained
        # graph is reused for every backward pass below.
        replicated_ids = input_ids.expand(dim_batch, -1)
        model.forward(replicated_ids)
        target_activation = recorder.activations[target_layer]  # [dim_batch, seq_len, d_model]
        source_activations = [recorder.activations[layer] for layer in source_layers]

        valid_positions = position_mask.nonzero(as_tuple=True)[0].to(target_activation.device)
        batch_indices = torch.arange(dim_batch, device=target_activation.device)
        cotangent = torch.zeros_like(target_activation)

        for pass_idx, dim_start in enumerate(range(0, d_model, dim_batch)):
            n_dims_this_pass = min(dim_batch, d_model - dim_start)
            # One-hot cotangent at dim (dim_start + b) for batch element b,
            # at every valid target position. Yields rows dim_start..+n of J_l.
            cotangent.zero_()
            cotangent[
                batch_indices[:n_dims_this_pass, None],
                valid_positions[None, :],
                dim_start + batch_indices[:n_dims_this_pass, None],
            ] = 1.0
            grads = torch.autograd.grad(
                outputs=target_activation,
                inputs=source_activations,
                grad_outputs=cotangent,
                retain_graph=(pass_idx < n_passes - 1),
            )
            for layer, grad in zip(source_layers, grads, strict=True):
                # grad: [dim_batch, seq_len, d_model] on whatever device this
                # layer lives on; mean over the valid positions -> dim_batch rows.
                positions_on_device = valid_positions.to(grad.device, non_blocking=True)
                rows = grad[:n_dims_this_pass, positions_on_device, :].float().mean(dim=1)
                jacobians[layer][dim_start : dim_start + n_dims_this_pass, :] = rows.cpu()
            del grads
            if pass_idx % 100 == 0 or pass_idx == n_passes - 1:
                logger.debug(
                    "    pass %d/%d (dims %d-%d)",
                    pass_idx + 1,
                    n_passes,
                    dim_start,
                    dim_start + n_dims_this_pass,
                )

    return jacobians, seq_len, n_valid_positions


def _atomic_save(obj: object, path: str) -> None:
    """``torch.save`` to a temp file then ``os.replace`` so a crash never
    leaves a half-written checkpoint."""
    tmp_path = f"{path}.tmp.{os.getpid()}"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def fit(
    model: LensModel,
    prompts: Sequence[str],
    *,
    source_layers: Sequence[int] | None = None,
    target_layer: int | None = None,
    dim_batch: int = 8,
    max_seq_len: int = 128,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
    checkpoint_path: str | None = None,
    checkpoint_every: int | None = 1,
    resume: bool = True,
    min_prompts: int = 100,
    stop_window: int = 10,
    stop_at_delta: float | None = None,
    fit_provenance: dict[str, Any] | None = None,
) -> JacobianLens:
    """Fit ``J_l`` over a list of prompts and return a :class:`JacobianLens`.

    Per-prompt Jacobians from :func:`jacobian_for_prompt` are accumulated as a
    running mean. If ``checkpoint_path`` is set, the running sum is written
    every ``checkpoint_every`` prompts (atomic) and resumed from on restart.

    Args:
        model: The model to fit on.
        prompts: Text prompts to average over. See the README for guidance on
            corpus size and distribution.
        source_layers: Layers to fit at. Defaults to every layer below
            ``target_layer``; negative indices count from the end.
        target_layer: See :func:`jacobian_for_prompt`. Defaults to the final
            layer; negative indices count from the end.
        dim_batch: See :func:`jacobian_for_prompt`.
        max_seq_len: Truncate each prompt to this many tokens.
        skip_first: See :func:`jacobian_for_prompt`.
        checkpoint_path: If set, write a resumable checkpoint here.
        checkpoint_every: Write the checkpoint every N prompts (default 1).
            ``None`` skips per-iteration writes and saves once at the end; the
            checkpoint can be large (``len(source_layers) * d_model**2 * 4``
            bytes), so raise this for large models.
        resume: If ``True`` and ``checkpoint_path`` exists, resume from it.
        min_prompts: Successful-prompt floor before convergence may stop the
            fit. Has no effect when ``stop_at_delta`` is ``None``.
        stop_window: Number of recent successful-prompt changes in the rolling
            convergence mean.
        stop_at_delta: Stop when the rolling mean is strictly below this
            positive threshold. ``None`` disables convergence stopping.
        fit_provenance: Optional JSON-safe model/corpus provenance to embed in
            the returned lens artifact.

    Returns:
        The fitted :class:`JacobianLens`.
    """
    n_layers, d_model = model.n_layers, model.d_model
    if type(min_prompts) is not int or min_prompts <= 0:
        raise ValueError(f"min_prompts must be a positive integer, got {min_prompts!r}")
    if type(stop_window) is not int or stop_window <= 0:
        raise ValueError(f"stop_window must be a positive integer, got {stop_window!r}")
    if checkpoint_every is not None and (
        type(checkpoint_every) is not int or checkpoint_every <= 0
    ):
        raise ValueError(
            f"checkpoint_every must be a positive integer or None, got {checkpoint_every!r}"
        )
    if isinstance(stop_at_delta, bool):
        raise ValueError(
            f"stop_at_delta must be a positive finite number or None, got {stop_at_delta!r}"
        )
    if stop_at_delta == 0:
        stop_at_delta = None
    if stop_at_delta is not None and (
        not isinstance(stop_at_delta, (int, float))
        or not math.isfinite(stop_at_delta)
        or stop_at_delta <= 0
    ):
        raise ValueError(
            f"stop_at_delta must be a positive finite number or None, got {stop_at_delta!r}"
        )
    if fit_provenance is not None:
        normalized_metadata = normalize_fit_metadata(
            {"schema_version": 1, "provenance": fit_provenance}
        )
        assert normalized_metadata is not None
        fit_provenance = normalized_metadata["provenance"]
    source_layers, target_layer = _check_layer_indices(source_layers, target_layer, n_layers)

    logger.info(
        "fit: n_layers=%d d_model=%d, fitting %d source layers (target=L%d) on %d prompts",
        n_layers,
        d_model,
        len(source_layers),
        target_layer,
        len(prompts),
    )

    # Running state: sum of per-prompt Jacobians, success count, and the list
    # index to resume from. ``next_idx`` is tracked separately from ``n_done``
    # so a too-short prompt that was skipped is not re-processed on resume.
    jacobian_sum: dict[int, torch.Tensor]
    n_done: int
    next_idx: int
    fit_history: list[dict[str, Any]]
    prompt_hasher = hashlib.sha256()
    if resume and checkpoint_path is not None and os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        safety_fields = {
            "source_layers",
            "target_layer",
            "skip_first",
            "max_seq_len",
            "d_model",
            "prompt_prefix_sha256",
            "fit_history",
            "fit_provenance",
            "estimator",
        }
        missing_safety_fields = sorted(safety_fields - set(state))
        for key, expected in (
            ("source_layers", source_layers),
            ("target_layer", target_layer),
            ("skip_first", skip_first),
            ("max_seq_len", max_seq_len),
            ("d_model", d_model),
            ("estimator", ESTIMATOR_VERSION),
        ):
            if key in state and state[key] != expected:
                raise ValueError(
                    f"checkpoint at {checkpoint_path} was fitted with {key}="
                    f"{state[key]!r}, not {expected!r}; pass resume=False "
                    "(CLI: --fresh) to discard it"
                )
        jacobian_sum, n_done, next_idx = (
            state["jacobian_sum"],
            state["n_done"],
            state["next_idx"],
        )
        jacobian_sum = _validate_checkpoint_accumulator(
            jacobian_sum,
            source_layers=source_layers,
            d_model=d_model,
            checkpoint_path=checkpoint_path,
        )
        if type(n_done) is not int or type(next_idx) is not int or n_done < 0 or next_idx < n_done:
            raise ValueError(f"checkpoint at {checkpoint_path} has invalid prompt counts")
        if next_idx > 0 and missing_safety_fields:
            raise ValueError(
                f"checkpoint at {checkpoint_path} lacks resume-safety metadata "
                f"{missing_safety_fields!r} and cannot be resumed safely; pass "
                "resume=False (CLI: --fresh) to start a new fit"
            )
        if missing_safety_fields:
            logger.warning(
                "  empty legacy checkpoint lacks resume-safety metadata %s; upgrading it",
                missing_safety_fields,
            )
        if next_idx > len(prompts):
            raise ValueError(
                f"checkpoint at {checkpoint_path} already processed {next_idx} "
                f"prompts, but only {len(prompts)} were provided"
            )
        stored_digest = state.get("prompt_prefix_sha256")
        if next_idx > 0 and stored_digest is None:
            raise ValueError(
                f"checkpoint at {checkpoint_path} predates prompt-prefix validation "
                "and cannot be resumed safely; pass resume=False (CLI: --fresh) "
                "to start a new fit"
            )
        for prompt in prompts[:next_idx]:
            _update_prompt_digest(prompt_hasher, prompt)
        if stored_digest is not None and stored_digest != prompt_hasher.hexdigest():
            raise ValueError(
                f"checkpoint at {checkpoint_path} was fitted on a different "
                "ordered prompt prefix; pass resume=False (CLI: --fresh) to discard it"
            )
        fit_history = _validate_checkpoint_history(
            state.get("fit_history", []),
            n_done=n_done,
            next_idx=next_idx,
            checkpoint_path=checkpoint_path,
        )
        _check_resume_provenance(state.get("fit_provenance"), fit_provenance, checkpoint_path)
        if fit_provenance is None:
            fit_provenance = state.get("fit_provenance")
        logger.info(
            "  resuming from checkpoint: %d/%d prompts processed (%d fitted)",
            next_idx,
            len(prompts),
            n_done,
        )
    else:
        jacobian_sum = {
            layer: torch.zeros(d_model, d_model, dtype=torch.float32) for layer in source_layers
        }
        n_done = 0
        next_idx = 0
        fit_history = []

    def write_checkpoint() -> None:
        if checkpoint_path is not None:
            _atomic_save(
                {
                    "jacobian_sum": jacobian_sum,
                    "n_done": n_done,
                    "next_idx": next_idx,
                    "source_layers": source_layers,
                    "target_layer": target_layer,
                    "skip_first": skip_first,
                    "max_seq_len": max_seq_len,
                    "d_model": d_model,
                    "prompt_prefix_sha256": prompt_hasher.hexdigest(),
                    "fit_history": fit_history,
                    "fit_provenance": fit_provenance,
                    "estimator": ESTIMATOR_VERSION,
                },
                checkpoint_path,
            )

    converged, rolling_mean = _convergence_state(
        fit_history,
        n_done=n_done,
        min_prompts=min_prompts,
        stop_window=stop_window,
        stop_at_delta=stop_at_delta,
    )
    sqrt_d = math.sqrt(d_model)
    for prompt_idx, prompt in enumerate(prompts):
        if converged:
            break
        if prompt_idx < next_idx:
            continue
        start_time = time.perf_counter()
        try:
            per_prompt_J, seq_len, n_valid = jacobian_for_prompt(
                model,
                prompt,
                source_layers,
                target_layer=target_layer,
                dim_batch=dim_batch,
                max_seq_len=max_seq_len,
                skip_first=skip_first,
            )
        except ValueError as exc:
            logger.warning("  skipping prompt %d: %s", prompt_idx, exc)
            _update_prompt_digest(prompt_hasher, prompt)
            next_idx = prompt_idx + 1
            write_checkpoint()
            continue

        layer_norms = {layer: matrix.norm().item() for layer, matrix in per_prompt_J.items()}
        nonfinite_layers = [layer for layer, norm in layer_norms.items() if not math.isfinite(norm)]
        if nonfinite_layers:
            logger.warning(
                "  skipping prompt %d: non-finite Jacobian values at layers %s",
                prompt_idx,
                nonfinite_layers,
            )
            _update_prompt_digest(prompt_hasher, prompt)
            next_idx = prompt_idx + 1
            write_checkpoint()
            continue

        # The prompt norm flags heavy-tailed outliers. Convergence matches
        # Neuronpedia's fitter: for each layer, measure the Frobenius movement
        # of the running mean relative to the *new* mean, then average layers.
        prompt_norm = max(layer_norms.values()) / sqrt_d
        if n_done > 0:
            layer_changes: list[float] = []
            new_count = n_done + 1
            for layer in source_layers:
                old_mean = jacobian_sum[layer] / n_done
                step_norm = (per_prompt_J[layer] - old_mean).norm().item() / new_count
                new_mean_norm = (
                    ((jacobian_sum[layer] + per_prompt_J[layer]) / new_count).norm().item()
                )
                if new_mean_norm > 0:
                    relative_change = step_norm / new_mean_norm
                    if math.isfinite(relative_change):
                        layer_changes.append(relative_change)
            mean_rel_change = (
                sum(layer_changes) / len(layer_changes)
                if len(layer_changes) == len(source_layers)
                else None
            )
        else:
            mean_rel_change = None

        for layer in source_layers:
            jacobian_sum[layer] += per_prompt_J[layer]
        n_done += 1
        _update_prompt_digest(prompt_hasher, prompt)
        next_idx = prompt_idx + 1
        fit_history.append(
            {
                "n_prompts": n_done,
                "prompt_index": prompt_idx,
                "seq_len": seq_len,
                "valid_positions": n_valid,
                "prompt_jacobian_norm_over_sqrt_d": (
                    prompt_norm if math.isfinite(prompt_norm) else None
                ),
                "mean_relative_change": mean_rel_change,
            }
        )
        converged, rolling_mean = _convergence_state(
            fit_history,
            n_done=n_done,
            min_prompts=min_prompts,
            stop_window=stop_window,
            stop_at_delta=stop_at_delta,
        )

        logger.info(
            "  prompt %d/%d  seq_len=%d n_valid=%d  %.0fs  "
            "max||J||/sqrt(d)=%.3f  d_mean=%s  rolling=%s",
            prompt_idx + 1,
            len(prompts),
            seq_len,
            n_valid,
            time.perf_counter() - start_time,
            prompt_norm,
            "n/a" if mean_rel_change is None else f"{mean_rel_change:.2e}",
            "n/a" if rolling_mean is None else f"{rolling_mean:.2e}",
        )
        if checkpoint_every is not None and next_idx % checkpoint_every == 0:
            write_checkpoint()
        if converged:
            # Always persist the exact stopping point even when checkpointing
            # was configured less frequently than once per prompt.
            write_checkpoint()
            logger.info(
                "fit: converged after %d prompts (rolling d_mean %.3g < %.3g)",
                n_done,
                rolling_mean,
                stop_at_delta,
            )
            break

    write_checkpoint()
    if n_done == 0:
        raise ValueError("no prompts were long enough to fit on")
    jacobian_mean = {layer: jacobian_sum[layer] / n_done for layer in source_layers}
    logger.info("fit: done, %d prompts", n_done)
    last_change = next(
        (
            point["mean_relative_change"]
            for point in reversed(fit_history)
            if point.get("mean_relative_change") is not None
        ),
        None,
    )
    fit_metadata = {
        "schema_version": 1,
        "provenance": fit_provenance or {},
        "fit": {
            "estimator": ESTIMATOR_VERSION,
            "position_reduction": "mean",
            "prompt_reduction": "uniform_mean",
            "target_layer": target_layer,
            "max_seq_len": max_seq_len,
            "skip_first": skip_first,
            "dim_batch": dim_batch,
            "processed_prompts": next_idx,
            "skipped_prompts": next_idx - n_done,
            "processed_prompt_prefix_sha256": prompt_hasher.hexdigest(),
        },
        "convergence": {
            "metric": CONVERGENCE_METRIC,
            "enabled": stop_at_delta is not None,
            "min_prompts": min_prompts,
            "window": stop_window,
            "threshold": stop_at_delta,
            "converged": converged,
            "last_mean_relative_change": last_change,
            "rolling_mean_relative_change": rolling_mean,
            "history_total_points": len(fit_history),
            "history_truncated": len(fit_history) > MAX_EMBEDDED_CONVERGENCE_POINTS,
            "history": fit_history[-MAX_EMBEDDED_CONVERGENCE_POINTS:],
        },
    }
    return JacobianLens(
        jacobians=jacobian_mean,
        n_prompts=n_done,
        d_model=d_model,
        fit_metadata=fit_metadata,
    )
