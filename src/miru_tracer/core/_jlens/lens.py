# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
# Modified by Miru Tracer: optional JSON-safe fit metadata.
"""Applying a fitted Jacobian lens.

A :class:`JacobianLens` holds the per-layer ``J_l`` matrices produced by
:func:`jlens.fitting.fit`. :meth:`JacobianLens.apply` runs a forward pass and
reads out the requested layers; :meth:`JacobianLens.transport` is the bare
``J_l @ h`` for callers that already have residuals.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import torch

from miru_tracer.core._jlens.fit_metadata import normalize_fit_metadata
from miru_tracer.core._jlens.hooks import ActivationRecorder
from miru_tracer.core._jlens.protocol import LensModel


class JacobianLens:
    """A fitted Jacobian lens: per-layer ``J_l`` matrices and the readout method.

    Attributes:
        jacobians: ``{layer_index: Tensor[d_model, d_model]}``. Each ``J_l``
            maps the residual at layer ``l`` into the final-layer basis.
        source_layers: Sorted list of fitted layer indices.
        n_prompts: Number of prompts the lens was averaged over.
        d_model: Residual-stream width.
        fit_metadata: Optional JSON-safe fitting provenance and convergence
            diagnostics. Older/upstream artifacts leave this as ``None``.
    """

    def __init__(
        self,
        jacobians: dict[int, torch.Tensor],
        *,
        n_prompts: int,
        d_model: int,
        fit_metadata: dict[str, Any] | None = None,
    ) -> None:
        if type(n_prompts) is not int or n_prompts <= 0:
            raise ValueError(f"n_prompts must be a positive integer, got {n_prompts!r}")
        if type(d_model) is not int or d_model <= 0:
            raise ValueError(f"d_model must be a positive integer, got {d_model!r}")
        if not isinstance(jacobians, dict) or not jacobians:
            raise ValueError("jacobians must be a non-empty layer-to-matrix mapping")
        expected_shape = (d_model, d_model)
        for layer, matrix in jacobians.items():
            if type(layer) is not int or layer < 0:
                raise ValueError(f"Jacobian layer keys must be nonnegative integers: {layer!r}")
            if not isinstance(matrix, torch.Tensor) or not matrix.is_floating_point():
                raise ValueError(f"Jacobian at layer {layer} must be a floating tensor")
            if tuple(matrix.shape) != expected_shape:
                raise ValueError(
                    f"Jacobian at layer {layer} must have shape {expected_shape}, "
                    f"got {tuple(matrix.shape)}"
                )
            if not torch.isfinite(matrix).all():
                raise ValueError(f"Jacobian at layer {layer} contains non-finite values")
        self.jacobians = {layer: J.float() for layer, J in jacobians.items()}
        self.source_layers = sorted(self.jacobians)
        self.n_prompts = n_prompts
        self.d_model = d_model
        self.fit_metadata = normalize_fit_metadata(fit_metadata, n_prompts=n_prompts)

    def __repr__(self) -> str:
        return (
            f"JacobianLens(d_model={self.d_model}, n_prompts={self.n_prompts}, "
            f"source_layers=[{self.source_layers[0]}..{self.source_layers[-1]}] "
            f"({len(self.source_layers)} layers))"
        )

    def _jacobians_for_save(self, dtype: torch.dtype) -> dict[int, torch.Tensor]:
        try:
            floating = torch.empty((), dtype=dtype).is_floating_point()
        except (TypeError, RuntimeError) as exc:
            raise ValueError(f"lens storage dtype must be floating, got {dtype!r}") from exc
        if not floating:
            raise ValueError(f"lens storage dtype must be floating, got {dtype}")
        serialized: dict[int, torch.Tensor] = {}
        for layer, matrix in self.jacobians.items():
            cast = matrix.to(dtype)
            if not torch.isfinite(cast).all():
                raise ValueError(
                    f"Jacobian at layer {layer} overflows or becomes non-finite "
                    f"when stored as {dtype}; choose a wider floating dtype"
                )
            serialized[layer] = cast
        return serialized

    def save(self, path: str, *, dtype: torch.dtype = torch.float16) -> None:
        """Save to ``path``. Jacobians are stored as ``dtype`` (default fp16:
        halves file size; entries are O(1) so the range is not a constraint
        and fp16's extra mantissa bits beat bf16 here)."""
        payload = {
            "J": self._jacobians_for_save(dtype),
            "n_prompts": self.n_prompts,
            "source_layers": self.source_layers,
            "d_model": self.d_model,
        }
        if self.fit_metadata is not None:
            payload["fit_metadata"] = normalize_fit_metadata(
                self.fit_metadata, n_prompts=self.n_prompts
            )
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str) -> JacobianLens:
        """Load a lens previously written by :meth:`save`."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if "J" not in checkpoint:
            raise ValueError(
                f"{path} is not a JacobianLens file "
                f"(found keys {sorted(checkpoint)!r}; a fit() checkpoint?)"
            )
        jacobians = checkpoint["J"]
        if not isinstance(jacobians, dict):
            raise ValueError(f"{path} is not a valid JacobianLens file ('J' must be a mapping)")
        if any(type(layer) is not int or layer < 0 for layer in jacobians):
            raise ValueError(
                f"{path} is not a valid JacobianLens file "
                "(Jacobian layer keys must be nonnegative integers)"
            )
        stored_layers = checkpoint.get("source_layers")
        actual_layers = sorted(jacobians)
        if stored_layers is not None and stored_layers != actual_layers:
            raise ValueError(
                f"{path} is not a valid JacobianLens file "
                f"(source_layers={stored_layers!r}, tensor layers={actual_layers!r})"
            )
        return cls(
            jacobians=jacobians,
            n_prompts=checkpoint["n_prompts"],
            d_model=checkpoint["d_model"],
            fit_metadata=checkpoint.get("fit_metadata"),
        )

    @classmethod
    def from_pretrained(
        cls,
        name_or_path: str,
        *,
        filename: str = "lens.pt",
        revision: str | None = None,
    ) -> JacobianLens:
        """Load a lens from a local file, a local directory, or a HuggingFace
        Hub ``repo_id``. ``filename`` is the path inside the directory or repo
        (so one Hub repo can host lenses for many models); ignored when
        ``name_or_path`` is itself a file. ``revision`` selects a Hub branch,
        tag, or commit. Deserialisation goes through :meth:`load`
        (``weights_only=True``)."""
        if os.path.isfile(name_or_path):
            return cls.load(name_or_path)
        if not os.path.isdir(name_or_path):
            from huggingface_hub import snapshot_download

            name_or_path = snapshot_download(
                name_or_path, allow_patterns=[filename], revision=revision
            )
        return cls.load(os.path.join(name_or_path, filename))

    @classmethod
    def merge(cls, lenses: Sequence[JacobianLens]) -> JacobianLens:
        """Combine lenses fitted on disjoint prompt subsets into one
        (``n_prompts``-weighted mean of the inputs).

        Args:
            lenses: Lenses to merge. Must agree on ``source_layers`` and
                ``d_model``.

        Raises:
            ValueError: If ``lenses`` is empty or the inputs disagree on shape.
        """
        if not lenses:
            raise ValueError("merge() needs at least one lens")
        first = lenses[0]
        for other in lenses[1:]:
            if other.source_layers != first.source_layers or other.d_model != first.d_model:
                raise ValueError("lenses disagree on source_layers / d_model")
        n_total = sum(lens.n_prompts for lens in lenses)
        merged: dict[int, torch.Tensor] = {}
        for layer in first.source_layers:
            weighted_sum = sum(lens.jacobians[layer] * lens.n_prompts for lens in lenses)
            merged[layer] = weighted_sum / n_total
        # Per-run convergence histories cannot be combined into a valid history
        # for the weighted mean, so merged artifacts deliberately omit them.
        return cls(jacobians=merged, n_prompts=n_total, d_model=first.d_model)

    def transport(self, residual: torch.Tensor, layer: int) -> torch.Tensor:
        """Map a residual at ``layer`` into the final-layer basis: ``J_l @ h``.

        Args:
            residual: Tensor of shape ``[..., d_model]``.
            layer: Source layer index (must be in :attr:`source_layers`).
        """
        J_bar = self.jacobians[layer].to(residual.device)
        return residual @ J_bar.T

    @torch.no_grad()
    def apply(
        self,
        model: LensModel,
        prompt: str,
        *,
        layers: Sequence[int] | None = None,
        positions: Sequence[int] | None = None,
        max_seq_len: int = 512,
        use_jacobian: bool = True,
    ) -> tuple[dict[int, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Run ``model`` on ``prompt`` and return lens logits at ``positions``.

        Args:
            model: The model to read out from.
            prompt: Input text.
            layers: Layers to read out at. Defaults to all of
                :attr:`source_layers`. Must be a subset of
                :attr:`source_layers` when ``use_jacobian`` is ``True``.
            positions: Token positions to read out (Python indexing into the
                sequence; negative indices count from the end). ``None`` returns
                every position.
            max_seq_len: Truncate the prompt to this many tokens.
            use_jacobian: If ``False``, skip the ``J_l`` transport (vanilla
                logit-lens baseline).

        Returns:
            A triple ``(lens_logits, model_logits, input_ids)``. ``lens_logits``
            maps each requested layer to a ``[n_positions, vocab_size]`` tensor;
            ``model_logits`` is the model's actual final-layer logits at the
            same positions (same shape). ``n_positions`` is ``len(positions)``,
            or the full sequence length when ``positions`` is ``None``.

        Raises:
            ValueError: If any requested layer is out of range for the model,
                or (with ``use_jacobian``) not in :attr:`source_layers`.
        """
        if layers is None:
            layers = self.source_layers
        out_of_range = sorted(l for l in set(layers) if not 0 <= l < model.n_layers)
        if out_of_range:
            raise ValueError(
                f"layers {out_of_range} out of range for a {model.n_layers}-layer model"
            )
        unknown = set(layers) - set(self.source_layers)
        if use_jacobian and unknown:
            raise ValueError(
                f"layers {sorted(unknown)} not in source_layers; "
                f"fitted layers are {self.source_layers}"
            )
        final_layer = model.n_layers - 1
        record_at = sorted(set(layers) | {final_layer})

        input_ids = model.encode(prompt, max_length=max_seq_len)
        with ActivationRecorder(model.layers, at=record_at) as recorder:
            model.forward(input_ids)
            activations = {i: recorder.activations[i].detach() for i in record_at}

        def select(layer: int) -> torch.Tensor:
            """Residuals at the requested positions: ``[n_positions, d_model]``."""
            full = activations[layer][0]  # [seq_len, d_model]
            return (full if positions is None else full[list(positions)]).float()

        lens_logits: dict[int, torch.Tensor] = {}
        for layer in layers:
            residual = select(layer)
            if use_jacobian:
                residual = self.transport(residual, layer)
            lens_logits[layer] = model.unembed(residual).float().cpu()

        model_logits = model.unembed(select(final_layer)).float().cpu()
        return lens_logits, model_logits, input_ids
