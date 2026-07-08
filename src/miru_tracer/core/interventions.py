"""Activation interventions: steer, ablate, and swap lens readouts.

Follows the intervention scheme from Anthropic's global-workspace paper:
for a token t at layer ℓ, the *lens vector* v̂_t is the (unit-normalized)
residual-stream direction whose lens readout maximally hits t. Edits applied
to the layer-ℓ block output:

- **steer**:  ``h ← h + α·‖h‖·v̂_t``   (α < 0 steers away)
- **ablate**: ``h ← h − (v̂_tᵀh)·v̂_t``  (remove the component)
- **swap**:   ``h ← h − (v̂_aᵀh)·v̂_a + (v̂_aᵀh)·v̂_b`` (transfer the
  coefficient from a to b, preserving orthogonal components)

Unlike Neuronpedia's single-intervention UI, any number of interventions can
be active at once; they are applied in order within each forward pass.

Lens vectors: in "logit" basis v̂_t is the unembedding direction û_t itself;
in "jacobian" basis it solves ``J_ℓ v = û_t`` (least squares, falling back to
``J_ℓᵀ û_t`` if the solve is ill-conditioned), i.e. the pre-transport
direction that reads out as t.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch

from miru_tracer.core._jlens import JacobianLens
from miru_tracer.core.logging_config import get_logger

logger = get_logger(__name__)

INTERVENTION_KINDS = ("steer", "ablate", "swap")
VECTOR_BASES = ("logit", "jacobian")


@dataclass(frozen=True)
class Intervention:
    """One activation edit at one layer.

    Attributes:
        kind: "steer", "ablate", or "swap".
        layer: Block index whose output is edited.
        token_id: The readout token defining the direction.
        strength: Steer coefficient α (ignored for ablate/swap).
        token_id_to: Swap target token (swap only).
        basis: "jacobian" (needs a fitted lens at ``layer``) or "logit".
    """

    kind: str
    layer: int
    token_id: int
    strength: float = 1.0
    token_id_to: int | None = None
    basis: str = "jacobian"

    def __post_init__(self):
        if self.kind not in INTERVENTION_KINDS:
            raise ValueError(
                f"Unknown intervention kind: {self.kind!r}. Use one of {INTERVENTION_KINDS}."
            )
        if self.basis not in VECTOR_BASES:
            raise ValueError(
                f"Unknown basis: {self.basis!r}. Use one of {VECTOR_BASES}."
            )
        if self.kind == "swap" and self.token_id_to is None:
            raise ValueError("swap interventions need token_id_to")

    def describe(self, tokenizer=None) -> str:
        def name(token_id):
            if tokenizer is None:
                return str(token_id)
            return repr(tokenizer.convert_ids_to_tokens([token_id])[0])

        if self.kind == "steer":
            return f"steer {name(self.token_id)} @L{self.layer} (α={self.strength:+g})"
        if self.kind == "ablate":
            return f"ablate {name(self.token_id)} @L{self.layer}"
        return f"swap {name(self.token_id)}→{name(self.token_id_to)} @L{self.layer}"


def unembed_direction(model, token_id: int) -> torch.Tensor:
    """The unembedding (lm_head) row for a token, unit-normalized."""
    lm_head = getattr(model, "lm_head", None)
    if lm_head is None:  # GPT-NeoX naming
        lm_head = model.embed_out
    direction = lm_head.weight[token_id].detach().float()
    return direction / direction.norm()


def lens_vector(
    model,
    token_id: int,
    layer: int,
    *,
    basis: str = "jacobian",
    jlens: JacobianLens | None = None,
    n_layers: int | None = None,
) -> torch.Tensor:
    """Unit residual-stream direction at ``layer`` that reads out as ``token_id``."""
    u_hat = unembed_direction(model, token_id)
    is_final = n_layers is not None and layer == n_layers - 1
    if basis == "logit" or is_final or jlens is None or layer not in jlens.jacobians:
        if basis == "jacobian" and not is_final and (
            jlens is None or layer not in jlens.jacobians
        ):
            raise ValueError(
                f"jacobian basis requires a fitted lens covering layer {layer}"
            )
        return u_hat

    J = jlens.jacobians[layer].float().to(u_hat.device)
    try:
        solution = torch.linalg.lstsq(J, u_hat.unsqueeze(-1)).solution.squeeze(-1)
    except Exception:  # pragma: no cover - driver-dependent
        solution = J.T @ u_hat
    if not torch.isfinite(solution).all() or solution.norm() < 1e-8:
        logger.warning(
            f"lstsq ill-conditioned for layer {layer}; falling back to J^T u"
        )
        solution = J.T @ u_hat
    return solution / solution.norm()


class InterventionSet:
    """A compiled, immutable batch of interventions grouped by layer."""

    def __init__(
        self,
        interventions: list[Intervention],
        model,
        *,
        jlens: JacobianLens | None = None,
    ):
        self.interventions = list(interventions)
        n_layers = model.config.get_text_config().num_hidden_layers
        self._edits: dict[int, list[tuple[Intervention, torch.Tensor, torch.Tensor | None]]] = {}
        for iv in self.interventions:
            if not 0 <= iv.layer < n_layers:
                raise ValueError(f"layer {iv.layer} out of range (n_layers={n_layers})")
            v = lens_vector(
                model, iv.token_id, iv.layer,
                basis=iv.basis, jlens=jlens, n_layers=n_layers,
            )
            v_to = None
            if iv.kind == "swap":
                v_to = lens_vector(
                    model, iv.token_id_to, iv.layer,
                    basis=iv.basis, jlens=jlens, n_layers=n_layers,
                )
            self._edits.setdefault(iv.layer, []).append((iv, v, v_to))

    def __len__(self) -> int:
        return len(self.interventions)

    @property
    def layers(self) -> set[int]:
        return set(self._edits)

    def edit(self, layer: int, hidden: torch.Tensor) -> torch.Tensor:
        """Apply this set's layer-``layer`` edits to a ``[..., d_model]`` tensor."""
        for iv, v, v_to in self._edits.get(layer, ()):
            v = v.to(hidden.device, hidden.dtype)
            if iv.kind == "steer":
                scale = hidden.norm(dim=-1, keepdim=True)
                hidden = hidden + iv.strength * scale * v
            elif iv.kind == "ablate":
                coef = (hidden * v).sum(dim=-1, keepdim=True)
                hidden = hidden - coef * v
            else:  # swap
                v_to = v_to.to(hidden.device, hidden.dtype)
                coef = (hidden * v).sum(dim=-1, keepdim=True)
                hidden = hidden - coef * v + coef * v_to
        return hidden


def _locate_blocks(model):
    """The residual block list of a *ForCausalLM (Layout('model') and friends)."""
    for path in ("model.layers", "model.language_model.layers", "transformer.h", "gpt_neox.layers"):
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
        except AttributeError:
            continue
        return obj
    raise ValueError(f"cannot locate residual blocks in {type(model).__name__}")


@contextmanager
def apply_interventions(model, intervention_set: InterventionSet | None):
    """Context manager installing forward hooks that edit block outputs.

    Hooks registered here run before any hooks registered inside the context
    (PyTorch calls forward hooks in registration order and later hooks see
    earlier hooks' modified output), so an ActivationRecorder entered inside
    this context records the *edited* residuals.
    """
    if intervention_set is None or len(intervention_set) == 0:
        yield
        return

    blocks = _locate_blocks(model)
    handles = []

    def make_hook(layer: int):
        def hook(module, inputs, output):
            if torch.is_tensor(output):
                return intervention_set.edit(layer, output)
            edited = intervention_set.edit(layer, output[0])
            return (edited, *output[1:])

        return hook

    try:
        for layer in intervention_set.layers:
            handles.append(blocks[layer].register_forward_hook(make_hook(layer)))
        yield
    finally:
        for handle in handles:
            handle.remove()
