"""Lens artifact I/O: safetensors by default, legacy torch.save ``.pt`` fallback.

New artifacts are ``.safetensors`` — no pickle, so they are safe to share
between machines. ``.pt`` files written by older versions (or by passing a
``.pt`` output path explicitly) still round-trip through the vendored
:class:`JacobianLens` codec. The format is chosen by file extension.

The safetensors layout flattens the lens payload (safetensors only stores a
flat name→tensor dict plus a string→string metadata header): each ``J_ℓ``
matrix is stored under ``J.<layer>`` and the scalars/lists go in metadata.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import torch

from miru_tracer.core._jlens import JacobianLens

FORMAT_MARKER = "miru-tracer/jacobian-lens"
FORMAT_VERSION = "1"
_LAYER_KEY_RE = re.compile(r"J\.(0|[1-9][0-9]*)\Z")


def _invalid_artifact(path: Path, detail: str) -> ValueError:
    return ValueError(f"invalid JacobianLens file {path}: {detail}")


def _positive_int_metadata(metadata: dict[str, str], key: str, path: Path) -> int:
    raw = metadata.get(key)
    if raw is None:
        raise _invalid_artifact(path, f"missing required {key!r} metadata")
    try:
        value = int(raw)
    except ValueError as exc:
        raise _invalid_artifact(
            path, f"{key!r} metadata must be a positive integer; got {raw!r}"
        ) from exc
    if value <= 0:
        raise _invalid_artifact(path, f"{key!r} metadata must be a positive integer; got {raw!r}")
    return value


def _source_layers_metadata(metadata: dict[str, str], path: Path) -> list[int]:
    raw = metadata.get("source_layers")
    if raw is None:
        raise _invalid_artifact(path, "missing required 'source_layers' metadata")
    try:
        source_layers = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise _invalid_artifact(
            path, f"'source_layers' metadata is not valid JSON: {raw!r}"
        ) from exc
    if not isinstance(source_layers, list):
        raise _invalid_artifact(path, "'source_layers' metadata must be a JSON list")
    if any(type(layer) is not int or layer < 0 for layer in source_layers):
        raise _invalid_artifact(
            path,
            "'source_layers' metadata must contain only nonnegative integers",
        )
    if source_layers != sorted(set(source_layers)):
        raise _invalid_artifact(
            path, "'source_layers' metadata must be sorted and contain no duplicates"
        )
    return source_layers


def save_lens(lens: JacobianLens, path: str | Path, *, dtype: torch.dtype = torch.float16) -> None:
    """Write a lens artifact. ``.pt`` paths use the legacy torch.save codec;
    everything else is safetensors. Jacobians are stored as ``dtype`` (fp16,
    matching :meth:`JacobianLens.save`). The safetensors write is atomic
    (tmp + ``os.replace``): fitting rewrites the artifact after every chunk
    while the app may be reading it."""
    path = Path(path)
    if path.suffix == ".pt":
        lens.save(str(path), dtype=dtype)
        return
    from safetensors.torch import save_file

    tensors = {f"J.{layer}": J.to(dtype).contiguous() for layer, J in lens.jacobians.items()}
    metadata = {
        "format": FORMAT_MARKER,
        "version": FORMAT_VERSION,
        "n_prompts": str(lens.n_prompts),
        "d_model": str(lens.d_model),
        "source_layers": json.dumps(lens.source_layers),
    }
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    save_file(tensors, str(tmp), metadata=metadata)
    os.replace(tmp, path)


def load_lens(path: str | Path) -> JacobianLens:
    """Load a lens artifact in either format.

    Raises ValueError with a "not a JacobianLens file" message on non-lens
    files, in both codecs."""
    path = Path(path)
    if path.suffix != ".safetensors":
        return JacobianLens.load(str(path))
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as f:
        metadata = f.metadata() or {}
        if metadata.get("format") != FORMAT_MARKER:
            raise ValueError(
                f"{path} is not a JacobianLens file (missing the {FORMAT_MARKER!r} metadata marker)"
            )
        version = metadata.get("version")
        if version is None:
            raise _invalid_artifact(path, "missing required 'version' metadata")
        if version != FORMAT_VERSION:
            raise _invalid_artifact(
                path,
                f"unsupported format version {version!r}; expected {FORMAT_VERSION!r}",
            )

        n_prompts = _positive_int_metadata(metadata, "n_prompts", path)
        d_model = _positive_int_metadata(metadata, "d_model", path)
        source_layers = _source_layers_metadata(metadata, path)

        # safe_open handles support .keys() but not iteration.
        keys = list(f.keys())  # noqa: SIM118
        if not keys:
            raise _invalid_artifact(path, "contains no Jacobian matrices")
        invalid_keys = [key for key in keys if _LAYER_KEY_RE.fullmatch(key) is None]
        if invalid_keys:
            raise _invalid_artifact(
                path,
                "tensor keys must be nonnegative integer layer keys of the form "
                f"'J.<layer>'; got {invalid_keys!r}",
            )
        key_layers = sorted(int(key[2:]) for key in keys)
        if source_layers != key_layers:
            raise _invalid_artifact(
                path,
                f"'source_layers' metadata {source_layers!r} does not match "
                f"tensor layers {key_layers!r}",
            )

        jacobians: dict[int, torch.Tensor] = {}
        expected_shape = (d_model, d_model)
        for key, layer in zip(keys, (int(key[2:]) for key in keys), strict=True):
            tensor = f.get_tensor(key)
            if not tensor.is_floating_point():
                raise _invalid_artifact(
                    path, f"tensor {key!r} must have a floating dtype; got {tensor.dtype}"
                )
            if tuple(tensor.shape) != expected_shape:
                raise _invalid_artifact(
                    path,
                    f"tensor {key!r} must have shape {expected_shape}; got {tuple(tensor.shape)}",
                )
            jacobians[layer] = tensor
    return JacobianLens(
        jacobians=jacobians,
        n_prompts=n_prompts,
        d_model=d_model,
    )


def convert_main(argv: list[str] | None = None) -> int:
    """Console entry point: miru-tracer-convert-lens."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="miru-tracer-convert-lens",
        description=(
            "Convert a lens artifact between formats — the output extension "
            "decides: .safetensors (default, safe to share) or legacy .pt."
        ),
    )
    parser.add_argument("src", help="input lens file (.safetensors or .pt)")
    parser.add_argument(
        "dst",
        nargs="?",
        help="output path (default: the input with a .safetensors extension)",
    )
    args = parser.parse_args(argv)

    src = Path(args.src)
    dst = Path(args.dst) if args.dst else src.with_suffix(".safetensors")
    if src.resolve() == dst.resolve():
        parser.error(f"source and destination are the same file: {src}")
    lens = load_lens(src)
    save_lens(lens, dst)
    print(f"Wrote {dst}: {lens}")
    return 0
