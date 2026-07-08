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
from pathlib import Path

import torch

from miru_tracer.core._jlens import JacobianLens

FORMAT_MARKER = "miru-tracer/jacobian-lens"
FORMAT_VERSION = "1"


def save_lens(
    lens: JacobianLens, path: str | Path, *, dtype: torch.dtype = torch.float16
) -> None:
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
                f"{path} is not a JacobianLens file "
                f"(missing the {FORMAT_MARKER!r} metadata marker)"
            )
        # safe_open handles support .keys() but not iteration
        jacobians = {
            int(key.removeprefix("J.")): f.get_tensor(key)
            for key in f.keys()  # noqa: SIM118
        }
    return JacobianLens(
        jacobians=jacobians,
        n_prompts=int(metadata["n_prompts"]),
        d_model=int(metadata["d_model"]),
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
        "dst", nargs="?",
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
