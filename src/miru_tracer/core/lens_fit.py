"""Fitting Jacobian-lens matrices: library API and the fit-lens CLI.

Fitting is the expensive part of the Jacobian lens (one forward plus
``d_model/dim_batch`` backward passes per prompt). This module wraps the
vendored :func:`miru_tracer.core._jlens.fit` with chunked progress,
cancellation, and resumable checkpoints, and exposes the
``miru-tracer-fit-lens`` console command.

The lens artifact saved after every chunk is a valid (partially averaged)
lens, so the app can pick it up before the full corpus is done. Artifacts
are written as safetensors by default (safe to share between machines);
passing an ``--out`` path ending in ``.pt`` writes the legacy torch.save
format instead.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

from miru_tracer.core._jlens import JacobianLens, fit, from_hf
from miru_tracer.core.lens import get_lens_store
from miru_tracer.core.lens_io import save_lens
from miru_tracer.core.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_NUM_PROMPTS = 1_000
DEFAULT_DIM_BATCH = 4  # CPU-friendly; raise on GPU
DEFAULT_MAX_SEQ_LEN = 128
DEFAULT_CHUNK_SIZE = 5
DEFAULT_MIN_PROMPTS = 100
DEFAULT_STOP_WINDOW = 10
DEFAULT_STOP_AT_DELTA = 0.002

# The default corpus rechunker mirrors Neuronpedia's 2,000-character contexts;
# its shorter final tail is retained only when it is still likely to exceed the
# fitter's skipped 16-token prefix.
MIN_PROMPT_CHARS = 200
MAX_PROMPT_CHARS = 2_000
WIKITEXT_TRAIN_SHARDS = (
    "wikitext-103-raw-v1/train-00000-of-00002.parquet",
    "wikitext-103-raw-v1/train-00001-of-00002.parquet",
)


@dataclass
class FitProgress:
    """State after each fitted chunk."""

    prompts_done: int
    prompts_total: int
    elapsed_s: float
    lens: JacobianLens
    prompts_processed: int
    prompts_processed_this_run: int
    converged: bool
    last_delta: float | None
    rolling_delta: float | None


def _prompt_sequence_sha256(prompts: list[str]) -> str:
    """Digest an ordered prompt list without retaining its text in metadata."""
    hasher = hashlib.sha256()
    for prompt in prompts:
        encoded = prompt.encode("utf-8")
        hasher.update(len(encoded).to_bytes(8, "big"))
        hasher.update(encoded)
    return hasher.hexdigest()


def _artifact_model_identifier(name_or_path: str) -> str:
    """Avoid embedding an absolute local filesystem path in shared artifacts."""
    path = Path(name_or_path).expanduser()
    return path.name if path.is_absolute() else name_or_path


def _local_model_location_sha256(name_or_path: str) -> str | None:
    """Checkpoint identity for local paths without publishing the path itself."""
    path = Path(name_or_path).expanduser()
    if not path.is_absolute() and not path.exists():
        return None
    return hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()


def _local_model_manifest_sha256(
    name_or_path: str, *, exclude: tuple[Path, ...] = ()
) -> str | None:
    """Fingerprint local model-file names, sizes, and mtimes cheaply."""
    path = Path(name_or_path).expanduser()
    if not path.exists():
        return None
    excluded = {item.expanduser().resolve() for item in exclude}
    files = (
        [path]
        if path.is_file()
        else sorted(
            item
            for item in path.iterdir()
            if item.is_file()
            and item.resolve() not in excluded
            and ".tmp." not in item.name
            and not item.name.endswith(".checkpoint.pt")
            and "lens" not in item.name.lower()
        )
    )
    hasher = hashlib.sha256()
    for item in files:
        stat = item.stat()
        name = item.name.encode("utf-8")
        hasher.update(len(name).to_bytes(8, "big"))
        hasher.update(name)
        hasher.update(stat.st_size.to_bytes(8, "big"))
        hasher.update(stat.st_mtime_ns.to_bytes(8, "big"))
    return hasher.hexdigest()


def _model_config_sha256(model) -> str | None:
    config = getattr(model, "config", None)
    if config is None or not hasattr(config, "to_json_string"):
        return None
    payload = config.to_json_string(use_diff=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _tokenizer_sha256(tokenizer) -> str:
    """Fingerprint tokenizer rules/vocabulary without storing their contents."""
    hasher = hashlib.sha256()
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None and hasattr(backend, "to_str"):
        payload = backend.to_str().encode("utf-8")
        hasher.update(len(payload).to_bytes(8, "big"))
        hasher.update(payload)
    else:
        for token, token_id in sorted(tokenizer.get_vocab().items()):
            encoded = token.encode("utf-8")
            hasher.update(len(encoded).to_bytes(8, "big"))
            hasher.update(encoded)
            hasher.update(int(token_id).to_bytes(8, "big", signed=True))
    chat_template = str(getattr(tokenizer, "chat_template", "")).encode("utf-8")
    hasher.update(len(chat_template).to_bytes(8, "big"))
    hasher.update(chat_template)
    return hasher.hexdigest()


def _with_runtime_provenance(
    model, tokenizer, provided: dict[str, object] | None
) -> dict[str, object]:
    provenance = dict(provided or {})
    config = getattr(model, "config", None)
    model_name = getattr(config, "_name_or_path", None)
    tokenizer_name = getattr(tokenizer, "name_or_path", None)
    if model_name:
        provenance.setdefault("model_name_or_path", _artifact_model_identifier(model_name))
        provenance.setdefault("model_location_sha256", _local_model_location_sha256(model_name))
        if "model_manifest_sha256" not in provenance:
            provenance["model_manifest_sha256"] = _local_model_manifest_sha256(model_name)
    commit = getattr(config, "_commit_hash", None)
    if isinstance(commit, str):
        provenance.setdefault("model_commit_hash", commit)
    if "model_config_sha256" not in provenance:
        provenance["model_config_sha256"] = _model_config_sha256(model)
    if tokenizer_name:
        provenance.setdefault("tokenizer_name_or_path", _artifact_model_identifier(tokenizer_name))
    if "tokenizer_sha256" not in provenance:
        provenance["tokenizer_sha256"] = _tokenizer_sha256(tokenizer)
    if "compute_dtype" not in provenance:
        try:
            dtype = next(model.parameters()).dtype
        except (AttributeError, StopIteration):
            pass
        else:
            provenance["compute_dtype"] = str(dtype).removeprefix("torch.")
    return provenance


def validate_lens_provenance(lens, model, tokenizer) -> tuple[str, str]:
    """Compare a fitted artifact with the loaded runtime.

    Returns ``(status, detail)`` where status is ``verified``, ``legacy`` or
    ``mismatch``. Legacy artifacts have no usable cryptographic provenance and
    therefore require an explicit UI opt-in before installation.
    """
    metadata = lens.fit_metadata or {}
    stored = metadata.get("provenance") or {}
    current = _with_runtime_provenance(model, tokenizer, None)
    strong_keys = (
        "model_config_sha256",
        "tokenizer_sha256",
        "model_commit_hash",
        "model_manifest_sha256",
    )
    compared = []
    for key in strong_keys:
        expected = stored.get(key)
        actual = current.get(key)
        if expected is None or actual is None:
            continue
        compared.append(key)
        if expected != actual:
            return "mismatch", f"{key} does not match the loaded model"
    if compared:
        return "verified", "verified by " + ", ".join(compared)
    return "legacy", "artifact has no comparable model/tokenizer provenance"


def prompts_from_file(path: str | Path) -> list[str]:
    """One prompt per non-empty line."""
    lines = Path(path).read_text().splitlines()
    return [line for line in (line.strip() for line in lines) if line]


def _chunk_text_records(
    records: Iterator[object],
    n: int,
    *,
    max_chars: int = MAX_PROMPT_CHARS,
    min_chars: int = MIN_PROMPT_CHARS,
) -> list[str]:
    """Concatenate text rows and rechunk them into stable fitting contexts."""
    if n <= 0:
        return []
    if max_chars <= 0 or min_chars <= 0:
        raise ValueError("max_chars and min_chars must be positive")
    prompts: list[str] = []
    buffer = ""
    for record in records:
        text = str(record).strip()
        if not text or text.startswith("="):
            continue
        buffer += " " + text
        while len(buffer) > max_chars:
            prompts.append(buffer[:max_chars].strip())
            buffer = buffer[max_chars:]
            if len(prompts) >= n:
                return prompts
    tail = buffer.strip()
    if tail and len(tail) >= min_chars and len(prompts) < n:
        prompts.append(tail)
    return prompts


def wikitext_prompts(
    n: int,
    *,
    max_chars: int = MAX_PROMPT_CHARS,
    min_chars: int = MIN_PROMPT_CHARS,
) -> list[str]:
    """Pull ``n`` rechunked prompts from the WikiText-103 training split.

    Reads the dataset's parquet shard directly (pandas + pyarrow) instead of
    the `datasets` library, which currently trips over the legacy repo id.
    Row concatenation/rechunking matches Neuronpedia's fitter and prevents
    WikiText's many short lines from becoming atypically short sequences.
    """
    import pandas as pd
    from huggingface_hub import hf_hub_download

    def records() -> Iterator[object]:
        for filename in WIKITEXT_TRAIN_SHARDS:
            shard = hf_hub_download(
                "Salesforce/wikitext",
                filename,
                repo_type="dataset",
            )
            frame = pd.read_parquet(shard, columns=["text"])
            yield from frame["text"]

    return _chunk_text_records(records(), n, max_chars=max_chars, min_chars=min_chars)


def iter_fit_lens(
    model,
    tokenizer,
    prompts: list[str],
    *,
    out_path: str | Path,
    dim_batch: int = DEFAULT_DIM_BATCH,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    min_prompts: int = DEFAULT_MIN_PROMPTS,
    stop_window: int = DEFAULT_STOP_WINDOW,
    stop_at_delta: float | None = DEFAULT_STOP_AT_DELTA,
    fit_provenance: dict[str, object] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> Iterator[FitProgress]:
    """Fit a lens in chunks, yielding progress after each.

    The vendored fitter's checkpoint records how many prompts are done, so
    passing a growing prefix of the same prompt list resumes exactly where the
    previous call (or a previous process) stopped. The saved artifact after
    each chunk is a valid lens averaged over the prompts so far.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_path.with_suffix(".checkpoint.pt")
    if not prompts:
        raise ValueError("at least one prompt is required")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if stop_at_delta == 0:
        stop_at_delta = None
    fit_provenance = _with_runtime_provenance(model, tokenizer, fit_provenance)

    wrapper = from_hf(model, tokenizer)  # force_bos: raw-text prompts want BOS
    start = time.perf_counter()

    # Skip directly to the first unfinished chunk on process restart. The
    # fitter itself validates that the ordered checkpointed prefix still
    # matches these prompts.
    next_idx = 0
    if checkpoint_path.exists():
        import torch

        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        next_idx = int(state.get("next_idx", 0))
    processed_at_start = next_idx
    first_end = min(max(next_idx + chunk_size, chunk_size), len(prompts))

    for end in range(first_end, len(prompts) + chunk_size, chunk_size):
        end = min(end, len(prompts))
        lens = fit(
            wrapper,
            prompts[:end],
            dim_batch=dim_batch,
            max_seq_len=max_seq_len,
            checkpoint_path=str(checkpoint_path),
            resume=True,
            min_prompts=min_prompts,
            stop_window=stop_window,
            stop_at_delta=stop_at_delta,
            fit_provenance=fit_provenance,
        )
        save_lens(lens, out_path)
        metadata = lens.fit_metadata or {}
        fit_state = metadata.get("fit", {})
        convergence = metadata.get("convergence", {})
        yield FitProgress(
            prompts_done=lens.n_prompts,
            prompts_total=len(prompts),
            elapsed_s=time.perf_counter() - start,
            lens=lens,
            prompts_processed=int(fit_state.get("processed_prompts", end)),
            prompts_processed_this_run=max(
                int(fit_state.get("processed_prompts", end)) - processed_at_start,
                0,
            ),
            converged=bool(convergence.get("converged", False)),
            last_delta=convergence.get("last_mean_relative_change"),
            rolling_delta=convergence.get("rolling_mean_relative_change"),
        )
        if bool(convergence.get("converged", False)) or end == len(prompts):
            break
        if should_stop is not None and should_stop():
            logger.info(f"Lens fitting stopped at {end}/{len(prompts)} prompts")
            break


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative finite number")
    return parsed


def main(argv: list[str] | None = None) -> int:
    """Console entry point: miru-tracer-fit-lens."""
    parser = argparse.ArgumentParser(
        prog="miru-tracer-fit-lens",
        description=(
            "Fit Jacobian-lens matrices for a HuggingFace model. "
            "Slow (backward passes per prompt) but one-off, checkpointed, "
            "and safe to interrupt and re-run."
        ),
    )
    parser.add_argument("model", help="HuggingFace model name, e.g. Qwen/Qwen3-0.6B")
    parser.add_argument(
        "--num-prompts",
        type=_positive_int,
        default=DEFAULT_NUM_PROMPTS,
        help=f"maximum prompt budget (default {DEFAULT_NUM_PROMPTS})",
    )
    parser.add_argument(
        "--min-prompts",
        type=_positive_int,
        default=DEFAULT_MIN_PROMPTS,
        help=f"successful-prompt floor before convergence may stop (default {DEFAULT_MIN_PROMPTS})",
    )
    parser.add_argument(
        "--stop-window",
        type=_positive_int,
        default=DEFAULT_STOP_WINDOW,
        help=f"recent prompt changes in the convergence mean (default {DEFAULT_STOP_WINDOW})",
    )
    parser.add_argument(
        "--stop-at-delta",
        type=_nonnegative_finite_float,
        default=DEFAULT_STOP_AT_DELTA,
        help="stop when rolling relative change is below this value "
        f"(default {DEFAULT_STOP_AT_DELTA}; 0 disables)",
    )
    parser.add_argument(
        "--prompts-file",
        help="text file with one prompt per line (overrides wikitext; capped by --num-prompts)",
    )
    parser.add_argument(
        "--dim-batch",
        type=_positive_int,
        default=DEFAULT_DIM_BATCH,
        help=f"Jacobian rows per backward pass (default {DEFAULT_DIM_BATCH}; "
        "raise to 16-64 on a GPU)",
    )
    parser.add_argument(
        "--max-length",
        type=_positive_int,
        default=DEFAULT_MAX_SEQ_LEN,
        help=f"maximum tokens per prompt (default {DEFAULT_MAX_SEQ_LEN})",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="where to run the model (auto = cuda if available)",
    )
    parser.add_argument(
        "--device-map",
        default=None,
        help='shard the model across devices, e.g. "auto" for multi-GPU '
        "(needed for models that don't fit on one GPU); overrides --device",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float32", "bfloat16", "float16"],
        help="model dtype (auto = bfloat16 on cuda, float32 on cpu)",
    )
    parser.add_argument(
        "--out",
        help="output lens path (default: the app's lens cache dir); "
        "a .pt extension writes the legacy torch.save format",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="discard any existing checkpoint instead of resuming",
    )
    args = parser.parse_args(argv)
    if args.max_length <= 17:
        parser.error(
            "--max-length must be at least 18 because the first 16 and final "
            "token positions are excluded"
        )

    import functools

    echo = functools.partial(print, flush=True)  # nohup/pipe friendliness

    from miru_tracer.core.logging_config import setup_logging

    setup_logging()  # surface the fitter's per-prompt INFO lines

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_path = Path(args.out) if args.out else get_lens_store().lens_path(args.model)
    checkpoint = out_path.with_suffix(".checkpoint.pt")
    if args.fresh and checkpoint.exists():
        checkpoint.unlink()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype_name = args.dtype
    if dtype_name == "auto":
        cuda = device == "cuda" or (args.device_map and torch.cuda.is_available())
        dtype_name = "bfloat16" if cuda else "float32"
    dtype = getattr(torch, dtype_name)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.device_map:
        echo(f"Loading {args.model} ({dtype_name}, device_map={args.device_map})...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=dtype, device_map=args.device_map
        ).eval()
    else:
        echo(f"Loading {args.model} ({dtype_name} on {device})...")
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device).eval()
    if not args.device_map and device == "cpu" and dtype_name == "float32":
        echo(
            "Note: fitting on CPU is slow (minutes per prompt). "
            "A GPU instance with --dim-batch 32 is strongly recommended; "
            "copy the resulting lens.safetensors back afterwards."
        )

    if args.prompts_file:
        prompts = prompts_from_file(args.prompts_file)[: args.num_prompts]
        corpus = {
            "kind": "text_file",
            "name": Path(args.prompts_file).name,
        }
    else:
        echo("Loading wikitext prompts...")
        prompts = wikitext_prompts(args.num_prompts)
        corpus = {
            "kind": "huggingface_dataset",
            "dataset_id": "Salesforce/wikitext",
            "dataset_config": "wikitext-103-raw-v1",
            "split": "train",
            "shards": list(WIKITEXT_TRAIN_SHARDS),
            "selection": {
                "min_chars": MIN_PROMPT_CHARS,
                "max_chars": MAX_PROMPT_CHARS,
                "exclude_headings": True,
                "concatenate_and_rechunk": True,
            },
        }
    if not prompts:
        parser.error("the selected corpus contains no non-empty prompts")

    model_commit = getattr(model.config, "_commit_hash", None)
    artifact_model_id = _artifact_model_identifier(args.model)
    fit_provenance = {
        "model_name_or_path": artifact_model_id,
        "model_location_sha256": _local_model_location_sha256(args.model),
        "model_manifest_sha256": _local_model_manifest_sha256(
            args.model, exclude=(out_path, checkpoint)
        ),
        "model_commit_hash": model_commit if isinstance(model_commit, str) else None,
        "model_config_sha256": _model_config_sha256(model),
        "tokenizer_name_or_path": artifact_model_id,
        "tokenizer_sha256": _tokenizer_sha256(tokenizer),
        "corpus": corpus,
        "prompt_budget": args.num_prompts,
        "selected_prompts": len(prompts),
        "prompt_sequence_sha256": _prompt_sequence_sha256(prompts),
        "compute_dtype": dtype_name,
    }
    threshold = None if args.stop_at_delta == 0 else args.stop_at_delta
    stop_description = (
        "disabled"
        if threshold is None
        else f"rolling {args.stop_window}-prompt d_mean < {threshold:g} "
        f"after {args.min_prompts} fitted prompts"
    )
    echo(f"Fitting up to {len(prompts)} prompts -> {out_path}")
    echo(f"Convergence stop: {stop_description}")
    echo(
        "This runs one forward + many backward passes per prompt; "
        "interrupt any time, re-run to resume."
    )

    last_progress: FitProgress | None = None
    for progress in iter_fit_lens(
        model,
        tokenizer,
        prompts,
        out_path=out_path,
        dim_batch=args.dim_batch,
        max_seq_len=args.max_length,
        min_prompts=args.min_prompts,
        stop_window=args.stop_window,
        stop_at_delta=threshold,
        fit_provenance=fit_provenance,
    ):
        last_progress = progress
        rate = progress.elapsed_s / max(progress.prompts_processed_this_run, 1)
        remaining = rate * (progress.prompts_total - progress.prompts_processed)
        delta = "n/a" if progress.last_delta is None else f"{progress.last_delta:.3g}"
        rolling = "n/a" if progress.rolling_delta is None else f"{progress.rolling_delta:.3g}"
        timing = (
            f"{progress.elapsed_s:.0f}s elapsed"
            if progress.converged
            else f"{progress.elapsed_s:.0f}s elapsed, ~{remaining:.0f}s left"
        )
        echo(
            f"  {progress.prompts_processed}/{progress.prompts_total} processed, "
            f"{progress.prompts_done} fitted; d_mean={delta}, rolling={rolling} "
            f"({timing}) -> saved {out_path}"
        )

    if last_progress is not None and last_progress.converged:
        echo(
            f"Converged after {last_progress.prompts_done} fitted prompts "
            f"(rolling d_mean={last_progress.rolling_delta:.3g})."
        )
    echo(
        "Done."
        if last_progress is None
        else f"Done: {last_progress.prompts_done} prompts in the final lens."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
