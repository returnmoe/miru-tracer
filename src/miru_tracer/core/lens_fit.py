"""Fitting Jacobian-lens matrices: library API and the fit-lens CLI.

Fitting is the expensive part of the Jacobian lens (one forward plus
``d_model/dim_batch`` backward passes per prompt). This module wraps the
vendored :func:`miru_tracer.core._jlens.fit` with chunked progress,
cancellation, and resumable checkpoints, and exposes the
``miru-tracer-fit-lens`` console command.

The lens artifact saved after every chunk is a valid (partially averaged)
lens, so the app can pick it up before the full corpus is done.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

from miru_tracer.core._jlens import JacobianLens, fit, from_hf
from miru_tracer.core.lens import get_lens_store
from miru_tracer.core.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_NUM_PROMPTS = 100
DEFAULT_DIM_BATCH = 4  # CPU-friendly; raise on GPU
DEFAULT_MAX_SEQ_LEN = 128
DEFAULT_CHUNK_SIZE = 5

# Fitting needs prompts longer than the fitter's skipped prefix (16 tokens).
MIN_PROMPT_CHARS = 200


@dataclass
class FitProgress:
    """State after each fitted chunk."""

    prompts_done: int
    prompts_total: int
    elapsed_s: float
    lens: JacobianLens


def prompts_from_file(path: str | Path) -> list[str]:
    """One prompt per non-empty line."""
    lines = Path(path).read_text().splitlines()
    return [line for line in (line.strip() for line in lines) if line]


def wikitext_prompts(n: int, *, min_chars: int = MIN_PROMPT_CHARS) -> list[str]:
    """Pull ``n`` paragraph-sized prompts from wikitext-103.

    Reads the dataset's parquet shard directly (pandas + pyarrow) instead of
    the `datasets` library, which currently trips over the legacy repo id.
    """
    import pandas as pd
    from huggingface_hub import hf_hub_download

    shard = hf_hub_download(
        "Salesforce/wikitext",
        "wikitext-103-raw-v1/train-00000-of-00002.parquet",
        repo_type="dataset",
    )
    frame = pd.read_parquet(shard, columns=["text"])
    prompts: list[str] = []
    for text in frame["text"]:
        text = text.strip()
        if len(text) >= min_chars and not text.startswith("="):
            prompts.append(text)
            if len(prompts) == n:
                break
    return prompts


def iter_fit_lens(
    model,
    tokenizer,
    prompts: list[str],
    *,
    out_path: str | Path,
    dim_batch: int = DEFAULT_DIM_BATCH,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
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

    wrapper = from_hf(model, tokenizer)  # force_bos: raw-text prompts want BOS
    start = time.perf_counter()

    for end in range(chunk_size, len(prompts) + chunk_size, chunk_size):
        end = min(end, len(prompts))
        lens = fit(
            wrapper,
            prompts[:end],
            dim_batch=dim_batch,
            max_seq_len=max_seq_len,
            checkpoint_path=str(checkpoint_path),
            resume=True,
        )
        lens.save(str(out_path))
        yield FitProgress(
            prompts_done=end,
            prompts_total=len(prompts),
            elapsed_s=time.perf_counter() - start,
            lens=lens,
        )
        if end == len(prompts):
            break
        if should_stop is not None and should_stop():
            logger.info(f"Lens fitting stopped at {end}/{len(prompts)} prompts")
            break


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
        "--num-prompts", type=int, default=DEFAULT_NUM_PROMPTS,
        help=f"wikitext prompts to average over (default {DEFAULT_NUM_PROMPTS})",
    )
    parser.add_argument(
        "--prompts-file", help="text file with one prompt per line (overrides wikitext)"
    )
    parser.add_argument(
        "--dim-batch", type=int, default=DEFAULT_DIM_BATCH,
        help=f"Jacobian rows per backward pass (default {DEFAULT_DIM_BATCH}; "
        "raise to 16-64 on a GPU)",
    )
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_SEQ_LEN)
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda"],
        help="where to run the model (auto = cuda if available)",
    )
    parser.add_argument(
        "--dtype", default="auto", choices=["auto", "float32", "bfloat16", "float16"],
        help="model dtype (auto = bfloat16 on cuda, float32 on cpu)",
    )
    parser.add_argument(
        "--out", help="output lens path (default: the app's lens cache dir)"
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="discard any existing checkpoint instead of resuming",
    )
    args = parser.parse_args(argv)

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
        dtype_name = "bfloat16" if device == "cuda" else "float32"
    dtype = getattr(torch, dtype_name)

    echo(f"Loading {args.model} ({dtype_name} on {device})...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = (
        AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device).eval()
    )
    if device == "cpu" and dtype_name == "float32":
        echo(
            "Note: fitting on CPU is slow (minutes per prompt). "
            "A GPU instance with --dim-batch 32 is strongly recommended; "
            "copy the resulting lens.pt back afterwards."
        )

    if args.prompts_file:
        prompts = prompts_from_file(args.prompts_file)
    else:
        echo("Loading wikitext prompts...")
        prompts = wikitext_prompts(args.num_prompts)
    echo(f"Fitting on {len(prompts)} prompts -> {out_path}")
    echo("This runs one forward + many backward passes per prompt; "
          "interrupt any time, re-run to resume.")

    for progress in iter_fit_lens(
        model,
        tokenizer,
        prompts,
        out_path=out_path,
        dim_batch=args.dim_batch,
        max_seq_len=args.max_length,
    ):
        rate = progress.elapsed_s / max(progress.prompts_done, 1)
        remaining = rate * (progress.prompts_total - progress.prompts_done)
        echo(
            f"  {progress.prompts_done}/{progress.prompts_total} prompts "
            f"({progress.elapsed_s:.0f}s elapsed, ~{remaining:.0f}s left) "
            f"-> saved {out_path}"
        )

    echo("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
