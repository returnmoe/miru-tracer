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
import os
import platform
import shutil
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import suppress
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from miru_tracer.core._jlens import JacobianLens, fit, from_hf
from miru_tracer.core._jlens.fitting import NoFittedPromptsError
from miru_tracer.core.fit_diagnostics import (
    DEFAULT_TELEMETRY_INTERVAL_S,
    cleanup_runtime,
    cuda_allocator_backend,
    filesystem_info,
    log_fit_telemetry,
)
from miru_tracer.core.lens import get_lens_store
from miru_tracer.core.lens_io import save_lens
from miru_tracer.core.lens_provenance import (
    MODEL_ARCHITECTURE_HASH_KIND,
    MODEL_CONFIG_HASH_KIND,
)
from miru_tracer.core.lens_provenance import (
    artifact_model_identifier as _artifact_model_identifier,
)
from miru_tracer.core.lens_provenance import (
    local_model_location_sha256 as _local_model_location_sha256,
)
from miru_tracer.core.lens_provenance import (
    local_model_manifest_sha256 as _local_model_manifest_sha256,
)
from miru_tracer.core.lens_provenance import (
    model_architecture_sha256 as _model_architecture_sha256,
)
from miru_tracer.core.lens_provenance import (
    model_config_sha256 as _model_config_sha256,
)
from miru_tracer.core.lens_provenance import (
    tokenizer_sha256 as _tokenizer_sha256,
)
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
WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
WIKITEXT_TRAIN_SHARDS = (
    "wikitext-103-raw-v1/train-00000-of-00002.parquet",
    "wikitext-103-raw-v1/train-00001-of-00002.parquet",
)

_HF_HOME_SUBDIRS = {
    "HF_HUB_CACHE": "hub",
    "HUGGINGFACE_HUB_CACHE": "hub",
    "TRANSFORMERS_CACHE": "hub",
    "HF_XET_CACHE": "xet",
    "HF_ASSETS_CACHE": "assets",
    "HF_MODULES_CACHE": "modules",
}


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
        provenance["model_config_sha256_kind"] = MODEL_CONFIG_HASH_KIND
    if "model_architecture_sha256" not in provenance:
        provenance["model_architecture_sha256"] = _model_architecture_sha256(model)
        provenance["model_architecture_sha256_kind"] = MODEL_ARCHITECTURE_HASH_KIND
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


def _installed_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "unknown"


def _device_map_summary(model) -> str:
    device_map = getattr(model, "hf_device_map", None)
    if not isinstance(device_map, dict):
        return "none"
    counts = Counter(str(device) for device in device_map.values())
    return ",".join(f"{device}:{counts[device]}" for device in sorted(counts))


def _linear_attention_backend_status(
    model,
) -> tuple[bool, set[tuple[str, str]], set[str]]:
    """Inspect the Gated DeltaNet functions selected on each model layer.

    Transformers binds these callables to the layer instance during
    initialization; its module-level imports can remain ``None`` even while an
    instance has selected the explicit PyTorch fallback.
    """
    has_gated_delta = False
    implementations: set[tuple[str, str]] = set()
    for module in model.modules():
        class_name = type(module).__name__.lower()
        if "gateddelta" not in class_name and "gated_delta" not in class_name:
            continue
        has_gated_delta = True
        for component in (
            "chunk_gated_delta_rule",
            "recurrent_gated_delta_rule",
            "causal_conv1d_fn",
        ):
            implementation = getattr(module, component, None)
            if implementation is None:
                name = "missing"
            else:
                name = (
                    f"{getattr(implementation, '__module__', type(implementation).__module__)}."
                    f"{getattr(implementation, '__name__', type(implementation).__name__)}"
                )
            implementations.add((component, name))

    fallback_components = {
        component
        for component, implementation in implementations
        if component in {"chunk_gated_delta_rule", "causal_conv1d_fn"}
        and (implementation == "missing" or "torch" in implementation.lower())
    }
    if has_gated_delta and not implementations:
        fallback_components.add("backend_introspection")
    return has_gated_delta, implementations, fallback_components


def _log_linear_attention_backends(model) -> bool:
    """Record Gated DeltaNet implementations and return fast-path readiness."""
    has_gated_delta, implementations, fallback_components = _linear_attention_backend_status(model)
    for component, implementation in sorted(implementations):
        logger.info(
            "fit_backend component=%s implementation=%s",
            component,
            implementation,
        )
        if component in fallback_components:
            logger.warning(
                "fit_runtime_warning kind=linear_attention_fallback component=%s "
                "implementation=%s detail=install_the_model_recommended_fast_kernels",
                component,
                implementation,
            )
    if "backend_introspection" in fallback_components:
        logger.warning(
            "fit_runtime_warning kind=linear_attention_fallback "
            "component=backend_introspection implementation=unknown "
            "detail=could_not_verify_model_fast_kernels"
        )
    return not has_gated_delta or not fallback_components


def _log_fit_environment(
    model,
    wrapper,
    *,
    out_path: Path,
    chunk_size: int,
    prompt_budget: int,
    requested_device_map: str | None,
    start_prompt: int,
    empty_cuda_cache: bool,
) -> None:
    """Log enough immutable run context to compare slow and healthy processes."""
    import torch

    config = getattr(model, "config", None)
    allocator_config = os.getenv(
        "PYTORCH_ALLOC_CONF",
        os.getenv("PYTORCH_CUDA_ALLOC_CONF", "unset"),
    )
    logger.info(
        "fit_environment host=%s pid=%d python=%s kernel=%s torch=%s "
        "transformers=%s accelerate=%s "
        "cuda_runtime=%s model_class=%s model_type=%s requested_device_map=%s "
        "resolved_device_map=%s cuda_visible_devices=%s allocator_config=%s "
        "allocator_backend=%s cuda_cache_cleanup=%s torch_threads=%d "
        "torch_interop_threads=%d slurm_job_id=%s",
        platform.node().replace(" ", "_") or "unknown",
        os.getpid(),
        platform.python_version(),
        platform.release().replace(" ", "_"),
        torch.__version__,
        _installed_version("transformers"),
        _installed_version("accelerate"),
        torch.version.cuda or "none",
        type(model).__name__,
        getattr(config, "model_type", "unknown"),
        requested_device_map or "none",
        _device_map_summary(model),
        os.getenv("CUDA_VISIBLE_DEVICES", "unset").replace(" ", "_"),
        allocator_config.replace(" ", "_"),
        cuda_allocator_backend(),
        "chunk" if empty_cuda_cache else "disabled",
        torch.get_num_threads(),
        torch.get_num_interop_threads(),
        os.getenv("SLURM_JOB_ID", "unset").replace(" ", "_"),
    )
    try:
        disk = shutil.disk_usage(out_path.parent)
    except OSError:
        disk_free_bytes = -1
    else:
        disk_free_bytes = disk.free
    source_layer_count = max(wrapper.n_layers - 1, 0)
    checkpoint_bytes = source_layer_count * wrapper.d_model**2 * 4
    artifact_bytes = source_layer_count * wrapper.d_model**2 * 2
    remaining_prompts = max(prompt_budget - start_prompt, 0)
    planned_chunks = math.ceil(remaining_prompts / chunk_size)
    planned_write_bytes = planned_chunks * (checkpoint_bytes + artifact_bytes)
    storage = filesystem_info(out_path.parent)
    checkpoint_path = out_path.with_suffix(".checkpoint.pt")
    stale_temp_paths = tuple(out_path.parent.glob(f"{out_path.name}.tmp.*")) + tuple(
        checkpoint_path.parent.glob(f"{checkpoint_path.name}.tmp.*")
    )
    stale_temp_bytes = 0
    for temp_path in stale_temp_paths:
        with suppress(OSError):
            stale_temp_bytes += temp_path.stat().st_size
    required_output_free_bytes = (
        max(checkpoint_bytes, artifact_bytes)
        if checkpoint_path.exists()
        else checkpoint_bytes + artifact_bytes
    )
    logger.info(
        "fit_storage_plan chunk_size=%d planned_chunks=%d source_layers=%d d_model=%d "
        "checkpoint_estimate_bytes=%d artifact_estimate_bytes=%d "
        "planned_logical_write_bytes=%d output_disk_free_bytes=%d "
        "atomic_rewrite_headroom_bytes=%d stale_temp_files=%d stale_temp_bytes=%d "
        "output_filesystem=%s output_mount=%s",
        chunk_size,
        planned_chunks,
        source_layer_count,
        wrapper.d_model,
        checkpoint_bytes,
        artifact_bytes,
        planned_write_bytes,
        disk_free_bytes,
        max(checkpoint_bytes, artifact_bytes),
        len(stale_temp_paths),
        stale_temp_bytes,
        storage.get("output_filesystem", "unknown"),
        storage.get("output_mount", "unknown").replace(" ", "_"),
    )
    if stale_temp_paths:
        logger.warning(
            "fit_runtime_warning kind=stale_atomic_temp_files files=%d bytes=%d "
            "detail=inspect_output_directory_and_remove_orphaned_tmp_pid_files_when_no_fit_is_active",
            len(stale_temp_paths),
            stale_temp_bytes,
        )
    if 0 <= disk_free_bytes < required_output_free_bytes:
        logger.warning(
            "fit_runtime_warning kind=low_output_disk_headroom output_disk_free_bytes=%d "
            "required_output_free_bytes=%d atomic_rewrite_headroom_bytes=%d "
            "detail=checkpoint_and_artifact_writes_may_exhaust_output_storage",
            disk_free_bytes,
            required_output_free_bytes,
            max(checkpoint_bytes, artifact_bytes),
        )
    if checkpoint_bytes >= 1024**3:
        logger.warning(
            "fit_runtime_warning kind=large_checkpoint checkpoint_estimate_bytes=%d "
            "chunk_size=%d detail=checkpoint_and_artifact_are_written_once_per_chunk",
            checkpoint_bytes,
            chunk_size,
        )
    if planned_write_bytes >= 100 * 1024**3:
        logger.warning(
            "fit_runtime_warning kind=large_planned_io planned_logical_write_bytes=%d "
            "chunk_size=%d detail=use_fast_local_output_storage_or_raise_chunk_size",
            planned_write_bytes,
            chunk_size,
        )
    if requested_device_map is not None:
        logger.warning(
            "fit_runtime_warning kind=inference_device_map detail="
            "accelerate_device_map_is_layer_dispatch_not_data_parallelism;"
            "multi_gpu_layers_usually_execute_serially"
        )
    if torch.cuda.is_available():
        for device in range(torch.cuda.device_count()):
            try:
                properties = torch.cuda.get_device_properties(device)
            except Exception:
                continue
            logger.info(
                "fit_environment_gpu device=%d name=%s total_memory_bytes=%d capability=%d.%d",
                device,
                properties.name.replace(" ", "_"),
                properties.total_memory,
                properties.major,
                properties.minor,
            )
    _log_linear_attention_backends(model)
    log_fit_telemetry(
        logger,
        "run_start",
        global_prompt=start_prompt,
        process_prompt=0,
    )


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
    cache_dir: str | Path | None = None,
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
                revision=WIKITEXT_REVISION,
                repo_type="dataset",
                cache_dir=cache_dir,
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
    device_map_label: str | None = None,
    empty_cuda_cache: bool = False,
    telemetry_interval_s: float | None = DEFAULT_TELEMETRY_INTERVAL_S,
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
    initial_resume_state = None
    if checkpoint_path.exists():
        import torch

        initial_resume_state = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(initial_resume_state, dict):
            raise ValueError(f"checkpoint at {checkpoint_path} has invalid state")
        next_idx = int(initial_resume_state.get("next_idx", 0))
    processed_at_start = next_idx
    _log_fit_environment(
        model,
        wrapper,
        out_path=out_path,
        chunk_size=chunk_size,
        prompt_budget=len(prompts),
        requested_device_map=(
            device_map_label or ("configured" if getattr(model, "hf_device_map", None) else None)
        ),
        start_prompt=processed_at_start,
        empty_cuda_cache=empty_cuda_cache,
    )
    first_end = min(max(next_idx + chunk_size, chunk_size), len(prompts))
    process_logical_write_bytes = 0
    diagnostic_prompt_durations: list[float] = []

    for end in range(first_end, len(prompts) + chunk_size, chunk_size):
        end = min(end, len(prompts))
        try:
            checkpoint_before = checkpoint_path.stat()
        except OSError:
            checkpoint_before = None
        fit_start = time.perf_counter()
        try:
            lens = fit(
                wrapper,
                prompts[:end],
                dim_batch=dim_batch,
                max_seq_len=max_seq_len,
                checkpoint_path=str(checkpoint_path),
                checkpoint_every=None,
                resume=True,
                min_prompts=min_prompts,
                stop_window=stop_window,
                stop_at_delta=stop_at_delta,
                fit_provenance=fit_provenance,
                log_diagnostics=True,
                diagnostic_run_start_idx=processed_at_start,
                diagnostic_interval_s=telemetry_interval_s,
                diagnostic_prompt_durations=diagnostic_prompt_durations,
                resume_state=initial_resume_state,
            )
        except NoFittedPromptsError:
            # A custom corpus can begin with a complete chunk of prompts that
            # are too short after tokenization. ``fit`` has checkpointed that
            # processed prefix, so continue growing the prefix rather than
            # aborting before a later usable prompt. If this was the final
            # prefix, preserve the clear library error.
            initial_resume_state = None
            if end == len(prompts):
                raise
            logger.warning(
                "fit_chunk_skipped chunk_end=%d detail=no_usable_prompts_yet;"
                "continuing_to_next_chunk",
                end,
            )
            continue
        initial_resume_state = None
        fit_elapsed_s = time.perf_counter() - fit_start
        artifact_start = time.perf_counter()
        save_lens(lens, out_path)
        artifact_elapsed_s = time.perf_counter() - artifact_start
        try:
            checkpoint_after = checkpoint_path.stat()
            checkpoint_bytes = checkpoint_after.st_size
        except OSError:
            checkpoint_after = None
            checkpoint_bytes = -1
        checkpoint_written = checkpoint_after is not None and (
            checkpoint_before is None
            or checkpoint_after.st_mtime_ns != checkpoint_before.st_mtime_ns
            or checkpoint_after.st_size != checkpoint_before.st_size
        )
        try:
            artifact_bytes = out_path.stat().st_size
        except OSError:
            artifact_bytes = -1
        try:
            output_disk_free_bytes = shutil.disk_usage(out_path.parent).free
        except OSError:
            output_disk_free_bytes = -1
        if checkpoint_written and checkpoint_bytes >= 0:
            process_logical_write_bytes += checkpoint_bytes
        if artifact_bytes >= 0:
            process_logical_write_bytes += artifact_bytes
        logger.info(
            "fit_io event=chunk_saved chunk_end=%d checkpoint_written=%s "
            "checkpoint_bytes=%d artifact_bytes=%d process_logical_write_bytes=%d "
            "output_disk_free_bytes=%d fit_elapsed_s=%.3f artifact_write_s=%.3f",
            end,
            str(checkpoint_written).lower(),
            checkpoint_bytes,
            artifact_bytes,
            process_logical_write_bytes,
            output_disk_free_bytes,
            fit_elapsed_s,
            artifact_elapsed_s,
        )
        metadata = lens.fit_metadata or {}
        fit_state = metadata.get("fit", {})
        convergence = metadata.get("convergence", {})
        converged = bool(convergence.get("converged", False))
        progress = FitProgress(
            prompts_done=lens.n_prompts,
            prompts_total=len(prompts),
            elapsed_s=time.perf_counter() - start,
            lens=lens,
            prompts_processed=int(fit_state.get("processed_prompts", end)),
            prompts_processed_this_run=max(
                int(fit_state.get("processed_prompts", end)) - processed_at_start,
                0,
            ),
            converged=converged,
            last_delta=convergence.get("last_mean_relative_change"),
            rolling_delta=convergence.get("rolling_mean_relative_change"),
        )
        cleanup_runtime(
            logger,
            chunk_end=end,
            empty_cuda_cache=empty_cuda_cache,
        )
        yield progress
        # Drop the full fp32 mean before evaluating the next ``fit(...)`` RHS.
        # The caller may retain it intentionally through ``progress.lens``;
        # the CLI consumes each progress item without doing so.
        del progress, lens
        log_fit_telemetry(logger, "chunk_progress_released", chunk_end=end)
        if converged or end == len(prompts):
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


def _configure_hf_home(path: str | Path) -> tuple[Path, Path]:
    """Set one explicit, process-wide Hugging Face cache root.

    The fit command is a short-lived process, so configuring the standard
    environment before importing Transformers/Hugging Face also covers side
    caches (Xet, assets, and remote modules). Callers still pass the returned
    Hub directory explicitly to downloads, which keeps this reliable if a
    Hugging Face module was imported earlier by an embedding process.
    """
    root = Path(path).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(root)
    for name, subdir in _HF_HOME_SUBDIRS.items():
        os.environ[name] = str(root / subdir)
    return root, root / "hub"


def main(argv: list[str] | None = None) -> int:
    """Console entry point: miru-tracer-fit-lens."""
    parser = argparse.ArgumentParser(
        prog="miru-tracer-fit-lens",
        description=(
            "Fit Jacobian-lens matrices for a HuggingFace model. "
            "Slow (backward passes per prompt) but one-off and resumable "
            "with configurable chunk checkpoints."
        ),
    )
    parser.add_argument("model", help="HuggingFace model name, e.g. Qwen/Qwen3-0.6B")
    parser.add_argument(
        "--revision",
        help="immutable Hugging Face model revision (commit recommended for reproducible fits)",
    )
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
        "--chunk-size",
        type=_positive_int,
        default=DEFAULT_CHUNK_SIZE,
        help="prompts between checkpoint/artifact writes "
        f"(default {DEFAULT_CHUNK_SIZE}; larger reduces I/O)",
    )
    parser.add_argument(
        "--empty-cuda-cache",
        action="store_true",
        help="release unused PyTorch CUDA cache after each saved chunk; "
        "off by default so allocator behavior remains measurable",
    )
    parser.add_argument(
        "--telemetry-interval",
        type=_nonnegative_finite_float,
        default=DEFAULT_TELEMETRY_INTERVAL_S,
        help="seconds between intra-prompt process/GPU samples "
        f"(default {DEFAULT_TELEMETRY_INTERVAL_S:g}; 0 disables periodic samples)",
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
        "--require-fast-kernels",
        action="store_true",
        help="fail before fitting a Gated DeltaNet model when its optional "
        "linear-attention or causal-convolution fast path is unavailable",
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
        "--hf-home",
        help="Hugging Face cache root (Hub, Xet, assets, and modules); "
        "independent of --out and standard HF environment defaults",
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

    hf_hub_cache: Path | None = None
    if args.hf_home:
        hf_home, hf_hub_cache = _configure_hf_home(args.hf_home)
        echo(f"Hugging Face cache: {hf_home}")

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

    cache_kwargs = {} if hf_hub_cache is None else {"cache_dir": hf_hub_cache}
    if args.revision:
        cache_kwargs["revision"] = args.revision
    tokenizer = AutoTokenizer.from_pretrained(args.model, **cache_kwargs)
    revision_label = f"@{args.revision}" if args.revision else ""
    if args.device_map:
        echo(
            f"Loading {args.model}{revision_label} ({dtype_name}, device_map={args.device_map})..."
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=dtype, device_map=args.device_map, **cache_kwargs
        ).eval()
    else:
        echo(f"Loading {args.model}{revision_label} ({dtype_name} on {device})...")
        model = (
            AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, **cache_kwargs)
            .to(device)
            .eval()
        )
    has_gated_delta, _implementations, fallback_components = _linear_attention_backend_status(model)
    if args.require_fast_kernels and has_gated_delta and fallback_components:
        parser.error(
            "--require-fast-kernels was requested, but the loaded model is using "
            "or may be using fallback components: " + ", ".join(sorted(fallback_components))
        )
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
        prompts = wikitext_prompts(args.num_prompts, cache_dir=hf_hub_cache)
        corpus = {
            "kind": "huggingface_dataset",
            "dataset_id": "Salesforce/wikitext",
            "dataset_revision": WIKITEXT_REVISION,
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
        "model_config_sha256_kind": MODEL_CONFIG_HASH_KIND,
        "model_architecture_sha256": _model_architecture_sha256(model),
        "model_architecture_sha256_kind": MODEL_ARCHITECTURE_HASH_KIND,
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
        f"checkpoint/artifact writes occur every {args.chunk_size} prompts. "
        "Interrupt any time, re-run to resume."
    )
    echo(
        "CUDA cache cleanup: "
        + (
            "empty unused cache after each chunk (--empty-cuda-cache enabled)."
            if args.empty_cuda_cache
            else "allocator-managed (use --empty-cuda-cache for an A/B run)."
        )
    )
    echo(
        "Intra-prompt telemetry: "
        + (
            f"every {args.telemetry_interval:g}s."
            if args.telemetry_interval
            else "disabled (prompt boundary records remain enabled)."
        )
    )

    progress_iterator = iter_fit_lens(
        model,
        tokenizer,
        prompts,
        out_path=out_path,
        dim_batch=args.dim_batch,
        max_seq_len=args.max_length,
        chunk_size=args.chunk_size,
        min_prompts=args.min_prompts,
        stop_window=args.stop_window,
        stop_at_delta=threshold,
        fit_provenance=fit_provenance,
        device_map_label=args.device_map,
        empty_cuda_cache=args.empty_cuda_cache,
        telemetry_interval_s=(args.telemetry_interval if args.telemetry_interval else None),
    )
    last_done: int | None = None
    last_converged = False
    last_rolling: float | None = None
    while True:
        try:
            progress = next(progress_iterator)
        except StopIteration:
            break
        last_done = progress.prompts_done
        last_converged = progress.converged
        last_rolling = progress.rolling_delta
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
        # Do not retain the full fp32 lens while the generator fits the next
        # chunk. The durable partial artifact is already on disk.
        del progress

    if last_converged:
        assert last_done is not None and last_rolling is not None
        echo(f"Converged after {last_done} fitted prompts (rolling d_mean={last_rolling:.3g}).")
    echo("Done." if last_done is None else f"Done: {last_done} prompts in the final lens.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
