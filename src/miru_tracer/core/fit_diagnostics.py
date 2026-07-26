"""Low-overhead runtime diagnostics for long Jacobian-lens fitting jobs.

The fitter can run for days, often on a remote multi-GPU node where a restart
destroys the most useful evidence about a slowdown.  These helpers emit stable
``key=value`` records using only Python, ``/proc`` (when available), and
PyTorch's allocator counters.  Diagnostics are deliberately best-effort:
failure to read one counter must never stop a fit.
"""

from __future__ import annotations

import gc
import logging
import math
import os
import shutil
import statistics
import subprocess
import threading
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

import torch

_MIB = 1024 * 1024
DEFAULT_TELEMETRY_INTERVAL_S = 30.0
SLOWDOWN_BASELINE_PROMPTS = 5
SLOWDOWN_WINDOW_PROMPTS = 20
SLOWDOWN_RATIO = 2.0
SLOWDOWN_MIN_EXCESS_S = 60.0
_NVIDIA_SMI_QUERY = (
    "index",
    "uuid",
    "driver_version",
    "pstate",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.total",
    "temperature.gpu",
    "power.draw",
    "clocks.current.sm",
    "clocks.current.memory",
)


def _read_key_value_file(path: str) -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        lines = Path(path).read_text().splitlines()
    except OSError:
        return values
    for line in lines:
        key, separator, value = line.partition(":")
        if separator:
            values[key] = value.strip()
    return values


def _kib_value(values: dict[str, str], key: str) -> float | None:
    raw = values.get(key)
    if raw is None:
        return None
    number, *_unit = raw.split()
    try:
        return int(number) / 1024
    except ValueError:
        return None


def _proc_io() -> dict[str, int]:
    values = _read_key_value_file("/proc/self/io")
    result: dict[str, int] = {}
    for source, destination in (
        ("read_bytes", "proc_read_bytes"),
        ("write_bytes", "proc_write_bytes"),
        ("cancelled_write_bytes", "proc_cancelled_write_bytes"),
    ):
        with suppress(KeyError, ValueError):
            result[destination] = int(values[source])
    return result


def _pressure_snapshot() -> dict[str, int | float]:
    """Linux pressure-stall counters, when the kernel exposes PSI."""
    result: dict[str, int | float] = {}
    for resource_name in ("cpu", "memory", "io"):
        try:
            lines = Path(f"/proc/pressure/{resource_name}").read_text().splitlines()
        except OSError:
            continue
        for line in lines:
            kind, *raw_fields = line.split()
            if kind not in {"some", "full"}:
                continue
            fields = dict(field.split("=", 1) for field in raw_fields if "=" in field)
            with suppress(KeyError, ValueError):
                result[f"host_psi_{resource_name}_{kind}_avg10"] = float(fields["avg10"])
            with suppress(KeyError, ValueError):
                result[f"host_psi_{resource_name}_{kind}_total_us"] = int(fields["total"])
    return result


def _nvidia_smi_snapshot() -> dict[str, int | float | str]:
    """One best-effort NVML snapshot via ``nvidia-smi``.

    Physical NVIDIA indices are deliberately prefixed ``nvidia`` rather than
    ``cuda``: ``CUDA_VISIBLE_DEVICES`` can reorder or hide devices. The run
    environment record includes that mapping.
    """
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {}
    try:
        completed = subprocess.run(
            [
                executable,
                f"--query-gpu={','.join(_NVIDIA_SMI_QUERY)}",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    if completed.returncode != 0:
        return {}

    result: dict[str, int | float | str] = {}
    numeric_fields: dict[str, tuple[str, type[int] | type[float]]] = {
        "utilization.gpu": ("utilization_pct", float),
        "utilization.memory": ("memory_utilization_pct", float),
        "memory.used": ("device_used_mib", float),
        "memory.total": ("device_total_mib", float),
        "temperature.gpu": ("temperature_c", float),
        "power.draw": ("power_w", float),
        "clocks.current.sm": ("sm_clock_mhz", float),
        "clocks.current.memory": ("memory_clock_mhz", float),
    }
    for line in completed.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != len(_NVIDIA_SMI_QUERY):
            continue
        row = dict(zip(_NVIDIA_SMI_QUERY, values, strict=True))
        try:
            index = int(row["index"])
        except ValueError:
            continue
        prefix = f"nvidia{index}_"
        for source, destination in (
            ("uuid", "uuid"),
            ("driver_version", "driver_version"),
            ("pstate", "pstate"),
        ):
            value = row[source]
            if value and value.upper() != "N/A":
                result[f"{prefix}{destination}"] = value
        for source, (destination, converter) in numeric_fields.items():
            value = row[source]
            if not value or value.upper() == "N/A":
                continue
            with suppress(ValueError):
                result[f"{prefix}{destination}"] = converter(value)
    return result


def filesystem_info(path: str | Path) -> dict[str, str]:
    """Return the longest matching Linux mount point and filesystem type."""
    try:
        resolved = str(Path(path).resolve())
        lines = Path("/proc/self/mountinfo").read_text().splitlines()
    except OSError:
        return {}

    def unescape_mount(value: str) -> str:
        return (
            value.replace("\\040", " ")
            .replace("\\011", "\t")
            .replace("\\012", "\n")
            .replace("\\134", "\\")
        )

    best: tuple[int, str, str] | None = None
    for line in lines:
        left, separator, right = line.partition(" - ")
        if not separator:
            continue
        left_fields = left.split()
        right_fields = right.split()
        if len(left_fields) < 5 or not right_fields:
            continue
        mount_point = unescape_mount(left_fields[4])
        if resolved != mount_point and not resolved.startswith(mount_point.rstrip("/") + "/"):
            continue
        candidate = (len(mount_point), mount_point, right_fields[0])
        if best is None or candidate[0] > best[0]:
            best = candidate
    if best is None:
        return {}
    return {"output_mount": best[1], "output_filesystem": best[2]}


def cuda_allocator_backend() -> str:
    """Return the active PyTorch CUDA allocator name without failing a fit."""
    try:
        if torch.cuda.is_available():
            return str(torch.cuda.get_allocator_backend())
    except Exception:
        pass
    return "unavailable"


def resource_snapshot() -> dict[str, int | float | str]:
    """Return process, host-memory, I/O, and CUDA allocator counters.

    Missing platform- or device-specific values are simply omitted. Memory
    capacities use MiB; process I/O counters remain exact cumulative bytes.
    """
    snapshot: dict[str, int | float | str] = {}
    with suppress(OSError):
        load_1m, load_5m, load_15m = os.getloadavg()
        snapshot.update(
            {
                "host_load_1m": round(load_1m, 3),
                "host_load_5m": round(load_5m, 3),
                "host_load_15m": round(load_15m, 3),
            }
        )
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
    except (ImportError, OSError):
        pass
    else:
        snapshot.update(
            {
                "cpu_user_s": round(usage.ru_utime, 3),
                "cpu_system_s": round(usage.ru_stime, 3),
                "minor_faults": usage.ru_minflt,
                "major_faults": usage.ru_majflt,
                "voluntary_context_switches": usage.ru_nvcsw,
                "involuntary_context_switches": usage.ru_nivcsw,
            }
        )

    status = _read_key_value_file("/proc/self/status")
    for source, destination in (
        ("VmRSS", "rss_mib"),
        ("VmHWM", "rss_peak_mib"),
    ):
        value = _kib_value(status, source)
        if value is not None:
            snapshot[destination] = round(value, 1)
    with suppress(KeyError, ValueError):
        snapshot["threads"] = int(status["Threads"])

    smaps = _read_key_value_file("/proc/self/smaps_rollup")
    for source, destination in (
        ("Pss", "pss_mib"),
        ("Pss_Anon", "pss_anon_mib"),
        ("Pss_File", "pss_file_mib"),
        ("Private_Dirty", "private_dirty_mib"),
        ("Swap", "swap_mib"),
    ):
        value = _kib_value(smaps, source)
        if value is not None:
            snapshot[destination] = round(value, 1)

    meminfo = _read_key_value_file("/proc/meminfo")
    for source, destination in (
        ("MemAvailable", "host_available_mib"),
        ("Cached", "host_cached_mib"),
        ("Dirty", "host_dirty_mib"),
        ("Writeback", "host_writeback_mib"),
    ):
        value = _kib_value(meminfo, source)
        if value is not None:
            snapshot[destination] = round(value, 1)

    snapshot.update(_proc_io())
    snapshot.update(_pressure_snapshot())
    with suppress(OSError):
        snapshot["open_fds"] = len(os.listdir("/proc/self/fd"))

    try:
        cuda_available = torch.cuda.is_available()
    except Exception:
        cuda_available = False
    if not cuda_available:
        return snapshot

    try:
        device_count = torch.cuda.device_count()
    except Exception:
        return snapshot
    for device in range(device_count):
        prefix = f"cuda{device}_"
        try:
            snapshot[f"{prefix}allocated_mib"] = round(
                torch.cuda.memory_allocated(device) / _MIB, 1
            )
            snapshot[f"{prefix}reserved_mib"] = round(torch.cuda.memory_reserved(device) / _MIB, 1)
            snapshot[f"{prefix}peak_allocated_mib"] = round(
                torch.cuda.max_memory_allocated(device) / _MIB, 1
            )
            snapshot[f"{prefix}peak_reserved_mib"] = round(
                torch.cuda.max_memory_reserved(device) / _MIB, 1
            )
            stats = torch.cuda.memory_stats(device)
            snapshot[f"{prefix}active_mib"] = round(
                stats.get("active_bytes.all.current", 0) / _MIB, 1
            )
            snapshot[f"{prefix}inactive_split_mib"] = round(
                stats.get("inactive_split_bytes.all.current", 0) / _MIB, 1
            )
            snapshot[f"{prefix}alloc_retries"] = stats.get("num_alloc_retries", 0)
            snapshot[f"{prefix}ooms"] = stats.get("num_ooms", 0)
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            snapshot[f"{prefix}free_mib"] = round(free_bytes / _MIB, 1)
            snapshot[f"{prefix}total_mib"] = round(total_bytes / _MIB, 1)
        except Exception:
            # A single device may disappear after a CUDA error. Keep the host
            # and other-device counters instead of masking the original fault.
            continue
    snapshot.update(_nvidia_smi_snapshot())
    return snapshot


def _log_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value).replace("\\", "/").replace(" ", "_").replace("\n", "_")


def log_fit_telemetry(
    logger: logging.Logger,
    event: str,
    *,
    level: int = logging.INFO,
    include_resources: bool = True,
    **fields: Any,
) -> None:
    """Emit one grep-friendly diagnostic record.

    The prefix and field names are intentionally stable so cluster logs can be
    loaded as whitespace-delimited ``key=value`` records.
    """
    values: dict[str, Any] = {"event": event, **fields}
    if include_resources:
        values.update(resource_snapshot())
    record = " ".join(f"{key}={_log_value(value)}" for key, value in values.items())
    logger.log(level, "fit_telemetry %s", record)


class PromptTelemetrySampler:
    """Periodically sample resources while one expensive prompt is running.

    Sampling happens on a daemon thread and never synchronizes CUDA. The main
    fitting thread updates a cheap phase label as it moves through encoding,
    forward, and backward/copy work.
    """

    def __init__(
        self,
        logger: logging.Logger,
        *,
        interval_s: float,
        global_prompt: int,
        process_prompt: int,
    ):
        if (
            isinstance(interval_s, bool)
            or not isinstance(interval_s, (int, float))
            or not math.isfinite(interval_s)
            or interval_s <= 0
        ):
            raise ValueError(
                f"telemetry interval must be a positive finite number, got {interval_s!r}"
            )
        self._logger = logger
        self._interval_s = float(interval_s)
        self._base_fields = {
            "global_prompt": global_prompt,
            "process_prompt": process_prompt,
        }
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at = 0.0
        self._phase_started_at = 0.0
        self._phase = "setup"
        self._phase_fields: dict[str, Any] = {}
        self._sample_index = 0

    def __enter__(self) -> PromptTelemetrySampler:
        self._started_at = self._phase_started_at = time.perf_counter()
        log_fit_telemetry(
            self._logger,
            "prompt_start",
            **self._base_fields,
            phase=self._phase,
            telemetry_interval_s=self._interval_s,
        )
        self._thread = threading.Thread(
            target=self._run,
            name=f"miru-fit-telemetry-{self._base_fields['process_prompt']}",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self._stop.set()
        if self._thread is not None:
            # A snapshot can be inside nvidia-smi (bounded to three seconds).
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                log_fit_telemetry(
                    self._logger,
                    "telemetry_sampler_stop_timeout",
                    level=logging.WARNING,
                    include_resources=False,
                    **self._base_fields,
                )

    def update_phase(self, phase: str, **fields: Any) -> None:
        """Publish the current prompt phase without touching CUDA."""
        now = time.perf_counter()
        with self._lock:
            if phase != self._phase:
                self._phase_started_at = now
            self._phase = phase
            self._phase_fields = dict(fields)

    def _run(self) -> None:
        while not self._stop.wait(self._interval_s):
            now = time.perf_counter()
            with self._lock:
                self._sample_index += 1
                fields = {
                    **self._base_fields,
                    "sample_index": self._sample_index,
                    "elapsed_s": now - self._started_at,
                    "phase": self._phase,
                    "phase_elapsed_s": now - self._phase_started_at,
                    **self._phase_fields,
                }
            try:
                log_fit_telemetry(self._logger, "prompt_sample", **fields)
            except Exception as exc:
                # Diagnostics must never terminate or perturb the fit.
                log_fit_telemetry(
                    self._logger,
                    "telemetry_sampler_error",
                    level=logging.WARNING,
                    include_resources=False,
                    **self._base_fields,
                    error_type=type(exc).__name__,
                )
                return


def prompt_slowdown(
    elapsed_s: float,
    previous_durations_s: list[float],
) -> tuple[float, float] | None:
    """Return ``(rolling_median, ratio)`` for a material process-local slowdown."""
    recent = previous_durations_s[-SLOWDOWN_WINDOW_PROMPTS:]
    if len(recent) < SLOWDOWN_BASELINE_PROMPTS:
        return None
    baseline = float(statistics.median(recent))
    if baseline <= 0:
        return None
    ratio = elapsed_s / baseline
    if ratio < SLOWDOWN_RATIO or elapsed_s - baseline < SLOWDOWN_MIN_EXCESS_S:
        return None
    return baseline, ratio


def reset_cuda_peak_stats() -> None:
    """Reset per-device peaks so the next prompt reports its own high-water mark."""
    try:
        if not torch.cuda.is_available():
            return
        device_count = torch.cuda.device_count()
    except Exception:
        return
    for device in range(device_count):
        with suppress(Exception):
            torch.cuda.reset_peak_memory_stats(device)


def cleanup_runtime(
    logger: logging.Logger,
    *,
    chunk_end: int,
    empty_cuda_cache: bool = False,
) -> None:
    """Collect dead graphs and optionally empty CUDA caches at a chunk boundary.

    Cache emptying is opt-in because it is an allocator intervention, not a
    general memory-leak fix. Keeping it configurable makes long-run A/B results
    interpretable and avoids forcing CUDA reallocations in every fit.
    """
    log_fit_telemetry(logger, "chunk_cleanup_before", chunk_end=chunk_end)
    collected = gc.collect()
    cuda_errors: list[str] = []
    try:
        cuda_available = torch.cuda.is_available()
    except Exception:
        cuda_available = False
    if cuda_available and empty_cuda_cache:
        try:
            device_count = torch.cuda.device_count()
        except Exception as exc:
            cuda_errors.append(type(exc).__name__)
            device_count = 0
        for device in range(device_count):
            try:
                torch.cuda.synchronize(device)
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
            except Exception as exc:
                cuda_errors.append(f"cuda{device}:{type(exc).__name__}")
    log_fit_telemetry(
        logger,
        "chunk_cleanup_after",
        chunk_end=chunk_end,
        gc_collected=collected,
        cuda_empty_cache=(
            "completed"
            if cuda_available and empty_cuda_cache and not cuda_errors
            else "failed"
            if cuda_available and empty_cuda_cache
            else "disabled"
        ),
        cuda_cleanup_errors="none" if not cuda_errors else ",".join(cuda_errors),
    )


def log_cuda_memory_summaries(logger: logging.Logger) -> None:
    """Dump PyTorch allocator summaries after an OOM, without hiding the OOM."""
    try:
        if not torch.cuda.is_available():
            return
        device_count = torch.cuda.device_count()
    except Exception:
        return
    for device in range(device_count):
        try:
            summary = torch.cuda.memory_summary(device=device, abbreviated=True)
        except Exception as exc:
            logger.error(
                "fit_cuda_memory_summary device=%d unavailable=%s",
                device,
                type(exc).__name__,
            )
        else:
            logger.error("fit_cuda_memory_summary device=%d\n%s", device, summary)
