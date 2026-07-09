"""Validation and JSON normalization for optional lens-fit metadata.

The Jacobian matrices and their required shape/count fields remain the stable
artifact payload.  Fit metadata is deliberately optional so lenses produced by
the upstream reference implementation and older Miru releases continue to
load unchanged.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

FIT_METADATA_SCHEMA_VERSION = 1
MAX_FIT_METADATA_BYTES = 1_048_576
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


def _require_mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an object with string keys")
    return value


def _optional_nonnegative_number(value: object, name: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a nonnegative finite number or null")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite or value < 0:
        raise ValueError(f"{name} must be a nonnegative finite number or null")


def _optional_integer(
    mapping: dict[str, Any], field: str, name: str, *, minimum: int
) -> int | None:
    value = mapping.get(field)
    if value is None:
        return None
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name}.{field} must be an integer >= {minimum}")
    return value


def _optional_boolean(mapping: dict[str, Any], field: str, name: str) -> bool | None:
    value = mapping.get(field)
    if value is None:
        return None
    if type(value) is not bool:
        raise ValueError(f"{name}.{field} must be a boolean")
    return value


def _optional_sha256(mapping: dict[str, Any], field: str, name: str) -> None:
    value = mapping.get(field)
    if value is not None and (not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None):
        raise ValueError(f"{name}.{field} must be a lowercase SHA-256 digest")


def normalize_fit_metadata(
    value: object | None,
    *,
    n_prompts: int | None = None,
) -> dict[str, Any] | None:
    """Return a detached, JSON-safe fit-metadata object.

    Validation intentionally fixes only the small v1 envelope and convergence
    history. Unknown keys remain allowed so v1 can gain additive provenance
    fields without invalidating existing readers.
    """
    if value is None:
        return None
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("fit_metadata must contain only finite JSON values") from exc
    if len(encoded.encode("utf-8")) > MAX_FIT_METADATA_BYTES:
        raise ValueError(f"fit_metadata exceeds the {MAX_FIT_METADATA_BYTES}-byte size limit")

    decoded = _require_mapping(json.loads(encoded), "fit_metadata")
    version = decoded.get("schema_version")
    if type(version) is not int or version != FIT_METADATA_SCHEMA_VERSION:
        raise ValueError(
            f"fit_metadata.schema_version must be {FIT_METADATA_SCHEMA_VERSION}, got {version!r}"
        )

    for section in ("provenance", "fit", "convergence"):
        if section in decoded:
            _require_mapping(decoded[section], f"fit_metadata.{section}")

    convergence = decoded.get("convergence")
    if isinstance(convergence, dict):
        for field in ("enabled", "converged", "history_truncated"):
            _optional_boolean(convergence, field, "fit_metadata.convergence")
        _optional_integer(convergence, "min_prompts", "fit_metadata.convergence", minimum=1)
        _optional_integer(convergence, "window", "fit_metadata.convergence", minimum=1)
        history_total = _optional_integer(
            convergence,
            "history_total_points",
            "fit_metadata.convergence",
            minimum=0,
        )
        for field in (
            "threshold",
            "last_mean_relative_change",
            "rolling_mean_relative_change",
        ):
            if field in convergence:
                _optional_nonnegative_number(
                    convergence[field], f"fit_metadata.convergence.{field}"
                )
        history = convergence.get("history", [])
        if not isinstance(history, list):
            raise ValueError("fit_metadata.convergence.history must be a list")
        previous_count = 0
        previous_prompt_index = -1
        for index, raw_point in enumerate(history):
            point = _require_mapping(raw_point, f"fit_metadata.convergence.history[{index}]")
            count = point.get("n_prompts")
            if type(count) is not int or count <= previous_count:
                raise ValueError(
                    "fit_metadata convergence history prompt counts must be "
                    "strictly increasing positive integers"
                )
            if n_prompts is not None and count > n_prompts:
                raise ValueError("fit_metadata convergence history cannot extend beyond n_prompts")
            previous_count = count
            prompt_index = _optional_integer(
                point,
                "prompt_index",
                f"fit_metadata.convergence.history[{index}]",
                minimum=0,
            )
            if prompt_index is not None:
                if prompt_index <= previous_prompt_index:
                    raise ValueError(
                        "fit_metadata convergence history prompt indices must be "
                        "strictly increasing"
                    )
                previous_prompt_index = prompt_index
            seq_len = _optional_integer(
                point,
                "seq_len",
                f"fit_metadata.convergence.history[{index}]",
                minimum=1,
            )
            valid_positions = _optional_integer(
                point,
                "valid_positions",
                f"fit_metadata.convergence.history[{index}]",
                minimum=0,
            )
            if seq_len is not None and valid_positions is not None and valid_positions > seq_len:
                raise ValueError("fit_metadata convergence valid_positions cannot exceed seq_len")
            _optional_nonnegative_number(
                point.get("mean_relative_change"),
                f"fit_metadata.convergence.history[{index}].mean_relative_change",
            )
            _optional_nonnegative_number(
                point.get("prompt_jacobian_norm_over_sqrt_d"),
                f"fit_metadata.convergence.history[{index}].prompt_jacobian_norm_over_sqrt_d",
            )
        if history and n_prompts is not None and previous_count != n_prompts:
            raise ValueError("fit_metadata convergence history must end at n_prompts")
        if history_total is not None and history_total < len(history):
            raise ValueError(
                "fit_metadata.convergence.history_total_points cannot be smaller than history"
            )
        if (
            history_total is not None
            and convergence.get("history_truncated") is False
            and history_total != len(history)
        ):
            raise ValueError(
                "fit_metadata non-truncated convergence history must contain all points"
            )

    fit_state = decoded.get("fit")
    if isinstance(fit_state, dict):
        processed = _optional_integer(fit_state, "processed_prompts", "fit_metadata.fit", minimum=0)
        skipped = _optional_integer(fit_state, "skipped_prompts", "fit_metadata.fit", minimum=0)
        _optional_integer(fit_state, "target_layer", "fit_metadata.fit", minimum=0)
        _optional_integer(fit_state, "max_seq_len", "fit_metadata.fit", minimum=1)
        _optional_integer(fit_state, "skip_first", "fit_metadata.fit", minimum=0)
        _optional_integer(fit_state, "dim_batch", "fit_metadata.fit", minimum=1)
        _optional_sha256(fit_state, "processed_prompt_prefix_sha256", "fit_metadata.fit")
        if (
            n_prompts is not None
            and processed is not None
            and skipped is not None
            and processed - skipped != n_prompts
        ):
            raise ValueError(
                "fit_metadata processed_prompts - skipped_prompts must equal n_prompts"
            )

    provenance = decoded.get("provenance")
    if isinstance(provenance, dict):
        _optional_sha256(provenance, "prompt_sequence_sha256", "fit_metadata.provenance")
        _optional_sha256(provenance, "model_location_sha256", "fit_metadata.provenance")
        _optional_sha256(provenance, "model_manifest_sha256", "fit_metadata.provenance")
        _optional_sha256(provenance, "model_config_sha256", "fit_metadata.provenance")
        _optional_sha256(provenance, "tokenizer_sha256", "fit_metadata.provenance")

    return decoded


def encode_fit_metadata(value: object, *, n_prompts: int) -> str:
    """Encode validated metadata canonically for a safetensors header."""
    normalized = normalize_fit_metadata(value, n_prompts=n_prompts)
    assert normalized is not None
    return json.dumps(
        normalized,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
