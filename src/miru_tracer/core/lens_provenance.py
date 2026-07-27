"""Lens-fit provenance fingerprints and loaded-model compatibility checks.

Fit provenance is optional so upstream and pre-v0.3 artifacts remain usable.
When it is present, however, Miru must not silently apply a same-shaped lens
from another model or revision.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from miru_tracer.core.logging_config import get_logger

logger = get_logger(__name__)

MODEL_CONFIG_HASH_KIND = "transformers-json-v1"
MODEL_ARCHITECTURE_HASH_KIND = "miru-semantic-v1"

# These values describe how/where a config was loaded, rather than the model
# function whose residual space the fitted matrices inhabit.
_VOLATILE_CONFIG_KEYS = {
    "_attn_implementation",
    "_attn_implementation_autoset",
    "_commit_hash",
    "_name_or_path",
    "attn_implementation",
    "device_map",
    "dtype",
    "output_attentions",
    "output_hidden_states",
    "quantization_config",
    "return_dict",
    "torch_dtype",
    "transformers_version",
    "use_cache",
}

_model_config_cache: dict[tuple[int, str], str | None] = {}
_tokenizer_cache: dict[tuple[int, int, str], str] = {}
_reported_warnings: set[tuple[int, str]] = set()


@dataclass(frozen=True)
class LensCompatibility:
    """Compatibility evidence for one lens/model/tokenizer combination."""

    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    compared_fields: tuple[str, ...] = ()
    structural_errors: tuple[str, ...] = ()
    provenance_errors: tuple[str, ...] = ()

    @property
    def compatible(self) -> bool:
        return not self.errors


def clear_provenance_caches() -> None:
    """Drop fingerprints and warning de-duplication for an unloaded model."""
    _model_config_cache.clear()
    _tokenizer_cache.clear()
    _reported_warnings.clear()


def artifact_model_identifier(name_or_path: str) -> str:
    """Avoid embedding an absolute local filesystem path in shared artifacts."""
    path = Path(name_or_path).expanduser()
    return path.name if path.is_absolute() else name_or_path


def local_model_location_sha256(name_or_path: str) -> str | None:
    """Checkpoint identity for local paths without publishing the path itself."""
    path = Path(name_or_path).expanduser()
    if not path.is_absolute() and not path.exists():
        return None
    return hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()


def local_model_manifest_sha256(name_or_path: str, *, exclude: tuple[Path, ...] = ()) -> str | None:
    """Fingerprint local model-file names, sizes, and mtimes cheaply.

    This is deliberately a cheap change detector, not a content hash of
    multi-gigabyte weight files, so a mismatch is advisory rather than fatal.
    """
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


def _without_volatile_config_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_volatile_config_fields(item)
            for key, item in value.items()
            if key not in _VOLATILE_CONFIG_KEYS
        }
    if isinstance(value, list):
        return [_without_volatile_config_fields(item) for item in value]
    return value


def model_architecture_sha256(model) -> str | None:
    """Hash architecture-affecting config while ignoring runtime/load details."""
    key = (id(model), MODEL_ARCHITECTURE_HASH_KIND)
    if key in _model_config_cache:
        return _model_config_cache[key]
    config = getattr(model, "config", None)
    if config is None or not hasattr(config, "to_dict"):
        result = None
    else:
        normalized = _without_volatile_config_fields(config.to_dict())
        payload = json.dumps(
            normalized,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        result = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    _model_config_cache[key] = result
    return result


def model_config_sha256(model) -> str | None:
    """Preserve the full-config fingerprint used by existing fit checkpoints."""
    key = (id(model), MODEL_CONFIG_HASH_KIND)
    if key in _model_config_cache:
        return _model_config_cache[key]
    config = getattr(model, "config", None)
    if config is None or not hasattr(config, "to_json_string"):
        result = None
    else:
        payload = config.to_json_string(use_diff=False)
        result = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    _model_config_cache[key] = result
    return result


def tokenizer_sha256(tokenizer) -> str:
    """Fingerprint tokenizer rules/vocabulary without storing their contents."""
    try:
        vocab_size = len(tokenizer)
    except (AttributeError, TypeError):
        vocab_size = -1
    chat_template_value = str(getattr(tokenizer, "chat_template", ""))
    key = (id(tokenizer), vocab_size, chat_template_value)
    if key in _tokenizer_cache:
        return _tokenizer_cache[key]
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
    chat_template = chat_template_value.encode("utf-8")
    hasher.update(len(chat_template).to_bytes(8, "big"))
    hasher.update(chat_template)
    result = hasher.hexdigest()
    _tokenizer_cache[key] = result
    return result


def _text_config(model):
    config = getattr(model, "config", None)
    if config is None:
        return None
    getter = getattr(config, "get_text_config", None)
    return getter() if callable(getter) else config


def _nonempty_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _is_local_model_path(value: str) -> bool:
    path = Path(value).expanduser()
    if path.is_absolute() or value.startswith((".", "~")):
        return True
    try:
        return path.exists()
    except OSError:
        return False


def _is_full_hub_commit(value: str | None) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _hub_commit_from_cache_path(value: str | Path) -> str | None:
    """Extract the immutable revision from a Hub ``snapshots/<sha>`` path."""
    parts = Path(value).parts
    for index, part in enumerate(parts[:-1]):
        if part == "snapshots":
            candidate = parts[index + 1]
            return candidate if _is_full_hub_commit(candidate) else None
    return None


def resolve_hub_model_commit(
    name_or_path: str,
    *,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
) -> str | None:
    """Resolve a Hub model reference to the commit containing ``config.json``.

    Local checkpoints have no Hub commit and return ``None``. A full commit is
    already immutable; mutable/default Hub revisions are resolved through the
    public Hub download API so the caller can pin every subsequent file load to
    the same snapshot.
    """
    revision = (revision or "").strip() or None
    if _is_local_model_path(name_or_path):
        return None
    if _is_full_hub_commit(revision):
        return revision

    from huggingface_hub import hf_hub_download

    resolved_config = hf_hub_download(
        repo_id=name_or_path,
        filename="config.json",
        revision=revision,
        cache_dir=cache_dir,
    )
    commit = _hub_commit_from_cache_path(resolved_config)
    if commit is None:
        raise RuntimeError(
            "Hugging Face resolved config.json outside a commit-addressed "
            f"snapshot for {name_or_path!r}; pass a full 40-character --revision"
        )
    return commit


def _append_once(items: list[str], message: str) -> None:
    if message not in items:
        items.append(message)


def check_lens_compatibility(
    lens,
    model,
    tokenizer,
    *,
    model_name_or_path: str | None = None,
) -> LensCompatibility:
    """Compare a fitted lens with the currently loaded model and tokenizer.

    Definite conflicts are errors. Missing provenance and local-path aliases
    are warnings so old/upstream artifacts and moved local checkpoints remain
    usable, but never silently so.
    """
    structural_errors: list[str] = []
    provenance_errors: list[str] = []
    warnings: list[str] = []
    compared: list[str] = []
    text_config = _text_config(model)
    label = model_name_or_path or type(model).__name__

    d_model = getattr(text_config, "hidden_size", None)
    if d_model is not None:
        compared.append("d_model")
        if lens.d_model != d_model:
            structural_errors.append(
                f"lens d_model={lens.d_model} does not match model d_model={d_model} "
                f"for {label}; "
                "this lens was fitted for a different model"
            )

    n_layers = getattr(text_config, "num_hidden_layers", None)
    if n_layers is not None:
        compared.append("layer_count")
        invalid_layers = [layer for layer in lens.source_layers if not 0 <= layer < n_layers]
        if invalid_layers:
            structural_errors.append(
                f"lens source layers {invalid_layers} are out of range for "
                f"{label} ({n_layers} layers)"
            )

    metadata = getattr(lens, "fit_metadata", None)
    provenance = metadata.get("provenance") if isinstance(metadata, dict) else None
    if not isinstance(provenance, dict) or not provenance:
        warnings.append(
            "lens has no model provenance (legacy/upstream artifact); "
            "model, revision, and tokenizer identity cannot be verified"
        )
        return LensCompatibility(
            errors=tuple(structural_errors),
            warnings=tuple(warnings),
            compared_fields=tuple(compared),
            structural_errors=tuple(structural_errors),
        )

    config = getattr(model, "config", None)
    runtime_model_name = _nonempty_string(model_name_or_path) or _nonempty_string(
        getattr(config, "_name_or_path", None)
    )
    artifact_model_name = _nonempty_string(provenance.get("model_name_or_path"))
    model_names_match = False
    if artifact_model_name and runtime_model_name:
        compared.append("model_name_or_path")
        runtime_identifier = artifact_model_identifier(runtime_model_name)
        model_names_match = artifact_model_name == runtime_identifier
        if not model_names_match:
            message = (
                f"lens provenance names model {artifact_model_name!r}, but the "
                f"loaded model is {runtime_identifier!r}"
            )
            if _is_local_model_path(runtime_model_name):
                warnings.append(
                    f"{message}; local paths and aliases cannot be compared by name alone"
                )
            else:
                provenance_errors.append(f"{message}; this lens was fitted for a different model")
    elif artifact_model_name:
        warnings.append(
            f"lens provenance names model {artifact_model_name!r}, but the loaded "
            "model exposes no comparable identifier"
        )

    artifact_commit = _nonempty_string(provenance.get("model_commit_hash"))
    runtime_commit = _nonempty_string(getattr(model, "_miru_model_commit", None))
    if runtime_commit is None:
        runtime_commit = _nonempty_string(getattr(config, "_commit_hash", None))
    commits_match = False
    if artifact_commit and runtime_commit:
        compared.append("model_commit_hash")
        commits_match = artifact_commit == runtime_commit
        if not commits_match:
            provenance_errors.append(
                f"lens revision {artifact_commit!r} does not match loaded model "
                f"revision {runtime_commit!r}"
            )
    elif artifact_commit:
        warnings.append(
            f"lens records model revision {artifact_commit!r}, but the loaded "
            "model exposes no resolved revision to compare"
        )

    architecture_fingerprint_matches: bool | None = None
    artifact_architecture_hash = _nonempty_string(provenance.get("model_architecture_sha256"))
    if artifact_architecture_hash:
        hash_kind = _nonempty_string(provenance.get("model_architecture_sha256_kind"))
        if hash_kind not in (None, MODEL_ARCHITECTURE_HASH_KIND):
            warnings.append(
                f"lens uses unsupported model-architecture fingerprint kind {hash_kind!r}"
            )
        else:
            try:
                runtime_architecture_hash = model_architecture_sha256(model)
            except Exception as exc:
                warnings.append(
                    "lens records a normalized model-architecture fingerprint, "
                    f"but the loaded configuration could not be fingerprinted "
                    f"({type(exc).__name__})"
                )
            else:
                if runtime_architecture_hash:
                    compared.append("model_architecture_sha256")
                    architecture_fingerprint_matches = (
                        artifact_architecture_hash == runtime_architecture_hash
                    )
                    if not architecture_fingerprint_matches:
                        provenance_errors.append(
                            "lens normalized model-architecture fingerprint does "
                            "not match the loaded model"
                        )
                else:
                    warnings.append(
                        "lens records a normalized model-architecture fingerprint, "
                        "but the loaded model configuration cannot be fingerprinted"
                    )

    artifact_config_hash = _nonempty_string(provenance.get("model_config_sha256"))
    if artifact_config_hash:
        hash_kind = _nonempty_string(provenance.get("model_config_sha256_kind"))
        try:
            if hash_kind in (None, MODEL_CONFIG_HASH_KIND):
                runtime_config_hash = model_config_sha256(model)
            elif hash_kind == MODEL_ARCHITECTURE_HASH_KIND:
                # A small number of pre-release v0.3 artifacts used the
                # normalized hash in this field. Continue to recognize them.
                runtime_config_hash = model_architecture_sha256(model)
            else:
                runtime_config_hash = None
                warnings.append(
                    f"lens uses unsupported model-config fingerprint kind {hash_kind!r}"
                )
        except Exception as exc:
            runtime_config_hash = None
            warnings.append(
                "lens records a model-configuration fingerprint, but the loaded "
                f"model configuration could not be fingerprinted ({type(exc).__name__})"
            )
        if runtime_config_hash:
            compared.append("model_config_sha256")
            if artifact_config_hash != runtime_config_hash:
                message = (
                    "lens full model-configuration fingerprint does not match the loaded model"
                )
                if hash_kind == MODEL_ARCHITECTURE_HASH_KIND:
                    provenance_errors.append(
                        "lens normalized model-architecture fingerprint in the "
                        "legacy configuration field does not match the loaded model"
                    )
                elif architecture_fingerprint_matches:
                    warnings.append(
                        f"{message}; the normalized architecture matches, so this "
                        "is treated as a runtime/load-configuration difference"
                    )
                elif artifact_architecture_hash is None:
                    if model_names_match and commits_match:
                        warnings.append(
                            f"{message}; this legacy artifact has no normalized "
                            "architecture fingerprint. The model name and immutable "
                            "revision match, but its full hash cannot distinguish an "
                            "architecture change from runtime/load settings"
                        )
                    else:
                        warnings.append(
                            f"{message}; this legacy artifact has no normalized "
                            "architecture fingerprint, so its full hash cannot distinguish "
                            "an architecture change from runtime/load settings. Other "
                            "recorded identity checks still apply, but the exact "
                            "architecture and revision remain unverified"
                        )
                elif model_names_match and commits_match:
                    warnings.append(f"{message}, despite matching model and revision identifiers")
                else:
                    provenance_errors.append(message)
        elif hash_kind in (None, MODEL_CONFIG_HASH_KIND, MODEL_ARCHITECTURE_HASH_KIND) and not any(
            "could not be fingerprinted" in item for item in warnings
        ):
            warnings.append(
                "lens records a model-configuration fingerprint, but the loaded "
                "model configuration cannot be fingerprinted"
            )

    artifact_tokenizer_hash = _nonempty_string(provenance.get("tokenizer_sha256"))
    if artifact_tokenizer_hash:
        try:
            runtime_tokenizer_hash = tokenizer_sha256(tokenizer)
        except Exception as exc:
            warnings.append(
                "lens records a tokenizer fingerprint, but the loaded tokenizer "
                f"could not be fingerprinted ({type(exc).__name__})"
            )
        else:
            compared.append("tokenizer_sha256")
            if artifact_tokenizer_hash != runtime_tokenizer_hash:
                provenance_errors.append(
                    "lens tokenizer fingerprint does not match the loaded tokenizer"
                )

    artifact_manifest = _nonempty_string(provenance.get("model_manifest_sha256"))
    if artifact_manifest and runtime_model_name:
        try:
            runtime_manifest = local_model_manifest_sha256(runtime_model_name)
        except OSError as exc:
            warnings.append(
                "lens records a local model manifest, but the loaded model files "
                f"could not be inspected ({type(exc).__name__})"
            )
        else:
            if runtime_manifest:
                compared.append("model_manifest_sha256")
                if artifact_manifest != runtime_manifest:
                    warnings.append(
                        "local model-file manifest changed since this lens was fitted; "
                        "file names, sizes, or modification times differ"
                    )

    strong_fields = {
        "model_name_or_path",
        "model_commit_hash",
        "model_config_sha256",
        "model_architecture_sha256",
        "tokenizer_sha256",
        "model_manifest_sha256",
    }
    if not strong_fields.intersection(compared):
        _append_once(
            warnings,
            "lens provenance contains no model identity field that this runtime can verify",
        )
    elif artifact_commit and not runtime_commit:
        _append_once(
            warnings,
            "the exact model revision remains unverified in this runtime",
        )

    all_errors = (*structural_errors, *provenance_errors)
    return LensCompatibility(
        errors=all_errors,
        warnings=tuple(warnings),
        compared_fields=tuple(compared),
        structural_errors=tuple(structural_errors),
        provenance_errors=tuple(provenance_errors),
    )


def require_lens_compatible(
    lens,
    model,
    tokenizer,
    *,
    model_name_or_path: str | None = None,
    force_provenance: bool = False,
) -> LensCompatibility:
    """Raise for incompatible structure or provenance unless explicitly forced.

    ``force_provenance`` bypasses only identity/fingerprint conflicts. Structural
    conflicts such as residual-width or layer-range mismatches always fail.
    """
    result = check_lens_compatibility(
        lens,
        model,
        tokenizer,
        model_name_or_path=model_name_or_path,
    )
    blocking_errors = result.structural_errors if force_provenance else result.errors
    if blocking_errors:
        raise ValueError("incompatible fitted lens: " + "; ".join(blocking_errors))
    if force_provenance:
        for error in result.provenance_errors:
            key = (id(lens), f"forced:{error}")
            if key not in _reported_warnings:
                logger.warning("Lens provenance check explicitly bypassed: %s", error)
                _reported_warnings.add(key)
    for warning in result.warnings:
        key = (id(lens), warning)
        if key not in _reported_warnings:
            logger.warning("Lens compatibility warning: %s", warning)
            _reported_warnings.add(key)
    return result
