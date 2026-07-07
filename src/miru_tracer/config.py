"""Environment-based configuration.

All environment parsing lives here so every consumer agrees on semantics.
Variables (also documented in .env.example):

- MIRU_DEBUG: "1"/"true"/"yes"/"on" enable debug logging and Gradio debug mode.
- MIRU_SERVER_NAME: bind address (falls back to GRADIO_SERVER_NAME, then 127.0.0.1).
- MIRU_SERVER_PORT: bind port (falls back to GRADIO_SERVER_PORT, then 7860).
- HF_TOKEN: read directly by huggingface_hub for gated models.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def env_bool(name: str, default: bool = False) -> bool:
    """Parse a boolean environment variable ("1"/"true"/"yes"/"on", case-insensitive)."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in _TRUTHY


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        return default


def env_str(name: str, default: str, *fallbacks: str) -> str:
    """Read a string variable, trying fallback variable names before the default."""
    for candidate in (name, *fallbacks):
        value = os.getenv(candidate)
        if value:
            return value
    return default


@dataclass(frozen=True)
class Settings:
    debug: bool
    server_name: str
    server_port: int

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            debug=env_bool("MIRU_DEBUG"),
            server_name=env_str("MIRU_SERVER_NAME", "127.0.0.1", "GRADIO_SERVER_NAME"),
            server_port=env_int(
                "MIRU_SERVER_PORT", env_int("GRADIO_SERVER_PORT", 7860)
            ),
        )
