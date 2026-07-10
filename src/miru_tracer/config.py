"""Environment-based configuration.

All environment parsing lives here so every consumer agrees on semantics.
Variables (also documented in .env.example):

- MIRU_DEBUG: "1"/"true"/"yes"/"on" enable debug logging and Gradio debug mode.
- MIRU_SERVER_NAME: bind address (falls back to GRADIO_SERVER_NAME, then 127.0.0.1).
- MIRU_SERVER_PORT: bind port (falls back to GRADIO_SERVER_PORT, then 7860).
- MIRU_ALLOW_REMOTE_CODE: allow the Model Loader to execute repository code.
- MIRU_AUTH_USERNAME / MIRU_AUTH_PASSWORD: optional paired Gradio credentials.
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
    allow_remote_code: bool
    auth_username: str | None
    auth_password: str | None
    max_new_tokens: int
    max_log_top_k: int
    max_full_prob_steps: int
    max_lens_cells: int

    @classmethod
    def from_env(cls) -> Settings:
        auth_username = os.getenv("MIRU_AUTH_USERNAME") or None
        auth_password = os.getenv("MIRU_AUTH_PASSWORD") or None
        if (auth_username is None) != (auth_password is None):
            raise ValueError(
                "MIRU_AUTH_USERNAME and MIRU_AUTH_PASSWORD must be set together"
            )
        return cls(
            debug=env_bool("MIRU_DEBUG"),
            server_name=env_str("MIRU_SERVER_NAME", "127.0.0.1", "GRADIO_SERVER_NAME"),
            server_port=env_int(
                "MIRU_SERVER_PORT", env_int("GRADIO_SERVER_PORT", 7860)
            ),
            allow_remote_code=env_bool("MIRU_ALLOW_REMOTE_CODE"),
            auth_username=auth_username,
            auth_password=auth_password,
            max_new_tokens=max(1, env_int("MIRU_MAX_NEW_TOKENS", 1000)),
            max_log_top_k=max(1, env_int("MIRU_MAX_LOG_TOP_K", 256)),
            max_full_prob_steps=max(
                1, env_int("MIRU_MAX_FULL_PROB_STEPS", 128)
            ),
            max_lens_cells=max(1, env_int("MIRU_MAX_LENS_CELLS", 8192)),
        )

    @property
    def auth(self) -> tuple[str, str] | None:
        if self.auth_username is None or self.auth_password is None:
            return None
        return self.auth_username, self.auth_password
