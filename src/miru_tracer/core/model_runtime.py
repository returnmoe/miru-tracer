"""Serialization primitives for operations on the one app-wide model.

Temporary intervention and activation-recorder hooks mutate shared PyTorch
modules. A single re-entrant lock keeps their registration/forward/removal
regions isolated and lets model replacement wait for in-flight inference.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")

MODEL_RUNTIME_LOCK = threading.RLock()


def serialized_model_operation(fn: Callable[P, R]) -> Callable[P, R]:
    """Run ``fn`` exclusively with respect to every shared-model operation."""

    @wraps(fn)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        with MODEL_RUNTIME_LOCK:
            return fn(*args, **kwargs)

    return wrapped
