# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
#
# Vendored from Anthropic's jacobian-lens (jlens v0.1.0) — see VENDORED.md.
"""Vendored Jacobian-lens reference implementation (Apache-2.0)."""

from miru_tracer.core._jlens.fitting import fit
from miru_tracer.core._jlens.hf import HFLensModel, Layout, from_hf
from miru_tracer.core._jlens.hooks import ActivationRecorder
from miru_tracer.core._jlens.lens import JacobianLens
from miru_tracer.core._jlens.protocol import LensModel

__all__ = [
    "ActivationRecorder",
    "HFLensModel",
    "JacobianLens",
    "Layout",
    "LensModel",
    "fit",
    "from_hf",
]
