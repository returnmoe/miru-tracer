"""Miru Tracer — interactive analysis of LLM text generation, token by token."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("miru-tracer")
except PackageNotFoundError:  # Source tree imported without installing the package.
    __version__ = "0.0.0+unknown"
