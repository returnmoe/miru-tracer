"""Logging configuration for Miru Tracer.

This module provides centralized logging setup based on the MIRU_DEBUG environment variable.
"""

import logging
import os
import sys


def setup_logging():
    """
    Configure logging for the application based on MIRU_DEBUG environment variable.

    Log Levels:
        - MIRU_DEBUG=0 or unset: INFO level (default)
        - MIRU_DEBUG=1 or "true": DEBUG level

    Format: YYYY-MM-DD HH:MM:SS [module_name] LEVEL: message
    Output: Console (stdout/stderr)
    """
    # Determine log level from environment variable
    debug_mode = os.getenv("MIRU_DEBUG", "0") in ("1", "true")
    log_level = logging.DEBUG if debug_mode else logging.INFO

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,  # Override any existing configuration
    )

    # Get root logger to confirm setup
    root_logger = logging.getLogger()

    # Log the logging configuration itself
    if debug_mode:
        root_logger.info("Logging configured: DEBUG level enabled")
    else:
        root_logger.info("Logging configured: INFO level")

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.

    Args:
        name: Module name (typically __name__)

    Returns:
        Logger instance configured with the application settings
    """
    return logging.getLogger(name)
