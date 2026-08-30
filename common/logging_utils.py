"""Tiny structured logger.

.NET note: this is the rough equivalent of `ILogger<T>` — you ask for a logger
named after your module and the host decides where it goes. There is no DI
container; `logging` is a process-global registry keyed by name.
"""

from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False
_FORMAT = "%(asctime)s %(levelname)-7s %(name)-28s %(message)s"
_DATEFMT = "%H:%M:%S"


def _configure() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    level = os.environ.get("FORGE_LOG_LEVEL", "INFO").upper()
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    root = logging.getLogger("forge")
    root.handlers[:] = [handler]
    root.setLevel(getattr(logging, level, logging.INFO))
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger. `name` is usually `__name__`."""
    _configure()
    suffix = name.removeprefix("forge.")
    return logging.getLogger(f"forge.{suffix}")
