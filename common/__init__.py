"""Shared utilities for the AI Systems Forge course.

Everything here is course infrastructure: configuration, logging, and the
market-data loaders every topic reuses. The *subjects* of the course
(attention, HNSW, PPO, paging, ...) are never implemented here — they live in
`phases/<phase>/<topic>/src/`.

AlphaDesk is a fictional educational simulation. Nothing in this package
places real orders, moves real money, or redistributes licensed market data.
"""

from common.config import CONFIG, Config, paths  # noqa: F401
from common.logging_utils import get_logger  # noqa: F401

__all__ = ["CONFIG", "Config", "paths", "get_logger", "__version__"]
__version__ = "0.1.0"
