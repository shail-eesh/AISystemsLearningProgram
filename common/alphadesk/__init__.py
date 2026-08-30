"""AlphaDesk — the capstone every topic plugs into.

**AlphaDesk is a fictional educational simulation.** It never places real
orders, never touches real money or a real brokerage system, and never
redistributes licensed market data. Every surface repeats that disclaimer.

The package starts empty on purpose. Each course topic registers exactly one
component here, so by Phase 9 the registry *is* the architecture diagram:

    from common.alphadesk import register, Surface

    @register(topic="T30", name="fintok", surface=Surface.MODELS,
              summary="BPE tokenizer trained on the filings corpus")
    class FinTok:
        ...

Nothing is imported eagerly — `load_all()` walks the registry manifest so a
broken topic can never break the whole desk.
"""

from common.alphadesk.disclaimer import DISCLAIMER, banner  # noqa: F401
from common.alphadesk.registry import (  # noqa: F401
    REGISTRY,
    Component,
    Registry,
    Surface,
    register,
)

__all__ = ["Component", "Registry", "Surface", "REGISTRY", "register", "DISCLAIMER", "banner"]
