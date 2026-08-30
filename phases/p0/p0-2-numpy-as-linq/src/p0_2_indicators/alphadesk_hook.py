"""AlphaDesk wiring for P0.2.

The desk's Data surface needs leak-free technical features over the price
history: this is the first, deliberately simple version of what T48 (feature
store) will make point-in-time correct in Phase 3.
"""

from __future__ import annotations

import pandas as pd

from common.alphadesk import Surface, register
from common.data import load_ohlcv

from .frames import INDICATOR_COLUMNS, indicator_frame


@register(
    topic="P0.2",
    name="technical_features",
    surface=Surface.DATA,
    summary=f"Vectorised indicator table ({len(INDICATOR_COLUMNS)} columns) over the price history",
)
def build_technical_features(symbols: list[str] | None = None) -> pd.DataFrame:
    """The feature table AlphaDesk's research surface reads. No lookahead."""
    return indicator_frame(load_ohlcv(symbols))
