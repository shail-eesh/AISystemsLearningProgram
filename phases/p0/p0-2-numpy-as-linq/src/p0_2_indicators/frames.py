"""Applying the array functions across a tidy, multi-symbol DataFrame.

The bridge from "NumPy on one series" to "a feature table for the desk". The
important detail is `groupby(...)` per symbol: computing a 20-day SMA across a
concatenated frame would silently average the end of one issuer's history into
the start of the next.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .core import atr, bollinger, ema, macd, rsi, sma, vwap

INDICATOR_COLUMNS = [
    "sma_20", "ema_12", "ema_26", "rsi_14",
    "bb_lower", "bb_mid", "bb_upper",
    "vwap_session", "vwap_20", "macd", "macd_signal", "macd_hist", "atr_14",
]


def _one_symbol(g: pd.DataFrame) -> pd.DataFrame:
    c = g["close"].to_numpy(dtype=np.float64)
    h = g["high"].to_numpy(dtype=np.float64)
    low = g["low"].to_numpy(dtype=np.float64)
    v = g["volume"].to_numpy(dtype=np.float64)
    pc = g["prev_close"].to_numpy(dtype=np.float64)
    lower, mid, upper = bollinger(c, 20, 2.0)
    line, signal, hist = macd(c)
    out = g.copy()
    out["sma_20"] = sma(c, 20)
    out["ema_12"] = ema(c, 12)
    out["ema_26"] = ema(c, 26)
    out["rsi_14"] = rsi(c, 14)
    out["bb_lower"], out["bb_mid"], out["bb_upper"] = lower, mid, upper
    out["vwap_session"] = vwap(h, low, c, v)
    out["vwap_20"] = vwap(h, low, c, v, window=20)
    out["macd"], out["macd_signal"], out["macd_hist"] = line, signal, hist
    out["atr_14"] = atr(h, low, pc, 14)
    return out


def indicator_frame(prices: pd.DataFrame) -> pd.DataFrame:
    """Add every indicator column, computed per symbol, order preserved."""
    required = {"symbol", "date", "open", "high", "low", "close", "prev_close", "volume"}
    missing = required - set(prices.columns)
    if missing:
        raise KeyError(f"missing columns: {sorted(missing)}")
    frames = [
        _one_symbol(g.sort_values("date", kind="stable"))
        for _, g in prices.groupby("symbol", sort=True)
    ]
    return pd.concat(frames, ignore_index=True)
