"""Reference implementations, written independently of `core.py`.

TA-Lib is a C library that will not install in this sandbox, so the reference
is **pandas** — a different codebase, different algorithms (`rolling`, `ewm`),
maintained by other people. Matching it to 1e-6 is the "done when" for P0.2.

Read these as the specification. `core.py` must agree with them, not the other
way round.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ref_sma(close: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(close).rolling(window).mean().to_numpy()


def ref_rolling_std(close: np.ndarray, window: int, ddof: int = 0) -> np.ndarray:
    return pd.Series(close).rolling(window).std(ddof=ddof).to_numpy()


def ref_ema(close: np.ndarray, span: int) -> np.ndarray:
    return pd.Series(close).ewm(span=span, adjust=False).mean().to_numpy()


def ref_wilder_rma(x: np.ndarray, period: int) -> np.ndarray:
    """Wilder smoothing seeded on the simple mean of the first `period` values."""
    s = pd.Series(x, dtype="float64")
    seeded = s.copy()
    seeded.iloc[: period - 1] = np.nan
    seeded.iloc[period - 1] = s.iloc[:period].mean()
    out = seeded.ewm(alpha=1.0 / period, adjust=False, ignore_na=True).mean()
    out.iloc[: period - 1] = np.nan
    return out.to_numpy()


def ref_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    s = pd.Series(close)
    delta = s.diff()
    gain = delta.clip(lower=0.0).iloc[1:].to_numpy()
    loss = (-delta.clip(upper=0.0)).iloc[1:].to_numpy()
    avg_gain = ref_wilder_rma(gain, period)
    avg_loss = ref_wilder_rma(loss, period)
    with np.errstate(divide="ignore", invalid="ignore"):
        rsi = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    rsi = np.where(avg_loss == 0.0, 100.0, rsi)
    rsi = np.where(np.isnan(avg_gain), np.nan, rsi)
    return np.concatenate([[np.nan], rsi])


def ref_bollinger(
    close: np.ndarray, window: int = 20, num_std: float = 2.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s = pd.Series(close)
    mid = s.rolling(window).mean()
    sd = s.rolling(window).std(ddof=0)
    return (mid - num_std * sd).to_numpy(), mid.to_numpy(), (mid + num_std * sd).to_numpy()


def ref_vwap(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray,
    window: int | None = None,
) -> np.ndarray:
    typical = (pd.Series(high) + pd.Series(low) + pd.Series(close)) / 3.0
    pv = typical * pd.Series(volume)
    if window is None:
        return (pv.cumsum() / pd.Series(volume).cumsum()).to_numpy()
    return (pv.rolling(window).sum() / pd.Series(volume).rolling(window).sum()).to_numpy()


def ref_macd(
    close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    line = pd.Series(ref_ema(close, fast) - ref_ema(close, slow))
    sig = line.ewm(span=signal, adjust=False).mean()
    return line.to_numpy(), sig.to_numpy(), (line - sig).to_numpy()


def ref_atr(
    high: np.ndarray, low: np.ndarray, prev_close: np.ndarray, period: int = 14
) -> np.ndarray:
    df = pd.DataFrame({"h": high, "l": low, "pc": prev_close})
    tr = pd.concat(
        [df.h - df.l, (df.h - df.pc).abs(), (df.l - df.pc).abs()], axis=1
    ).max(axis=1)
    return ref_wilder_rma(tr.to_numpy(), period)


def ref_sma_python_loop(close: np.ndarray, window: int) -> np.ndarray:
    """The loop everyone writes first. Kept so the benchmark can time it."""
    out = [float("nan")] * len(close)
    for i in range(window - 1, len(close)):
        total = 0.0
        for j in range(i - window + 1, i + 1):
            total += close[j]
        out[i] = total / window
    return np.array(out)
