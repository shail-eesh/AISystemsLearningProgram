"""Vectorised technical indicators.

Every function here takes a 1-D `float64` array and returns a 1-D array of the
same length, NaN-padded at the front where the indicator is not yet defined.
No Python-level loop touches a bar — that is the whole exercise.

Three vectorisation tools cover almost everything:

* **`sliding_window_view`** — a *view* over overlapping windows, shape
  `(n - w + 1, w)`, costing no memory. Reductions then run on `axis=-1`.
* **`cumsum` differencing** — O(n) rolling sums, at the price of catastrophic
  cancellation on long series (see NOTES); used only where it is safe.
* **`scipy.signal.lfilter`** — the escape hatch for *recurrences*. An EMA is
  `y[t] = a*x[t] + (1-a)*y[t-1]`, which genuinely cannot be expressed as a
  reduction over independent windows. The algebraic "vectorised EMA" overflows
  after a few hundred bars; `lfilter` pushes the loop into C where it belongs.
  Knowing which of these three a problem is, is the skill.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import lfilter

__all__ = [
    "sliding_windows", "sma", "rolling_std", "rolling_min", "rolling_max",
    "ema", "wilder_rma", "rsi", "bollinger", "vwap", "macd", "true_range", "atr",
]


def _as_1d(x: np.ndarray | list[float], name: str = "x") -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {a.shape}")
    return a


def _check_window(w: int, n: int) -> int:
    if not isinstance(w, int | np.integer) or isinstance(w, bool):
        raise TypeError(f"window must be an int, got {type(w).__name__}")
    if w < 1:
        raise ValueError(f"window must be >= 1, got {w}")
    if w > n:
        raise ValueError(f"window {w} exceeds series length {n}")
    return int(w)


def sliding_windows(x: np.ndarray, window: int) -> np.ndarray:
    """`(n-window+1, window)` view of overlapping windows. Zero copies."""
    a = _as_1d(x)
    _check_window(window, a.size)
    return sliding_window_view(a, window)


def _front_pad(values: np.ndarray, n: int) -> np.ndarray:
    """Right-align `values` in a length-`n` NaN array (the warm-up convention)."""
    out = np.full(n, np.nan)
    out[n - values.size:] = values
    return out


def sma(x: np.ndarray, window: int) -> np.ndarray:
    """Simple moving average — the mean over each window."""
    a = _as_1d(x)
    w = _check_window(window, a.size)
    return _front_pad(sliding_windows(a, w).mean(axis=-1), a.size)


def rolling_std(x: np.ndarray, window: int, ddof: int = 0) -> np.ndarray:
    """Rolling standard deviation. `ddof=0` (population) matches Bollinger."""
    a = _as_1d(x)
    w = _check_window(window, a.size)
    return _front_pad(sliding_windows(a, w).std(axis=-1, ddof=ddof), a.size)


def rolling_min(x: np.ndarray, window: int) -> np.ndarray:
    a = _as_1d(x)
    w = _check_window(window, a.size)
    return _front_pad(sliding_windows(a, w).min(axis=-1), a.size)


def rolling_max(x: np.ndarray, window: int) -> np.ndarray:
    a = _as_1d(x)
    w = _check_window(window, a.size)
    return _front_pad(sliding_windows(a, w).max(axis=-1), a.size)


def _recursive_filter(x: np.ndarray, alpha: float, seed: float) -> np.ndarray:
    """y[t] = alpha*x[t] + (1-alpha)*y[t-1], seeded with `seed` at t = -1.

    `lfilter`'s `zi` is the filter's initial delay state; for this one-pole
    filter it equals (1-alpha)*seed, which is what makes y[0] come out as
    alpha*x[0] + (1-alpha)*seed exactly.
    """
    b = np.array([alpha])
    a = np.array([1.0, -(1.0 - alpha)])
    zi = np.array([(1.0 - alpha) * seed])
    y, _ = lfilter(b, a, x, zi=zi)
    return y


def ema(x: np.ndarray, span: int, *, seed: str = "first") -> np.ndarray:
    """Exponential moving average, `alpha = 2 / (span + 1)`.

    `seed="first"` starts from x[0] (pandas `ewm(span=..., adjust=False)`);
    `seed="sma"` starts from the SMA of the first `span` bars and NaN-pads the
    warm-up, which is the convention charting packages use.
    """
    a = _as_1d(x)
    w = _check_window(span, a.size)
    alpha = 2.0 / (w + 1.0)
    if seed == "first":
        return _recursive_filter(a, alpha, a[0])
    if seed == "sma":
        start = float(a[:w].mean())
        tail = _recursive_filter(a[w:], alpha, start)
        out = np.full(a.size, np.nan)
        out[w - 1] = start
        out[w:] = tail
        return out
    raise ValueError(f"seed must be 'first' or 'sma', got {seed!r}")


def wilder_rma(x: np.ndarray, period: int, *, seed: str = "sma") -> np.ndarray:
    """Wilder's smoothing: an EMA with `alpha = 1/period`.

    Wilder's 14-period average is *not* a 14-span EMA — `1/14` vs `2/15`. Every
    RSI/ATR mismatch between two libraries starts here.
    """
    a = _as_1d(x)
    p = _check_window(period, a.size)
    alpha = 1.0 / p
    if seed == "first":
        return _recursive_filter(a, alpha, a[0])
    if seed == "sma":
        start = float(a[:p].mean())
        out = np.full(a.size, np.nan)
        out[p - 1] = start
        out[p:] = _recursive_filter(a[p:], alpha, start)
        return out
    raise ValueError(f"seed must be 'first' or 'sma', got {seed!r}")


def rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder's Relative Strength Index.

    RSI = 100 - 100 / (1 + RS), RS = smoothed gain / smoothed loss, both
    smoothed with `wilder_rma` seeded on the simple mean of the first `period`
    changes. A zero average loss pins RSI at 100 (not a division error).
    """
    c = _as_1d(close, "close")
    p = _check_window(period, c.size - 1 if c.size > 1 else 1)
    delta = np.diff(c)
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    avg_gain = wilder_rma(gains, p, seed="sma")
    avg_loss = wilder_rma(losses, p, seed="sma")
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = np.divide(avg_gain, avg_loss)
        out = 100.0 - 100.0 / (1.0 + rs)
    out = np.where(avg_loss == 0.0, 100.0, out)
    out = np.where(np.isnan(avg_gain), np.nan, out)
    return np.concatenate([[np.nan], out])   # realign to the price series


def bollinger(
    close: np.ndarray, window: int = 20, num_std: float = 2.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(lower, middle, upper) bands. Population std, matching the original."""
    c = _as_1d(close, "close")
    mid = sma(c, window)
    sd = rolling_std(c, window, ddof=0)
    return mid - num_std * sd, mid, mid + num_std * sd


def vwap(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray,
    window: int | None = None,
) -> np.ndarray:
    """Volume-weighted average price over the typical price (H+L+C)/3.

    `window=None` is the session-cumulative VWAP (the anchored one a trader
    watches intraday); an int gives the rolling variant.
    """
    h, low_, c, v = (_as_1d(a, n) for a, n in
                     ((high, "high"), (low, "low"), (close, "close"), (volume, "volume")))
    if not (h.size == low_.size == c.size == v.size):
        raise ValueError("high/low/close/volume must be the same length")
    typical = (h + low_ + c) / 3.0
    pv = typical * v
    if window is None:
        cum_v = np.cumsum(v)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(cum_v > 0, np.cumsum(pv) / cum_v, np.nan)
    w = _check_window(window, c.size)
    num = sliding_windows(pv, w).sum(axis=-1)
    den = sliding_windows(v, w).sum(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return _front_pad(np.where(den > 0, num / den, np.nan), c.size)


def macd(
    close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(macd, signal, histogram) using `adjust=False` EMAs seeded at x[0]."""
    c = _as_1d(close, "close")
    if fast >= slow:
        raise ValueError(f"fast ({fast}) must be shorter than slow ({slow})")
    line = ema(c, fast) - ema(c, slow)
    sig = ema(line, signal)
    return line, sig, line - sig


def true_range(high: np.ndarray, low: np.ndarray, prev_close: np.ndarray) -> np.ndarray:
    """max(H-L, |H-Cp|, |L-Cp|) — vectorised with a single `np.maximum.reduce`."""
    h, low_, pc = (_as_1d(a, n) for a, n in
                   ((high, "high"), (low, "low"), (prev_close, "prev_close")))
    return np.maximum.reduce([h - low_, np.abs(h - pc), np.abs(low_ - pc)])


def atr(
    high: np.ndarray, low: np.ndarray, prev_close: np.ndarray, period: int = 14
) -> np.ndarray:
    """Average True Range — Wilder-smoothed true range."""
    return wilder_rma(true_range(high, low, prev_close), period, seed="sma")
