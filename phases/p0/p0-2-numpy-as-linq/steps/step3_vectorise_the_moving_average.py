#!/usr/bin/env python3
"""Step 3 — three ways to compute an SMA, and why only one of them is right.

Run:  python3 steps/step3_vectorise_the_moving_average.py

The same 20-day simple moving average, written three ways:

1. **The nested loop** — what everyone ports from C# first. O(n*w).
2. **`cumsum` differencing** — O(n), beautifully short, and numerically
   dangerous: it subtracts two large, nearly-equal running totals, so the
   result loses precision exactly where prices are large and windows are long
   (catastrophic cancellation).
3. **`sliding_window_view` + `mean`** — O(n*w) work but entirely in C, no
   cancellation, and it reads like the definition. This is what `core.sma`
   uses.

The lesson is not "vectorise everything". It is that the fastest formulation
and the numerically sound one are often different, and you have to know which
you picked.
"""

import time

import _bootstrap  # noqa: F401
import numpy as np
from p0_2_indicators import sma
from p0_2_indicators.references import ref_sma, ref_sma_python_loop

from common.data import load_ohlcv

WINDOW = 20


def cumsum_sma(x: np.ndarray, window: int) -> np.ndarray:
    """O(n) but subtracts two big running totals — accurate only while they are small."""
    csum = np.concatenate([[0.0], np.cumsum(x)])
    out = np.full(x.size, np.nan)
    out[window - 1:] = (csum[window:] - csum[:-window]) / window
    return out


def demo_three_implementations() -> None:
    close = load_ohlcv(["ALPHAINFRA"])["close"].to_numpy(dtype=np.float64)
    a = ref_sma_python_loop(close, WINDOW)
    b = cumsum_sma(close, WINDOW)
    c = sma(close, WINDOW)
    ref = ref_sma(close, WINDOW)
    for name, v in [("python loop", a), ("cumsum", b), ("sliding_window", c)]:
        m = ~np.isnan(v)
        print(f"  {name:15s} max |diff| vs pandas = {np.nanmax(np.abs(v[m] - ref[m])):.3e}")


def demo_timing() -> None:
    n = 200_000
    rng = np.random.default_rng(1729)
    x = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    timings = {}
    for name, fn in [
        ("python loop", lambda: ref_sma_python_loop(x[:20_000], WINDOW)),
        ("cumsum", lambda: cumsum_sma(x, WINDOW)),
        ("sliding_window", lambda: sma(x, WINDOW)),
        ("pandas rolling", lambda: ref_sma(x, WINDOW)),
    ]:
        t0 = time.perf_counter()
        fn()
        timings[name] = time.perf_counter() - t0
    scale = timings["python loop"] * 10          # loop ran on 1/10th the data
    print(f"  python loop (scaled to {n:,} bars): {scale * 1e3:8.1f} ms")
    for name in ("cumsum", "sliding_window", "pandas rolling"):
        print(f"  {name:28s}: {timings[name] * 1e3:8.1f} ms  ({scale / timings[name]:6.0f}x)")


def demo_catastrophic_cancellation() -> None:
    """Where cumsum differencing actually breaks: big offset, long series."""
    rng = np.random.default_rng(7)
    n, window = 2_000_000, 500
    x = 1e9 + rng.normal(0, 1.0, n)              # index level ~1e9, noise ~1
    fast = cumsum_sma(x, window)
    exact = sma(x, window)
    err = np.nanmax(np.abs(fast - exact))
    print(f"  series: n={n:,} level~1e9 window={window}")
    print(f"  max |cumsum - sliding_window| = {err:.3e}")
    print(f"  running total reaches ~{np.cumsum(x)[-1]:.3e}; float64 has ~15-16 digits,")
    print("  so differencing two such totals throws away the low-order bits you wanted.")


def demo_window_edges() -> None:
    x = np.arange(1.0, 8.0)
    print(f"  x        = {x}")
    print(f"  sma(x,3) = {sma(x, 3)}")
    print("  the first w-1 slots are NaN by convention: the average is not yet defined.")
    print(f"  windows view shape for w=3: {np.lib.stride_tricks.sliding_window_view(x, 3).shape}"
          " (a view — 0 bytes copied)")


if __name__ == "__main__":
    print("three implementations agree:")
    demo_three_implementations()
    print("timing:")
    demo_timing()
    print("where cumsum breaks:")
    demo_catastrophic_cancellation()
    print("edges:")
    demo_window_edges()
    print("\nstep 3 OK")
