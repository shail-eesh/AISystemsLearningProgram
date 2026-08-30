#!/usr/bin/env python3
"""Step 4 — the recurrence that will not vectorise.

Run:  python3 steps/step4_recurrences_and_ema.py

An SMA is a reduction over independent windows, so it vectorises trivially. An
EMA is not:

    y[t] = a * x[t] + (1 - a) * y[t-1]

Every output depends on the previous output. There is a closed form —

    y[t] = (1-a)^(t+1) * y_seed  +  a * sum_{k<=t} (1-a)^(t-k) * x[k]

— and it can be written as a cumsum of `x[k] / (1-a)^k`, which is the trick
that shows up in every "vectorised EMA" blog post. It is also a numerical trap.
`(1-a)^-k` grows exponentially: for a 12-day span it passes float64's 1.8e308
around bar 4,200. Up to that point the answer is *exact to 1e-14*, and one bar
later the entire tail is NaN. Tests on a year of data never see it.

The professional answer is to keep the loop but move it into C:
`scipy.signal.lfilter` implements exactly this one-pole IIR filter. This is the
general shape of the skill — recognise a *scan*, and reach for the primitive
that already implements it, rather than forcing it into a reduction.
"""

import time

import _bootstrap  # noqa: F401
import numpy as np
from p0_2_indicators import ema, wilder_rma
from p0_2_indicators.references import ref_ema

SPAN = 12
ALPHA = 2.0 / (SPAN + 1.0)


def naive_loop_ema(x: np.ndarray, alpha: float) -> np.ndarray:
    out = np.empty_like(x)
    y = x[0]
    for i, xi in enumerate(x):
        y = alpha * xi + (1 - alpha) * y if i else xi
        out[i] = y
    return out


def algebraic_ema(x: np.ndarray, alpha: float) -> np.ndarray:
    """The 'clever' closed form. Correct on paper, doomed in float64.

        y[t] = (1-a)^t * ( x[0] + sum_{k=1..t} a * x[k] / (1-a)^k )

    One cumsum, no loop — and `(1-a)^-k` grows without bound.
    """
    n = x.size
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        decay = (1 - alpha) ** np.arange(n, dtype=np.float64)
        tail = np.cumsum(np.concatenate([[0.0], alpha * x[1:] / decay[1:]]))
        return decay * (x[0] + tail)


def demo_agreement_on_short_series() -> None:
    rng = np.random.default_rng(1729)
    x = 100.0 + np.cumsum(rng.normal(0, 0.5, 50))
    ours = ema(x, SPAN)
    loop = naive_loop_ema(x, ALPHA)
    alg = algebraic_ema(x, ALPHA)
    ref = ref_ema(x, SPAN)
    print(f"  n=50:  lfilter vs loop      {np.max(np.abs(ours - loop)):.3e}")
    print(f"         lfilter vs algebraic {np.max(np.abs(ours - alg)):.3e}")
    print(f"         lfilter vs pandas    {np.max(np.abs(ours - ref)):.3e}")
    print("  all four agree while the decay factor is still small.")


def demo_algebraic_blows_up() -> None:
    rng = np.random.default_rng(1729)
    for n in (200, 1_000, 3_000, 4_000, 4_500, 6_000):
        x = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
        alg, good = algebraic_ema(x, ALPHA), ema(x, SPAN)
        err = np.nanmax(np.abs(alg - good))
        with np.errstate(over="ignore"):
            biggest = np.float64(1 - ALPHA) ** np.float64(-(n - 1))
        state = "ok" if err < 1e-6 else ("degraded" if np.isfinite(err) else "INF/NaN")
        print(f"  n={n:5d}  (1-a)^-(n-1) = {biggest:9.2e}  max err = {err:9.3e}  [{state}]")
    print("  float64 tops out near 1.8e308. For span=12 that is ~4,200 bars — about")
    print("  seventeen years of daily data, and sooner for a shorter span. The error")
    print("  stays tiny right up to the cliff, then the whole tail becomes NaN at once:")
    print("  the worst kind of numerical bug, because small-scale tests never see it.")


def demo_timing() -> None:
    rng = np.random.default_rng(3)
    x = 100.0 + np.cumsum(rng.normal(0, 0.5, 500_000))
    t0 = time.perf_counter()
    naive_loop_ema(x[:50_000], ALPHA)
    loop = (time.perf_counter() - t0) * 10
    t0 = time.perf_counter()
    ema(x, SPAN)
    fast = time.perf_counter() - t0
    print(f"  python loop (scaled to 500k): {loop * 1e3:7.1f} ms")
    print(f"  scipy lfilter:                {fast * 1e3:7.1f} ms  ({loop / fast:.0f}x)")
    print("  same algorithm, same sequential dependency — the loop just moved into C.")


def demo_wilder_is_not_an_ema() -> None:
    rng = np.random.default_rng(11)
    x = 100.0 + np.cumsum(rng.normal(0, 0.5, 300))
    w = wilder_rma(x, 14)
    e = ema(x, 14)
    print(f"  Wilder alpha = 1/14 = {1 / 14:.4f}")
    print(f"  EMA(14) alpha = 2/15 = {2 / 15:.4f}")
    print(f"  max |Wilder(14) - EMA(14)| = {np.nanmax(np.abs(w - e)):.4f}")
    print("  An 'RSI mismatch between two libraries' is almost always these two swapped.")
    print(f"  EMA(27) matches Wilder(14) closely: {np.nanmax(np.abs(w - ema(x, 27))):.4f}")


if __name__ == "__main__":
    print("all four agree on a short series:")
    demo_agreement_on_short_series()
    print("the closed form degrades:")
    demo_algebraic_blows_up()
    print("timing:")
    demo_timing()
    print("Wilder vs EMA:")
    demo_wilder_is_not_an_ema()
    print("\nstep 4 OK")
