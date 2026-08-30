#!/usr/bin/env python3
"""Step 1 — an ndarray is not a `List<double>`, and that is the whole point.

Run:  python3 steps/step1_arrays_vs_lists.py

A `List<double>` in .NET is already a contiguous `double[]` with a count — so
why is a Python list slow? Because a Python list is an array of *pointers to
boxed objects*. Every element is a full `PyFloat` on the heap: 32 bytes for 8
bytes of payload, and every arithmetic operation is a virtual dispatch.

`np.ndarray` is what you assumed `List<double>` was: one contiguous block of
unboxed float64, with the loop living in C. The speed-up is not "NumPy is
optimised"; it is "the interpreter got out of the way".
"""

import sys
import time

import _bootstrap  # noqa: F401
import numpy as np

from common.data import load_ohlcv


def demo_memory_layout() -> None:
    py = [1.0] * 1_000_000
    np_ = np.ones(1_000_000, dtype=np.float64)
    py_bytes = sys.getsizeof(py) + sys.getsizeof(1.0) * 1_000_000
    print(f"  python list: ~{py_bytes / 1e6:.1f} MB (pointers + boxed floats)")
    print(f"  ndarray:      {np_.nbytes / 1e6:.1f} MB (raw float64)")
    print(f"  dtype={np_.dtype} itemsize={np_.itemsize} contiguous={np_.flags['C_CONTIGUOUS']}")


def demo_why_loops_die() -> None:
    n = 1_000_000
    xs = [float(i) for i in range(n)]
    arr = np.arange(n, dtype=np.float64)

    t0 = time.perf_counter()
    total = 0.0
    for x in xs:
        total += x * 1.0001
    loop = time.perf_counter() - t0

    t0 = time.perf_counter()
    vec = float((arr * 1.0001).sum())
    vector = time.perf_counter() - t0

    print(f"  python loop: {loop * 1e3:7.1f} ms")
    print(f"  vectorised:  {vector * 1e3:7.1f} ms  -> {loop / vector:.0f}x")
    assert abs(total - vec) / vec < 1e-9


def demo_dtypes_bite() -> None:
    ints = np.array([1, 2, 3])
    print(f"  np.array([1,2,3]).dtype = {ints.dtype}  <- integer, not float")
    ints_div = ints / 2
    print(f"  ...but `/` always promotes: {ints_div} ({ints_div.dtype})")
    small = np.array([250, 10], dtype=np.uint8)
    with np.errstate(over="ignore"):
        print(f"  uint8 wraps around: 250 + 10 = {small[0] + small[1]} (NumPy 2 warns; "
              "older versions were silent)")
    print("  rule: state the dtype at the boundary — np.asarray(x, dtype=np.float64)")


def demo_views_vs_copies() -> None:
    # pandas 3 hands back a read-only array (copy-on-write); .copy() makes it ours.
    prices = load_ohlcv(["ALPHAINFRA"])["close"].to_numpy(dtype=np.float64).copy()
    window = prices[10:20]                 # a VIEW: no data copied
    window[0] = -1.0
    print(f"  slicing gives a view: prices[10] is now {prices[10]}")
    prices[10] = window[0] = 100.0

    fancy = prices[[10, 11, 12]]           # fancy indexing COPIES
    fancy[0] = -999.0
    print(f"  fancy indexing copies: prices[10] still {prices[10]}")
    print(f"  check with `.base`: view={window.base is not None}, copy={fancy.base is None}")


def demo_nan_semantics() -> None:
    x = np.array([1.0, np.nan, 3.0])
    print(f"  x.mean()    = {x.mean()}      <- NaN poisons the reduction")
    print(f"  np.nanmean  = {np.nanmean(x)}")
    print(f"  np.nan == np.nan -> {np.nan == np.nan}; use np.isnan()")
    print("  every indicator here NaN-pads its warm-up, so this matters immediately")


if __name__ == "__main__":
    print("memory layout:")
    demo_memory_layout()
    print("why loops die:")
    demo_why_loops_die()
    print("dtypes:")
    demo_dtypes_bite()
    print("views vs copies:")
    demo_views_vs_copies()
    print("NaN:")
    demo_nan_semantics()
    print("\nstep 1 OK")
