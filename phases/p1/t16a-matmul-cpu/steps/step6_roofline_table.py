#!/usr/bin/env python3
"""Step 6 — the table, and reading the memory hierarchy off your own numbers.

Run:  python3 steps/step6_roofline_table.py

The deliverable of this topic is not a fast kernel; it is a chart you can
defend. This step measures GFLOP/s across sizes for every kernel, and separately
measures the memory system by timing strided reads until the working set falls
out of each cache level.
"""

import time

import _bootstrap  # noqa: F401
import numpy as np
from t16a_matmul import ALL_KERNELS, AVAILABLE, kernels, random_pair, time_call


def table(sizes=(256, 512, 1024)) -> None:
    k = kernels()
    print(f"  {'n':>6} " + "".join(f"{name.replace('matmul_', ''):>17}" for name in ALL_KERNELS)
          + f"{'numpy':>17}")
    for n in sizes:
        A, B = random_pair(n)
        row = [f"{n:>6} "]
        for name in ALL_KERNELS:
            t = time_call(name, lambda name=name, A=A, B=B: k.call(name, A, B), n, n, n, repeats=2)
            row.append(f"{t.gflops:>16.2f} ")
        blas = time_call("numpy", lambda A=A, B=B: A @ B, n, n, n)
        row.append(f"{blas.gflops:>16.2f} ")
        print("".join(row))
    print("\n  (GFLOP/s, higher is better)")


def measure_the_memory_hierarchy() -> None:
    """Read+write bandwidth against working-set size — the chart, from your machine.

    Each row scales one array in place (`a *= 1.0000001`) enough times to be
    measurable, and reports effective bandwidth: 2 bytes moved per byte of
    array (one read, one write). Small arrays live in L1/L2 and go fast; once
    the array no longer fits, the number collapses to DRAM bandwidth.

    Vectorised on purpose: a pointer chase in Python measures the interpreter,
    not the cache. Here the loop is inside NumPy, so what varies row to row is
    the memory system and nothing else.
    """
    print("\n  working set    passes   effective GB/s   (read+write, in place)")
    for kb in (16, 64, 256, 1024, 4096, 16384, 65536):
        a = np.ones(kb * 1024 // 8, dtype=np.float64)
        passes = max(3, int(2e8 // a.nbytes))
        a *= 1.0000001                       # warm / first-touch
        t0 = time.perf_counter()
        for _ in range(passes):
            a *= 1.0000001
        elapsed = time.perf_counter() - t0
        gbs = 2 * a.nbytes * passes / elapsed / 1e9
        print(f"  {kb:>8} KB {passes:>9}   {gbs:>12.1f}")
    print("\n  (The smallest rows understate: at 16 KB the per-call overhead of")
    print("  NumPy is a real fraction of the time. The cliffs are the signal.)")
    print("\n  Read the cliffs. Each one is a cache level running out, and the")
    print("  block sizes that won the sweep in step 4 are the ones that keep the")
    print("  B tile on the fast side of the last cliff.")


if __name__ == "__main__":
    if not AVAILABLE:
        print("  no C compiler available; skipping")
    else:
        print("== throughput table ==")
        table()
        measure_the_memory_hierarchy()
