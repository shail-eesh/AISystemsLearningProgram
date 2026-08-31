#!/usr/bin/env python3
"""Step 2 — the NumPy baseline, and what "fast" would even mean.

Run:  python3 steps/step2_numpy_and_flops.py

`A @ B` calls into whatever BLAS NumPy was linked against. Before trying to
beat it, work out the ceiling: cores x clock x FLOPs-per-cycle. Then every
later measurement is a percentage of a real number instead of a vibe.
"""

import os

import _bootstrap  # noqa: F401
from t16a_matmul import matmul_numpy, random_pair, time_call


def numpy_across_sizes() -> float:
    best = 0.0
    for n in (128, 256, 512, 1024, 1536):
        A, B = random_pair(n)
        t = time_call(f"numpy n={n}", lambda A=A, B=B: matmul_numpy(A, B), n, n, n)
        best = max(best, t.gflops)
        print(f"  {t}")
    return best


def estimate_the_ceiling(measured_peak: float) -> None:
    cores = os.cpu_count() or 1
    print(f"\n  logical cores visible: {cores}")
    print(f"  best measured (BLAS):  {measured_peak:.1f} GFLOP/s")
    print("\n  A modern x86 core retires up to 2 FMAs per cycle. With AVX2 that is")
    print("  4 doubles x 2 flops x 2 units = 16 FLOPs/cycle; with AVX-512, 32.")
    for width, label in ((16, "AVX2"), (32, "AVX-512")):
        implied = measured_peak / (cores * width)
        print(f"    if {label:<8}: implied clock {implied:.2f} GHz")
    print("\n  Whichever line lands near the real clock tells you which ISA BLAS")
    print("  is using -- and that number is the 100% mark for everything below.")


if __name__ == "__main__":
    print("== NumPy / BLAS across sizes ==")
    peak = numpy_across_sizes()
    print("\n== the ceiling ==")
    estimate_the_ceiling(peak)
