#!/usr/bin/env python3
"""Step 1 — the triple loop in pure Python. Feel the pain, then measure it.

Run:  python3 steps/step1_naive_python.py

Nobody needs convincing that Python loops are slow. What is worth having is the
*number*, because it sets the scale for everything after: how many floating
point operations per second does the interpreter manage, and what would a
1024x1024 matmul cost at that rate?
"""

import _bootstrap  # noqa: F401
import numpy as np
from t16a_matmul import arithmetic_intensity, flops, matmul_python_naive, time_call


def measure_small() -> float:
    rng = np.random.default_rng(0)
    n = 128
    A = rng.standard_normal((n, n)).tolist()
    B = rng.standard_normal((n, n)).tolist()

    t = time_call("python triple loop", lambda: matmul_python_naive(A, B), n, n, n, repeats=1)
    print(f"  n=128: {t}")

    ref = np.array(A) @ np.array(B)
    got = np.array(matmul_python_naive(A, B))
    print(f"  max abs error vs NumPy: {np.abs(got - ref).max():.2e}")
    return t.gflops


def extrapolate(gflops: float) -> None:
    for n in (256, 512, 1024, 4096):
        seconds = flops(n, n, n) / (gflops * 1e9)
        unit = f"{seconds:.1f} s" if seconds < 600 else f"{seconds / 3600:.1f} hours"
        print(f"  n={n:<5} would take {unit:>12}   ({flops(n, n, n) / 1e9:.1f} GFLOP)")


def why_matmul_is_worth_optimising() -> None:
    print("  arithmetic intensity (FLOPs per byte, at perfect reuse):")
    for n in (64, 256, 1024, 4096):
        print(f"    n={n:<5} {arithmetic_intensity(n, n, n):8.1f}")
    print("\n  It grows with n. That is the whole reason matmul can be made fast:")
    print("  there is enough arithmetic per byte to hide the memory system --")
    print("  if, and only if, you actually reuse the bytes you loaded.")


if __name__ == "__main__":
    print("== pure Python ==")
    g = measure_small()
    print("\n== what that rate implies ==")
    extrapolate(g)
    print("\n== why bother ==")
    why_matmul_is_worth_optimising()
