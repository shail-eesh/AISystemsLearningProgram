#!/usr/bin/env python3
"""Step 2 — the shift identity, proved and then tested.

Run:  python3 steps/step2_max_subtraction.py

    softmax(x)_i = exp(x_i - c) / sum_j exp(x_j - c)      for any constant c

Proof: multiply top and bottom by exp(-c). Nothing is approximated; softmax is
*invariant* under adding a constant to every input. So pick c = max(x): the
largest exponent becomes exactly exp(0) = 1 and every other one is below it.
Overflow is now impossible by construction.
"""

import _bootstrap  # noqa: F401
import numpy as np
from t45a_softmax.softmax import stable_softmax


def invariance_holds_for_any_shift() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 8)) * 2
    base = stable_softmax(x)
    print("  shift      max |softmax(x+shift) - softmax(x)|")
    for shift in (0.0, 1.0, 50.0, 500.0, -500.0):
        got = stable_softmax(x + shift)
        print(f"  {shift:>8.1f}   {np.abs(got - base).max():.3e}")
    print("\n  Invariance is exact in exact arithmetic and near-exact in float64;")
    print("  the residual is rounding in exp, not error in the identity.")


def why_the_max_and_not_the_mean() -> None:
    x = np.array([[0.0, 1000.0]])
    print("\n  x = [0, 1000]")
    for name, c in (("mean", x.mean()), ("max", x.max())):
        shifted = x - c
        with np.errstate(over="ignore"):
            e = np.exp(shifted)
        print(f"    c = {name:<5} ({c:7.1f}) -> exp(x - c) = {e[0]}")
    print("\n  Any c removes the *scale* problem; only c = max guarantees every")
    print("  exponent is <= 0, so nothing can overflow no matter how spread out")
    print("  the inputs are. Underflow to 0 is fine: those terms are negligible.")


def adversarial_grid() -> None:
    print("\n  stable softmax on adversarial inputs, three dtypes:")
    print("  (reference = float64 stable softmax)")
    rng = np.random.default_rng(3)
    x64 = np.where(rng.random((4, 64)) > 0.5, 1e4, -1e4) + rng.standard_normal((4, 64))
    ref = stable_softmax(x64)
    for dtype in (np.float64, np.float32, np.float16):
        got = stable_softmax(x64.astype(dtype)).astype(np.float64)
        print(f"    {np.dtype(dtype).name:<9} max abs err {np.abs(got - ref).max():.3e}"
              f"   rows sum to {got.sum(axis=-1).min():.6f}..{got.sum(axis=-1).max():.6f}")


if __name__ == "__main__":
    print("== invariance ==")
    invariance_holds_for_any_shift()
    why_the_max_and_not_the_mean()
    adversarial_grid()
