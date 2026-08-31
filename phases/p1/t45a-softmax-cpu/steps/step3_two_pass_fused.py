#!/usr/bin/env python3
"""Step 3 — count the passes over memory, then remove one.

Run:  python3 steps/step3_two_pass_fused.py

The naive-but-stable version reads x three times: once for the max, once to
exponentiate and sum, once to divide. Fusing the second and third gets it to
two. For a kernel that is memory-bound (and softmax always is — one exp per
element is nothing), passes over memory *are* the runtime.
"""

import _bootstrap  # noqa: F401
import numpy as np
from t45a_softmax.softmax import stable_softmax, two_pass_softmax


def count_the_passes() -> None:
    print("  three-pass (textbook stable):")
    print("    pass 1: c = max(x)")
    print("    pass 2: e = exp(x - c);  s = sum(e)")
    print("    pass 3: out = e / s")
    print("  ...but pass 2 already materialised e, so pass 3 reads e, not x.")
    print("\n  two-pass (fused):")
    print("    pass 1: c = max(x)")
    print("    pass 2: e = exp(x - c) and accumulate s in the same sweep")
    print("    then:   out = e / s  (e is still in cache if the row fits)")


def measure(n_rows: int = 512, n_cols: int = 4096) -> None:

    rng = np.random.default_rng(0)
    x = rng.standard_normal((n_rows, n_cols)).astype(np.float32) * 5
    print(f"\n  {n_rows}x{n_cols} float32 ({x.nbytes / 1e6:.1f} MB):")
    for name, fn in (("stable_softmax", stable_softmax), ("two_pass_softmax", two_pass_softmax)):
        fn(x)
        best = min(_time(fn, x) for _ in range(3))
        print(f"    {name:<18} {best * 1e3:7.2f} ms   {x.nbytes / best / 1e9:5.1f} GB/s of input")
    np.testing.assert_allclose(two_pass_softmax(x), stable_softmax(x), rtol=1e-6)
    print("    (identical outputs; NumPy already fuses much of this, so the gap")
    print("     is small here -- in a hand-written kernel it is the whole game)")


def _time(fn, x) -> float:
    import time

    t0 = time.perf_counter()
    fn(x)
    return time.perf_counter() - t0


if __name__ == "__main__":
    print("== passes over memory ==")
    count_the_passes()
    measure()
