#!/usr/bin/env python3
"""Step 4 — one swapped loop, then cache blocking. Measured, not asserted.

Run:  python3 steps/step4_loop_order_and_blocking.py

i-j-k to i-k-j is a single line and it is the largest single win in the whole
ladder, because it turns a strided column walk into a contiguous row walk that
the compiler can vectorise.

Blocking is the second win, and it is a different problem: ikj has the right
*stride* but streams all of B past the cache once per row of A. Working in
tiles keeps a panel of B resident while several rows of A consume it.
"""

import _bootstrap  # noqa: F401
import numpy as np
from t16a_matmul import AVAILABLE, kernels, random_pair, time_call


def loop_order(n: int = 1024) -> float:
    k = kernels()
    A, B = random_pair(n)
    a = time_call("i-j-k (naive)", lambda A=A, B=B: k.call("matmul_naive_ijk", A, B), n, n, n, repeats=2)
    b = time_call("i-k-j (swapped)", lambda: k.call("matmul_ikj", A, B), n, n, n, repeats=2)
    print(f"  {a}")
    print(f"  {b}")
    print(f"  -> {a.seconds / b.seconds:.1f}x from reordering two loops")
    # Not bit-identical to BLAS and never will be: a different summation order
    # over 1024 terms moves the last couple of digits. That is float addition
    # not being associative, not a bug.
    np.testing.assert_allclose(k.call("matmul_ikj", A, B), A @ B, rtol=1e-8, atol=1e-10)
    return b.seconds


def sweep_block_sizes(n: int = 1024) -> None:
    k = kernels()
    A, B = random_pair(n)
    print(f"\n  block-size sweep at n={n} (single-threaded kernel):")
    print("    mc    kc    nc     ms   GFLOP/s   B-tile")
    best = None
    for mc, kc, nc in [(64, 64, 128), (128, 128, 256), (256, 128, 256),
                       (128, 256, 512), (64, 256, 1024), (256, 256, 256)]:
        t = time_call("blocked", lambda mc=mc, kc=kc, nc=nc:
                      k.call("matmul_blocked", A, B, mc=mc, kc=kc, nc=nc), n, n, n, repeats=2)
        tile_kb = kc * nc * 8 / 1024
        print(f"    {mc:<5} {kc:<5} {nc:<5} {t.seconds * 1e3:6.1f} {t.gflops:8.2f}   {tile_kb:6.0f} KB")
        if best is None or t.seconds < best[0]:
            best = (t.seconds, mc, kc, nc)
    print(f"\n  best: mc={best[1]} kc={best[2]} nc={best[3]}")
    print("  The winner is the one whose B tile fits the cache you are aiming at.")
    print("  Notice how flat the middle of the table is -- within a factor of two,")
    print("  block size is not a magic number, it is 'roughly L2-sized'.")


if __name__ == "__main__":
    if not AVAILABLE:
        print("  no C compiler available; skipping")
    else:
        print("== loop order ==")
        loop_order()
        sweep_block_sizes()
