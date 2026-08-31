#!/usr/bin/env python3
"""Step 5 — register tiling and threads, and where the remaining gap lives.

Run:  python3 steps/step5_simd_and_threads.py

Two more multipliers, both cheap:

* **Register tiling.** Process four rows of A at once so a single load of
  `B[k][j]` feeds four FMAs instead of one. This is the germ of what OpenBLAS
  does with hand-written assembly and much bigger tiles.
* **Threads.** The column tiles of C are disjoint, so `#pragma omp parallel
  for` over `jj` needs no locks and no reduction.

Then the honest part: measure how far from the machine's ceiling we still are,
and say why.
"""

import os

import _bootstrap  # noqa: F401
from t16a_matmul import AVAILABLE, kernels, random_pair, time_call


def main(n: int = 1024) -> None:
    k = kernels()
    A, B = random_pair(n)
    results = {}
    for name in ("matmul_naive_ijk", "matmul_ikj", "matmul_blocked",
                 "matmul_blocked_omp", "matmul_blocked_regtile"):
        t = time_call(name, lambda name=name, A=A, B=B: k.call(name, A, B), n, n, n, repeats=2)
        results[name] = t
        print(f"  {t}")
    blas = time_call("numpy (OpenBLAS)", lambda A=A, B=B: A @ B, n, n, n)
    print(f"  {blas}")

    naive = results["matmul_naive_ijk"]
    best = min(results.values(), key=lambda t: t.seconds)
    print(f"\n  best hand-written kernel: {best.name}")
    print(f"  vs naive C:  {naive.seconds / best.seconds:5.1f}x")
    print(f"  vs OpenBLAS: {best.gflops / blas.gflops:5.1%} of its throughput")

    print("\n  Thread scaling (OMP_NUM_THREADS is read at process start, so this")
    print("  reports the single number this process was given):")
    print(f"    OpenMP threads: {k.threads} of {os.cpu_count()} logical cores")

    print("\n  Where the rest of the gap is: this kernel still writes C back to")
    print("  memory on every k step. A real micro-kernel holds a tile of C in")
    print("  vector registers across the whole k loop and stores it once. That")
    print("  is the single change worth the most, and it is what Phase 7 rebuilds")
    print("  on a GPU, where the same idea is called 'accumulate in registers'.")


if __name__ == "__main__":
    if not AVAILABLE:
        print("  no C compiler available; skipping")
    else:
        main()
