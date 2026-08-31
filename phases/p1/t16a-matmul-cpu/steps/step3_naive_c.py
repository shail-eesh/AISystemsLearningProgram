"""Step 3 — the same triple loop in C, called through ctypes.

Run:  python3 steps/step3_naive_c.py

Moving to C buys a factor of hundreds over the interpreter and then stops
dead, at a small fraction of what the machine can do. The gap is not the
language. It is that the inner loop walks a column of B, touching one useful
double per 64-byte cache line.
"""

import _bootstrap  # noqa: F401
import numpy as np
from t16a_matmul import AVAILABLE, environment, kernels, random_pair, time_call


def main() -> None:
    if not AVAILABLE:
        print(f"  no compiler available: {environment()['reason']}")
        return
    k = kernels()
    print(f"  built with: {' '.join(k.flags)}   OpenMP threads: {k.threads}")

    for n in (256, 512, 1024):
        A, B = random_pair(n)
        ref = A @ B
        got = k.call("matmul_naive_ijk", A, B)
        err = np.abs(got - ref).max()
        t = time_call(f"naive C  n={n}", lambda A=A, B=B: k.call("matmul_naive_ijk", A, B), n, n, n, repeats=2)
        nt = time_call(f"numpy    n={n}", lambda A=A, B=B: A @ B, n, n, n)
        print(f"  {t}   (max err {err:.1e})")
        print(f"  {nt}   -> BLAS is {t.seconds / nt.seconds:.0f}x faster")

    print("\n  Same flops. Same language, near enough. The difference is memory order.")


if __name__ == "__main__":
    main()
