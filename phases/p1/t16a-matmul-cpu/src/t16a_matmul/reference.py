"""The Python end of the ladder, plus honest FLOP accounting.

Two things live here and nothing else: the implementations you write before
reaching for C, and the arithmetic that turns "seconds" into a number you can
compare against hardware — GFLOP/s and arithmetic intensity.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


def flops(m: int, n: int, k: int) -> int:
    """A matmul is m*n*k multiply-adds; each counts as 2 flops by convention."""
    return 2 * m * n * k


def bytes_touched(m: int, n: int, k: int, itemsize: int = 8) -> int:
    """Compulsory traffic if every matrix is read/written exactly once."""
    return (m * k + k * n + m * n) * itemsize


def arithmetic_intensity(m: int, n: int, k: int) -> float:
    """FLOPs per byte at best-case reuse — the x-axis of the roofline chart.

    For a square n x n matmul this grows like n/1.5, which is why matmul is the
    canonical *compute-bound* kernel and why every accelerator is designed
    around it. Naive code throws that reuse away and lands memory-bound; the
    whole topic is about getting it back.
    """
    return flops(m, n, k) / bytes_touched(m, n, k)


def matmul_python_naive(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """Pure Python, no NumPy. Feel the pain once; never write it again."""
    m, k = len(A), len(A[0])
    n = len(B[0])
    C = [[0.0] * n for _ in range(m)]
    for i in range(m):
        Ai = A[i]
        Ci = C[i]
        for kk in range(k):
            a = Ai[kk]
            Bk = B[kk]
            for j in range(n):
                Ci[j] += a * Bk[j]
    return C


def matmul_numpy(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """The baseline every kernel is measured against: whatever BLAS ships."""
    return A @ B


@dataclass(frozen=True)
class Timing:
    name: str
    seconds: float
    gflops: float
    repeats: int

    def __str__(self) -> str:
        return f"{self.name:<26} {self.seconds * 1e3:9.2f} ms  {self.gflops:7.2f} GFLOP/s"


def time_call(name: str, fn, m: int, n: int, k: int, *, repeats: int = 3) -> Timing:
    """Best-of-N wall clock. Best, not mean: we are measuring the kernel, and
    the slow runs measure the machine's other tenants."""
    fn()  # warm the caches and the first-touch pages
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return Timing(name, best, flops(m, n, k) / best / 1e9, repeats)


def random_pair(n: int, *, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return (
        np.ascontiguousarray(rng.standard_normal((n, n))),
        np.ascontiguousarray(rng.standard_normal((n, n))),
    )
