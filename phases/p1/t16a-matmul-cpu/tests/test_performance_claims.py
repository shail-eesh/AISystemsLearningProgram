"""The capsule's 'done when' as an executable claim.

These are timing tests, which are usually a bad idea. They earn their place
here because the *claim* of the topic is a speed ratio, and the margin asked
for (10x) is far outside any plausible noise on a shared machine.
"""

from __future__ import annotations

import numpy as np
import pytest
from t16a_matmul import (
    AVAILABLE,
    arithmetic_intensity,
    bytes_touched,
    flops,
    random_pair,
    time_call,
)
from t16a_matmul.native import UNAVAILABLE_REASON, kernels

pytestmark = pytest.mark.skipif(not AVAILABLE, reason=f"no C compiler: {UNAVAILABLE_REASON}")


def test_flop_accounting():
    assert flops(2, 3, 4) == 2 * 24
    assert bytes_touched(2, 3, 4) == (2 * 4 + 4 * 3 + 2 * 3) * 8
    # Arithmetic intensity grows with n -- the reason matmul is compute-bound.
    assert arithmetic_intensity(1024, 1024, 1024) > 10 * arithmetic_intensity(64, 64, 64)


@pytest.mark.slow
def test_blocked_beats_naive_c_by_at_least_ten_times():
    n = 512
    lib = kernels()
    A, B = random_pair(n)
    naive = time_call("naive", lambda: lib.call("matmul_naive_ijk", A, B), n, n, n, repeats=2)
    best = min(
        (
            time_call(name, lambda name=name: lib.call(name, A, B), n, n, n, repeats=2)
            for name in ("matmul_blocked", "matmul_blocked_omp", "matmul_blocked_regtile")
        ),
        key=lambda t: t.seconds,
    )
    assert naive.seconds / best.seconds >= 10.0, (
        f"blocked kernel only {naive.seconds / best.seconds:.1f}x faster than naive"
    )


@pytest.mark.slow
def test_loop_reordering_alone_is_worth_several_times():
    n = 512
    lib = kernels()
    A, B = random_pair(n)
    ijk = time_call("ijk", lambda: lib.call("matmul_naive_ijk", A, B), n, n, n, repeats=2)
    ikj = time_call("ikj", lambda: lib.call("matmul_ikj", A, B), n, n, n, repeats=2)
    assert ijk.seconds / ikj.seconds >= 3.0


def test_python_naive_agrees_with_numpy():
    from t16a_matmul import matmul_python_naive

    rng = np.random.default_rng(1)
    A = rng.standard_normal((12, 9))
    B = rng.standard_normal((9, 7))
    got = np.array(matmul_python_naive(A.tolist(), B.tolist()))
    np.testing.assert_allclose(got, A @ B, rtol=1e-12, atol=1e-14)


def test_timing_helper_reports_positive_throughput():
    t = time_call("noop", lambda: np.ones((8, 8)) @ np.ones((8, 8)), 8, 8, 8, repeats=2)
    assert t.gflops > 0
    assert "GFLOP/s" in str(t)
