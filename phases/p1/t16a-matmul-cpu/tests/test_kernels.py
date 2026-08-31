"""Every kernel computes the same product; only the memory order differs."""

from __future__ import annotations

import numpy as np
import pytest
from t16a_matmul import ALL_KERNELS, AVAILABLE
from t16a_matmul.native import UNAVAILABLE_REASON, kernels

pytestmark = pytest.mark.skipif(not AVAILABLE, reason=f"no C compiler: {UNAVAILABLE_REASON}")


@pytest.fixture(scope="module")
def lib():
    return kernels()


@pytest.mark.parametrize("name", ALL_KERNELS)
@pytest.mark.parametrize("shape", [(1, 1, 1), (3, 5, 7), (64, 64, 64), (65, 33, 129)])
def test_kernel_matches_numpy(lib, name, shape):
    """Odd sizes on purpose: the tails of every blocking loop are where bugs live."""
    m, k, n = shape
    rng = np.random.default_rng(hash(shape) % 2**31)
    A = rng.standard_normal((m, k))
    B = rng.standard_normal((k, n))
    got = lib.call(name, A, B, mc=16, kc=16, nc=32)
    np.testing.assert_allclose(got, A @ B, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("name", ALL_KERNELS)
def test_kernel_handles_non_multiple_block_sizes(lib, name):
    rng = np.random.default_rng(4)
    A, B = rng.standard_normal((37, 53)), rng.standard_normal((53, 41))
    np.testing.assert_allclose(lib.call(name, A, B, mc=8, kc=8, nc=16), A @ B, rtol=1e-10, atol=1e-12)


def test_shape_mismatch_is_rejected(lib):
    with pytest.raises(ValueError, match="shape mismatch"):
        lib.call("matmul_ikj", np.ones((2, 3)), np.ones((4, 5)))


def test_unknown_kernel_is_rejected(lib):
    with pytest.raises(KeyError):
        lib.call("matmul_does_not_exist", np.ones((2, 2)), np.ones((2, 2)))


def test_non_contiguous_and_float32_inputs_are_coerced(lib):
    rng = np.random.default_rng(0)
    A = rng.standard_normal((16, 32)).T.T[:, ::1]
    B = np.asfortranarray(rng.standard_normal((32, 8)))
    C32 = rng.standard_normal((8, 4)).astype(np.float32)
    out = lib.call("matmul_ikj", A, B)
    np.testing.assert_allclose(out, A @ B, rtol=1e-10, atol=1e-12)
    out2 = lib.call("matmul_ikj", B, C32)
    np.testing.assert_allclose(out2, B @ C32.astype(np.float64), rtol=1e-6, atol=1e-8)


def test_threads_reported(lib):
    assert lib.threads >= 1
    if lib.openmp:
        assert lib.threads == max(1, lib.threads)
