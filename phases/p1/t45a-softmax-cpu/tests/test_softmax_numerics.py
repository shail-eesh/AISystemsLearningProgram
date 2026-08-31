"""Softmax: the identity, the failure it prevents, and every dtype."""

from __future__ import annotations

import numpy as np
import pytest
from t45a_softmax import (
    naive_softmax,
    online_softmax,
    stable_softmax,
    two_pass_softmax,
)

IMPLEMENTATIONS = {
    "stable": stable_softmax,
    "two_pass": two_pass_softmax,
    "online": online_softmax,
}


@pytest.fixture
def rng():
    return np.random.default_rng(45)


@pytest.mark.parametrize("name", list(IMPLEMENTATIONS))
def test_rows_are_probability_distributions(name, rng):
    x = rng.standard_normal((6, 33)) * 7
    p = IMPLEMENTATIONS[name](x)
    assert np.all(p >= 0)
    np.testing.assert_allclose(p.sum(axis=-1), 1.0, rtol=1e-12)


@pytest.mark.parametrize("name", list(IMPLEMENTATIONS))
def test_matches_the_float64_reference(name, rng):
    x = rng.standard_normal((4, 128)) * 12
    np.testing.assert_allclose(IMPLEMENTATIONS[name](x), stable_softmax(x), rtol=1e-12, atol=1e-15)


def test_naive_really_does_overflow():
    """Kept as a test so the failure is pinned, not merely described."""
    with np.errstate(over="ignore", invalid="ignore"):
        out = naive_softmax(np.array([[1e4, -1e4, 0.0]]))
    assert not np.all(np.isfinite(out))


@pytest.mark.parametrize("name", ["stable", "two_pass", "online"])
def test_survives_adversarial_logits(name):
    x = np.array([[1e4, -1e4, 0.0, 9999.0, -1e4]])
    p = IMPLEMENTATIONS[name](x)
    assert np.all(np.isfinite(p))
    np.testing.assert_allclose(p.sum(), 1.0, rtol=1e-12)
    # exp(-1) / (1 + exp(-1)) for the 9999 entry against the 1e4 entry
    np.testing.assert_allclose(p[0, 0], 1 / (1 + np.exp(-1.0)), rtol=1e-12)


def test_shift_invariance_is_exact_enough():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((3, 16)) * 3
    base = stable_softmax(x)
    for shift in (0.0, 50.0, 500.0, -500.0):
        np.testing.assert_allclose(stable_softmax(x + shift), base, rtol=1e-12, atol=1e-15)


def test_uniform_input_gives_uniform_output():
    p = stable_softmax(np.full((2, 5), 800.0))
    np.testing.assert_allclose(p, 0.2, rtol=1e-12)


ADVERSARIAL = np.array(
    [
        [1e4, -1e4, 0.0, 1.0, -3.5, 1e4, -1e4, 2.25],
        [-1e4, -1e4, -1e4, -1e4, 5.0, 4.0, 3.0, -1e4],
        [800.0, 800.0, 800.0, -800.0, 0.0, 0.0, 0.0, 0.0],
    ]
)


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.float16])
def test_online_equals_two_pass_in_every_dtype(dtype):
    """The topic's central claim: the *algorithm* costs no accuracy.

    Compared within the dtype, not against float64 — the question is whether
    streaming the normaliser is as accurate as computing the max up front, and
    that has nothing to do with how much precision the dtype has to begin with.
    """
    x = ADVERSARIAL.astype(dtype)
    got = online_softmax(x).astype(np.float64)
    want = stable_softmax(x).astype(np.float64)
    assert np.all(np.isfinite(got))
    tol = {np.float64: 1e-15, np.float32: 1e-7, np.float16: 1e-3}[dtype]
    assert np.abs(got - want).max() <= tol


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.float16])
def test_adversarial_logits_against_the_float64_reference(dtype):
    """The capsule's 'done when': fp32/fp16 on +/-1e4 logits."""
    ref = stable_softmax(ADVERSARIAL)
    got = online_softmax(ADVERSARIAL.astype(dtype)).astype(np.float64)
    assert np.all(np.isfinite(got))
    tol = {np.float64: 1e-15, np.float32: 1e-7, np.float16: 1e-3}[dtype]
    assert np.abs(got - ref).max() <= tol
    np.testing.assert_allclose(got.sum(axis=-1), 1.0, rtol=1e-3)


def test_float16_cannot_resolve_neighbouring_giant_logits():
    """An expected limitation, pinned so nobody mistakes it for a bug.

    float16 has ~11 bits of mantissa, so the spacing between representable
    values near 1e4 is 8. Two logits of 10000 and 9999 are the *same float16
    number*, and the softmax over them is 0.5/0.5 — correct for the inputs the
    dtype can actually hold, and wrong for the inputs you meant.
    """
    x = np.array([[1e4, 9999.0]], dtype=np.float16)
    np.testing.assert_allclose(online_softmax(x).astype(np.float64), [[0.5, 0.5]])
    exact = stable_softmax(np.array([[1e4, 9999.0]]))
    assert abs(exact[0, 0] - 0.5) > 0.2
    # ...which is why attention logits are accumulated in fp32 even when the
    # weights are fp16. Phase 7 leans on this fact.


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_axis_argument(axis, rng):
    x = rng.standard_normal((5, 9))
    for fn in (stable_softmax, two_pass_softmax, online_softmax):
        p = fn(x, axis=axis)
        np.testing.assert_allclose(p.sum(axis=axis), 1.0, rtol=1e-12)


def test_single_element_row():
    np.testing.assert_allclose(stable_softmax(np.array([[42.0]])), [[1.0]])
    np.testing.assert_allclose(online_softmax(np.array([[42.0]])), [[1.0]])
