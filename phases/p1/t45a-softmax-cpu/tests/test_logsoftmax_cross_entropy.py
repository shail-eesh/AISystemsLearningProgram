"""log-softmax, the fused loss, and the gradient that falls out of it."""

from __future__ import annotations

import numpy as np
import pytest
from t45a_softmax import cross_entropy, cross_entropy_grad, log_softmax, logsumexp, stable_softmax


def test_logsumexp_matches_the_direct_computation():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 9))
    np.testing.assert_allclose(logsumexp(x), np.log(np.exp(x).sum(axis=-1)), rtol=1e-12)


def test_logsumexp_survives_huge_inputs():
    x = np.array([[1e4, 1e4]])
    np.testing.assert_allclose(logsumexp(x), [1e4 + np.log(2)], rtol=1e-15)


def test_log_softmax_keeps_precision_where_log_of_softmax_loses_it():
    x = np.array([[0.0, -800.0]])
    with np.errstate(divide="ignore"):
        naive = np.log(stable_softmax(x))
    fused = log_softmax(x)
    assert np.isneginf(naive[0, 1])
    assert np.isfinite(fused[0, 1])
    assert fused[0, 1] == pytest.approx(-800.0, rel=1e-12)


def test_log_softmax_rows_exponentiate_to_one():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((3, 11)) * 5
    np.testing.assert_allclose(np.exp(log_softmax(x)).sum(axis=-1), 1.0, rtol=1e-12)


def test_cross_entropy_matches_the_unfused_form_on_tame_inputs():
    rng = np.random.default_rng(2)
    logits = rng.standard_normal((5, 7)) * 3
    targets = rng.integers(0, 7, 5)
    manual = -np.log(stable_softmax(logits))[np.arange(5), targets].mean()
    assert float(cross_entropy(logits, targets)) == pytest.approx(manual, rel=1e-12)


def test_cross_entropy_is_finite_where_the_unfused_form_is_not():
    logits = np.array([[0.0, -900.0]])
    assert np.isfinite(float(cross_entropy(logits, np.array([1]))))
    with np.errstate(divide="ignore"):
        assert np.isinf(-np.log(stable_softmax(logits))[0, 1])


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_reductions(reduction):
    rng = np.random.default_rng(3)
    logits = rng.standard_normal((4, 3))
    targets = rng.integers(0, 3, 4)
    out = cross_entropy(logits, targets, reduction=reduction)
    if reduction == "none":
        assert out.shape == (4,)
    else:
        assert np.isscalar(out) or out.shape == ()


def test_perfect_prediction_costs_almost_nothing():
    logits = np.array([[100.0, 0.0, 0.0]])
    assert float(cross_entropy(logits, np.array([0]))) == pytest.approx(0.0, abs=1e-12)


def test_uniform_prediction_costs_log_k():
    logits = np.zeros((1, 8))
    assert float(cross_entropy(logits, np.array([3]))) == pytest.approx(np.log(8), rel=1e-12)


def test_gradient_matches_central_differences():
    rng = np.random.default_rng(4)
    logits = rng.standard_normal((4, 5))
    targets = rng.integers(0, 5, 4)
    analytic = cross_entropy_grad(logits, targets)
    numeric = np.zeros_like(logits)
    h = 1e-6
    for i in range(logits.shape[0]):
        for j in range(logits.shape[1]):
            up, down = logits.copy(), logits.copy()
            up[i, j] += h
            down[i, j] -= h
            numeric[i, j] = (cross_entropy(up, targets) - cross_entropy(down, targets)) / (2 * h)
    assert np.abs(analytic - numeric).max() < 1e-8


def test_gradient_rows_sum_to_zero():
    """softmax sums to 1 and onehot sums to 1, so the difference sums to 0.

    A cheap invariant that catches a surprising number of implementation slips.
    """
    rng = np.random.default_rng(5)
    g = cross_entropy_grad(rng.standard_normal((6, 4)), rng.integers(0, 4, 6))
    np.testing.assert_allclose(g.sum(axis=-1), 0.0, atol=1e-15)
