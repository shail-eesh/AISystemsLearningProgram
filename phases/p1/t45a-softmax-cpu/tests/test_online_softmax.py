"""The online algorithm: the running state, the rescale, and the merge."""

from __future__ import annotations

import numpy as np
import pytest
from t45a_softmax import SoftmaxState, online_normalizer, online_softmax, stable_softmax


def _state_of(values: np.ndarray) -> SoftmaxState:
    return SoftmaxState.empty((), dtype=np.float64).update(values)


def test_running_denominator_matches_the_direct_sum():
    x = np.array([3.0, 1.0, 9.0, 2.0])
    state = SoftmaxState.empty((), dtype=np.float64)
    for v in x:
        state = state.update(np.array([v]))
    assert float(state.m) == 9.0
    assert float(state.d) == pytest.approx(float(np.exp(x - x.max()).sum()), rel=1e-15)


def test_first_update_from_empty_does_not_produce_nan():
    """exp(-inf - -inf) is nan; the guard in `update` is the reason this passes."""
    state = SoftmaxState.empty((3,), dtype=np.float64).update(np.zeros((3, 4)))
    assert np.all(np.isfinite(state.d))
    np.testing.assert_allclose(state.d, 4.0)


@pytest.mark.parametrize("chunk", [1, 2, 7, 64, 100_000])
def test_chunk_size_is_a_scheduling_choice_only(chunk):
    rng = np.random.default_rng(1)
    x = rng.standard_normal((3, 500)) * 20
    np.testing.assert_allclose(online_softmax(x, chunk=chunk), stable_softmax(x),
                               rtol=1e-12, atol=1e-15)


def test_merge_is_associative_and_order_independent():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(600) * 30
    whole = _state_of(x)
    a, b, c = x[:137], x[137:400], x[400:]
    left = _state_of(a).merge(_state_of(b)).merge(_state_of(c))
    right = _state_of(a).merge(_state_of(b).merge(_state_of(c)))
    for merged in (left, right):
        assert float(merged.m) == pytest.approx(float(whole.m))
        assert float(merged.d) == pytest.approx(float(whole.d), rel=1e-12)


def test_merge_with_an_empty_state_is_the_identity():
    rng = np.random.default_rng(3)
    x = rng.standard_normal(50)
    s = _state_of(x)
    empty = SoftmaxState.empty((), dtype=np.float64)
    merged = s.merge(empty)
    assert float(merged.d) == pytest.approx(float(s.d))
    assert float(merged.m) == pytest.approx(float(s.m))


def test_online_normalizer_on_adversarial_input():
    x = np.array([[1e4, -1e4, 0.0], [-1e4, -1e4, -1e4]])
    m, d = online_normalizer(x)
    assert np.all(np.isfinite(d))
    np.testing.assert_allclose(m, [1e4, -1e4])
    np.testing.assert_allclose(d, [1.0, 3.0], rtol=1e-12)


def test_online_never_overflows_where_naive_does():
    x = np.full((2, 300), 900.0)
    p = online_softmax(x)
    assert np.all(np.isfinite(p))
    np.testing.assert_allclose(p, 1 / 300, rtol=1e-12)
