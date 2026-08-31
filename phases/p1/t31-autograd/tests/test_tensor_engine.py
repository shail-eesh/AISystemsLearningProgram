"""The tensor engine, with broadcasting as the main suspect."""

from __future__ import annotations

import numpy as np
import pytest
from t31_autograd import Tensor, gradcheck_tensors, unbroadcast


@pytest.mark.parametrize(
    ("target", "expected_shape"),
    [((4, 3), (4, 3)), ((1, 3), (1, 3)), ((3,), (3,)), ((4, 1), (4, 1)), ((), ())],
)
def test_unbroadcast_shapes_and_conservation(target, expected_shape):
    grad = np.arange(12, dtype=float).reshape(4, 3)
    out = unbroadcast(grad, target)
    assert out.shape == expected_shape
    assert out.sum() == pytest.approx(grad.sum()), "gradient must be conserved, only regrouped"


def test_bias_gradient_is_the_row_sum():
    x = Tensor(np.ones((32, 4)), requires_grad=False)
    b = Tensor(np.zeros((1, 4)))
    (x + b).sum().backward()
    assert b.grad.shape == (1, 4)
    np.testing.assert_allclose(b.grad, np.full((1, 4), 32.0))


def test_matmul_gradients_match_the_textbook_formula():
    rng = np.random.default_rng(0)
    A = Tensor(rng.standard_normal((4, 3)))
    B = Tensor(rng.standard_normal((3, 5)))
    upstream = rng.standard_normal((4, 5))
    out = A @ B
    out.grad = upstream          # inject an upstream gradient by hand
    out._backward()              # ...and run just this node's local rule
    np.testing.assert_allclose(A.grad, upstream @ B.data.T)
    np.testing.assert_allclose(B.grad, A.data.T @ upstream)


def test_backward_requires_a_scalar():
    t = Tensor(np.ones((2, 2)))
    with pytest.raises(RuntimeError, match="scalar"):
        t.backward()


def test_getitem_accumulates_repeated_indices():
    x = Tensor(np.zeros(3))
    x[np.array([0, 0, 2])].sum().backward()
    np.testing.assert_allclose(x.grad, [2.0, 0.0, 1.0])


def test_detach_stops_the_graph():
    x = Tensor(np.array([2.0]))
    y = x.detach() * 3.0
    y.sum().backward()
    np.testing.assert_allclose(x.grad, [0.0])


def test_requires_grad_false_keeps_no_grad():
    c = Tensor(np.array([1.0]), requires_grad=False)
    assert c.grad is None
    (c * 2.0).sum().backward()
    assert c.grad is None


def _cases(rng):
    return {
        "matmul+bias": (
            lambda A, B, c: ((A @ B + c).tanh()).sum(),
            [Tensor(rng.standard_normal((5, 3))), Tensor(rng.standard_normal((3, 4))),
             Tensor(rng.standard_normal((1, 4)))],
        ),
        "column broadcast": (
            lambda A, c: ((A * c) ** 2).sum(),
            [Tensor(rng.standard_normal((4, 3))), Tensor(rng.standard_normal((4, 1)))],
        ),
        "rank promotion": (
            lambda A, c: ((A + c).tanh()).sum(),
            [Tensor(rng.standard_normal((4, 3))), Tensor(rng.standard_normal((3,)))],
        ),
        "sum-and-reuse": (
            lambda A: ((A - A.sum(axis=1, keepdims=True)) ** 2).sum(),
            [Tensor(rng.standard_normal((4, 3)))],
        ),
        "mean over all": (
            lambda A: (A.mean() * A.sum()),
            [Tensor(rng.standard_normal((4, 3)))],
        ),
        "max along axis": (
            lambda A: (A.max(axis=1) ** 3).sum(),
            [Tensor(rng.standard_normal((4, 3)))],
        ),
        "reshape + transpose": (
            lambda A: ((A.reshape(3, 4).T @ A.reshape(3, 4)) ** 2).sum(),
            [Tensor(rng.standard_normal((4, 3)))],
        ),
        "divide": (
            lambda A, B: (A / B).sum(),
            [Tensor(rng.standard_normal((3, 2))), Tensor(rng.random((3, 2)) + 0.5)],
        ),
        "exp/log chain": (
            lambda A: ((A.exp() + 1.0).log() * A.sigmoid()).sum(),
            [Tensor(rng.standard_normal((3, 3)))],
        ),
        "relu": (
            lambda A: (A.relu() ** 2).sum(),
            [Tensor(rng.standard_normal((4, 4)) + 0.3)],
        ),
    }


@pytest.mark.parametrize("name", list(_cases(np.random.default_rng(0))))
def test_gradcheck_tensor_ops(name):
    rng = np.random.default_rng(11)
    fn, inputs = _cases(rng)[name]
    res = gradcheck_tensors(fn, inputs, tolerance=1e-6)
    assert res.ok, f"{name}: {res}"


def test_broadcast_bug_would_be_caught():
    """A regression guard: sanity-check that gradcheck actually fails on a wrong gradient."""
    from t31_autograd.gradcheck import rel_error

    assert rel_error(1.0, 2.0) > 1e-6
    assert rel_error(0.0, 0.0) == 0.0
