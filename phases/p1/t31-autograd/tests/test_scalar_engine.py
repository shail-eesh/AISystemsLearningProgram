"""The scalar engine: local rules, accumulation, and topological order."""

from __future__ import annotations

import math

import pytest
from t31_autograd import Value, gradcheck_values


def test_add_routes_gradient():
    a, b = Value(2.0), Value(3.0)
    (a + b).backward()
    assert (a.grad, b.grad) == (1.0, 1.0)


def test_mul_switches_gradient():
    a, b = Value(2.0), Value(3.0)
    (a * b).backward()
    assert (a.grad, b.grad) == (3.0, 2.0)


def test_reuse_accumulates():
    x = Value(3.0)
    (x * x + x).backward()
    assert x.grad == pytest.approx(7.0)


def test_repeated_use_three_ways():
    x = Value(2.0)
    (x * x * x).backward()
    assert x.grad == pytest.approx(12.0)  # 3x^2


def test_backward_resets_between_calls():
    x = Value(3.0)
    y = x * x
    y.backward()
    first = x.grad
    y.backward()
    assert x.grad == pytest.approx(first), "backward() must zero before accumulating"


@pytest.mark.parametrize(
    ("fn", "at", "expected"),
    [
        (lambda v: v.tanh(), 0.7, 1 - math.tanh(0.7) ** 2),
        (lambda v: v.exp(), 1.3, math.exp(1.3)),
        (lambda v: v.log(), 2.5, 1 / 2.5),
        (lambda v: v.sigmoid(), -0.4, (lambda s: s * (1 - s))(1 / (1 + math.exp(0.4)))),
        (lambda v: v.relu(), 1.5, 1.0),
        (lambda v: v.relu(), -1.5, 0.0),
        (lambda v: v**3, 2.0, 12.0),
    ],
)
def test_unary_derivatives(fn, at, expected):
    x = Value(at)
    fn(x).backward()
    assert x.grad == pytest.approx(expected, rel=1e-12)


def test_division_and_rsub():
    a, b = Value(6.0), Value(3.0)
    (a / b).backward()
    assert a.grad == pytest.approx(1 / 3)
    assert b.grad == pytest.approx(-6 / 9)

    c = Value(4.0)
    (10 - c).backward()
    assert c.grad == pytest.approx(-1.0)


def test_log_of_non_positive_is_an_error():
    with pytest.raises(ValueError, match="non-positive"):
        Value(-1.0).log()


def test_pow_by_a_value_is_rejected():
    with pytest.raises(TypeError):
        Value(2.0) ** Value(3.0)  # type: ignore[operator]


def test_topological_order_puts_children_first():
    x = Value(1.5)
    a = x * 2
    b = x.tanh()
    y = a * b
    order = y.topo_order()
    pos = {id(n): i for i, n in enumerate(order)}
    assert pos[id(x)] < pos[id(a)] < pos[id(y)]
    assert pos[id(b)] < pos[id(y)]
    assert len(order) == len(set(map(id, order)))


def test_deep_chain_does_not_recurse():
    v = Value(1.0)
    for _ in range(5000):
        v = v * 1.0
    v.backward()  # a recursive implementation raises RecursionError here
    assert v.topo_order()[0].grad == pytest.approx(1.0)


def test_gradcheck_on_thirty_random_graphs():
    import random

    random.seed(31)
    builders = [
        lambda a, b: a * b + a.tanh(),
        lambda a, b: (a + b) ** 3,
        lambda a, b: a / b - b.exp(),
        lambda a, b: (a * b).sigmoid(),
        lambda a, b: a.relu() * b + b / a,
        lambda a, b: ((a * a + b).log()) * (a - b),
    ]
    worst = 0.0
    for fn in builders:
        for _ in range(5):
            a, b = Value(random.uniform(0.4, 2.0)), Value(random.uniform(0.4, 2.0))
            res = gradcheck_values(fn, [a, b])
            worst = max(worst, res.max_rel_error)
            assert res.ok, str(res)
    assert worst < 1e-6
