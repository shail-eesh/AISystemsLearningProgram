#!/usr/bin/env python3
"""Step 2 — why the *order* of the backward pass is the whole algorithm.

Run:  python3 steps/step2_topological_backward.py

A node may only push gradient into its children once every one of its own
parents has pushed into it. Get that wrong and you do not crash — you get a
gradient that is partly right, which is the worst kind.

This step builds a diamond graph (one node consumed twice), runs the correct
reverse-topological pass, then deliberately runs a *wrong* order to show the
size of the error it produces.
"""

import _bootstrap  # noqa: F401
from t31_autograd import Value


def diamond():
    x = Value(1.5, label="x")
    a = x * 2
    a.label = "a"
    b = x.tanh()
    b.label = "b"
    y = a * b
    y.label = "y"
    return x, a, b, y


def correct_order() -> float:
    x, a, b, y = diamond()
    y.backward()
    # d/dx [2x * tanh(x)] = 2*tanh(x) + 2x*(1 - tanh(x)^2)
    import math

    t = math.tanh(1.5)
    analytic = 2 * t + 2 * 1.5 * (1 - t * t)
    print(f"  reverse-topological: x.grad = {x.grad:.6f}   analytic = {analytic:.6f}")
    assert abs(x.grad - analytic) < 1e-12
    return x.grad


def wrong_order() -> None:
    """Insertion order instead of topological order."""
    x, a, b, y = diamond()
    for node in (x, a, b, y):
        node.grad = 0.0
    y.grad = 1.0
    for node in (x, a, b, y):        # forward order -- exactly backwards
        node._backward()
    print(f"  insertion order:     x.grad = {x.grad:.6f}   <- silently wrong")


def deep_chain_does_not_blow_the_stack() -> None:
    """20k-deep graph: proof the topo sort is iterative, not recursive."""
    v = Value(1.0)
    for _ in range(20_000):
        v = v * 1.0001
    v.backward()
    print(f"  20,000-node chain backward ok (grad {v.topo_order()[0].grad:.6f})")


if __name__ == "__main__":
    print("== the diamond ==")
    correct_order()
    wrong_order()
    print("\n== depth ==")
    deep_chain_does_not_blow_the_stack()
