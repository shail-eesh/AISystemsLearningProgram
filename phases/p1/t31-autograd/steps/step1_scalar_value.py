#!/usr/bin/env python3
"""Step 1 — a scalar that remembers where it came from.

Run:  python3 steps/step1_scalar_value.py

Build `Value` with `+`, `*` and `tanh`, then check every derivative against a
central finite difference. Nothing here is deep; the point is to see that
"backpropagation" is three lines of bookkeeping per operation, and that the
bookkeeping is *local* — `__mul__` knows nothing about the loss.
"""

import _bootstrap  # noqa: F401
from t31_autograd import Value, draw_ascii, gradcheck_values


def worked_example() -> None:
    """The one Karpathy draws: a two-input neuron with a tanh."""
    x1, x2 = Value(2.0, label="x1"), Value(0.0, label="x2")
    w1, w2 = Value(-3.0, label="w1"), Value(1.0, label="w2")
    b = Value(6.8813735870195432, label="b")
    n = x1 * w1 + x2 * w2 + b
    n.label = "n"
    out = n.tanh()
    out.label = "out"
    out.backward()

    print("  forward:  out =", f"{out.data:.4f}")
    print("  expected: x1.grad = w1 * (1 - tanh(n)^2) =", f"{-3.0 * (1 - out.data**2):.4f}")
    print("  got:      x1.grad =", f"{x1.grad:.4f}")
    print("\n  the graph, in reverse topological order:")
    print(draw_ascii(out))


def reuse_is_where_engines_break() -> None:
    """`y = x*x + x` uses x twice. Assign instead of accumulate and you get 2x, not 2x+1."""
    x = Value(3.0, label="x")
    y = x * x + x
    y.backward()
    print(f"  d/dx (x^2 + x) at x=3 -> {x.grad}  (analytic: {2 * 3 + 1})")
    assert x.grad == 7.0, "gradient accumulation is broken"


def gradcheck_a_pile_of_graphs() -> None:
    import random

    random.seed(7)
    ops = [
        ("a*b + a.tanh()", lambda a, b: a * b + a.tanh()),
        ("(a+b)**3", lambda a, b: (a + b) ** 3),
        ("a/b - b.exp()", lambda a, b: a / b - b.exp()),
        ("(a*b).sigmoid()", lambda a, b: (a * b).sigmoid()),
        ("a.relu()*b + b/a", lambda a, b: a.relu() * b + b / a),
    ]
    worst = 0.0
    for name, fn in ops:
        for _ in range(6):
            a = Value(random.uniform(0.3, 2.0))
            b = Value(random.uniform(0.3, 2.0))
            res = gradcheck_values(fn, [a, b])
            worst = max(worst, res.max_rel_error)
            assert res.ok, f"{name}: {res}"
        print(f"  {name:<22} ok")
    print(f"\n  30 graphs, worst relative error {worst:.2e}")


if __name__ == "__main__":
    print("== the worked example ==")
    worked_example()
    print("\n== accumulation, not assignment ==")
    reuse_is_where_engines_break()
    print("\n== gradcheck 30 random graphs ==")
    gradcheck_a_pile_of_graphs()
