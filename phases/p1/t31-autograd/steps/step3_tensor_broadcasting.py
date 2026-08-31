#!/usr/bin/env python3
"""Step 3 — broadcasting, the part that actually costs you a day.

Run:  python3 steps/step3_tensor_broadcasting.py

Forward, `(32, 8) + (8,)` is free. Backward it is a *sum*: the bias was used
by all 32 rows, so its gradient is the sum of 32 upstream gradients. The rule:

    forward broadcast  ==  backward sum-and-reshape

This step shows the shape bookkeeping explicitly, then gradchecks the ops that
depend on it.
"""

import _bootstrap  # noqa: F401
import numpy as np
from t31_autograd import Tensor, gradcheck_tensors, unbroadcast


def shapes_are_the_lesson() -> None:
    grad = np.ones((32, 8))
    for target in [(32, 8), (1, 8), (8,), (32, 1), ()]:
        out = unbroadcast(grad, target)
        print(f"  grad(32,8) -> {str(target):<8} = shape {out.shape}, sum {out.sum():.0f}")
    # Every reduction preserves the total: gradient is conserved, only regrouped.


def the_bias_case() -> None:
    rng = np.random.default_rng(0)
    x = Tensor(rng.standard_normal((32, 8)), requires_grad=False)
    W = Tensor(rng.standard_normal((8, 4)))
    b = Tensor(np.zeros((1, 4)))
    loss = ((x @ W + b).tanh() ** 2).sum()
    loss.backward()
    print(f"  W.grad {W.grad.shape} (matches W {W.shape})")
    print(f"  b.grad {b.grad.shape} (matches b {b.shape}) -- 32 rows summed into 1")


def gradcheck_the_hard_ops() -> None:
    rng = np.random.default_rng(3)
    cases = {
        "matmul + bias broadcast": (
            lambda A, B, c: ((A @ B + c).tanh()).sum(),
            [Tensor(rng.standard_normal((5, 3))), Tensor(rng.standard_normal((3, 4))), Tensor(rng.standard_normal((1, 4)))],
        ),
        "column broadcast (n,1)": (
            lambda A, c: ((A * c) ** 2).sum(),
            [Tensor(rng.standard_normal((4, 3))), Tensor(rng.standard_normal((4, 1)))],
        ),
        "rank promotion (3,) -> (4,3)": (
            lambda A, c: ((A + c).tanh()).sum(),
            [Tensor(rng.standard_normal((4, 3))), Tensor(rng.standard_normal((3,)))],
        ),
        "sum along an axis, then reuse": (
            lambda A: ((A - A.sum(axis=1, keepdims=True)) ** 2).sum(),
            [Tensor(rng.standard_normal((4, 3)))],
        ),
        "max along an axis": (
            lambda A: (A.max(axis=1) ** 3).sum(),
            [Tensor(rng.standard_normal((4, 3)))],
        ),
        "fancy indexing (repeats)": (
            lambda A: (A[np.array([0, 0, 2, 1])] ** 2).sum(),
            [Tensor(rng.standard_normal((3, 2)))],
        ),
    }
    for name, (fn, inputs) in cases.items():
        res = gradcheck_tensors(fn, inputs)
        print(f"  {name:<30} {res}")
        assert res.ok, name


if __name__ == "__main__":
    print("== unbroadcast, shape by shape ==")
    shapes_are_the_lesson()
    print("\n== the bias case ==")
    the_bias_case()
    print("\n== gradcheck ==")
    gradcheck_the_hard_ops()
