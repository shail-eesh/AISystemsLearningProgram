#!/usr/bin/env python3
"""Step 5 — fuse log-softmax and cross-entropy, and watch the gradient simplify.

Run:  python3 steps/step5_logsoftmax_and_cross_entropy.py

`log(softmax(x))` is the wrong way to compute log-softmax, for the reason step 1
already showed: any probability that underflowed to zero becomes -inf. Subtract
the log-sum-exp instead and full relative precision survives all the way down.

Then the payoff: -log(softmax(x)_t) = logsumexp(x) - x_t, whose gradient is
just `softmax(x) - onehot(t)`. The fused loss has a *simpler* derivative than
either half. That is not a coincidence — it is the log and the exp cancelling.
"""

import _bootstrap  # noqa: F401
import numpy as np
from t45a_softmax.softmax import (
    cross_entropy,
    cross_entropy_grad,
    log_softmax,
    stable_softmax,
)


def naive_log_softmax_loses_everything() -> None:
    x = np.array([[0.0, -50.0, -400.0, -800.0]])
    with np.errstate(divide="ignore"):
        naive = np.log(stable_softmax(x))
    fused = log_softmax(x)
    print("  x            =", x[0])
    print("  log(softmax) =", naive[0])
    print("  log_softmax  =", fused[0])
    print("\n  The -800 entry is the point: its probability underflowed to exactly")
    print("  0, so the log is -inf and any loss built on it is inf. The fused")
    print("  version never forms the probability at all.")


def cross_entropy_is_two_lookups() -> None:
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((5, 7)) * 3
    targets = rng.integers(0, 7, 5)
    fused = cross_entropy(logits, targets)
    manual = -np.log(stable_softmax(logits))[np.arange(5), targets].mean()
    print(f"\n  fused cross-entropy : {fused:.12f}")
    print(f"  -log(softmax)[t]    : {manual:.12f}")
    print("  Same value here, because these logits are tame. Push them to +/-800")
    print("  and the second one becomes inf while the first does not.")

    hard = np.array([[0.0, -900.0]])
    print(f"\n  hard case, target=1: fused = {cross_entropy(hard, np.array([1])):.4f}")
    with np.errstate(divide="ignore"):
        print(f"                        naive = {-np.log(stable_softmax(hard))[0, 1]}")


def the_gradient_is_embarrassingly_simple() -> None:
    rng = np.random.default_rng(1)
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

    print("\n  analytic grad = (softmax(x) - onehot(t)) / batch")
    print(f"  max |analytic - finite difference| = {np.abs(analytic - numeric).max():.3e}")
    print("\n  Every framework fuses these two ops for exactly this reason: the")
    print("  fused backward pass is one subtraction, and it needs no exp at all.")


if __name__ == "__main__":
    print("== log-softmax ==")
    naive_log_softmax_loses_everything()
    cross_entropy_is_two_lookups()
    the_gradient_is_embarrassingly_simple()
