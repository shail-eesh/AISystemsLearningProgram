"""The only thing standing between you and a plausible, wrong engine.

An autodiff bug does not raise. It returns a number of the right shape and
roughly the right size, the loss still goes down a bit, and you lose a week.
The defence is central finite differences:

    df/dx  ~=  (f(x + h) - f(x - h)) / (2h)     error O(h^2)

Central differences rather than forward (`(f(x+h)-f(x))/h`, error O(h)) because
the second-order cancellation buys ~7 digits instead of ~4 at float64. `h` is
scaled to the magnitude of the parameter so it survives both x=1e-6 and x=1e6.

The comparison metric is the standard *relative* one from CS231n:

    |a - b| / max(eps, |a| + |b|)

which is scale-free and does not blow up when both gradients are near zero.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .engine import Value
from .tensor import Tensor


@dataclass(frozen=True)
class GradCheckResult:
    max_rel_error: float
    worst_index: tuple
    n_checked: int
    tolerance: float

    @property
    def ok(self) -> bool:
        return self.max_rel_error <= self.tolerance

    def __str__(self) -> str:
        verdict = "PASS" if self.ok else "FAIL"
        return (
            f"{verdict}: {self.n_checked} entries, worst rel error "
            f"{self.max_rel_error:.3e} at {self.worst_index} (tol {self.tolerance:.0e})"
        )


def rel_error(a: float, b: float, eps: float = 1e-12) -> float:
    return abs(a - b) / max(eps, abs(a) + abs(b))


def gradcheck_tensors(
    fn: Callable[..., Tensor],
    inputs: Sequence[Tensor],
    *,
    h: float | None = None,
    tolerance: float = 1e-6,
    max_entries: int = 200,
    rng: np.random.Generator | None = None,
) -> GradCheckResult:
    """Compare analytic gradients from `backward()` against finite differences.

    `fn(*inputs)` must return a scalar Tensor. For tensors larger than
    `max_entries`, a random subset of coordinates is probed — each finite
    difference costs two full forward passes, so checking a 512x512 weight
    exhaustively is half a million forwards.
    """
    rng = rng or np.random.default_rng(0)

    out = fn(*inputs)
    out.backward()
    analytic = [t.grad.copy() for t in inputs]

    worst, worst_idx, checked = 0.0, (), 0
    for t_i, t in enumerate(inputs):
        flat_n = t.data.size
        idxs = range(flat_n) if flat_n <= max_entries else rng.choice(flat_n, max_entries, replace=False)
        for flat in idxs:
            idx = np.unravel_index(int(flat), t.data.shape)
            original = t.data[idx]
            step = h if h is not None else max(1e-6, 1e-6 * abs(original))

            t.data[idx] = original + step
            plus = float(np.asarray(fn(*inputs).data).reshape(()))
            t.data[idx] = original - step
            minus = float(np.asarray(fn(*inputs).data).reshape(()))
            t.data[idx] = original

            numeric = (plus - minus) / (2 * step)
            err = rel_error(float(analytic[t_i][idx]), numeric)
            checked += 1
            if err > worst:
                worst, worst_idx = err, (t_i, *idx)

    return GradCheckResult(worst, worst_idx, checked, tolerance)


def gradcheck_values(
    fn: Callable[..., Value],
    inputs: Sequence[Value],
    *,
    h: float = 1e-6,
    tolerance: float = 1e-6,
) -> GradCheckResult:
    """The scalar-engine equivalent, used by step 1."""
    out = fn(*inputs)
    out.backward()
    analytic = [v.grad for v in inputs]

    worst, worst_idx = 0.0, ()
    for i, v in enumerate(inputs):
        original = v.data
        v.data = original + h
        plus = fn(*inputs).data
        v.data = original - h
        minus = fn(*inputs).data
        v.data = original
        numeric = (plus - minus) / (2 * h)
        err = rel_error(analytic[i], numeric)
        if err > worst:
            worst, worst_idx = err, (i,)
    return GradCheckResult(worst, worst_idx, len(inputs), tolerance)
