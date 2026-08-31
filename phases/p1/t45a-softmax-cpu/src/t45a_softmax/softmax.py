"""T45A · softmax, from the naive formula to the one-pass online algorithm.

Four implementations of one function, in the order you should meet them:

1. :func:`naive_softmax` — the textbook formula. Overflows at x > 709 in
   float64, x > 88 in float32, x > 11 in float16.
2. :func:`stable_softmax` — subtract the row max first. Mathematically the
   identical function; numerically the difference between working and not.
3. :func:`two_pass_softmax` — one pass to find max and sum, one to normalise.
   The version every framework shipped until ~2018.
4. :func:`online_softmax` — **one** pass. Track a running max and a running
   denominator together, rescaling the denominator whenever the max moves.

(4) is the whole reason this topic exists. It is the trick Flash Attention
rests on: once you can compute a softmax denominator while streaming the
inputs, you never have to materialise the n x n attention matrix.

Reference for the online formulation: Milakov & Gimelshein, *Online normalizer
calculation for softmax* (2018).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def naive_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """exp(x) / sum(exp(x)). Correct on paper, unusable on a computer.

    Kept — and tested — precisely so the failure is something you have seen
    rather than something you have been warned about.
    """
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def stable_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """The max-subtraction identity.

        softmax(x)_i = exp(x_i - c) / sum_j exp(x_j - c)   for any constant c

    Proof, in one line: multiply numerator and denominator by exp(-c). The
    function is *invariant* to a shift, so we are free to choose c = max(x),
    which makes the largest exponent exactly 0 and every other one negative.
    Overflow becomes impossible; the worst that can happen is that a tiny term
    underflows to zero, which is the right answer to within the dtype anyway.
    """
    c = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - c)
    return e / e.sum(axis=axis, keepdims=True)


def two_pass_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Fused: one pass computes max *and* the shifted sum, one pass normalises.

    Identical arithmetic to :func:`stable_softmax`; the difference is that the
    exponentials are computed once and reused rather than recomputed. Three
    reads of x become two.
    """
    x = np.moveaxis(x, axis, -1)
    c = x.max(axis=-1, keepdims=True)
    e = np.exp(x - c)
    out = e / e.sum(axis=-1, keepdims=True)
    return np.moveaxis(out, -1, axis)


# --------------------------------------------------------------------------
# The online algorithm
# --------------------------------------------------------------------------
@dataclass
class SoftmaxState:
    """The running (max, denominator) pair for one row.

    `d` is the sum of exp(x_j - m) over everything seen so far, where `m` is
    the largest value seen so far. When a bigger value arrives, `m` changes and
    every term already accumulated in `d` was scaled by the *old* max — so `d`
    is rescaled by exp(m_old - m_new). One multiply, and the invariant holds
    again.
    """

    m: np.ndarray  # running maximum
    d: np.ndarray  # running sum of exp(x - m)

    @classmethod
    def empty(cls, shape: tuple[int, ...], dtype=np.float64) -> SoftmaxState:
        return cls(np.full(shape, -np.inf, dtype=dtype), np.zeros(shape, dtype=dtype))

    def update(self, block: np.ndarray) -> SoftmaxState:
        """Absorb a block of new values (shape: (*self.m.shape, chunk))."""
        block_max = block.max(axis=-1)
        new_m = np.maximum(self.m, block_max)
        # exp(-inf - -inf) is nan, so guard the very first update.
        scale = np.where(np.isfinite(self.m), np.exp(self.m - new_m), 0.0)
        block_d = np.exp(block - new_m[..., None]).sum(axis=-1)
        return SoftmaxState(new_m, self.d * scale + block_d)

    def merge(self, other: SoftmaxState) -> SoftmaxState:
        """Combine two independently-computed partial states.

        This is the associative operation that makes the algorithm
        parallelisable — and it is exactly what Flash Attention's inner loop
        does when it combines the running output of one K/V tile with the next.
        """
        new_m = np.maximum(self.m, other.m)
        a = np.where(np.isfinite(self.m), np.exp(self.m - new_m), 0.0)
        b = np.where(np.isfinite(other.m), np.exp(other.m - new_m), 0.0)
        return SoftmaxState(new_m, self.d * a + other.d * b)


def online_softmax(x: np.ndarray, axis: int = -1, chunk: int = 64) -> np.ndarray:
    """One streaming pass for the normaliser, then one pass to write the output.

    "One pass" is about the *normaliser*: the maximum and the denominator are
    computed together while the data streams by, so nothing needs to be revisited
    to find the max before summing. That is what lets Flash Attention consume an
    attention row tile by tile and never store the whole row.
    """
    x = np.moveaxis(x, axis, -1)
    n = x.shape[-1]
    state = SoftmaxState.empty(x.shape[:-1], dtype=x.dtype if x.dtype != np.float16 else np.float32)
    for start in range(0, n, chunk):
        state = state.update(x[..., start : start + chunk].astype(state.m.dtype))
    out = np.exp(x - state.m[..., None].astype(x.dtype)) / state.d[..., None].astype(x.dtype)
    return np.moveaxis(out, -1, axis)


def online_normalizer(x: np.ndarray, axis: int = -1, chunk: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Just the (max, denominator) pair — what a fused kernel actually returns."""
    x = np.moveaxis(x, axis, -1)
    state = SoftmaxState.empty(x.shape[:-1], dtype=np.float64)
    for start in range(0, x.shape[-1], chunk):
        state = state.update(x[..., start : start + chunk].astype(np.float64))
    return state.m, state.d


# --------------------------------------------------------------------------
# log-softmax and the cross-entropy fusion
# --------------------------------------------------------------------------
def logsumexp(x: np.ndarray, axis: int = -1, keepdims: bool = False) -> np.ndarray:
    """log(sum(exp(x))), computed as c + log(sum(exp(x - c))) with c = max."""
    c = np.max(x, axis=axis, keepdims=True)
    finite = np.where(np.isfinite(c), c, 0.0)
    out = finite + np.log(np.exp(x - finite).sum(axis=axis, keepdims=True))
    return out if keepdims else np.squeeze(out, axis=axis)


def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """log(softmax(x)) without ever forming softmax(x).

    `np.log(softmax(x))` loses everything below the dtype's smallest normal:
    any probability that underflowed to 0 becomes -inf, and cross-entropy
    becomes inf. Subtracting the log-sum-exp keeps full relative precision all
    the way down to -745 in float64.
    """
    return x - logsumexp(x, axis=axis, keepdims=True)


def cross_entropy(logits: np.ndarray, targets: np.ndarray, *, reduction: str = "mean") -> np.ndarray:
    """Fused softmax + negative-log-likelihood over class indices.

    Fused because the two halves cancel: -log(softmax(x)_t) = logsumexp(x) - x_t,
    and computing it that way needs no exponentials of the target term at all.
    """
    targets = np.asarray(targets, dtype=int)
    lse = logsumexp(logits, axis=-1)
    picked = np.take_along_axis(logits, targets[..., None], axis=-1)[..., 0]
    losses = lse - picked
    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    return losses.mean()


def cross_entropy_grad(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """d(mean cross-entropy)/d(logits) = (softmax(x) - onehot(t)) / batch.

    Worth deriving once: the softmax and the log cancel so completely that the
    gradient of the fused loss is *simpler* than the gradient of either half.
    This is why every framework fuses them.
    """
    targets = np.asarray(targets, dtype=int)
    p = stable_softmax(logits, axis=-1)
    np.put_along_axis(p, targets[..., None], np.take_along_axis(p, targets[..., None], -1) - 1.0, -1)
    n = int(np.prod(logits.shape[:-1])) or 1
    return p / n
