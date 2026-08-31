#!/usr/bin/env python3
"""Step 4 — one pass. The trick Flash Attention is built on.

Run:  python3 steps/step4_online_softmax.py

Two passes still means you must see the whole row before you can start summing,
because the max is needed first. The online algorithm removes that: carry the
running max `m` and the running denominator `d` *together*, and whenever a
bigger value shows up, rescale what you already have.

    seeing a new block B:
        m_new = max(m, max(B))
        d_new = d * exp(m - m_new) + sum(exp(B - m_new))

That is it. One multiply repairs the whole accumulated denominator, because
every term in `d` was scaled by exp(-m) and now needs exp(-m_new) instead.

Reference: Milakov & Gimelshein, "Online normalizer calculation for softmax"
(2018) — the paper Flash Attention cites for this step.
"""

import _bootstrap  # noqa: F401
import numpy as np
from t45a_softmax.softmax import (
    SoftmaxState,
    online_normalizer,
    online_softmax,
    stable_softmax,
)


def walk_through_it_by_hand() -> None:
    x = np.array([3.0, 1.0, 9.0, 2.0])
    print(f"  x = {x}\n")
    state = SoftmaxState.empty((), dtype=np.float64)
    print("  step   value    m (running max)   d (running denominator)")
    for i, v in enumerate(x):
        prev_m, prev_d = float(state.m), float(state.d)
        state = state.update(np.array([v]))
        note = ""
        if np.isfinite(prev_m) and float(state.m) > prev_m:
            note = f"   <- max moved, d rescaled by exp({prev_m:.0f}-{float(state.m):.0f})={np.exp(prev_m - float(state.m)):.4f}"
        print(f"  {i:>4}   {v:6.1f}   {float(state.m):>13.4f}   {float(state.d):>20.6f}{note}")
        _ = prev_d
    direct = np.exp(x - x.max()).sum()
    print(f"\n  final d = {float(state.d):.10f}")
    print(f"  direct  = {direct:.10f}   (identical, and computed in one pass)")


def equivalence_on_random_rows() -> None:
    rng = np.random.default_rng(0)
    print("\n  online vs two-pass, over 200 random rows of 512:")
    worst = 0.0
    for _ in range(200):
        x = rng.standard_normal((1, 512)) * rng.uniform(1, 50)
        worst = max(worst, float(np.abs(online_softmax(x) - stable_softmax(x)).max()))
    print(f"    worst absolute difference: {worst:.3e}")
    print("    (not 'close enough' -- the same arithmetic in a different order)")


def chunk_size_does_not_change_the_answer() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal((3, 500)) * 20
    ref = stable_softmax(x)
    print("\n  chunk   max abs error vs two-pass")
    for chunk in (1, 7, 64, 256, 4096):
        got = online_softmax(x, chunk=chunk)
        print(f"  {chunk:>5}   {np.abs(got - ref).max():.3e}")
    print("\n  Chunk size is a *scheduling* choice, not a numerical one. That is")
    print("  what makes the algorithm safe to tile across a GPU's SRAM budget.")


def the_merge_is_associative() -> None:
    """Two partial states combine — this is the parallel/tiled version."""
    rng = np.random.default_rng(2)
    x = rng.standard_normal(1000) * 30
    left, right = x[:337], x[337:]

    def state_of(v):
        s = SoftmaxState.empty((), dtype=np.float64)
        return s.update(v)

    merged = state_of(left).merge(state_of(right))
    whole = state_of(x)
    print(f"\n  merge(partial(left), partial(right)).d = {float(merged.d):.12f}")
    print(f"  partial(whole).d                       = {float(whole.d):.12f}")
    print("\n  Associative merge => the row can be split across threads, blocks or")
    print("  tiles in any order. Flash Attention's inner loop is exactly this")
    print("  merge, carrying an output accumulator alongside (m, d).")


if __name__ == "__main__":
    print("== the algorithm, one value at a time ==")
    walk_through_it_by_hand()
    equivalence_on_random_rows()
    chunk_size_does_not_change_the_answer()
    the_merge_is_associative()
    m, d = online_normalizer(np.zeros((2, 8)))
    assert np.allclose(d, 8.0)
