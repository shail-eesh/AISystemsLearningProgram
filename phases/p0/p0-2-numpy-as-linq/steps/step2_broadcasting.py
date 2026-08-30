#!/usr/bin/env python3
"""Step 2 — broadcasting: the loop you do not write.

Run:  python3 steps/step2_broadcasting.py

Broadcasting is the rule that lets arrays of different shapes combine without
copying. Compare shapes right-to-left; two dimensions are compatible when they
are equal or one of them is 1; a missing leading dimension counts as 1.

    (5, 260)  prices per symbol
    (5,   1)  a weight per symbol
    -------
    (5, 260)  weighted prices — no tiling, no allocation

There is no C# analogue. The closest habit — writing the nested loop and
trusting the JIT — is exactly the habit to unlearn: the loop is not slow
because it is a loop, it is slow because each iteration re-enters the
interpreter.
"""

import _bootstrap  # noqa: F401
import numpy as np

from common.data import load_ohlcv

SYMBOLS = ["ALPHAINFRA", "BHARATCHEM", "COASTBANK", "DECCANMOT", "EASTPOWER"]


def price_matrix() -> np.ndarray:
    """(n_symbols, n_days) closes — the shape every portfolio calc wants."""
    df = load_ohlcv(SYMBOLS)
    wide = df.pivot(index="date", columns="symbol", values="close")
    return wide[SYMBOLS].to_numpy(dtype=np.float64).T


def demo_shape_rules() -> None:
    prices = price_matrix()
    weights = np.array([0.30, 0.25, 0.20, 0.15, 0.10])
    print(f"  prices {prices.shape} * weights {weights.shape} -> ", end="")
    try:
        _ = prices * weights
    except ValueError as exc:
        print(f"ValueError\n    {exc}")
    contributions = prices * weights[:, None]     # (5,1) broadcasts across days
    print(f"  prices {prices.shape} * weights[:, None] {weights[:, None].shape} "
          f"-> {contributions.shape}")
    nav = contributions.sum(axis=0)
    print(f"  portfolio NAV per day: {nav[:3].round(2)} ... (len {nav.size})")


def demo_normalisation() -> None:
    prices = price_matrix()
    rebased = prices / prices[:, [0]] * 100.0        # each row to 100 at t=0
    print(f"  rebased to 100: first column = {rebased[:, 0]}")
    print(f"  final levels    = {rebased[:, -1].round(1)}")

    returns = np.diff(np.log(prices), axis=1)
    z = (returns - returns.mean(axis=1, keepdims=True)) / returns.std(axis=1, keepdims=True)
    print(f"  z-scored returns: mean~{z.mean():.1e} std~{z.std():.3f}")
    print("  `keepdims=True` is what keeps the (5,1) shape broadcastable")


def demo_outer_without_loops() -> None:
    last = price_matrix()[:, -1]
    ratio = last[:, None] / last[None, :]            # (5,1) / (1,5) -> (5,5)
    print("  pairwise price ratio matrix (5x5) built with two Nones:")
    print(np.array2string(ratio, precision=2, suppress_small=True, prefix="    "))


def demo_masking_is_where() -> None:
    df = load_ohlcv(["ALPHAINFRA"])
    close = df["close"].to_numpy(dtype=np.float64)
    prev = df["prev_close"].to_numpy(dtype=np.float64)
    ret = close / prev - 1.0

    # C#: returns.Where(r => Math.Abs(r) > 0.03).Count()
    big = np.abs(ret) > 0.03
    print(f"  days moving >3%: {big.sum()} of {ret.size}")
    print(f"  their mean move: {ret[big].mean():+.4f}")
    signal = np.where(ret > 0.03, 1, np.where(ret < -0.03, -1, 0))
    print(f"  np.where as a ternary: signal counts {np.bincount(signal + 1)}")


def demo_broadcast_memory_trap() -> None:
    a = np.ones((10_000, 1))
    b = np.ones((1, 10_000))
    print(f"  a {a.shape} ({a.nbytes / 1e3:.0f} KB) + b {b.shape} ({b.nbytes / 1e3:.0f} KB)")
    print(f"  -> result would be {(a + b).nbytes / 1e6:.0f} MB. Broadcasting does not")
    print("     allocate the inputs, but it does allocate the OUTPUT. Watch the shapes.")


if __name__ == "__main__":
    print("shape rules:")
    demo_shape_rules()
    print("normalisation:")
    demo_normalisation()
    print("outer products:")
    demo_outer_without_loops()
    print("masking:")
    demo_masking_is_where()
    print("the trap:")
    demo_broadcast_memory_trap()
    print("\nstep 2 OK")
