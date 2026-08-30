#!/usr/bin/env python3
"""Step 5 — einsum, and the LINQ → NumPy phrasebook.

Run:  python3 steps/step5_einsum_and_linq.py

`np.einsum` writes a whole class of index gymnastics as a subscript string:
name each axis, repeat a name to contract over it, omit a name from the output
to sum it away. `"ij,jk->ik"` is a matmul; `"ij,ij->i"` is a row-wise dot
product; `"ii->i"` is a diagonal.

It matters far beyond portfolio maths: attention in Phase 2 is
`einsum("bhqd,bhkd->bhqk")` — batch, head, query, key, depth — and being able
to read that line is the difference between following T4 and copying it.

The second half is the phrasebook: every LINQ operator you reach for daily and
its NumPy counterpart.
"""

import _bootstrap  # noqa: F401
import numpy as np

from common.data import load_ohlcv

SYMBOLS = ["ALPHAINFRA", "BHARATCHEM", "COASTBANK", "DECCANMOT", "EASTPOWER"]


def returns_matrix() -> np.ndarray:
    """(n_symbols, n_days-1) log returns."""
    df = load_ohlcv(SYMBOLS)
    wide = df.pivot(index="date", columns="symbol", values="close")[SYMBOLS]
    return np.diff(np.log(wide.to_numpy(dtype=np.float64)), axis=0).T


def demo_einsum_basics() -> None:
    a = np.arange(6.0).reshape(2, 3)
    b = np.arange(12.0).reshape(3, 4)
    print(f"  matmul      einsum('ij,jk->ik') == a @ b  -> "
          f"{np.allclose(np.einsum('ij,jk->ik', a, b), a @ b)}")
    print(f"  row sums    einsum('ij->i')     == a.sum(1) -> "
          f"{np.allclose(np.einsum('ij->i', a), a.sum(axis=1))}")
    print(f"  transpose   einsum('ij->ji')    == a.T    -> "
          f"{np.allclose(np.einsum('ij->ji', a), a.T)}")
    c = np.arange(9.0).reshape(3, 3)
    print(f"  diagonal    einsum('ii->i')     == diag   -> "
          f"{np.allclose(np.einsum('ii->i', c), np.diag(c))}")
    x = np.arange(8.0).reshape(2, 4)
    print(f"  row-wise dot einsum('ij,ij->i') == (x*x).sum(1) -> "
          f"{np.allclose(np.einsum('ij,ij->i', x, x), (x * x).sum(axis=1))}")


def demo_portfolio_risk() -> None:
    r = returns_matrix()
    w = np.array([0.30, 0.25, 0.20, 0.15, 0.10])
    cov = np.cov(r)                                  # (5,5), sample covariance
    var_matmul = float(w @ cov @ w)
    var_einsum = float(np.einsum("i,ij,j->", w, cov, w))
    print(f"  portfolio variance: matmul {var_matmul:.6e}  einsum {var_einsum:.6e}")
    print(f"  agree: {np.isclose(var_matmul, var_einsum)}")
    ann_vol = np.sqrt(var_matmul * 252)
    print(f"  annualised vol: {ann_vol:.2%}  (5 fictional issuers, synthetic prices)")

    contrib = np.einsum("i,ij,j->i", w, cov, w) / var_matmul
    print("  risk contribution by symbol:")
    for s, c in zip(SYMBOLS, contrib, strict=True):
        print(f"    {s:12s} {c:6.1%}")
    assert np.isclose(contrib.sum(), 1.0)


def demo_batched_shapes() -> None:
    """The shape pattern Phase 2's attention uses, on toy numbers."""
    rng = np.random.default_rng(1729)
    batch, heads, seq, depth = 2, 3, 4, 5
    q = rng.normal(size=(batch, heads, seq, depth))
    k = rng.normal(size=(batch, heads, seq, depth))
    scores = np.einsum("bhqd,bhkd->bhqk", q, k) / np.sqrt(depth)
    print(f"  q{q.shape} x k{k.shape} -> scores{scores.shape}")
    print("  read it as: keep batch and head, contract depth, pair query with key.")
    print("  That single line is scaled dot-product attention's numerator (T4, Phase 2).")


PHRASEBOOK = [
    ("xs.Select(x => x * 2)",              "x * 2"),
    ("xs.Where(x => x > 0)",               "x[x > 0]   (or np.where for a ternary)"),
    ("xs.Sum() / Average() / Max()",       "x.sum() / x.mean() / x.max()"),
    ("xs.Count(x => x > 0)",               "(x > 0).sum()"),
    ("xs.Any(...) / All(...)",             "(cond).any() / (cond).all()"),
    ("xs.OrderBy(x => x)",                 "np.sort(x)   /  np.argsort(x) for the indices"),
    ("xs.Take(n) / Skip(n)",               "x[:n] / x[n:]"),
    ("xs.Zip(ys, (a,b) => a*b).Sum()",     "x @ y   (or np.einsum('i,i->', x, y))"),
    ("xs.Aggregate((a,b) => a+b)",         "x.sum()  — a running total is np.cumsum"),
    ("xs.GroupBy(k).Select(g => g.Sum())", "np.add.at / np.bincount / pandas groupby"),
    ("xs.Distinct()",                      "np.unique(x)"),
    ("xs.SelectMany(...)",                 "x.ravel() / np.concatenate"),
    ("xs.Reverse()",                       "x[::-1]   (a view, not a copy)"),
    ("Enumerable.Range(0, n)",             "np.arange(n)"),
    ("xs.Select((x,i) => ...)",            "operate on x and np.arange(len(x)) together"),
]


def demo_phrasebook() -> None:
    width = max(len(a) for a, _ in PHRASEBOOK)
    for csharp, numpy_ in PHRASEBOOK:
        print(f"  {csharp:<{width}}  ->  {numpy_}")
    print("\n  The one that is NOT a translation: LINQ is lazy and streams; NumPy is")
    print("  eager and materialises every intermediate. Chaining ten operations over a")
    print("  million-row array allocates ten million-row arrays. That is the trade you")
    print("  make for the 100x, and why in-place ops (`x *= 2`, `out=`) exist.")


if __name__ == "__main__":
    print("einsum basics:")
    demo_einsum_basics()
    print("portfolio risk:")
    demo_portfolio_risk()
    print("batched shapes (a preview of attention):")
    demo_batched_shapes()
    print("LINQ phrasebook:")
    demo_phrasebook()
    print("\nstep 5 OK")
