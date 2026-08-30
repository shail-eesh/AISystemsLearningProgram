# P0.2 · NumPy as your new LINQ — Notes

*Intuition and gotchas. This file is the source the video script is written from.*

---

## The one-sentence version

LINQ let you *describe* a computation instead of writing its loop; NumPy lets you describe it and
then genuinely deletes the loop — at the cost of making memory layout, dtype and floating-point
error your problem.

## 1. Why the Python loop is slow (it is not the loop)

A `List<double>` in .NET is a contiguous `double[]`. A Python `list` is an array of **pointers to
heap-allocated boxed floats**: 8 bytes of payload wrapped in ~32 bytes of object header, scattered
across memory. Every `a + b` is a dynamic type lookup, a virtual dispatch, and an allocation for
the result.

`np.ndarray` is what you assumed the list was. The 70–100× is not clever optimisation; it is the
interpreter getting out of the way, plus contiguous memory that the CPU can prefetch and
vectorise. Two consequences worth internalising:

* Speeding up a Python loop by *micro-optimising the loop body* is nearly pointless. You have to
  remove the interpreter from the inner loop entirely.
* A NumPy operation on a 10-element array is *slower* than the equivalent Python — the per-call
  overhead dominates. Vectorisation is a scale play.

## 2. dtype is a contract, and integers wrap

`np.array([1, 2, 3])` is `int64`, not float. `/` promotes to float, but `//`, `+=` and in-place
ops do not — and `uint8(250) + uint8(10)` is `4`. NumPy 2 warns on scalar overflow; older versions
were silent, and array-level overflow still is.

Rule: **state the dtype at every boundary.** Every function in `core.py` starts with
`np.asarray(x, dtype=np.float64)`. It costs nothing when the input already matches and it removes
an entire category of bug.

Related: pandas 3 hands back **read-only** arrays from `.to_numpy()` (copy-on-write). If you need
to mutate, `.copy()` explicitly rather than fighting the error.

## 3. Views vs copies — the aliasing rules you already half-know

Basic slicing (`x[10:20]`) returns a **view**: no data is copied, and writing through it changes
the original. Fancy indexing (`x[[10,11,12]]`) and boolean masks return **copies**. `arr.base is
None` tells you which you have.

This is `Span<T>` versus `ToArray()`, except the syntax gives you no hint about which one you
got. When a function must not mutate its input, copy deliberately at the top.

## 4. Broadcasting is the loop you do not write

Compare shapes right-to-left; dimensions match if equal or one is 1; missing leading dimensions
count as 1.

```
prices  (5, 260)
weights (5,   1)      # weights[:, None]
------- ---------
result  (5, 260)
```

`keepdims=True` on a reduction is what keeps the result broadcastable — `x.mean(axis=1)` gives
`(5,)`, which does *not* line up with `(5, 260)`, while `keepdims=True` gives `(5, 1)`, which
does. Half of all "operands could not be broadcast" errors are a missing `keepdims`.

The trap: broadcasting does not allocate the *inputs*, but it does allocate the **output**. A
`(10_000, 1) + (1, 10_000)` is two 80 KB arrays producing 800 MB. Read the output shape before
you run it.

## 5. Three ways to roll a window, and only one is right

| approach | cost | accuracy | verdict |
|---|---|---|---|
| nested Python loop | O(n·w), interpreted | exact | the thing you are replacing |
| `cumsum` differencing | O(n) | **catastrophic cancellation** | tempting, dangerous |
| `sliding_window_view` + reduce | O(n·w) in C | exact | what ships |

The cumsum trick subtracts two large, nearly-equal running totals. On 2 M bars at an index level
of 1e9 the running total reaches 2e15; float64 carries ~15–16 significant digits, so differencing
throws away exactly the low-order bits you were trying to keep. Measured error in step 3:
**1.3e-2** — rupees, on a price. On a year of two-digit prices it is invisible. That is what makes
it dangerous.

`sliding_window_view` gives an `(n-w+1, w)` **view** built from stride tricks: no copy, no
allocation, and `.mean(axis=-1)` runs the reduction in C. It is 72× the Python loop and exact.

## 6. Some things are scans, not reductions

An SMA is a reduction over *independent* windows. An EMA is a **scan**:

```
y[t] = a·x[t] + (1-a)·y[t-1]
```

Every output needs the previous output. There is a closed form —
`y[t] = (1-a)^t · (x[0] + Σ a·x[k]/(1-a)^k)` — and it is the "vectorised EMA" you will find
online. It is also a landmine: `(1-a)^-k` grows exponentially and passes float64's 1.8e308
around bar **4,200** for a 12-day span. Up to that point the answer is exact to 1e-14; one bar
later the entire output is NaN. A test on one year of data will never see it.

`scipy.signal.lfilter` implements this one-pole IIR filter directly in C — same sequential
dependency, same algorithm, 30× faster than the Python loop, and numerically stable. **The skill
is recognising the shape of the problem** (reduction? scan? convolution?) and reaching for the
primitive that already implements it. Phase 1 rebuilds this instinct for autograd; Phase 6 uses
the same distinction for KV-cache paging.

`lfilter`'s `zi` argument is the filter's initial delay state, not the initial output; for this
filter `zi = (1-a)·seed`, which is what makes `y[0]` come out as `a·x[0] + (1-a)·seed` exactly.

## 7. Wilder's smoothing is not an EMA

Wilder's 14-period average uses `alpha = 1/14 = 0.0714`. A 14-span EMA uses `alpha = 2/15 =
0.1333` — nearly double the responsiveness. `EMA(2n-1)` is the equivalent span, so Wilder(14) ≈
EMA(27).

Essentially every "my RSI does not match TradingView / TA-Lib / the other library" report is one
of three things:

1. `alpha = 1/n` vs `2/(n+1)`;
2. the **seed** — simple mean of the first `n` values (Wilder's own convention, used here) vs
   starting the recursion at `x[0]`;
3. `ddof` on the Bollinger standard deviation — population (0) is the original definition,
   pandas' default is sample (1).

None of these is *wrong*; they are conventions. The lesson: pin the convention explicitly in
code, and write the reference implementation before the fast one.

## 8. einsum, and why it matters for Phase 2

Name every axis; a repeated name is contracted (summed over); a name missing from the output is
summed away.

```
"ij,jk->ik"      matmul
"ij->i"          row sums
"ii->i"          diagonal
"ij,ij->i"       row-wise dot product
"i,ij,j->"       quadratic form  wᵀΣw   (portfolio variance)
"bhqd,bhkd->bhqk"  batched attention scores
```

That last line is T4. `b` batch, `h` head, `q` query position, `k` key position, `d` depth: keep
batch and head, contract depth, pair every query with every key. Reading it fluently is the
difference between following the transformer topic and transcribing it.

## 9. The LINQ phrasebook — and where the analogy ends

`Select` → arithmetic on the array · `Where` → boolean mask · `Sum/Average/Max` → `.sum()/.mean()/.max()`
· `Count(pred)` → `(pred).sum()` · `OrderBy` → `np.sort` / `np.argsort` · `Take/Skip` → slicing
· `Zip().Sum()` → `x @ y` · `Aggregate` → a reduction, or `np.cumsum` for a running one ·
`Distinct` → `np.unique` · `SelectMany` → `ravel`/`concatenate` · `Reverse` → `x[::-1]` (a view).

**Where it ends:** LINQ is lazy and streams; NumPy is eager and materialises. Ten chained
operations over a million-row array allocate ten million-row arrays. That is the price of the
100×, and it is why in-place operators (`x *= 2`) and the `out=` parameter exist. When memory
matters, chain fewer, bigger steps.

## 10. The finance gotchas, which are not NumPy gotchas

* **Lookahead.** An indicator at time *t* must depend only on data up to *t*. The test that
  catches violations is trivial and almost nobody writes it: truncate the series, recompute, and
  demand the earlier values be bit-identical. Centred rolling windows, `bfill`, and a
  `groupby().transform('mean')` over the whole history all fail it silently.
* **Symbol boundaries.** A 20-day average over a concatenated multi-symbol frame blends one
  issuer's last 19 bars into the next issuer's first. Group first, always.
* **Warm-up.** The first `w-1` values are NaN because the indicator is *undefined*, not because
  data is missing. Filling them with zeros puts a fictitious signal at the start of every series.

## Gotchas checklist

- [ ] `np.asarray(x, dtype=np.float64)` at every boundary.
- [ ] `keepdims=True` when a reduction result must broadcast back.
- [ ] Check the *output* shape before running a broadcast.
- [ ] `cumsum` differencing loses precision at scale — prefer `sliding_window_view`.
- [ ] A scan is not a reduction; use `lfilter` (or accept the loop).
- [ ] Wilder ≠ EMA; pin `alpha`, the seed and `ddof` explicitly.
- [ ] NaN poisons reductions; `np.isnan`, never `== np.nan`.
- [ ] Slices are views, fancy indexing copies.
- [ ] Test for lookahead. Group by symbol.

## What breaks if we skip this

P0.3 and all of Phase 1 assume you read shapes fluently. An autograd bug in T31 usually presents
as a shape mismatch in a backward pass, and a transformer in T4 is 90% shape bookkeeping. If
`keepdims`, broadcasting and `einsum` are still unfamiliar, every later topic will look like a
framework problem when it is an array problem.
