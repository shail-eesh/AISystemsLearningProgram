# P0.2 · NumPy as your new LINQ

**Phase:** Python & Tensor Ramp (P0) · **Generation day:** Day 1 · **Video episodes:** 1
· **Status:** ✅ code · ✅ tests · ✅ bench

> [← Back to course home](../../../index.html) · [Master plan](../../../MASTER_PLAN.md) · [Progress ledger](../../../EXECUTION/LEDGER.md)

## What you build

A **fully vectorised technical-indicator suite** — SMA, EMA, Wilder's RMA, RSI, Bollinger bands,
VWAP (session and rolling), MACD, True Range and ATR — with **no Python-level loop touching a
bar**, plus the multi-symbol feature table AlphaDesk reads.

The suite is the vehicle; the subject is array thinking. LINQ already taught you to describe
*what* you want rather than how to iterate. NumPy asks for the same description and then
actually removes the loop, which is a stronger promise and a sharper set of edges.

## Why it matters beyond indicators

`einsum("bhqd,bhkd->bhqk")` is scaled dot-product attention. If that string is opaque now, T4 in
Phase 2 will read as magic. Step 5 builds up to exactly that line on toy shapes.

## How to run

```bash
# from the repo root
python3 -m pytest phases/p0/p0-2-numpy-as-linq -q      # 116 tests
python3 phases/p0/p0-2-numpy-as-linq/bench/indicator_parity.py

cd phases/p0/p0-2-numpy-as-linq
python3 steps/step1_arrays_vs_lists.py
python3 steps/step2_broadcasting.py
python3 steps/step3_vectorise_the_moving_average.py
python3 steps/step4_recurrences_and_ema.py
python3 steps/step5_einsum_and_linq.py
```

## The step ladder

1. **`step1_arrays_vs_lists.py`** — why a `list[float]` is 4× the memory and ~100× the time of a
   `float64[]`; dtypes and silent integer wrap-around; views vs copies; NaN semantics.
2. **`step2_broadcasting.py`** — the right-to-left shape rule, `keepdims`, `[:, None]`, masking
   with `np.where`, and the memory trap where a `(10000,1) + (1,10000)` allocates 800 MB.
3. **`step3_vectorise_the_moving_average.py`** — one SMA, three implementations. The `cumsum`
   trick is the fastest *and* loses 1.3e-2 of accuracy on a 2 M-bar series at index level 1e9;
   `sliding_window_view` is the one that ships.
4. **`step4_recurrences_and_ema.py`** — the recurrence that will not vectorise. The algebraic
   closed form is exact to 1e-14 for 4,000 bars and then becomes NaN *all at once*;
   `scipy.signal.lfilter` is the honest answer. Also: Wilder's `1/n` is not an EMA's `2/(n+1)`.
5. **`step5_einsum_and_linq.py`** — einsum subscripts, portfolio variance and risk contributions,
   the batched attention shape, and a 15-line LINQ → NumPy phrasebook.

## Verification benchmark ("done when")

`bench/indicator_parity.py` runs **70 parity checks** — 14 indicators × 5 symbols — against
`references.py`, an independently written pandas implementation (`rolling`, `ewm`).

| claim | result |
|---|---|
| NaN warm-up masks identical | 70/70 |
| worst relative error | **4.7e-14** (tolerance 1e-6) |
| 20-day SMA, 200k bars: Python loop → vectorised | 457 ms → **6.4 ms (72×)** |

Committed output: [`bench/results.json`](bench/results.json), re-run inside pytest by
`tests/test_bench_parity.py`.

> **On TA-Lib.** The master plan names TA-Lib as the reference. It is a C library that does not
> build in this sandbox, so pandas stands in — a genuinely independent implementation by other
> authors, which is what a reference is for. On your own machine, `pip install TA-Lib` and point
> `references.py` at it; the definitions here (population std for Bollinger, Wilder seeding for
> RSI/ATR) were chosen to match its conventions.

## Tests worth reading

* `test_no_lookahead` — truncate the series and every earlier value must be **bit-identical**.
  This is the one that catches the bug that quietly inflates every backtest you will ever write.
* `test_naive_whole_frame_computation_would_be_wrong` — proof that the per-symbol `groupby` is
  load-bearing: without it, a 20-day average silently blends the end of one issuer's history
  into the start of the next.
* `test_rsi_flat_series_has_no_division_error` — zero average loss pins RSI at 100, by
  convention, not by `ZeroDivisionError`.

## AlphaDesk hook

`data.technical_features` — the 13-column indicator table over the price history, computed per
symbol with no lookahead. T48 (feature store, Phase 3) makes it point-in-time correct; T42
(two-tower recsys) and the research surface read from it.

## Layout

- `src/p0_2_indicators/` — `core.py` (the vectorised implementations), `references.py` (the
  pandas specification), `frames.py` (the multi-symbol table), `alphadesk_hook.py`
- `steps/` · `tests/` · `bench/` · `NOTES.md`

## Videos

Episode script: [`video/topics/p0.2/script.md`](../../../video/topics/p0.2/script.md).

---
*AlphaDesk is a fictional educational simulation — no real orders, money, brokerage systems, or market-data redistribution.*
