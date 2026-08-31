# T45A · Softmax & online softmax — CPU

**Phase:** Foundations (P1) · **Generation day:** Day 2 · **Video episodes:** 2
· *(the fused GPU kernel is T45B, Phase 7)*

> [← Back to course home](../../../index.html) · [Master plan](../../../MASTER_PLAN.md) · [Progress ledger](../../../EXECUTION/LEDGER.md)

## What you build

One function, four implementations, in the order you should meet them:

1. **naive** — `exp(x) / sum(exp(x))`. Correct on paper. Dies at x > 709 in
   float64, x > 88 in float32, **x > 11 in float16**.
2. **stable** — subtract the row max. Softmax is invariant to a constant shift,
   so this changes nothing mathematically and everything numerically.
3. **two-pass fused** — find the max, then exponentiate-and-sum in one sweep.
4. **online (one-pass)** — carry a running max `m` and running denominator `d`
   together; when a larger value arrives, rescale `d` by `exp(m_old − m_new)`.

Plus `log_softmax`, a fused `cross_entropy`, and its gradient — which turns out
to be simply `softmax(x) − onehot(t)`, because the log and the exp cancel.

**(4) is why this topic exists.** Once the normaliser can be computed while the
data streams past, you never have to hold the whole row — which is exactly what
lets Flash Attention (T7) consume an attention matrix tile by tile without ever
materialising it. *Ref: Milakov & Gimelshein, "Online normalizer calculation for
softmax" (2018).*

## Results

On a 64×128 adversarial grid (logits in [−10⁴, +10⁴], plus rows that are
entirely 800 and entirely −10⁴):

| dtype | naive breaks on | online vs two-pass | online vs float64 ref | tolerance |
|:--|--:|--:|--:|--:|
| float64 | 64/64 rows | **0.0** | 0.0 | 1e-15 |
| float32 | 64/64 rows | **0.0** | 6.4e-09 | 1e-7 |
| float16 | 64/64 rows | **0.0** | 4.9e-05 | 1e-3 |

The zeros are the headline: **the online algorithm is bit-identical to the
two-pass version in every dtype.** It is not an approximation you trade accuracy
for — it is the same arithmetic, reassociated. Chunk size (1 … 512 … whole row)
changes the answer by exactly 0.0, which is what makes it safe to tile.

The honest counterweight, also measured: on a CPU with NumPy the online version
is **1.8× slower** (4.04 ms vs 2.29 ms on 256×4096 float32). It pays Python
overhead per chunk to save a pass it does not need here. Its value shows up only
in a fused kernel where the row does not fit in fast memory — T45B and T7.

## AlphaDesk hook

No product surface, by design. What is registered is
`foundation.softmax_numerics`: the routines that T4's attention block, T15's
loss, T25's logit processors and T45B's GPU kernel all depend on. Registering it
makes that dependency visible in `Registry.describe()` rather than implicit in
five files.

AlphaDesk is a fictional educational simulation — no real orders, money, or
venues anywhere in this repository.

## How to run

```bash
python3 phases/p1/t45a-softmax-cpu/steps/step1_watch_it_overflow.py
python3 phases/p1/t45a-softmax-cpu/steps/step2_max_subtraction.py
python3 phases/p1/t45a-softmax-cpu/steps/step3_two_pass_fused.py
python3 phases/p1/t45a-softmax-cpu/steps/step4_online_softmax.py
python3 phases/p1/t45a-softmax-cpu/steps/step5_logsoftmax_and_cross_entropy.py

python3 -m pytest phases/p1/t45a-softmax-cpu/tests -q
python3 phases/p1/t45a-softmax-cpu/bench/numerics.py
```

## Layout

- `src/t45a_softmax/softmax.py` — all four implementations, `SoftmaxState`, the losses
- `steps/` — the five checkpoints; step 4 walks the running `(m, d)` pair value by value
- `tests/` — 49 tests, including the float16 resolution limit pinned as expected behaviour
- `bench/numerics.py` + `results.json` — the adversarial grid
- `NOTES.md` — the derivations and the gotchas

## Videos

Episode scripts live in [`video/topics/t45a/`](../../../video/topics/t45a/).
Episode 2 (online softmax, derived on one whiteboard) is deliberately the most
patient in the series — it is the episode Phase 7 cashes in.
