# P0.3 · PyTorch tensors & the training loop

**Phase:** Python & Tensor Ramp (P0) · **Generation day:** Day 1 · **Video episodes:** 1
· **Status:** ✅ code · ✅ tests · ✅ bench

> [← Back to course home](../../../index.html) · [Master plan](../../../MASTER_PLAN.md) · [Progress ledger](../../../EXECUTION/LEDGER.md)

## What you build

The **training-loop skeleton** — the five lines every later topic reuses — and then the
experiment that teaches you not to trust it: a ~800-parameter MLP that memorises 64 rows
perfectly and predicts tomorrow's market direction at exactly chance.

Both halves matter. The loop is the mechanical skill (you must be able to write it from an empty
file). The second half is the judgement: **a falling loss is evidence that gradient descent
works, and nothing else.**

## The five lines

```python
for xb, yb in loader:
    optimiser.zero_grad()          # gradients ACCUMULATE — clear them first
    logits = model(xb)             # forward
    loss = criterion(logits, yb)   # a scalar
    loss.backward()                # reverse-mode autodiff fills .grad
    optimiser.step()               # p -= lr * p.grad  (+ the optimiser's state)
```

Everything else in `loop.py` — seeding, batching, eval mode, best-checkpoint restore — is
bookkeeping you can look up.

## How to run

```bash
# from the repo root
python3 -m pytest phases/p0/p0-3-pytorch-training-loop -q     # 27 tests, ~30s
python3 phases/p0/p0-3-pytorch-training-loop/bench/train_and_report.py   # ~8s

cd phases/p0/p0-3-pytorch-training-loop
python3 steps/step1_tensors_vs_ndarray.py
python3 steps/step2_autograd_as_a_user.py
python3 steps/step3_module_and_optimizer.py
python3 steps/step4_the_training_loop.py
python3 steps/step5_financial_ml_humility.py
```

PyTorch is an optional dependency: `pip install -e ".[torch]"`. The CPU wheel is enough for all
of Phase 0; the 4070 lane starts in Phase 7.

## The step ladder

1. **`step1_tensors_vs_ndarray.py`** — everything from P0.2 transfers; what a tensor adds
   (device, gradient tracking, in-place ops), the float32 default, and which conversions share
   memory.
2. **`step2_autograd_as_a_user.py`** — `.backward()` **adds** into `.grad` (demonstrated three
   times in a row), non-scalar backward, `no_grad` vs `detach`, the graph being freed, and why
   intermediates have no `.grad`.
3. **`step3_module_and_optimizer.py`** — `nn.Module` is a parameter registry (and an unregistered
   `self.x = torch.zeros(3)` is invisible); `optimiser.step()` for SGD is *literally*
   `p -= lr * p.grad`, proved to 7e-09; AdamW costs 2× the model in state; hand-written gradients
   matching autograd to 2.8e-17.
4. **`step4_the_training_loop.py`** — the loop written longhand, batch size vs update count, then
   the standard first diagnostic: **overfit a tiny batch** (100% on 64 rows by epoch 63).
5. **`step5_financial_ml_humility.py`** — the same model, honestly evaluated, at chance. Then
   three leaks measured side by side.

## Verification benchmark ("done when")

`bench/train_and_report.py` asserts **two** conditions, and the second is the unusual one:

| | claim | result |
|---|---|---|
| **A** | 64 rows must be memorised to ≥99% train accuracy | **100%**, first reached at epoch 63 |
| **B** | honest test accuracy must be indistinguishable from a coin flip (\|z\| < 2) | **51.1%**, z = **+0.30**, below the 55% majority baseline |
| **C** | the leakage demo must actually leak | centred window: **81.1%** (+30.0 pts) |

Condition B is deliberately inverted from the usual benchmark. The labels come from a synthetic
random walk, so the true signal is exactly zero; a model that "beats" the baseline here is
measuring leakage, and the benchmark fails rather than celebrates. Committed output:
[`bench/results.json`](bench/results.json).

## The leakage audit

| variant | test accuracy | uplift | severity |
|---|---:|---:|---|
| honest chronological split | 0.511 | — | — |
| **centred rolling feature** (`rolling(3, center=True)`) | **0.811** | **+0.300** | catastrophic |
| shuffled train/test split | 0.483 | −0.028 | silent *on this data* |
| scaler fitted before the split | 0.517 | +0.006 | silent *on this data* |

The asymmetry is the lesson. Two of these leaks barely register — on a random walk with eight
stationary features. On data with real autocorrelation they do not stay quiet. **"It didn't
change my score" is evidence about your dataset, not about your methodology.**

And note how the big one is written:

```python
ret.rolling(3).mean()                # t-2, t-1, t     — fine
ret.rolling(3, center=True).mean()   # t-1, t,   t+1   — contains the label
```

Nobody types `center=True` intending to cheat. It is a smoothing default.

## Tests worth reading

* `test_zero_grad_is_load_bearing` — runs the loop with and without `zero_grad()` and proves the
  weights differ. The loss still falls without it; it just falls wrongly.
* `test_features_are_causal` — truncate the price history, recompute, demand bit-identical
  earlier rows.
* `test_standardise_uses_training_statistics_only` — the training split centres to zero and the
  **test split does not**, which is the proof that no future statistics leaked in.
* `test_evaluate_does_not_build_a_graph_or_leave_train_mode` — the two things `evaluate` must
  guarantee.

## AlphaDesk hook

| key | surface | what it gives AlphaDesk |
|---|---|---|
| `foundation.training_loop` | Foundation | the reusable train/evaluate skeleton every training topic reuses (T15, T17, T19, T14…) |
| `foundation.toy_signal_model` | Foundation | the tiny MLP — registered explicitly as a teaching artefact, never wired to an order |

## Layout

- `src/p0_3_training/` — `data.py` (causal features + honest splits), `model.py` (`TinyMLP` and
  the hand-differentiated `ManualLinear`), `loop.py` (the skeleton), `experiments.py` (A/B/C),
  `alphadesk_hook.py`
- `steps/` · `tests/` · `bench/` · `NOTES.md`

## Videos

Episode script: [`video/topics/p0.3/script.md`](../../../video/topics/p0.3/script.md).

---
*AlphaDesk is a fictional educational simulation — no real orders, money, brokerage systems, or market-data redistribution.*
