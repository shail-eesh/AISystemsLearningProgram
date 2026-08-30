# P0.3 · PyTorch tensors & the training loop — Notes

*Intuition and gotchas. This file is the source the video script is written from.*

---

## The one-sentence version

The training loop is five lines and takes an afternoon to learn; knowing when its output means
nothing takes considerably longer, and this topic front-loads the second part because in finance
it is the part that costs money.

## 1. A tensor is an ndarray that remembers

Everything from P0.2 transfers unchanged — shapes, broadcasting, reductions, `einsum`. A
`torch.Tensor` adds three things:

* a **device** the storage lives on (`cpu`, `cuda:0`);
* optional **gradient tracking** — the tensor records the ops applied to it so they can be
  replayed in reverse;
* **in-place operations** spelled with a trailing underscore (`add_`, `zero_`, `clamp_`), which
  are fast and which autograd sometimes refuses to differentiate through.

Two spelling traps: torch says `dim`/`keepdim` where NumPy says `axis`/`keepdims`, and torch
defaults to **float32** where NumPy defaults to float64. The float32 default is not carelessness —
it halves memory and bandwidth, and gradient noise dwarfs the extra precision. Phase 5 pushes
further into bf16.

`torch.from_numpy(a)` and `t.numpy()` **share** memory; `torch.tensor(a)` copies. Mutating one
view and being surprised by the other is a rite of passage.

## 2. Autograd, from the user's side

`requires_grad=True` makes a tensor a **leaf**. Operations on it build a graph. `loss.backward()`
walks that graph in reverse applying the chain rule and **adds** the result into each leaf's
`.grad`.

*Adds.* Not sets. Run `(x**2).backward()` three times on the same leaf and `x.grad` reads 4, 8,
12. This is not a wart — it is what makes gradient accumulation across micro-batches work, which
is how you train a model that does not fit in memory (Phase 5). But it means every loop must
start with `zero_grad()`, and omitting it is the single most common PyTorch bug. The loss still
goes down. It just goes down using the sum of every gradient so far, and nothing tells you.

Three more facts worth memorising:

* **`backward()` on a non-scalar raises.** A loss is scalar precisely so the upstream gradient is
  implicitly a `1`. For a vector you must supply it: `z.backward(torch.ones_like(z))`.
* **The graph is freed after `backward()`.** Each forward pass builds a fresh one. Reaching for
  `retain_graph=True` is nearly always a sign you meant to restructure something.
* **Intermediates get no `.grad`.** Storing gradients for every intermediate would explode
  memory. `h.retain_grad()` opts one in — useful for debugging and for the interpretability work
  in T22.

`torch.no_grad()` stops recording (use it for evaluation); `.detach()` cuts one tensor out of the
graph (use it when storing a value for logging). Confusing the two costs memory, not correctness.

## 3. `nn.Module` is a registry, and `step()` is one line

Assigning an `nn.Parameter` or a submodule to `self` registers it. `model.parameters()` walks that
tree. That is the whole mechanism — there is no metaclass magic and no attribute interception
beyond `__setattr__`.

The corollary bites: `self.buffer = torch.zeros(3)` is **not** registered. It will not appear in
`parameters()`, will not be moved by `.to(device)`, will not be saved in `state_dict()`, and will
never receive a gradient. Use `nn.Parameter` for things that learn and `register_buffer` for
things that travel with the model but do not.

`optimiser.step()` for plain SGD is literally:

```python
with torch.no_grad():
    for p in params:
        p -= lr * p.grad
```

Step 3 runs both and gets agreement to 7e-09 (float32 accumulation order). AdamW adds two
exponential moving averages per parameter — hence **2× the model's memory in optimiser state**,
which at 40M parameters (T15) is the difference between fitting on a 12 GB card and not.

Finally: `model.eval()` and `model.train()` switch dropout and batch-norm behaviour. Forgetting
`eval()` before validation is a *silent* accuracy bug, not a crash.

## 4. Logits, not probabilities

`TinyMLP` returns raw logits and the loss is `BCEWithLogitsLoss`. Applying `sigmoid` yourself and
using `BCELoss` is mathematically identical and numerically worse: the fused version uses the
log-sum-exp trick internally, while the split version overflows once a logit reaches about ±40 and
produces NaN. Same for `CrossEntropyLoss`, which expects logits and applies log-softmax itself.

The manual derivation in `ManualLinear` shows *why* this pairing is standard: the sigmoid's
derivative cancels against the log in the cross-entropy, leaving `dL/dz = (p - y)/n`. The elegance
is not decorative — it is the reason the gradient is numerically well behaved. T31 rebuilds that
cancellation as a graph.

## 5. Overfit a tiny batch, always, first

Before any real training run: take 32–128 examples and drive training accuracy to 100%. It takes
seconds and it separates "my model cannot learn" from "my data has no signal", which are
completely different bugs with completely different fixes.

Here: 64 rows, ~1,000 parameters, 100% by epoch 63. What that proves is that the loop is wired
correctly — forward, loss, backward, step, zero_grad, the DataLoader, the shapes. What it proves
about the data is **nothing at all**. The same network would memorise those 64 rows just as
happily if the labels had been shuffled first.

## 6. The humility lesson, stated plainly

Same model, same data, honest protocol:

| | |
|---|---|
| train accuracy | 0.513 |
| test accuracy | **0.511** |
| always-predict-up baseline | **0.550** |
| z vs a coin flip | +0.30 |
| test loss | 0.6965 (ln 2 = 0.6931) |

The model is **worse than a constant**. That is the correct result: the prices are a synthetic
geometric random walk, so the true predictability of tomorrow's direction is exactly zero, and
anything above chance would be a bug in the protocol.

Three habits follow:

1. **Always report a baseline on the same window.** "55% accuracy" is meaningless alone; here it
   is precisely the score of predicting "up" every day.
2. **Report the loss too.** Accuracy hides calibration; a model at 0.6965 against ln 2 has learned
   nothing, and the loss says so more clearly than the accuracy does.
3. **Ask what the sample size supports.** 180 test rows give a coin-flip standard error of 3.7
   points, so *anything* under ~57% is unremarkable. Most reported edges in retail backtests are
   inside their own error bars.

## 7. Leakage does not look like cheating

| variant | accuracy | uplift |
|---|---:|---:|
| honest | 0.511 | — |
| centred rolling window | **0.811** | **+0.300** |
| shuffled split | 0.483 | −0.028 |
| scaler fit before split | 0.517 | +0.006 |

The catastrophic one is a *smoothing default*:

```python
ret.rolling(3).mean()                # t-2, t-1, t     — causal
ret.rolling(3, center=True).mean()   # t-1, t,   t+1   — contains the label
```

Nobody writes that intending to cheat. It is one keyword, it makes the feature look nicer on a
chart, and it hands the model tomorrow's return.

The two quiet ones matter differently. They are quiet *because this series is a random walk with
no autocorrelation to exploit and stationary feature scales*. Change the data and they wake up.
The conclusion to carry forward: **a leak that does not move your number is still a leak**, and
"it made no difference when I tried it" is a statement about your dataset.

The general detector is cheap and almost nobody runs it: **truncate the input, recompute, and
demand that every earlier value is bit-identical.** Both `test_features_are_causal` here and
`test_no_lookahead` in P0.2 are five lines each.

## 8. Splits in time series

* Never `train_test_split(shuffle=True)`. Shuffling lets the model interpolate between days it
  has already seen.
* Split at **date** boundaries, not row indices. Five symbols share every trading day; a
  row-count cut puts the same day on both sides and gives every test row a near-twin in training.
* Fit the scaler (and any imputation, any encoder) on **training data only**. The mean and
  standard deviation of the full series are themselves future information. The tell that you got
  this right: the training split centres to zero and the test split does *not*.
* One split is not enough for a real claim — walk-forward is. T48's point-in-time feature store
  makes that mechanical.

## Gotchas checklist

- [ ] `zero_grad()` first, every iteration.
- [ ] Logits + `*WithLogitsLoss`; never your own sigmoid + `BCELoss`.
- [ ] `model.eval()` **and** `torch.no_grad()` for evaluation.
- [ ] `nn.Parameter` / `register_buffer` — a bare tensor attribute is invisible.
- [ ] `dim`/`keepdim` in torch, `axis`/`keepdims` in NumPy.
- [ ] Overfit a tiny batch before trusting any training run.
- [ ] Quote a baseline and a standard error with every accuracy.
- [ ] Split by date; fit the scaler on train only; truncate-and-compare to test for lookahead.

## What breaks if we skip this

T31 (autograd) is this topic inverted: you build the machinery whose *user interface* you just
learned. T15 (AlphaSLM) reuses `loop.py` almost verbatim at 40M parameters, where a missing
`zero_grad()` costs a night of compute instead of a second. And every evaluation claim from
Phase 4's eval harness onwards rests on the protocol discipline in §6–§8 — without it, the desk
would confidently report edges that are entirely artefacts.
