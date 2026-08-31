# T31 · Autograd engine (micrograd-style)

**Phase:** Foundations (P1) · **Generation day:** Day 2 · **Video episodes:** 3

> [← Back to course home](../../../index.html) · [Master plan](../../../MASTER_PLAN.md) · [Progress ledger](../../../EXECUTION/LEDGER.md)

## What you build

Reverse-mode automatic differentiation, twice. First on scalars — a `Value` that
remembers the operation that produced it and carries a three-line closure for
its own local derivative. Then the same structure on NumPy arrays, where a
single node holds a whole matrix and the inner loops finally happen in C.

On top of that, the library that falls out for free: `Linear`, `Sequential`,
`MLP`, three losses, `SGD` and `Adam` — about 250 lines, because the engine
already did the hard part.

The point is not to produce a framework. It is that after this topic, `.backward()`
is not magic anywhere else in the course: it is a topological sort and a pile of
local rules, and you have written both.

**The paper trail:** the scalar engine follows the micrograd lineage (Karpathy);
the tensor gradients and the finite-difference check follow the CS231n
backpropagation notes. Neither is cited as an authority — both are checked here
against central differences and against a hand-derived reference network.

## Results

| Check | Result |
|:--|:--|
| Scalar gradcheck, 30 random graphs | worst relative error **3.6e-10** |
| Tensor gradcheck, 10 shape-mixing cases / 147 entries | worst **7.8e-08** |
| Whole-MLP gradcheck through every parameter | worst **8.7e-09** |
| Loss curve vs an independently hand-derived network, 200 Adam steps | max divergence **6.1e-17** |
| Cost of the engine vs hand-written gradients | **3.7×** slower |
| Cost of the *scalar* engine vs the tensor engine, same 64×8→16 layer | **498×** slower (27,777 graph nodes) |

The loss-curve number is the one worth staring at. "Matches within noise" was
the target; the actual answer is machine epsilon, because autodiff is not an
approximation of the derivative — it is the derivative, evaluated in a different
order.

## AlphaDesk hook

Two components on the **foundation** surface:

- `foundation.autograd` — the engine itself, so later topics can point at it.
- `foundation.handrolled_signal_model` — a toy up/down MLP trained end to end by
  this engine rather than by PyTorch.

The P0.3 caveat carries forward unchanged and is registered *with* the component:
this model memorises its training window and means nothing out of sample. It is a
teaching artefact. AlphaDesk is a fictional educational simulation — no real
orders, no money, no brokerage, nothing routed anywhere.

## How to run

```bash
# the step ladder, in order
python3 phases/p1/t31-autograd/steps/step1_scalar_value.py
python3 phases/p1/t31-autograd/steps/step2_topological_backward.py
python3 phases/p1/t31-autograd/steps/step3_tensor_broadcasting.py
python3 phases/p1/t31-autograd/steps/step4_nn_and_optimisers.py
python3 phases/p1/t31-autograd/steps/step5_train_on_your_engine.py

# tests + the verification benchmark
python3 -m pytest phases/p1/t31-autograd/tests -q
python3 phases/p1/t31-autograd/bench/gradcheck_suite.py
```

No PyTorch required — the reference implementation this engine is checked
against is hand-derived NumPy in `src/t31_autograd/train.py`, on purpose.

## Layout

- `src/t31_autograd/engine.py` — the scalar `Value`
- `src/t31_autograd/tensor.py` — the NumPy `Tensor` and `unbroadcast`
- `src/t31_autograd/nn.py`, `optim.py` — layers, losses, SGD/Adam
- `src/t31_autograd/gradcheck.py` — central finite differences
- `src/t31_autograd/train.py` — step 5 plus the hand-derived reference network
- `steps/` — the five checkpoints, each runnable on its own
- `tests/` — 61 tests
- `bench/` — the gradcheck suite and its `results.json`
- `NOTES.md` — the intuition and the gotchas (source for the video scripts)

## Videos

Episode scripts live in [`video/topics/t31/`](../../../video/topics/t31/). Rendered
`.mp4`s are delivered in chat, not committed.
