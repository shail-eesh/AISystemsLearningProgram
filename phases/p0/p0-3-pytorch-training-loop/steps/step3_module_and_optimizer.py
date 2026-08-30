#!/usr/bin/env python3
"""Step 3 — what `nn.Module` and `optimiser.step()` are actually doing.

Run:  python3 steps/step3_module_and_optimizer.py

Two demystifications:

* **`nn.Module` is a parameter registry.** Assigning a `Parameter` (or a
  submodule) to `self` registers it; `model.parameters()` walks that tree.
  That is the entire mechanism. A tensor you assign as a plain attribute is
  *not* registered, does not move with `.to(device)`, and never gets a
  gradient — a silent, popular bug.
* **`optimiser.step()` is `p -= lr * p.grad`, plus state.** For plain SGD it is
  literally that line; this step proves it by running both and comparing.
  Adam adds two exponential moving averages per parameter (which is why the
  optimiser costs 2x the model in memory — see Phase 5).

`ManualLinear` closes the circle: forward and backward written out by hand,
matching torch's gradients to 1e-6.
"""

import _bootstrap  # noqa: F401
import numpy as np
import torch
from p0_3_training import TinyMLP, count_parameters
from p0_3_training.model import ManualLinear
from torch import nn


def demo_module_is_a_registry() -> None:
    model = TinyMLP(8, hidden=32)
    print(f"  TinyMLP(8, hidden=32): {count_parameters(model)} trainable parameters")
    for name, p in model.named_parameters():
        print(f"    {name:16s} {tuple(p.shape)}  requires_grad={p.requires_grad}")

    class Broken(nn.Module):
        def __init__(self):
            super().__init__()
            self.good = nn.Parameter(torch.zeros(3))
            self.bad = torch.zeros(3)          # NOT registered

    b = Broken()
    print(f"  registered parameters of Broken: {[n for n, _ in b.named_parameters()]}")
    print("  `self.bad` is invisible to the optimiser, to .to(device) and to state_dict().")


def demo_step_is_one_line() -> None:
    torch.manual_seed(0)
    x = torch.randn(16, 4)
    y = torch.randn(16, 1)

    def fresh():
        torch.manual_seed(0)
        return nn.Linear(4, 1)

    auto, manual = fresh(), fresh()
    lr = 0.1
    opt = torch.optim.SGD(auto.parameters(), lr=lr)

    for _ in range(5):
        opt.zero_grad()
        nn.functional.mse_loss(auto(x), y).backward()
        opt.step()

        for p in manual.parameters():
            if p.grad is not None:
                p.grad.zero_()
        nn.functional.mse_loss(manual(x), y).backward()
        with torch.no_grad():                       # the update is not differentiable
            for p in manual.parameters():
                p -= lr * p.grad

    diff = max(
        float((a - b).detach().abs().max())
        for a, b in zip(auto.parameters(), manual.parameters(), strict=True)
    )
    print(f"  after 5 SGD steps, |optimiser - hand-written| = {diff:.2e}")
    print("  `optimiser.step()` for SGD is exactly `p -= lr * p.grad` under no_grad.")


def demo_optimiser_state_costs_memory() -> None:
    model = TinyMLP(8, hidden=32)
    n = count_parameters(model)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x, y = torch.randn(8, 8), torch.randint(0, 2, (8, 1)).float()
    nn.functional.binary_cross_entropy_with_logits(model(x), y).backward()
    opt.step()
    slots = sum(len(v) - 1 for v in opt.state.values())   # exp_avg, exp_avg_sq (minus 'step')
    print(f"  model parameters: {n}")
    print(f"  AdamW state tensors per parameter tensor: {slots // len(opt.state)}"
          " (exp_avg, exp_avg_sq)")
    print(f"  optimiser memory ~= {slots // len(opt.state)}x the model. At 40M params (T15)")
    print("  that is the difference between fitting on the 4070 and not.")


def demo_manual_gradients_match() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, 5))
    y = (rng.normal(size=(64, 1)) > 0).astype(np.float64)

    manual = ManualLinear(5, seed=0)
    W = torch.tensor(manual.W, requires_grad=True)
    b = torch.tensor(manual.b, requires_grad=True)
    Xt, yt = torch.tensor(X), torch.tensor(y)
    loss = nn.functional.binary_cross_entropy_with_logits(Xt @ W + b, yt)
    loss.backward()

    p = manual.forward(X)
    dz = (p - y) / X.shape[0]
    dW, db = X.T @ dz, dz.sum(axis=0)
    print(f"  loss: torch {loss.item():.10f}  manual {manual.loss(p, y):.10f}")
    print(f"  |dW_torch - dW_manual| = {np.abs(W.grad.numpy() - dW).max():.3e}")
    print(f"  |db_torch - db_manual| = {np.abs(b.grad.numpy() - db).max():.3e}")
    print("  (p - y) is the whole gradient because the sigmoid's derivative cancels")
    print("  against the log in the cross-entropy. T31 rebuilds that cancellation.")


def demo_train_eval_mode() -> None:
    model = TinyMLP(8, hidden=16, dropout=0.5)
    x = torch.randn(4, 8)
    model.train()
    a, b = model(x), model(x)
    model.eval()
    c, d = model(x), model(x)
    print(f"  train(): two forward passes differ? {not torch.allclose(a, b)}  <- dropout is on")
    print(f"  eval():  two forward passes differ? {not torch.allclose(c, d)}  <- deterministic")
    print("  forgetting .eval() before validation is a silent accuracy bug, not a crash.")


if __name__ == "__main__":
    print("nn.Module is a registry:")
    demo_module_is_a_registry()
    print("optimiser.step() is one line:")
    demo_step_is_one_line()
    print("optimiser state:")
    demo_optimiser_state_costs_memory()
    print("hand-written gradients match:")
    demo_manual_gradients_match()
    print("train vs eval mode:")
    demo_train_eval_mode()
    print("\nstep 3 OK")
