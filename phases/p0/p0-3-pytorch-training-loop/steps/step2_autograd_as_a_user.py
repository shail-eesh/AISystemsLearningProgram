#!/usr/bin/env python3
"""Step 2 — autograd from the outside: what `.backward()` actually does.

Run:  python3 steps/step2_autograd_as_a_user.py

You will *build* autograd in T31. Today you only need the user's model of it:

* A tensor with `requires_grad=True` is a **leaf** of a graph.
* Every operation on it records a node. `loss.backward()` walks that graph in
  reverse, applying the chain rule, and **adds** the result into each leaf's
  `.grad`.
* Adds. Not sets. That accumulation is why every training loop starts with
  `zero_grad()`, and forgetting it is the single most common PyTorch bug.
* The graph is freed after `backward()`. Calling it twice raises, unless you
  ask for `retain_graph=True` (usually a sign you meant something else).
* `torch.no_grad()` stops recording; `.detach()` cuts one tensor out of the
  graph. Both are how you evaluate without paying for a graph you will discard.
"""

import warnings

import _bootstrap  # noqa: F401
import torch


def demo_scalar_chain_rule() -> None:
    x = torch.tensor(3.0, requires_grad=True)
    y = x ** 2 + 2 * x + 1          # dy/dx = 2x + 2 = 8
    y.backward()
    print(f"  y = x^2 + 2x + 1 at x=3 -> y={y.item():.1f}, dy/dx={x.grad.item():.1f} (expect 8)")
    print(f"  x is a leaf: {x.is_leaf}; y is not: {y.is_leaf}; y.grad_fn={type(y.grad_fn).__name__}")


def demo_gradients_accumulate() -> None:
    x = torch.tensor(2.0, requires_grad=True)
    for i in range(1, 4):
        (x ** 2).backward()
        print(f"  backward #{i}: x.grad = {x.grad.item():.1f}  (each call adds 2x = 4)")
    x.grad = None
    (x ** 2).backward()
    print(f"  after clearing:  x.grad = {x.grad.item():.1f}")
    print("  THIS is what optimiser.zero_grad() is for. Omit it and your gradient is")
    print("  the sum over every batch so far — the loss still falls, just wrongly.")


def demo_vector_jacobian() -> None:
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = (x ** 2).sum()
    y.backward()
    print(f"  d(sum x^2)/dx = 2x = {x.grad.tolist()}")
    x.grad = None
    z = x ** 2
    try:
        z.backward()
    except RuntimeError as exc:
        print(f"  backward on a non-scalar raises: {str(exc).splitlines()[0]}")
    z.backward(torch.ones_like(z))     # supply the upstream gradient yourself
    print(f"  with an explicit upstream vector of ones: {x.grad.tolist()}")
    print("  a loss is scalar precisely so this vector is implicitly ones.")


def demo_no_grad_and_detach() -> None:
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    with torch.no_grad():
        y = x * 2
    print(f"  inside no_grad: y.requires_grad={y.requires_grad} (no graph built)")
    z = (x * 3).detach()
    print(f"  after .detach(): z.requires_grad={z.requires_grad}, x untouched")
    w = x * 3
    print(f"  without detach:  w.requires_grad={w.requires_grad}")
    print("  evaluation uses no_grad; storing a value for later logging uses detach.")


def demo_graph_is_freed() -> None:
    x = torch.tensor(1.0, requires_grad=True)
    y = x * 4
    y.backward()
    try:
        y.backward()
    except RuntimeError as exc:
        print(f"  second backward: {str(exc).splitlines()[0][:96]}...")
    print("  the graph is a one-shot recording; each forward pass builds a fresh one.")


def demo_leaf_vs_intermediate() -> None:
    x = torch.tensor(2.0, requires_grad=True)
    h = x * 3
    out = h ** 2
    out.backward()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        h_grad = h.grad
    print(f"  x.grad = {x.grad.item():.1f}   h.grad = {h_grad}")
    if caught:
        first = str(caught[0].message).split(" See ")[0].strip()
        print(f"  torch warns: {first[:150]}")
    print("  intermediates get no .grad by default — memory would explode. Use")
    print("  h.retain_grad() before backward if you actually need it (debugging,")
    print("  and the interpretability work in T22).")


if __name__ == "__main__":
    print("the chain rule:")
    demo_scalar_chain_rule()
    print("gradients accumulate:")
    demo_gradients_accumulate()
    print("non-scalar backward:")
    demo_vector_jacobian()
    print("no_grad and detach:")
    demo_no_grad_and_detach()
    print("the graph is freed:")
    demo_graph_is_freed()
    print("leaves vs intermediates:")
    demo_leaf_vs_intermediate()
    print("\nstep 2 OK")
