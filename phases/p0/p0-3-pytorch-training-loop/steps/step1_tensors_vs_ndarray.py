#!/usr/bin/env python3
"""Step 1 — a tensor is an ndarray that remembers what happened to it.

Run:  python3 steps/step1_tensors_vs_ndarray.py

Everything you learned in P0.2 transfers: shapes, broadcasting, dtypes,
reductions, `einsum`. A `torch.Tensor` adds three things an `np.ndarray` does
not have:

1. a **device** (cpu / cuda) the data lives on;
2. optional **gradient tracking** — the tensor records the operations applied
   to it so they can be replayed backwards;
3. **in-place ops** marked with a trailing underscore (`add_`, `zero_`), which
   are fast and which autograd sometimes refuses to differentiate through.

The dtype default also flips: NumPy gives you float64, torch gives you
float32, and mixing them is the first error most people hit.
"""

import _bootstrap  # noqa: F401
import numpy as np
import torch


def demo_defaults_and_conversion() -> None:
    a = np.array([1.0, 2.0, 3.0])
    t = torch.tensor([1.0, 2.0, 3.0])
    print(f"  numpy default dtype: {a.dtype}   torch default dtype: {t.dtype}")
    print("  float32 is not carelessness: it halves memory and bandwidth, and")
    print("  gradient noise dwarfs the extra precision. Phase 5 goes further (bf16).")

    shared = torch.from_numpy(a.copy())
    src = shared.numpy()                     # .numpy() also shares
    shared[0] = 99.0
    print(f"  torch.from_numpy / .numpy() SHARE memory: source is now {src}")

    b = np.array([1.0, 2.0, 3.0])
    copied = torch.tensor(b)                 # torch.tensor() always copies
    copied[0] = -1.0
    print(f"  torch.tensor(ndarray) COPIES:            source is still {b}")


def demo_shapes_carry_over() -> None:
    x = torch.arange(12.0).reshape(3, 4)
    print(f"  reshape/permute/broadcast all work: {tuple(x.shape)} -> "
          f"{tuple(x.T.shape)} -> {tuple((x * torch.ones(3, 1)).shape)}")
    q = torch.randn(2, 3, 4, 5)
    k = torch.randn(2, 3, 4, 5)
    scores = torch.einsum("bhqd,bhkd->bhqk", q, k)
    print(f"  einsum is identical to NumPy's: {tuple(scores.shape)}")
    print(f"  unsqueeze(1) is the [:, None] you used in P0.2: "
          f"{tuple(torch.arange(5.0).unsqueeze(1).shape)}")


def demo_device_and_dtype_errors() -> None:
    print(f"  cuda available here: {torch.cuda.is_available()} (this sandbox is CPU-only;")
    print("  the 4070 lane in Phase 7 is where device placement starts to matter)")
    try:
        torch.ones(3, dtype=torch.float32) @ torch.ones(3, dtype=torch.float64)
    except RuntimeError as exc:
        print(f"  dtype mismatch is a hard error: {str(exc).splitlines()[0]}")


def demo_inplace_ops() -> None:
    x = torch.ones(3)
    before = x.data_ptr()
    y = x.add(1)          # returns a NEW tensor; x untouched
    print(f"  after x.add(1):  x={x.tolist()}  y={y.tolist()}")
    x.add_(1)             # mutates x in place
    print(f"  after x.add_(1): x={x.tolist()}  same storage? {x.data_ptr() == before}")
    print(f"  y is a different buffer: {y.data_ptr() != before}")
    print("  the trailing underscore is the whole convention — and autograd will")
    print("  refuse some in-place ops on tensors it still needs for the backward pass.")


def demo_reductions_and_keepdims() -> None:
    x = torch.arange(6.0).reshape(2, 3)
    print(f"  x.mean(dim=1)              -> {tuple(x.mean(dim=1).shape)}")
    print(f"  x.mean(dim=1, keepdim=True) -> {tuple(x.mean(dim=1, keepdim=True).shape)}")
    print("  note the spelling: torch says `dim` and `keepdim`, NumPy says `axis`")
    print("  and `keepdims`. Same idea, and a constant source of typos.")


if __name__ == "__main__":
    print("defaults and conversion:")
    demo_defaults_and_conversion()
    print("shapes carry over:")
    demo_shapes_carry_over()
    print("device and dtype:")
    demo_device_and_dtype_errors()
    print("in-place ops:")
    demo_inplace_ops()
    print("reductions:")
    demo_reductions_and_keepdims()
    print("\nstep 1 OK")
