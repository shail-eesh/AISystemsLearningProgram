#!/usr/bin/env python3
"""Step 4 — the loop, written from an empty file, then used properly.

Run:  python3 steps/step4_the_training_loop.py

The whole thing is five lines inside two `for`s:

    for epoch in range(epochs):
        for xb, yb in loader:
            optimiser.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimiser.step()

Everything else — seeding, batching, eval mode, checkpointing — is bookkeeping
you can look up. The five lines you must be able to type from memory.

This step then runs the standard first diagnostic for any new setup:
**overfit a tiny batch**. If a model with spare capacity cannot memorise 64
rows, the bug is in your loop. If it can, your loop is correct — and you have
learned precisely nothing about your data. Step 5 is the other half.
"""

import time

import _bootstrap  # noqa: F401
import numpy as np
import torch
from p0_3_training import (
    ReturnsDataset,
    TinyMLP,
    TrainConfig,
    build_dataset,
    count_parameters,
    evaluate,
    seed_everything,
    standardise,
    train,
)
from p0_3_training.data import Dataset
from p0_3_training.experiments import overfit_tiny_batch
from torch import nn
from torch.utils.data import DataLoader


def demo_the_loop_from_scratch() -> None:
    """Written out longhand once, so the abstraction never becomes a mystery."""
    seed_everything(1729)
    data = build_dataset()
    tiny = Dataset(X=data.X[:128], y=data.y[:128], dates=data.dates[:128],
                   symbols=data.symbols[:128], feature_names=data.feature_names)
    (scaled,), _, _ = standardise(tiny)

    model = TinyMLP(len(data.feature_names), hidden=32)
    loader = DataLoader(ReturnsDataset(scaled), batch_size=32, shuffle=True,
                        generator=torch.Generator().manual_seed(0))
    criterion = nn.BCEWithLogitsLoss()
    optimiser = torch.optim.AdamW(model.parameters(), lr=3e-3)

    print(f"  model: {count_parameters(model)} parameters, {len(scaled)} rows, batch 32")
    for epoch in range(1, 41):
        model.train()
        for xb, yb in loader:
            optimiser.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimiser.step()
        if epoch % 10 == 0:
            X = torch.from_numpy(np.ascontiguousarray(scaled.X))
            y = torch.from_numpy(np.ascontiguousarray(scaled.y)).unsqueeze(1)
            ep_loss, ep_acc = evaluate(model, X, y)
            print(f"    epoch {epoch:3d}  loss {ep_loss:.4f}  train acc {ep_acc:.3f}")


def demo_batch_size_changes_the_path() -> None:
    seed_everything(1729)
    data = build_dataset()
    (scaled,), _, _ = standardise(
        Dataset(X=data.X[:512], y=data.y[:512], dates=data.dates[:512],
                symbols=data.symbols[:512], feature_names=data.feature_names)
    )
    for bs in (8, 64, 512):
        model = TinyMLP(len(data.feature_names), hidden=32)
        t0 = time.perf_counter()
        _, hist = train(model, scaled, None, TrainConfig(epochs=30, batch_size=bs, lr=3e-3))
        dt = time.perf_counter() - t0
        steps = 30 * int(np.ceil(len(scaled) / bs))
        print(f"    batch {bs:3d}: {steps:4d} updates, {dt:5.2f}s, "
              f"final train loss {hist.train_loss[-1]:.4f}, acc {hist.train_acc[-1]:.3f}")
    print("  small batches = more updates per epoch = faster progress per epoch, more noise.")
    print("  the noise is not purely a cost: it is part of why SGD generalises at all.")


def demo_overfit_tiny_batch() -> None:
    result = overfit_tiny_batch(n=64, epochs=400, hidden=64)
    print(f"    rows: {result['rows']}   epochs: {result['epochs']}")
    print(f"    final train loss:     {result['final_train_loss']:.6f}")
    print(f"    final train accuracy: {result['final_train_accuracy']:.3f}")
    print(f"    reached 100% at epoch {result['epochs_to_perfect']}")
    print(f"    verdict: {result['verdict']}")
    print("  A 1,000-parameter network memorised 64 arbitrary labels. It would have")
    print("  memorised them just as happily if you had shuffled the labels first —")
    print("  which is exactly the experiment step 5 makes you take seriously.")


if __name__ == "__main__":
    print("the loop, longhand:")
    demo_the_loop_from_scratch()
    print("batch size:")
    demo_batch_size_changes_the_path()
    print("overfit a tiny batch (the standard first diagnostic):")
    demo_overfit_tiny_batch()
    print("\nstep 4 OK")
