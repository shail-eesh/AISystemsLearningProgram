"""The training loop skeleton — the thing you must be able to write from empty.

Six lines carry the whole idea:

    optimiser.zero_grad()          # gradients ACCUMULATE; clear them first
    logits = model(xb)             # forward
    loss = criterion(logits, yb)   # scalar
    loss.backward()                # reverse-mode autodiff fills .grad
    optimiser.step()               # p -= lr * p.grad  (plus the optimiser's state)

Everything else in this file is bookkeeping: seeding, batching, eval mode, and
keeping the best checkpoint. Learn the six lines; the bookkeeping is the part
you can look up.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import Dataset, ReturnsDataset


def seed_everything(seed: int = 1729) -> int:
    """Determinism on CPU. On GPU you also need the cuDNN flags."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    return seed


@dataclass
class TrainConfig:
    epochs: int = 60
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.0
    hidden: int = 32
    dropout: float = 0.0
    seed: int = 1729
    shuffle: bool = True
    log_every: int = 0            # 0 = silent


@dataclass
class History:
    train_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, list[float]]:
        return {
            "train_loss": self.train_loss, "train_acc": self.train_acc,
            "val_loss": self.val_loss, "val_acc": self.val_acc,
        }


@torch.no_grad()
def evaluate(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    """Loss and accuracy. `no_grad` + `eval()` — both, every time.

    `eval()` switches dropout and batch-norm to inference behaviour;
    `no_grad()` stops the graph being built. Forgetting `eval()` gives you
    quietly wrong numbers; forgetting `no_grad()` just wastes memory.
    """
    was_training = model.training
    model.eval()
    logits = model(X)
    loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
    acc = ((logits > 0).float() == y).float().mean()
    if was_training:
        model.train()
    return float(loss), float(acc)


def _tensors(d: Dataset) -> tuple[torch.Tensor, torch.Tensor]:
    ds = ReturnsDataset(d)
    return ds.X, ds.y


def train(
    model: nn.Module,
    train_data: Dataset,
    val_data: Dataset | None = None,
    config: TrainConfig | None = None,
) -> tuple[nn.Module, History]:
    """Fit `model`, returning it with the best-validation weights restored."""
    cfg = config or TrainConfig()
    seed_everything(cfg.seed)

    loader = DataLoader(
        ReturnsDataset(train_data),
        batch_size=min(cfg.batch_size, len(train_data)),
        shuffle=cfg.shuffle,
        generator=torch.Generator().manual_seed(cfg.seed),
        drop_last=False,
    )
    criterion = nn.BCEWithLogitsLoss()
    optimiser = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    Xtr, ytr = _tensors(train_data)
    Xva, yva = _tensors(val_data) if val_data is not None and len(val_data) else (None, None)

    history = History()
    best_val, best_state = float("inf"), None

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for xb, yb in loader:
            optimiser.zero_grad()                 # 1. gradients accumulate
            logits = model(xb)                    # 2. forward
            loss = criterion(logits, yb)          # 3. scalar loss
            loss.backward()                       # 4. reverse-mode autodiff
            optimiser.step()                      # 5. parameter update

        tr_loss, tr_acc = evaluate(model, Xtr, ytr)
        history.train_loss.append(tr_loss)
        history.train_acc.append(tr_acc)
        if Xva is not None:
            va_loss, va_acc = evaluate(model, Xva, yva)
            history.val_loss.append(va_loss)
            history.val_acc.append(va_acc)
            if va_loss < best_val:
                best_val = va_loss
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if cfg.log_every and epoch % cfg.log_every == 0:
            tail = f" val {history.val_loss[-1]:.4f}/{history.val_acc[-1]:.3f}" if Xva is not None else ""
            print(f"    epoch {epoch:3d}  train {tr_loss:.4f}/{tr_acc:.3f}{tail}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history
