"""A training loop small enough to read, honest enough to trust.

Everything here also appears in T15's harness with checkpointing and resume
bolted on. The four things this loop does that a naive one does not:

* **warmup then cosine decay.** Adam's second-moment estimate is garbage for
  the first few dozen steps (it has seen almost no gradients), so a full
  learning rate at step 0 is a step in a confidently wrong direction. Warmup is
  the fix; cosine decay to ~10% is what nanoGPT/GPT-3 use to land softly.
* **gradient clipping at global norm 1.0.** One bad batch in a language corpus
  (a long run of rare characters) produces a gradient ten times the usual size,
  and Adam's normalisation *amplifies* rather than damps it. Clipping is the
  seatbelt; the loss spike you avoid is real.
* **eval in ``no_grad`` and ``eval()`` mode**, averaged over several batches,
  because a single validation batch on a 90 KB corpus is noise with a decimal
  point.
* **a fixed seed and a returned history**, so two runs are comparable and the
  bench can assert on the curve rather than on a final number.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field

import torch
from torch import Tensor

from .model import GPT


@dataclass
class TrainConfig:
    steps: int = 400
    batch_size: int = 16
    lr: float = 3e-3
    min_lr_ratio: float = 0.1
    warmup: int = 40
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_every: int = 50
    eval_batches: int = 8
    seed: int = 1337
    log: bool = False


@dataclass
class History:
    step: list[int] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    eval_step: list[int] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    lr: list[float] = field(default_factory=list)
    seconds: float = 0.0
    config: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def final_val(self) -> float:
        return self.val_loss[-1] if self.val_loss else float("nan")


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Linear warmup, then cosine decay from ``lr`` to ``min_lr_ratio * lr``."""
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / max(cfg.warmup, 1)
    progress = (step - cfg.warmup) / max(cfg.steps - cfg.warmup, 1)
    progress = min(progress, 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    min_lr = cfg.lr * cfg.min_lr_ratio
    return min_lr + coeff * (cfg.lr - min_lr)


@torch.no_grad()
def estimate_loss(model: GPT, data: Tensor, cfg: TrainConfig, *,
                  generator: torch.Generator | None = None) -> float:
    from .data import get_batch

    was_training = model.training
    model.eval()
    total = 0.0
    for _ in range(cfg.eval_batches):
        x, y = get_batch(data, cfg.batch_size, model.config.block_size, generator=generator)
        _, loss = model(x, y)
        total += float(loss.detach())
    if was_training:
        model.train()
    return total / cfg.eval_batches


def train(model: GPT, train_data: Tensor, val_data: Tensor | None = None,
          cfg: TrainConfig | None = None) -> History:
    from .data import get_batch

    cfg = cfg or TrainConfig()
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    eval_gen_seed = cfg.seed + 7_919      # a *fixed* eval draw, so val is comparable
    opt = model.configure_optimizers(weight_decay=cfg.weight_decay, lr=cfg.lr)
    hist = History(config=asdict(cfg) | {"model": model.config.summary()})
    model.train()
    t0 = time.perf_counter()
    for step in range(cfg.steps):
        lr = lr_at(step, cfg)
        for group in opt.param_groups:
            group["lr"] = lr
        x, y = get_batch(train_data, cfg.batch_size, model.config.block_size, generator=gen)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        hist.step.append(step)
        hist.train_loss.append(float(loss.detach()))
        hist.lr.append(lr)
        last = step == cfg.steps - 1
        if val_data is not None and (step % cfg.eval_every == 0 or last):
            v = estimate_loss(model, val_data, cfg,
                              generator=torch.Generator().manual_seed(eval_gen_seed))
            hist.eval_step.append(step)
            hist.val_loss.append(v)
            if cfg.log:
                print(f"  step {step:4d}  lr {lr:.2e}  train {float(loss.detach()):.4f}  val {v:.4f}")
    hist.seconds = time.perf_counter() - t0
    return hist


def smoothed(values: list[float], window: int = 20) -> list[float]:
    """Trailing mean — the only honest way to compare two noisy loss curves."""
    out = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        chunk = values[lo : i + 1]
        out.append(sum(chunk) / len(chunk))
    return out
