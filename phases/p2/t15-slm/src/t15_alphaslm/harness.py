"""The pretraining harness: schedule, clipping, checkpoints, resume, logging.

T4's ``train()`` is a teaching loop — 40 lines, no state, run it and read the
number. This is the version you leave running overnight on a 4070, and the
difference is entirely about **what happens when it stops**.

Four things it does that the teaching loop does not:

* **Checkpoints that actually resume.** Model weights are the easy part. A
  checkpoint that omits the optimiser state restarts Adam's moment estimates
  from zero, and the first few dozen steps after a "resume" are effectively
  untrained ones. This saves model, optimiser, step, and both RNG states, and
  ``test_resume_is_bit_identical`` proves that resuming at step *k* produces the
  same weights as never stopping.
* **Gradient accumulation.** The 4070 has 12 GB. A batch that does not fit is
  run as several micro-batches whose gradients are summed before one optimiser
  step — mathematically the same update, arbitrarily less memory. The loss is
  scaled by ``1/accum`` so the gradient is a mean and not a sum, which is the
  detail that silently multiplies your effective learning rate if you skip it.
* **A local metrics log.** One JSON object per line: step, loss, lr, grad norm,
  tokens seen, seconds, memory. No service, no account, no network — but
  everything a wandb chart would have shown, and greppable.
* **A wall-clock budget.** ``max_seconds`` stops cleanly at a checkpoint rather
  than being killed at an arbitrary point, which matters when the run is a
  scheduled job.
"""

from __future__ import annotations

import json
import math
import pathlib
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch


@dataclass
class TrainSpec:
    """Everything that is not the model or the data."""

    steps: int = 2000
    batch_size: int = 16
    micro_batches: int = 1          # gradient accumulation
    lr: float = 3e-3
    min_lr_ratio: float = 0.1
    warmup: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_every: int = 200
    eval_batches: int = 16
    checkpoint_every: int = 500
    seed: int = 15
    max_seconds: float | None = None
    log_every: int = 50

    @property
    def tokens_per_step(self) -> int:
        return self.batch_size * self.micro_batches


def lr_at(step: int, spec: TrainSpec) -> float:
    """Linear warmup, then cosine decay to ``min_lr_ratio * lr``.

    Warmup exists because Adam's second-moment estimate is meaningless before it
    has seen gradients: at step 0 the update is essentially ``sign(g) * lr``, and
    a full learning rate there is a confident step in an arbitrary direction.
    """
    if step < spec.warmup:
        return spec.lr * (step + 1) / max(spec.warmup, 1)
    progress = min((step - spec.warmup) / max(spec.steps - spec.warmup, 1), 1.0)
    floor = spec.lr * spec.min_lr_ratio
    return floor + 0.5 * (1.0 + math.cos(math.pi * progress)) * (spec.lr - floor)


@dataclass
class RunState:
    """What a checkpoint has to contain for a resume to be indistinguishable."""

    step: int = 0
    tokens: int = 0
    seconds: float = 0.0
    history: list[dict] = field(default_factory=list)
    evals: list[dict] = field(default_factory=list)


class Trainer:
    """Train one model on one packed shard set, restartably."""

    def __init__(self, model, train_shard, val_shard=None, spec: TrainSpec | None = None,
                 *, run_dir: pathlib.Path | str | None = None, device: str = "cpu") -> None:
        self.model = model
        self.train_shard = train_shard
        self.val_shard = val_shard
        self.spec = spec or TrainSpec()
        self.device = device
        self.run_dir = pathlib.Path(run_dir) if run_dir else None
        if self.run_dir:
            self.run_dir.mkdir(parents=True, exist_ok=True)
        self.opt = model.configure_optimizers(weight_decay=self.spec.weight_decay,
                                              lr=self.spec.lr)
        self.state = RunState()
        self._rng = np.random.default_rng(self.spec.seed)
        torch.manual_seed(self.spec.seed)

    # -- checkpointing ----------------------------------------------------
    def checkpoint_path(self, name: str = "ckpt.pt") -> pathlib.Path:
        if not self.run_dir:
            raise ValueError("this Trainer has no run_dir, so it cannot checkpoint")
        return self.run_dir / name

    def save(self, name: str = "ckpt.pt") -> pathlib.Path:
        path = self.checkpoint_path(name)
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.opt.state_dict(),
            "state": asdict(self.state),
            "spec": asdict(self.spec),
            "config": asdict(self.model.config),
            "numpy_rng": self._rng.bit_generator.state,
            "torch_rng": torch.get_rng_state(),
        }, path)
        return path

    def load(self, name: str = "ckpt.pt") -> RunState:
        ckpt = torch.load(self.checkpoint_path(name), map_location=self.device,
                          weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.opt.load_state_dict(ckpt["optimizer"])
        self.state = RunState(**ckpt["state"])
        self._rng = np.random.default_rng()
        self._rng.bit_generator.state = ckpt["numpy_rng"]
        torch.set_rng_state(ckpt["torch_rng"])
        return self.state

    # -- evaluation -------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, shard=None, batches: int | None = None) -> dict:
        """Mean loss over *non-overlapping* windows — see ``sequential_batches``."""
        shard = shard or self.val_shard
        if shard is None:
            raise ValueError("no validation shard")
        batches = batches or self.spec.eval_batches
        was_training = self.model.training
        self.model.eval()
        total, n = 0.0, 0
        for x, y in shard.sequential_batches(self.spec.batch_size, limit=batches):
            x, y = x.to(self.device), y.to(self.device)
            _, loss = self.model(x, y)
            total += float(loss.detach())
            n += 1
            if n >= batches:
                break
        if was_training:
            self.model.train()
        mean = total / max(n, 1)
        return {"loss": mean, "perplexity": math.exp(mean), "batches": n}

    # -- the loop ---------------------------------------------------------
    def _log(self, record: dict) -> None:
        if self.run_dir:
            with (self.run_dir / "metrics.jsonl").open("a") as fh:
                fh.write(json.dumps(record) + "\n")

    def train(self, *, until: int | None = None, verbose: bool = False) -> RunState:
        """Run to ``until`` (default: ``spec.steps``), resuming wherever we are."""
        spec = self.spec
        target = until or spec.steps
        block = self.model.config.block_size
        self.model.to(self.device).train()
        t0 = time.perf_counter()
        while self.state.step < target:
            step = self.state.step
            lr = lr_at(step, spec)
            for group in self.opt.param_groups:
                group["lr"] = lr
            self.opt.zero_grad(set_to_none=True)
            total_loss = 0.0
            for _ in range(spec.micro_batches):
                x, y = self.train_shard.batch(spec.batch_size, self._rng)
                x, y = x.to(self.device), y.to(self.device)
                _, loss = self.model(x, y)
                # /micro_batches so the accumulated gradient is the mean over the
                # full batch, not its sum. Forget this and the effective learning
                # rate is multiplied by the accumulation factor.
                (loss / spec.micro_batches).backward()
                total_loss += float(loss.detach()) / spec.micro_batches
            grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                             spec.grad_clip))
            self.opt.step()

            self.state.step += 1
            self.state.tokens += spec.tokens_per_step * block
            self.state.seconds = self.state.seconds + (time.perf_counter() - t0)
            t0 = time.perf_counter()
            record = {"step": self.state.step, "loss": total_loss, "lr": lr,
                      "grad_norm": grad_norm, "tokens": self.state.tokens,
                      "seconds": round(self.state.seconds, 2)}
            self.state.history.append(record)
            if self.state.step % spec.log_every == 0 or self.state.step == target:
                self._log(record)
                if verbose:
                    print(f"    step {self.state.step:5d}  loss {total_loss:.4f}  "
                          f"lr {lr:.2e}  |g| {grad_norm:.2f}")

            if self.val_shard is not None and (self.state.step % spec.eval_every == 0
                                               or self.state.step == target):
                ev = self.evaluate() | {"step": self.state.step,
                                        "tokens": self.state.tokens}
                self.state.evals.append(ev)
                self._log({"eval": ev})
                if verbose:
                    print(f"      eval  loss {ev['loss']:.4f}  ppl {ev['perplexity']:.2f}")

            if self.run_dir and spec.checkpoint_every and (
                    self.state.step % spec.checkpoint_every == 0):
                self.save()
            if spec.max_seconds and self.state.seconds >= spec.max_seconds:
                if self.run_dir:
                    self.save()
                break
        if self.run_dir:
            self.save()
        return self.state


def train_rung(rung, shards, *, spec: TrainSpec | None = None, run_dir=None,
               vocab_size: int = 3495, device: str = "cpu", verbose: bool = False):
    """Build one rung of the ladder and train it. Returns ``(model, state)``."""
    import sys
    from pathlib import Path

    t4 = Path(__file__).resolve().parents[5] / "phases/p2/t4-transformer/src"
    if str(t4) not in sys.path:
        sys.path.insert(0, str(t4))
    from t4_transformer import GPT

    train_shard, val_shard = shards
    torch.manual_seed((spec or TrainSpec()).seed)
    model = GPT(rung.gpt_config(vocab_size))
    trainer = Trainer(model, train_shard, val_shard, spec, run_dir=run_dir, device=device)
    state = trainer.train(verbose=verbose)
    return model, state, trainer
