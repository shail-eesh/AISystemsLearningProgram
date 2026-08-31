"""The scaling mini-study: three model sizes, one corpus, one honest chart.

The claim being tested is the least glamorous and most useful one in machine
learning: **more parameters, trained the same way on the same data, reach a
lower loss** — and they do it along a curve regular enough to extrapolate from.
That regularity is what lets somebody decide to spend a month of GPU time before
spending it.

What this study can and cannot say, stated up front because the numbers do not
say it themselves:

* It **can** say that the ordering holds and by how much, at this corpus size,
  with this tokenizer, under a matched schedule.
* It **cannot** say anything about the compute-optimal frontier. Chinchilla's
  ~20 tokens per parameter would need 119M tokens for the 5M rung; the corpus
  has 2.4M. Every rung here is deep in the *data-limited* regime, which is
  exactly where a larger model starts to lose its advantage — so if the ordering
  survives here it is a conservative result, and if it inverts, that inversion is
  the finding.
* The two-point power-law fit is a **description of three points**, not a law.
  It is reported with its residuals so it cannot be mistaken for one.
"""

from __future__ import annotations

import math
import time

from .config import CPU_RUNGS, Rung
from .harness import TrainSpec, Trainer, train_rung


def run_study(shards, *, rungs: list[Rung] | None = None, steps: int = 1200,
              batch_size: int = 16, vocab_size: int = 3495, seed: int = 15,
              verbose: bool = True) -> dict:
    """Train each rung under an identical schedule and collect the curves."""
    rungs = rungs or CPU_RUNGS
    train_shard, _ = shards
    out: dict = {"steps": steps, "batch_size": batch_size, "rungs": [],
                 "train_tokens": len(train_shard)}
    for rung in rungs:
        spec = TrainSpec(steps=steps, batch_size=batch_size, lr=3e-3, warmup=100,
                         eval_every=max(steps // 6, 1), eval_batches=24,
                         checkpoint_every=0, seed=seed)
        t0 = time.perf_counter()
        model, state, trainer = train_rung(rung, shards, spec=spec, vocab_size=vocab_size)
        final = trainer.evaluate(batches=48)
        params = rung.parameters(vocab_size)
        row = {
            "name": rung.name,
            "params": params,
            "n_layer": rung.n_layer,
            "n_embd": rung.n_embd,
            "tokens_seen": state.tokens,
            "tokens_per_param": state.tokens / params,
            "chinchilla_tokens": rung.chinchilla_tokens(vocab_size),
            "chinchilla_fraction": state.tokens / rung.chinchilla_tokens(vocab_size),
            "final_train_loss": sum(h["loss"] for h in state.history[-50:]) / 50,
            "val_loss": final["loss"],
            "val_perplexity": final["perplexity"],
            "eval_curve": [{"step": e["step"], "loss": round(e["loss"], 4)}
                           for e in state.evals],
            "seconds": time.perf_counter() - t0,
        }
        out["rungs"].append(row)
        if verbose:
            print(f"    {rung.name:<15} {params:>10,} params  "
                  f"val {row['val_loss']:.4f}  ppl {row['val_perplexity']:7.2f}  "
                  f"({row['seconds']:.0f}s)")
    out |= analyse(out["rungs"])
    return out


def analyse(rows: list[dict]) -> dict:
    """Ordering, margins, generalisation gap, and a two-point power-law fit."""
    by_size = sorted(rows, key=lambda r: r["params"])
    losses = [r["val_loss"] for r in by_size]
    monotone = all(a > b for a, b in zip(losses, losses[1:], strict=True))

    # L(N) = a * N^(-b), fitted on the smallest and largest rung only, then
    # *checked* on the middle one. A fit through all three points would hide
    # its own error.
    small, large = by_size[0], by_size[-1]
    b = (math.log(small["val_loss"]) - math.log(large["val_loss"])) / (
        math.log(large["params"]) - math.log(small["params"]))
    a = small["val_loss"] * small["params"] ** b
    residuals = [{"name": r["name"], "predicted": a * r["params"] ** -b,
                  "actual": r["val_loss"],
                  "relative_error": (a * r["params"] ** -b - r["val_loss"]) / r["val_loss"]}
                 for r in by_size]

    return {
        "ordering_holds": monotone,
        "smallest": small["name"],
        "largest": large["name"],
        "loss_improvement": small["val_loss"] - large["val_loss"],
        "perplexity_ratio": small["val_perplexity"] / large["val_perplexity"],
        "params_ratio": large["params"] / small["params"],
        "power_law": {"a": a, "exponent_b": b,
                      "note": "fitted on the two end rungs, checked on the middle one"},
        "power_law_residuals": residuals,
        "max_relative_fit_error": max(abs(r["relative_error"]) for r in residuals),
        "generalisation_gap": [
            {"name": r["name"], "train": r["final_train_loss"], "val": r["val_loss"],
             "gap": r["val_loss"] - r["final_train_loss"]}
            for r in by_size],
    }


def format_table(study: dict) -> str:
    lines = [f"{'rung':<16} {'params':>10} {'tok/param':>10} {'train':>8} {'val':>8} "
             f"{'ppl':>9} {'gap':>7} {'sec':>6}"]
    for r in sorted(study["rungs"], key=lambda x: x["params"]):
        gap = r["val_loss"] - r["final_train_loss"]
        lines.append(f"{r['name']:<16} {r['params']:>10,} {r['tokens_per_param']:>10.2f} "
                     f"{r['final_train_loss']:>8.4f} {r['val_loss']:>8.4f} "
                     f"{r['val_perplexity']:>9.2f} {gap:>+7.4f} {r['seconds']:>6.0f}")
    return "\n".join(lines)


def extrapolate(study: dict, params: int) -> float:
    """What the fitted curve predicts for an unseen size — e.g. the 40M rung.

    Printed with a warning wherever it is used: extrapolating a two-point fit by
    an order of magnitude is how people talk themselves into month-long runs.
    """
    pl = study["power_law"]
    return pl["a"] * params ** -pl["exponent_b"]


__all__ = ["Trainer", "analyse", "extrapolate", "format_table", "run_study"]
