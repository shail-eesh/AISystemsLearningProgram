"""The two experiments that make Phase 0 land.

**Experiment A — overfit a tiny batch.** Take 64 rows and drive the training
accuracy to 100%. This is the standard first diagnostic for *any* new training
setup: if a model with enough capacity cannot memorise a handful of examples,
the bug is in your loop, not your data. It proves the plumbing works and
proves nothing else.

**Experiment B — the honest split.** The same model, trained on a
chronological split with the scaler fit on training data only, evaluated
against two baselines: always-predict-up, and a coin flip. Next-day equity
direction from eight technical features is very close to unpredictable, so the
honest score sits at the base rate. Feeling that gap — 100% on the memorised
batch, chance on the real task — is the entire point of P0.3.

**Experiment C — how the number gets inflated.** Two one-line changes, each
producing a better-looking score with no more information: shuffling the split,
and fitting the scaler before splitting.
"""

from __future__ import annotations

import numpy as np
import torch

from .data import Dataset, build_dataset, chronological_split, standardise
from .loop import TrainConfig, evaluate, seed_everything, train
from .model import TinyMLP


def _as_tensors(d: Dataset) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.from_numpy(np.ascontiguousarray(d.X)),
        torch.from_numpy(np.ascontiguousarray(d.y)).unsqueeze(1),
    )


def _subset(d: Dataset, idx: np.ndarray) -> Dataset:
    return Dataset(X=d.X[idx], y=d.y[idx], dates=d.dates[idx], symbols=d.symbols[idx],
                   feature_names=d.feature_names)


def overfit_tiny_batch(n: int = 64, epochs: int = 400, hidden: int = 64) -> dict:
    """Experiment A. Memorise `n` rows; report how quickly and how completely."""
    seed_everything(1729)
    data = build_dataset()
    tiny = _subset(data, np.arange(min(n, len(data))))
    (scaled,), _, _ = standardise(tiny)

    model = TinyMLP(len(data.feature_names), hidden=hidden)
    cfg = TrainConfig(epochs=epochs, batch_size=n, lr=3e-3, hidden=hidden, seed=1729)
    model, history = train(model, scaled, None, cfg)
    loss, acc = evaluate(model, *_as_tensors(scaled))

    reached = next((i + 1 for i, a in enumerate(history.train_acc) if a >= 1.0), None)
    return {
        "rows": len(tiny),
        "epochs": epochs,
        "final_train_loss": loss,
        "final_train_accuracy": acc,
        "epochs_to_perfect": reached,
        "verdict": "loop is wired correctly" if acc >= 0.99 else "the loop has a bug",
    }


def honest_evaluation(epochs: int = 80) -> dict:
    """Experiment B. The number you are allowed to quote."""
    seed_everything(1729)
    data = build_dataset()
    tr, va, te = chronological_split(data)
    (tr_s, va_s, te_s), _, _ = standardise(tr, va, te)

    model = TinyMLP(len(data.feature_names), hidden=32, dropout=0.1)
    cfg = TrainConfig(epochs=epochs, batch_size=64, lr=1e-3, weight_decay=1e-2, seed=1729)
    model, history = train(model, tr_s, va_s, cfg)

    tr_loss, tr_acc = evaluate(model, *_as_tensors(tr_s))
    va_loss, va_acc = evaluate(model, *_as_tensors(va_s))
    te_loss, te_acc = evaluate(model, *_as_tensors(te_s))

    always_up = float(te.y.mean())
    majority = max(always_up, 1.0 - always_up)
    n = len(te)
    # Binomial standard error of a coin flip at this sample size.
    se = float(np.sqrt(0.25 / n))
    return {
        "rows": {"train": len(tr), "val": len(va), "test": len(te)},
        "train_accuracy": tr_acc,
        "val_accuracy": va_acc,
        "test_accuracy": te_acc,
        "test_loss": te_loss,
        "train_loss": tr_loss,
        "val_loss": va_loss,
        "baseline_always_up": always_up,
        "baseline_majority_class": majority,
        "coinflip_se": se,
        "z_vs_coinflip": (te_acc - 0.5) / se,
        "beats_majority_baseline": bool(te_acc > majority),
        "history_tail": {k: v[-5:] for k, v in history.as_dict().items()},
    }


def _centred_lookahead_feature(data: Dataset) -> Dataset:
    """Append one feature computed with a CENTRED window.

    `rolling(3, center=True).mean()` of the daily return averages t-1, t and
    **t+1**. It looks like ordinary smoothing, it is one keyword away from the
    causal version, and it contains the answer.
    """
    import pandas as pd

    from common.data import load_ohlcv

    prices = load_ohlcv()
    frames = []
    for _sym, g in prices.groupby("symbol", sort=True):
        g = g.sort_values("date", kind="stable")
        ret = g["close"].pct_change()
        frames.append(
            pd.DataFrame(
                {
                    "symbol": g["symbol"].to_numpy(),
                    "date": g["date"].to_numpy(),
                    "smoothed_ret": ret.rolling(3, center=True).mean().to_numpy(),
                }
            )
        )
    leaked = pd.concat(frames, ignore_index=True)
    key = pd.DataFrame({"symbol": data.symbols, "date": data.dates})
    merged = key.merge(leaked, on=["symbol", "date"], how="left")
    col = merged["smoothed_ret"].to_numpy(dtype=np.float32)
    col = np.nan_to_num(col, nan=0.0)
    return Dataset(
        X=np.column_stack([data.X, col]).astype(np.float32),
        y=data.y, dates=data.dates, symbols=data.symbols,
        feature_names=[*data.feature_names, "smoothed_ret_CENTRED"],
    )


def _fit_and_score(tr: Dataset, va: Dataset, te: Dataset, epochs: int) -> float:
    model = TinyMLP(tr.X.shape[1], hidden=32, dropout=0.1)
    model, _ = train(model, tr, va, TrainConfig(epochs=epochs, weight_decay=1e-2, seed=1729))
    return evaluate(model, *_as_tensors(te))[1]


def leakage_demonstrations(epochs: int = 80) -> dict:
    """Experiment C. Three ways to make the same model look better than it is.

    The headline result is deliberately uneven: one leak is catastrophic and
    two barely register **on this dataset**. That asymmetry is the lesson.
    A leak that does not move the number on a random walk will move it a great
    deal on data with real autocorrelation — "it did not change my score" is
    evidence about your data, not about your methodology.
    """
    seed_everything(1729)
    data = build_dataset()
    honest = honest_evaluation(epochs=epochs)
    baseline = honest["test_accuracy"]

    # -- C1: a feature computed with a centred (future-touching) window ------
    leaked = _centred_lookahead_feature(data)
    l_tr, l_va, l_te = chronological_split(leaked)
    (l_tr, l_va, l_te), _, _ = standardise(l_tr, l_va, l_te)
    centred_acc = _fit_and_score(l_tr, l_va, l_te, epochs)

    # -- C2: shuffled split (the train_test_split default) -------------------
    rng = np.random.default_rng(1729)
    perm = rng.permutation(len(data))
    cut_a, cut_b = int(0.7 * len(data)), int(0.85 * len(data))
    sh = [_subset(data, perm[s]) for s in
          (slice(None, cut_a), slice(cut_a, cut_b), slice(cut_b, None))]
    (sh_tr, sh_va, sh_te), _, _ = standardise(*sh)
    shuffled_acc = _fit_and_score(sh_tr, sh_va, sh_te, epochs)

    # -- C3: scaler fitted on everything, then split ------------------------
    tr, va, te = chronological_split(data)
    mu = data.X.mean(axis=0).astype(np.float32)
    sd = data.X.std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0).astype(np.float32)

    def leak(d: Dataset) -> Dataset:
        return Dataset(X=(d.X - mu) / sd, y=d.y, dates=d.dates,
                       symbols=d.symbols, feature_names=d.feature_names)

    scaler_acc = _fit_and_score(leak(tr), leak(va), leak(te), epochs)

    return {
        "honest_test_accuracy": baseline,
        "variants": [
            {
                "name": "centred rolling feature (uses t+1)",
                "accuracy": centred_acc,
                "uplift": centred_acc - baseline,
                "severity": "catastrophic",
            },
            {
                "name": "shuffled train/test split",
                "accuracy": shuffled_acc,
                "uplift": shuffled_acc - baseline,
                "severity": "silent here (this series has no autocorrelation to exploit)",
            },
            {
                "name": "scaler fitted before the split",
                "accuracy": scaler_acc,
                "uplift": scaler_acc - baseline,
                "severity": "silent here (eight features, stationary-ish scales)",
            },
        ],
        "note": (
            "None of these variants is given more information about tomorrow by the "
            "problem statement; all three obtain some. The centred window obtains a lot. "
            "The other two obtain almost nothing ON THIS DATA — which is a fact about a "
            "synthetic random walk, not a licence to ship them."
        ),
    }
