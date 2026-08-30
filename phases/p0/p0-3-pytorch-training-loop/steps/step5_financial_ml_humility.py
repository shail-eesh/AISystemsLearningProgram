#!/usr/bin/env python3
"""Step 5 — the first lesson in financial-ML humility.

Run:  python3 steps/step5_financial_ml_humility.py   (~1 minute)

Step 4 drove a tiny network to 100% training accuracy on 64 rows. This step
runs the *same* model on the *same* data with an honest protocol and watches
the number collapse to chance.

Three things to take away, in order of how much money they save you:

1. **A falling training loss is not evidence of anything.** It is evidence
   that gradient descent works, which was never in doubt.
2. **The baseline is the finding.** "55% accuracy" means nothing until you
   know that always predicting "up" scores 55% on the same test window.
3. **Leakage does not look like cheating; it looks like a good result.** One
   `center=True` on a rolling window takes this model from chance to 81%.

The labels here come from a synthetic random walk, so the true signal is
exactly zero and every point above chance is noise or leakage. Real markets are
not *quite* that clean — but they are much closer to it than a first backtest
suggests, and a protocol that cannot detect zero signal cannot be trusted to
measure small signal.
"""

import _bootstrap  # noqa: F401
import numpy as np
from p0_3_training import build_dataset, chronological_split
from p0_3_training.experiments import honest_evaluation, leakage_demonstrations


def demo_the_data_is_almost_nothing() -> None:
    data = build_dataset()
    tr, va, te = chronological_split(data)
    print(f"  rows: {len(data)}  (train {len(tr)} / val {len(va)} / test {len(te)})")
    print(f"  features: {', '.join(data.feature_names)}")
    print(f"  base rate (fraction of up-days): overall {data.base_rate:.3f}, "
          f"test window {te.y.mean():.3f}")
    print(f"  splits are chronological — no date appears in two splits: "
          f"{not (set(tr.dates) & set(te.dates))}")


def demo_honest_number() -> None:
    r = honest_evaluation(epochs=80)
    print(f"    train accuracy {r['train_accuracy']:.3f}   "
          f"val {r['val_accuracy']:.3f}   test {r['test_accuracy']:.3f}")
    print(f"    always-predict-up baseline on the test window: {r['baseline_always_up']:.3f}")
    print(f"    majority-class baseline:                       {r['baseline_majority_class']:.3f}")
    print(f"    beats the majority baseline? {r['beats_majority_baseline']}")
    print(f"    z vs a coin flip: {r['z_vs_coinflip']:+.2f} "
          f"(|z| > 2 would be interesting; this is not)")
    print(f"    test loss {r['test_loss']:.4f} — a coin flip scores ln(2) = {np.log(2):.4f}")
    print("  The model is not broken. The task has no signal, and the honest protocol")
    print("  is what lets you see that instead of shipping it.")


def demo_leakage() -> None:
    r = leakage_demonstrations(epochs=80)
    print(f"    honest test accuracy: {r['honest_test_accuracy']:.3f}")
    for v in r["variants"]:
        print(f"    {v['name']:38s} {v['accuracy']:.3f}  ({v['uplift']:+.3f})  [{v['severity']}]")
    print("\n  The centred rolling window is one keyword away from the causal version:")
    print("      ret.rolling(3).mean()                # uses t-2, t-1, t   -- fine")
    print("      ret.rolling(3, center=True).mean()   # uses t-1, t,  t+1  -- the label")
    print("  Nobody writes `center=True` intending to cheat. It is a smoothing default.")


def demo_what_would_convince_you() -> None:
    print("  Before believing any result on market data, ask for:")
    print("    - a baseline (majority class, buy-and-hold, last-value) on the SAME window")
    print("    - a walk-forward evaluation, not one split (T48 makes this point-in-time)")
    print("    - costs: spread, slippage, borrow. A 52% hit rate does not survive them.")
    print("    - the same protocol on shuffled labels — it must score chance")
    print("    - out-of-sample data the model was never tuned against, used ONCE")
    print("  Phase 4's eval harness (T27) turns this checklist into code.")


if __name__ == "__main__":
    print("the dataset:")
    demo_the_data_is_almost_nothing()
    print("the honest number:")
    demo_honest_number()
    print("how the number gets inflated:")
    demo_leakage()
    print("what would actually convince you:")
    demo_what_would_convince_you()
    print("\nstep 5 OK")
