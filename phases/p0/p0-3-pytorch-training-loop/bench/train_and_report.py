#!/usr/bin/env python3
"""Verification benchmark for P0.3 — can it memorise, and is it honest?

    python3 phases/p0/p0-3-pytorch-training-loop/bench/train_and_report.py

Two conditions, both of which must hold for the topic to be done:

* **A. The loop works.** 64 rows, ~1k parameters: training accuracy must reach
  100%. If it cannot memorise a tiny batch, the loop has a bug.
* **B. The result is honest.** On a chronological split with the scaler fitted
  on training data only, test accuracy must be statistically indistinguishable
  from a coin flip (|z| < 2). The labels come from a synthetic random walk, so
  a model that "beats" this is measuring leakage, not skill — and the benchmark
  fails loudly in that case rather than celebrating.

A third section quantifies three leaks, one of which is worth 30 accuracy
points.
"""

from __future__ import annotations

import json
import pathlib
import platform
import sys
import time
from datetime import UTC, datetime

TOPIC = pathlib.Path(__file__).resolve().parent.parent
REPO = TOPIC.parents[2]
sys.path[:0] = [str(REPO), str(TOPIC / "src")]

import torch  # noqa: E402
from p0_3_training.experiments import (  # noqa: E402
    honest_evaluation,
    leakage_demonstrations,
    overfit_tiny_batch,
)

Z_LIMIT = 2.0


def main() -> int:
    t0 = time.perf_counter()
    overfit = overfit_tiny_batch(n=64, epochs=400, hidden=64)
    honest = honest_evaluation(epochs=80)
    leaks = leakage_demonstrations(epochs=80)
    elapsed = time.perf_counter() - t0

    a_pass = overfit["final_train_accuracy"] >= 0.99
    b_pass = abs(honest["z_vs_coinflip"]) < Z_LIMIT
    centred = next(v for v in leaks["variants"] if "centred" in v["name"])
    c_pass = centred["uplift"] > 0.10        # the leak must be visible, or the demo is broken
    passed = a_pass and b_pass and c_pass

    results = {
        "topic": "P0.3",
        "benchmark": "overfit-a-tiny-batch + honest chronological evaluation + leakage audit",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "elapsed_seconds": round(elapsed, 1),
        "a_overfit_tiny_batch": {**overfit, "passed": a_pass},
        "b_honest_evaluation": {
            **{k: v for k, v in honest.items() if k != "history_tail"},
            "z_limit": Z_LIMIT,
            "passed": b_pass,
            "interpretation": (
                "Indistinguishable from a coin flip, and below the majority-class "
                "baseline. That is the correct answer for a synthetic random walk."
            ),
        },
        "c_leakage_audit": {**leaks, "passed": c_pass},
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "passed": passed,
    }
    out = pathlib.Path(__file__).parent / "results.json"
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print(f"A. overfit 64 rows: train acc {overfit['final_train_accuracy']:.3f} "
          f"(100% at epoch {overfit['epochs_to_perfect']})  -> {'PASS' if a_pass else 'FAIL'}")
    print(f"B. honest split:    test acc {honest['test_accuracy']:.3f}  "
          f"baseline {honest['baseline_majority_class']:.3f}  "
          f"z={honest['z_vs_coinflip']:+.2f}  -> {'PASS' if b_pass else 'FAIL'}")
    print("C. leakage audit:")
    for v in leaks["variants"]:
        print(f"     {v['name']:38s} {v['accuracy']:.3f} ({v['uplift']:+.3f})")
    print(f"   -> {'PASS' if c_pass else 'FAIL'}")
    print(f"total {elapsed:.1f}s; wrote {out.relative_to(REPO)}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
