#!/usr/bin/env python3
"""Step 3 — the scaling mini-study: does bigger actually win, and by how much?

Run:  python3 steps/step3_scaling_ministudy.py     (~22 min on 2 CPU cores)
      python3 steps/step3_scaling_ministudy.py --quick   (~4 min)

Three models, 0.6M / 1.8M / 5.9M parameters, identical corpus, identical
schedule, identical seed. Nothing varies but size.

Read the result with the caveat attached: the corpus is 2.4M tokens, and
Chinchilla's rule of thumb wants ~20 tokens per parameter — 119M for the largest
rung. Every model here is deep in the **data-limited** regime, which is exactly
where extra parameters stop paying. If the ordering survives *here*, it is a
conservative result. If it inverts, that inversion is the finding, not a
failure.
"""

import argparse
import json
import pathlib

import _bootstrap  # noqa: F401
from t15_alphaslm import CPU_RUNGS, ensure_shards, extrapolate, format_table, run_study
from t15_alphaslm.config import LADDER


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--steps", type=int, default=None)
    args = ap.parse_args()
    steps = args.steps or (250 if args.quick else 1200)

    print(__doc__)
    train_shard, val_shard, meta = ensure_shards(block_size=128)
    print(f"  corpus: {len(train_shard):,} train tokens, {len(val_shard):,} val tokens, "
          f"vocab {meta['vocab_size']}")
    print(f"  schedule: {steps} steps, batch 16, block 128, lr 3e-3 cosine, "
          f"warmup 100, clip 1.0\n")

    study = run_study((train_shard, val_shard), rungs=CPU_RUNGS, steps=steps,
                      vocab_size=meta["vocab_size"])

    print("\n  the table:\n")
    for line in format_table(study).splitlines():
        print(f"    {line}")

    print("\n  the claim:\n")
    print(f"    ordering holds (bigger -> lower val loss): {study['ordering_holds']}")
    print(f"    {study['largest']} beats {study['smallest']} by "
          f"{study['loss_improvement']:.4f} nats "
          f"({study['perplexity_ratio']:.2f}x lower perplexity) "
          f"for {study['params_ratio']:.1f}x the parameters")

    print("\n  how far under Chinchilla each rung is:\n")
    print(f"    {'rung':<16} {'tokens seen':>14} {'chinchilla':>14} {'fraction':>10}")
    for r in sorted(study["rungs"], key=lambda x: x["params"]):
        print(f"    {r['name']:<16} {r['tokens_seen']:>14,} "
              f"{r['chinchilla_tokens']:>14,} {r['chinchilla_fraction']:>9.1%}")
    print("\n    Single-digit percentages. These models are starved of data, not of")
    print("    parameters, which is the regime almost every hobby project lives in and")
    print("    almost no scaling chart is drawn from.")

    print("\n  the power law, fitted on the two ends and *checked* on the middle:\n")
    pl = study["power_law"]
    print(f"    L(N) = {pl['a']:.4g} * N^-{pl['exponent_b']:.4f}\n")
    print(f"    {'rung':<16} {'predicted':>10} {'actual':>10} {'rel. error':>11}")
    for r in study["power_law_residuals"]:
        print(f"    {r['name']:<16} {r['predicted']:>10.4f} {r['actual']:>10.4f} "
              f"{r['relative_error']:>+10.2%}")
    print("\n    Three points and two of them were used to draw the line, so the middle")
    print("    row is the only honest evidence here. It is a description, not a law.")
    for name in ("alphaslm-15m", "alphaslm-40m"):
        rung = LADDER[name]
        n = rung.parameters(meta["vocab_size"])
        print(f"    extrapolated to {name} ({n:,} params): "
              f"{extrapolate(study, n):.4f}")
    print("\n    That extrapolation is exactly the kind of number people commit a")
    print("    month of GPU time to. It is a two-point fit stretched by an order of")
    print("    magnitude on a corpus none of those models could saturate. Step 4 runs")
    print("    the real thing on the 4070 and writes down what actually happened.")

    print("\n  train vs held-out (is the biggest model just memorising?):\n")
    print(f"    {'rung':<16} {'train':>9} {'val':>9} {'gap':>9}")
    for g in study["generalisation_gap"]:
        print(f"    {g['name']:<16} {g['train']:>9.4f} {g['val']:>9.4f} {g['gap']:>+9.4f}")
    print("\n    A gap that grows faster than the val loss falls is the signal to stop")
    print("    adding parameters and start adding data — the moment T38 (curation) and")
    print("    T23 (synthetic data) become the interesting topics rather than the dull")
    print("    ones.")

    out = pathlib.Path(__file__).resolve().parent.parent / "bench" / "scaling_study.json"
    if not args.quick:
        out.write_text(json.dumps(study, indent=1) + "\n")
        print(f"\n  written to {out.relative_to(out.parents[4])}")


if __name__ == "__main__":
    main()
