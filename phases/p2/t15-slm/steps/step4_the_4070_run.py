#!/usr/bin/env python3
"""Step 4 — planning the 4070 run, which is most of doing it.

Run:  python3 steps/step4_the_4070_run.py     (~15 s; trains nothing)

This step does not train the 40M model — that happens on your RTX 4070, via
`gpu-runner/t15_alphaslm_40m.py`. What it does is the part people skip: work out
in advance what the run will cost and whether it can possibly succeed.

Three questions, answered with arithmetic rather than optimism:

1. **Does it fit in 12 GB?** Weights + gradients + two Adam moments, plus
   activations. The optimiser state is the surprise: it is three times the
   weights.
2. **How long will it take?** Measured on this CPU, scaled by an assumed
   throughput ratio, and reported as a range with the assumption named.
3. **Is there enough data?** Chinchilla wants ~20 tokens per parameter. The
   corpus has 2.4M. This is where the run stops being a compute problem.
"""

import subprocess
import sys

import _bootstrap  # noqa: F401
import torch
from t15_alphaslm import CPU_RUNGS, LADDER, TrainSpec, Trainer, ensure_shards
from t4_transformer import GPT

GPU_RUNNER = "gpu-runner/t15_alphaslm_40m.py"
# A 4070 is roughly 30 TFLOP/s fp32 with tensor cores idle; these two cores do
# perhaps 0.13 TFLOP/s sustained on this workload. The ratio is a planning
# figure, deliberately conservative, and named so it can be argued with.
ASSUMED_GPU_SPEEDUP = 120.0


def the_memory_arithmetic():
    print("  where 12 GB goes, per rung:\n")
    print(f"    {'rung':<16} {'params':>12} {'weights':>10} {'grads':>10} "
          f"{'Adam':>10} {'total':>10}")
    for rung in LADDER.values():
        n = rung.parameters(3495)
        w = n * 4
        print(f"    {rung.name:<16} {n:>12,} {w / 1e6:>9.1f}M {w / 1e6:>9.1f}M "
              f"{2 * w / 1e6:>9.1f}M {4 * w / 1e6:>9.1f}M")
    print("\n    Four copies of every parameter before a single activation exists:")
    print("    the weight, its gradient, and Adam's two moment estimates. This is why")
    print("    'the model is only 40M parameters, it will fit' is the wrong sentence —")
    print("    and why every memory-saving technique in Phase 5 and 6 (8-bit Adam,")
    print("    LoRA, quantization, activation checkpointing) targets one of these four.")


def activations_are_the_other_half():
    print("\n  activations, which scale with batch x context and not with parameters:\n")
    print(f"    {'rung':<16} {'ctx':>6} {'batch':>6} {'attention matrix':>18} "
          f"{'residual stream':>17}")
    for rung in LADDER.values():
        for batch in (8, 24):
            attn = batch * rung.n_layer * rung.n_head * rung.block_size ** 2 * 4
            resid = batch * rung.block_size * rung.n_layer * 10 * rung.n_embd * 4
            print(f"    {rung.name:<16} {rung.block_size:>6} {batch:>6} "
                  f"{attn / 1e9:>17.2f}G {resid / 1e9:>16.2f}G")
    print("\n    The attention matrix column is quadratic in context and it is the")
    print("    reason a 40M model at 512 tokens costs more activation memory than a")
    print("    40M model at 128 tokens costs in total. T7 (Flash Attention) exists to")
    print("    delete that column by never materialising the matrix.")


def measure_here_and_extrapolate():
    print("\n  measuring throughput on this machine, then extrapolating (carefully):\n")
    train_shard, val_shard, meta = ensure_shards(block_size=128)
    rung = CPU_RUNGS[-1]
    torch.manual_seed(15)
    model = GPT(rung.gpt_config(meta["vocab_size"]))
    trainer = Trainer(model, train_shard, val_shard,
                      TrainSpec(steps=12, batch_size=8, warmup=2, eval_every=100,
                                checkpoint_every=0))
    state = trainer.train()
    tps = state.tokens / state.seconds
    print(f"    {rung.name}: {tps:,.0f} tokens/s on {torch.get_num_threads()} CPU threads")
    print(f"    assumed 4070 speedup for this workload: {ASSUMED_GPU_SPEEDUP:.0f}x "
          f"-> {tps * ASSUMED_GPU_SPEEDUP:,.0f} tokens/s\n")
    big = LADDER["alphaslm-40m"]
    ratio = big.parameters(3495) / rung.parameters(meta["vocab_size"])
    est_tps = tps * ASSUMED_GPU_SPEEDUP / ratio
    print(f"    {'target tokens':>16} {'hours at that rate':>20}")
    for tokens in (50_000_000, 200_000_000, 491_520_000):
        print(f"    {tokens:>16,} {tokens / est_tps / 3600:>20.1f}")
    print("\n    Every number in that table rests on one assumed constant, stated")
    print("    above so you can replace it with a measurement after your first hour on")
    print("    the 4070. A plan whose assumption is written down is a plan; a plan")
    print("    whose assumption is implicit is a wish.")


def the_data_wall():
    print("\n  and the thing compute cannot fix:\n")
    train_shard, _, meta = ensure_shards(block_size=128)
    corpus = len(train_shard)
    print(f"    corpus: {corpus:,} tokens\n")
    print(f"    {'rung':<16} {'params':>12} {'chinchilla':>14} {'epochs to reach it':>20}")
    for rung in LADDER.values():
        target = rung.chinchilla_tokens(meta["vocab_size"])
        print(f"    {rung.name:<16} {rung.parameters(meta['vocab_size']):>12,} "
              f"{target:>14,} {target / corpus:>20.0f}")
    print("\n    The 40M rung would need 333 passes over this corpus to hit its")
    print("    compute-optimal token budget. It will memorise long before that. The")
    print("    honest reading: **this corpus is the binding constraint, not the GPU**,")
    print("    and the useful next move is T23 (synthetic data) and T38 (curation),")
    print("    not a bigger model. Run the 40M anyway — watching held-out perplexity")
    print("    flatten while training loss keeps falling is worth one night of")
    print("    electricity, and it is the clearest lesson in the phase.")


def the_command():
    print("\n  what to actually run on the 4070:\n")
    print(f"    python3 {GPU_RUNNER} --dry-run")
    print(f"    python3 {GPU_RUNNER} --rung alphaslm-15m --hours 1")
    print(f"    python3 {GPU_RUNNER} --rung alphaslm-40m --hours 8")
    print(f"    python3 {GPU_RUNNER} --rung alphaslm-40m --resume\n")
    print("    The runner imports this topic's harness — same loop, same checkpoints,")
    print("    same shards. It refuses to start if the plan does not fit in the")
    print("    device's memory, tells you exactly what to change on an OOM (halve the")
    print("    batch, double the accumulation — same update, half the memory) and")
    print("    writes bench/gpu_results.json in the same shape as the CPU bench so the")
    print("    two lanes are directly comparable.\n")
    print("    Its plan, from here:")
    root = _bootstrap.ROOT
    try:
        out = subprocess.run([sys.executable, str(root / GPU_RUNNER), "--dry-run"],
                             capture_output=True, text=True, timeout=300)
        for line in out.stdout.splitlines():
            print(f"    {line}")
    except Exception as exc:                      # pragma: no cover - convenience only
        print(f"      (could not run the planner here: {exc})")


if __name__ == "__main__":
    print(__doc__)
    the_memory_arithmetic()
    activations_are_the_other_half()
    measure_here_and_extrapolate()
    the_data_wall()
    the_command()
