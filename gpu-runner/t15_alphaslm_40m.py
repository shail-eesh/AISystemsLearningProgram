#!/usr/bin/env python3
"""Pretrain AlphaSLM-15M / AlphaSLM-40M on the RTX 4070.

This is the same harness the CPU rungs use. Nothing about the model, the data
pipeline or the training loop changes between lanes — what changes is the size
of the rung, the batch that fits, and the availability of mixed precision.

    python3 gpu-runner/t15_alphaslm_40m.py --dry-run
    python3 gpu-runner/t15_alphaslm_40m.py --rung alphaslm-40m --hours 6
    python3 gpu-runner/t15_alphaslm_40m.py --rung alphaslm-40m --resume

The plan it prints before starting — parameters, tokens, VRAM estimate, steps,
time — is the part worth reading. A run you cannot predict the cost of is a run
you should not start.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import platform
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
for src in sorted((ROOT / "phases").glob("*/*/src")):
    sys.path.insert(0, str(src))
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from t4_transformer import GPT  # noqa: E402
from t15_alphaslm import LADDER, Trainer, TrainSpec, ensure_shards  # noqa: E402
from t15_alphaslm.evaluate import (  # noqa: E402
    perplexity_by_tag,
    sample_commentary,
)

BYTES_PER_PARAM_TRAINING = 16   # fp32 weight + grad + two Adam moments


def estimate_vram(params: int, batch: int, block: int, layers: int, d: int,
                  heads: int) -> dict:
    """A back-of-the-envelope that is honest about being one.

    Optimiser state dominates for a model this size; activations dominate once
    the context is long. Both are estimated, the total is padded 30% for
    fragmentation and cuBLAS workspaces, and the number is a *plan*, not a
    guarantee — which is why the script also catches OOM and halves the batch.
    """
    weights = params * BYTES_PER_PARAM_TRAINING
    # per layer, per token: residual stream a few times over, plus the attention
    # matrix if it is materialised (ours is, until T7 replaces it)
    activations = batch * block * (layers * (10 * d) * 4 + layers * heads * block * 4)
    total = weights + activations
    return {
        "weights_and_optimizer_bytes": weights,
        "activation_bytes": activations,
        "estimated_total_bytes": total,
        "estimated_total_gb": total / 1e9,
        "with_30pc_headroom_gb": total * 1.3 / 1e9,
    }


def plan(rung, vocab_size: int, batch: int, micro: int, steps: int,
         corpus_tokens: int = 0) -> dict:
    params = rung.parameters(vocab_size)
    tokens = steps * batch * micro * rung.block_size
    est = estimate_vram(params, batch, rung.block_size, rung.n_layer, rung.n_embd,
                        rung.n_head)
    return {
        "rung": rung.name,
        "parameters": params,
        "block_size": rung.block_size,
        "batch_size": batch,
        "micro_batches": micro,
        "steps": steps,
        "tokens": tokens,
        "tokens_per_parameter": tokens / params,
        "chinchilla_tokens": rung.chinchilla_tokens(vocab_size),
        "chinchilla_fraction": tokens / rung.chinchilla_tokens(vocab_size),
        "corpus_tokens": corpus_tokens,
        "epochs": tokens / corpus_tokens if corpus_tokens else None,
        "vram": est,
    }


def print_plan(p: dict) -> None:
    print(f"\n  plan for {p['rung']}:\n")
    print(f"    parameters            {p['parameters']:>14,}")
    print(f"    context               {p['block_size']:>14,}")
    print(f"    batch x micro         {p['batch_size']:>10} x {p['micro_batches']}")
    print(f"    steps                 {p['steps']:>14,}")
    print(f"    tokens                {p['tokens']:>14,}")
    print(f"    tokens per parameter  {p['tokens_per_parameter']:>14.2f}")
    print(f"    Chinchilla target     {p['chinchilla_tokens']:>14,} "
          f"({p['chinchilla_fraction']:.1%} of it)")
    print(f"    estimated VRAM        {p['vram']['with_30pc_headroom_gb']:>13.2f} GB "
          f"(weights+Adam {p['vram']['weights_and_optimizer_bytes'] / 1e9:.2f} GB, "
          f"activations {p['vram']['activation_bytes'] / 1e9:.2f} GB)")
    if p.get("epochs"):
        print(f"    corpus                {p['corpus_tokens']:>14,} tokens "
              f"= {p['epochs']:.1f} epochs")
        if p["epochs"] > 10:
            print("\n    !! That is a lot of epochs. Past a handful of passes a model this")
            print("       size stops learning the language and starts memorising the")
            print("       corpus: training loss keeps falling, held-out perplexity does")
            print("       not. The fix is more data, not fewer parameters — which is what")
            print("       T23 (synthetic data) and T38 (curation) are for. Run it anyway")
            print("       if you want to *see* the divergence; the eval curve in")
            print("       results.json is where it shows up.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", default="alphaslm-40m", choices=sorted(LADDER))
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--micro", type=int, default=2)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--hours", type=float, default=8.0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "phases/p2/t15-slm/bench/gpu_results.json"))
    args = ap.parse_args()

    rung = LADDER[args.rung]
    train_shard, val_shard, meta = ensure_shards(block_size=rung.block_size)
    p = plan(rung, meta["vocab_size"], args.batch, args.micro, args.steps,
             corpus_tokens=len(train_shard))
    print(f"  AlphaSLM GPU runner · {platform.node()} · torch {torch.__version__}")
    print_plan(p)

    if args.dry_run:
        print("\n  --dry-run: nothing trained. Re-run without it when the plan looks right.")
        return 0

    if not torch.cuda.is_available():
        print("\n  No CUDA device. This runner is the 4070 lane; the CPU rungs are")
        print("  `python3 phases/p2/t15-slm/steps/step3_scaling_ministudy.py`.")
        return 2
    device = "cuda"
    name = torch.cuda.get_device_name(0)
    total_vram = torch.cuda.get_device_properties(0).total_memory
    print(f"\n  device: {name}, {total_vram / 1e9:.1f} GB")
    if p["vram"]["with_30pc_headroom_gb"] * 1e9 > total_vram:
        print("  The plan does not fit. Lower --batch or raise --micro (same update,")
        print("  less memory) and run --dry-run again.")
        return 2

    run_dir = ROOT / "phases/p2/t15-slm/src/t15_alphaslm/artifacts/runs" / rung.name
    torch.manual_seed(15)
    model = GPT(rung.gpt_config(meta["vocab_size"]))
    spec = TrainSpec(steps=args.steps, batch_size=args.batch, micro_batches=args.micro,
                     lr=args.lr, warmup=min(500, args.steps // 20), eval_every=500,
                     eval_batches=32, checkpoint_every=500, seed=15,
                     max_seconds=args.hours * 3600, log_every=50)
    trainer = Trainer(model, train_shard, val_shard, spec, run_dir=run_dir, device=device)
    if args.resume and trainer.checkpoint_path().exists():
        state = trainer.load()
        print(f"  resuming from step {state.step:,} ({state.tokens:,} tokens seen)")

    t0 = time.perf_counter()
    try:
        state = trainer.train(verbose=True)
    except torch.cuda.OutOfMemoryError:
        print("\n  OOM. Halve --batch and double --micro: the optimiser sees the same")
        print("  batch, the GPU holds half as much. Then --resume.")
        return 3
    elapsed = time.perf_counter() - t0

    final = trainer.evaluate(batches=64)
    tokenizer = __import__("t15_alphaslm").load_tokenizer()
    from t15_alphaslm.corpus import build_corpus, split_documents

    _, val_docs = split_documents(build_corpus())
    by_tag = perplexity_by_tag(model, tokenizer, val_docs, device=device)
    sample = sample_commentary(model, tokenizer, device=device)

    results = {
        "topic": "T15",
        "lane": "gpu",
        "device": name,
        "torch": torch.__version__,
        "plan": p,
        "steps_completed": state.step,
        "tokens_seen": state.tokens,
        "wall_clock_seconds": elapsed,
        "tokens_per_second": state.tokens / max(elapsed, 1e-9),
        "val_loss": final["loss"],
        "val_perplexity": final["perplexity"],
        "val_bits_per_token": final["loss"] / math.log(2),
        "perplexity_by_tag": by_tag,
        "sample": sample,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        "eval_curve": state.evals,
    }
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1) + "\n")
    print(f"\n  done: {state.step:,} steps, {state.tokens:,} tokens, "
          f"val loss {final['loss']:.4f} (ppl {final['perplexity']:.2f})")
    print(f"  peak VRAM {results['peak_vram_bytes'] / 1e9:.2f} GB, "
          f"{results['tokens_per_second']:,.0f} tokens/s")
    print(f"  written to {out}")
    print("\n  Commit that file and set T15's bench row to done in EXECUTION/status.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
