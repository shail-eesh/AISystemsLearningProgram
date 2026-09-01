#!/usr/bin/env python3
"""Step 5 — evaluating a language model without fooling yourself.

Run:  python3 steps/step5_evaluate.py            (~9 min)
      python3 steps/step5_evaluate.py --quick    (~2 min)

Four ways to be wrong about a perplexity number, each demonstrated:

1. **Report one number for a mixed corpus.** Perplexity is dominated by whatever
   register is most predictable. Broken out by document type, the same model can
   be at 2 on the price tape and 12 on filing prose, and only the second is a
   statement about language.
2. **Evaluate on overlapping random windows.** Random draws re-count some tokens
   and skip others, so the number wobbles between runs of the same model.
3. **Compare across tokenizers.** Perplexity is per *token*; a bigger vocabulary
   gets a better number for free. Bits per character is the comparable unit.
4. **Stop at the number.** Read what the model writes. A model can have a good
   perplexity and produce text that is obviously broken in a way no aggregate
   will show you.
"""

import argparse
import math

import _bootstrap  # noqa: F401
import numpy as np
import torch
from t15_alphaslm import (
    CPU_RUNGS,
    Trainer,
    TrainSpec,
    build_corpus,
    compare_models,
    ensure_shards,
    load_tokenizer,
    perplexity_by_tag,
    perplexity_on_documents,
    sample_commentary,
    split_documents,
    train_rung,
)


def train_the_ladder(shards, steps: int):
    print(f"  training the three CPU rungs for {steps} steps each:\n")
    models = {}
    for rung in CPU_RUNGS:
        spec = TrainSpec(steps=steps, batch_size=16, lr=3e-3, warmup=min(100, steps // 8),
                         eval_every=steps, eval_batches=16, checkpoint_every=0, seed=15)
        model, state, trainer = train_rung(rung, shards, spec=spec)
        models[rung.name] = model
        print(f"    {rung.name:<16} {rung.parameters(3495):>10,} params, "
              f"{state.seconds:>5.0f}s, final train loss "
              f"{state.history[-1]['loss']:.4f}")
    return models


def one_number_hides_the_answer(models, tokenizer, val_docs):
    print("\n  1 · the same model, split by register:\n")
    biggest = list(models)[-1]
    by_tag = perplexity_by_tag(models[biggest], tokenizer, val_docs)
    overall = perplexity_on_documents(models[biggest], tokenizer, val_docs, max_docs=200)
    print(f"    {biggest}\n")
    print(f"    {'register':<16} {'documents':>10} {'tokens':>9} {'loss':>8} {'ppl':>9}")
    for name, s in by_tag.items():
        print(f"    {name:<16} {s['documents']:>10} {s['tokens']:>9,} "
              f"{s['loss']:>8.4f} {s['perplexity']:>9.2f}")
    print(f"    {'all held-out':<16} {overall['documents']:>10} "
          f"{overall['tokens']:>9,} {overall['loss']:>8.4f} "
          f"{overall['perplexity']:>9.2f}")
    print("\n    The held-out split is filings by construction, so 'all' and 'filing'")
    print("    agree here — which is the point of having chosen the split that way.")
    print("    Evaluate on a mixed corpus instead and the tape's near-deterministic")
    print("    lines would drag the headline number down while telling you nothing.")


def random_windows_wobble(shards):
    print("\n  2 · why the evaluation walks the split instead of sampling it:\n")
    train_shard, val_shard = shards
    print("    Random windows overlap, so some tokens are counted several times and")
    print("    some not at all. Same model, same data, five different draws:\n")
    torch.manual_seed(15)
    from t4_transformer import GPT
    from t15_alphaslm.config import LADDER

    model = GPT(LADDER["alphaslm-0.6m"].gpt_config(3495))
    trainer = Trainer(model, train_shard, val_shard,
                      TrainSpec(steps=150, batch_size=16, warmup=20, eval_every=1000,
                                checkpoint_every=0))
    trainer.train()
    model.eval()
    randoms = []
    with torch.no_grad():
        for seed in range(5):
            rng = np.random.default_rng(seed)
            total = 0.0
            for _ in range(12):
                x, y = val_shard.batch(16, rng)
                total += float(model(x, y)[1])
            randoms.append(total / 12)
    seq = [trainer.evaluate(batches=12)["loss"] for _ in range(5)]
    print(f"    random windows:     {[f'{v:.4f}' for v in randoms]}")
    print(f"      spread {max(randoms) - min(randoms):.4f}")
    print(f"    sequential windows: {[f'{v:.4f}' for v in seq]}")
    print(f"      spread {max(seq) - min(seq):.4f}")
    print("\n    The second is deterministic because it walks the split once, in order,")
    print("    with no overlap. Random draws are right for *training* and wrong for a")
    print("    number you are going to put in a table.")


def the_ladder_compared(models, tokenizer, val_docs, max_docs: int):
    print("\n  3 · held-out perplexity across the ladder:\n")
    result = compare_models(models, tokenizer, val_docs, max_docs=max_docs)
    print(f"    {'rung':<16} {'loss':>8} {'perplexity':>12} {'bits/char':>11}")
    for name, s in result["scores"].items():
        print(f"    {name:<16} {s['loss']:>8.4f} {s['perplexity']:>12.3f} "
              f"{s['bits_per_char']:>11.4f}")
    print("\n    margins:")
    for pair, m in result["margins"].items():
        print(f"      {pair:<36} {m['loss_delta']:>+8.4f} nats  "
              f"{m['perplexity_ratio']:>6.3f}x")
    print("\n    bits/char is the column to trust across a tokenizer change: perplexity")
    print("    is per token, so a model with a 50k vocabulary looks better than one with")
    print("    a 3.5k vocabulary on identical text. Every one of these shares FinTok, so")
    print("    here the two columns tell the same story — but the habit is the point.")
    return result


def read_what_it_writes(models, tokenizer):
    print("\n  4 · and then actually read the output:\n")
    prompts = [
        "<|commentary|> Market commentary for 2024-06-12.",
        "<|filing|> COASTBANK 10-K FY2024Q1 risk factors.",
    ]
    for name, model in models.items():
        print(f"    --- {name} ---")
        for prompt in prompts:
            text = sample_commentary(model, tokenizer, prompt=prompt,
                                     max_new_tokens=70, temperature=0.8, top_p=0.9)
            body = text.replace(prompt, "").strip().replace("\n", " ")
            print(f"      {prompt}")
            print(f"        {body[:260]}")
        print()
    print("    AlphaDesk is a fictional educational simulation. This is a language")
    print("    model producing text about invented issuers — not a market view, not")
    print("    advice, and never routed to anything that could act on it.\n")
    print("    What to look for, in order: is it the right register? are the fields in")
    print("    the right order? are the numbers plausible for that issuer? does it")
    print("    contradict itself inside one sentence? Perplexity answers none of those")
    print("    and every one of them is visible in ten seconds of reading.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    steps = 250 if args.quick else 1200
    max_docs = 40 if args.quick else 150

    print(__doc__)
    train_shard, val_shard, meta = ensure_shards(block_size=128)
    tokenizer = load_tokenizer()
    _, val_docs = split_documents(build_corpus())
    print(f"  held-out set: {len(val_docs):,} documents, "
          f"{sum(len(d) for d in val_docs):,} characters\n")

    models = train_the_ladder((train_shard, val_shard), steps)
    one_number_hides_the_answer(models, tokenizer, val_docs)
    random_windows_wobble((train_shard, val_shard))
    self = the_ladder_compared(models, tokenizer, val_docs, max_docs)
    read_what_it_writes(models, tokenizer)
    biggest = list(models)[-1]
    smallest = list(models)[0]
    ratio = (self["scores"][smallest]["perplexity"] /
             self["scores"][biggest]["perplexity"])
    print(f"\n  headline: {biggest} reaches perplexity "
          f"{self['scores'][biggest]['perplexity']:.3f} on held-out filings, "
          f"{ratio:.3f}x better than {smallest} "
          f"({math.log(ratio):+.4f} nats).")


if __name__ == "__main__":
    main()
