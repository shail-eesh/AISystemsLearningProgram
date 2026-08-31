#!/usr/bin/env python3
"""Step 3 — the full block, and the first real training run on the tape.

Run:  python3 steps/step3_block_and_first_train.py   (~3 min on 2 CPU cores)

The block is attention + MLP, each wrapped in a norm and a residual add. Three
things get demonstrated here, all measured:

1. where the parameters actually live (the MLP, roughly 2:1 over attention);
2. a model trained for 500 steps on the market tape, sampled, so you can read
   whether the grammar was learned;
3. pre-norm vs post-norm across depth and learning rate — where the textbook
   answer is wrong at small scale and right at large scale.
"""

import _bootstrap  # noqa: F401
import torch
from t4_transformer import GPT, GPTConfig, TrainConfig, char_dataset, train
from t4_transformer.blocks import Block

TAPE_HEAD = "2024-03-0"


def where_the_parameters_are() -> None:
    cfg = GPTConfig(vocab_size=66, block_size=64, n_layer=4, n_head=4, n_embd=128)
    model = GPT(cfg)
    rep = model.parameter_report()
    total = rep["total"]
    print("  parameter budget of a 4-layer, d=128 model:\n")
    for k in ("embedding", "attention", "mlp", "norm", "head"):
        bar = "#" * round(40 * rep[k] / total)
        print(f"    {k:<10} {rep[k]:>8,}  {100 * rep[k] / total:>5.1f}%  {bar}")
    print(f"    {'total':<10} {total:>8,}")
    print("\n  The head is 0 because it is *tied* to the embedding — the same tensor,")
    print("  read forwards to embed and backwards to unembed.")
    print(f"  Attention:MLP ratio is 1 : {rep['mlp'] / rep['attention']:.2f}. Attention moves")
    print("  information between positions; the MLP does the thinking, with the budget.")


def one_block_is_two_residual_writes() -> None:
    print("\n  what a block does to the residual stream (pre-norm):")
    torch.manual_seed(0)
    blk = Block(n_embd=64, n_head=4, block_size=16).eval()
    x = torch.randn(1, 8, 64)
    with torch.no_grad():
        after_attn = x + blk.attn(blk.ln1(x))
        out = after_attn + blk.mlp(blk.ln2(after_attn))
    print(f"    ||x||             {float(x.norm()):8.3f}")
    print(f"    ||x + attn(..)||  {float(after_attn.norm()):8.3f}   "
          f"(attention added {float((after_attn - x).norm()):.3f})")
    print(f"    ||.. + mlp(..)||  {float(out.norm()):8.3f}   "
          f"(the MLP added {float((out - after_attn).norm()):.3f})")
    print("\n  Nothing was replaced. Each sublayer *adds* to a stream that runs unbroken")
    print("  from the embeddings to the final norm — which is why gradients reach layer 0.")


def train_the_tape() -> None:
    print("\n  training a 4-layer model on the market tape (500 steps):")
    tr, va, vocab = char_dataset()
    print(f"    corpus {len(tr) + len(va):,} chars, vocab {vocab.size}, "
          f"train/val {len(tr):,}/{len(va):,}")
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=vocab.size, block_size=64, n_layer=4,
                          n_head=4, n_embd=128, dropout=0.0))
    hist = train(model, tr, va, TrainConfig(steps=500, batch_size=16, eval_every=100))
    print(f"    {model.num_params():,} params, {hist.seconds:.1f}s")
    print(f"\n    {'step':>6} {'val loss':>10} {'bits/char':>11}")
    for s, v in zip(hist.eval_step, hist.val_loss, strict=True):
        print(f"    {s:>6} {v:>10.4f} {v / 0.6931:>11.3f}")
    ids = torch.tensor([vocab.encode(TAPE_HEAD)])
    print("\n  greedy continuation of a date prefix:\n")
    for line in vocab.decode(model.generate(ids, 130, greedy=True)[0]).splitlines():
        print(f"      {line}")
    print("\n  sampled at temperature 0.8, top-k 20:\n")
    g = torch.Generator().manual_seed(4)
    sampled = model.generate(ids, 130, temperature=0.8, top_k=20, generator=g)
    for line in vocab.decode(sampled[0]).splitlines():
        print(f"      {line}")
    print("\n  Read the *shape*, not the numbers: field order, two decimal places, the")
    print("  space-delimited O/H/L/C/V grammar. A char model that has learned the")
    print("  language of the tape has learned something a bag of unigrams cannot.")
    print("  Note the greedy run repeating one digit forever: the argmax of a language")
    print("  model is not fluent text. Step 6 is entirely about fixing that.")


def pre_norm_vs_post_norm() -> None:
    print("\n  pre-norm vs post-norm — 2x2 over depth and learning rate, 150 steps each,")
    print("  identical seed and data, no warmup (warmup is what usually hides this):\n")
    tr, va, vocab = char_dataset()
    print(f"    {'depth':>6} {'lr':>8} {'pre-norm':>10} {'post-norm':>11}   verdict")
    for depth in (6, 12):
        for lr in (3e-3, 1e-2):
            finals = []
            for post in (False, True):
                torch.manual_seed(0)
                cfg = GPTConfig(vocab_size=vocab.size, block_size=64, n_layer=depth,
                                n_head=4, n_embd=128)
                model = GPT(cfg)
                for blk in model.blocks:
                    blk.post_norm = post
                hist = train(model, tr, va, TrainConfig(steps=150, batch_size=16,
                                                        eval_every=150, warmup=0, lr=lr))
                finals.append(hist.final_val)
            pre, post_v = finals
            verdict = "post-norm wins" if post_v < pre - 0.05 else (
                "post-norm stalls" if post_v > pre + 0.3 else "tie")
            print(f"    {depth:>6} {lr:>8.0e} {pre:>10.4f} {post_v:>11.4f}   {verdict}")
    print("\n  The honest result: at 6 layers and a mild learning rate post-norm is")
    print("  *better*. Push either knob — deeper, or hotter — and it stops training,")
    print("  parking near a degenerate solution while pre-norm keeps descending.")
    print("  Pre-norm is not free quality; it is robustness. It buys you the right to")
    print("  stack 12, 48, 96 layers and raise the learning rate without a babysitter,")
    print("  and that is the trade every model after GPT-2 took.")


if __name__ == "__main__":
    print(__doc__)
    where_the_parameters_are()
    one_block_is_two_residual_writes()
    train_the_tape()
    pre_norm_vs_post_norm()
