#!/usr/bin/env python3
"""Step 6 — the decoder is trained; now decide what to do with the logits.

Run:  python3 steps/step6_sampling.py     (~1 min)

A trained model gives you a distribution over the next token. Turning that into
text is a separate design problem, and it is where "the model is bad" usually
turns out to mean "the sampler is wrong". Four knobs, measured on the same
model, with a grammar checker as the referee.
"""

import re
from collections import Counter

import _bootstrap  # noqa: F401
import torch
from t4_transformer import (
    GPT,
    GPTConfig,
    TrainConfig,
    apply_temperature,
    char_dataset,
    top_k_filter,
    top_p_filter,
    train,
)
from t4_transformer.sampling import distribution_stats

LINE = re.compile(
    r"^\d{4}-\d{2}-\d{2} [A-Z]+ O \d+\.\d{2} H \d+\.\d{2} L \d+\.\d{2} C \d+\.\d{2} V \d+$"
)


def a_trained_model():
    tr, va, vocab = char_dataset()
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=vocab.size, block_size=64, n_layer=4, n_head=4,
                          n_embd=128, position="rope"))
    hist = train(model, tr, va, TrainConfig(steps=600, batch_size=16, lr=3e-3,
                                            eval_every=600, seed=0))
    print(f"  a 4-layer RoPE model, 600 steps, val loss {hist.final_val:.4f}\n")
    return model, vocab


def two_rows_one_certain_one_not(model, vocab) -> None:
    """The argument for nucleus sampling, on two real rows of one real model."""
    prompts = {
        "certain  ('2024-05-14 ALPHAINF' -> ?)": "2024-05-14 ALPHAINF",
        "uncertain ('2024-05-14 ' -> which symbol?)": "2024-05-14 ",
    }
    for label, prompt in prompts.items():
        ids = torch.tensor([vocab.encode(prompt)])
        with torch.no_grad():
            logits = model(ids)[0][:, -1, :]
        probs = logits.softmax(-1)[0]
        top = probs.topk(5)
        shown = "  ".join(f"{vocab.itos[int(i)]!r}={float(pv):.3f}"
                          for pv, i in zip(top.values, top.indices, strict=True))
        print(f"  {label}")
        print(f"    top-5: {shown}")
        print(f"    {'sampler':<22} {'kept':>5} {'entropy':>9} {'eff. choices':>13}")
        for name, lg in [
            ("raw", logits),
            ("T = 0.8", apply_temperature(logits, 0.8)),
            ("T = 1.5", apply_temperature(logits, 1.5)),
            ("top-k 5", top_k_filter(logits, 5)),
            ("top-k 20", top_k_filter(logits, 20)),
            ("top-p 0.9", top_p_filter(logits, 0.9)),
            ("T = 0.8 + top-p 0.9", top_p_filter(apply_temperature(logits, 0.8), 0.9)),
        ]:
            st = distribution_stats(lg)
            kept = int((lg > float("-inf")).sum())
            print(f"    {name:<22} {kept:>5} {st['entropy_nats']:>9.3f} "
                  f"{st['effective_choices']:>13.2f}")
        print()
    print("  This is the whole case for nucleus sampling, in two tables.")
    print("  Where the model is certain, top-k 20 keeps 20 candidates it has already")
    print("  ruled out — a fixed budget spends probability mass the model did not want")
    print("  to spend, and every one of those 19 alternatives is a typo waiting to be")
    print("  drawn. top-p keeps one. Where the model is genuinely unsure, top-p opens")
    print("  up instead. The budget follows the model's own confidence, which is the")
    print("  thing a constant k cannot do.")
    print("  ('kept' = characters still drawable; 'eff. choices' = exp(entropy), how")
    print("  many the model is really choosing between.)\n")


def the_sweep(model, vocab) -> None:
    print("\n  600 sampled characters per setting, scored by the tape grammar:\n")
    ids = torch.tensor([vocab.encode("2024-05-14 ")])
    settings = [
        ("greedy", dict(greedy=True)),
        ("T = 0.2", dict(temperature=0.2)),
        ("T = 0.5", dict(temperature=0.5)),
        ("T = 0.8", dict(temperature=0.8)),
        ("T = 1.0", dict(temperature=1.0)),
        ("T = 1.5", dict(temperature=1.5)),
        ("T = 1.0, top-k 5", dict(temperature=1.0, top_k=5)),
        ("T = 1.0, top-k 20", dict(temperature=1.0, top_k=20)),
        ("T = 1.0, top-p 0.9", dict(temperature=1.0, top_p=0.9)),
        ("T = 1.2, top-p 0.9", dict(temperature=1.2, top_p=0.9)),
        ("T = 0.8, top-k 20", dict(temperature=0.8, top_k=20)),
    ]
    print(f"    {'sampler':<20} {'well-formed':>12} {'distinct lines':>15} "
          f"{'longest repeat':>15}")
    for name, kw in settings:
        g = torch.Generator().manual_seed(11)
        text = vocab.decode(model.generate(ids, 600, generator=g, **kw)[0])
        lines = [ln for ln in text.split("\n")[1:-1] if ln.strip()]
        wf = sum(bool(LINE.match(ln)) for ln in lines) / max(len(lines), 1)
        distinct = len(set(lines)) / max(len(lines), 1)
        counts = Counter(text[i:i + 8] for i in range(len(text) - 8))
        repeat = counts.most_common(1)[0][1] if counts else 0
        print(f"    {name:<20} {wf:>11.0%} {distinct:>14.0%} {repeat:>15}")
    print("\n    Greedy scores well on grammar and badly on everything else: it emits")
    print("    the same safe line over and over, which the 'distinct' and 'longest")
    print("    repeat' columns catch. High temperature does the opposite — variety")
    print("    bought by breaking the grammar. The useful settings sit in between,")
    print("    and *that* is why nucleus sampling with a mild temperature is the")
    print("    default in every serving stack you will meet.")


def side_by_side(model, vocab) -> None:
    print("\n  the same prompt, three samplers:\n")
    ids = torch.tensor([vocab.encode("2024-05-14 ")])
    for name, kw in (("greedy", dict(greedy=True)),
                     ("T = 1.5 (no filter)", dict(temperature=1.5)),
                     ("T = 0.8, top-p 0.9", dict(temperature=0.8, top_p=0.9))):
        g = torch.Generator().manual_seed(3)
        text = vocab.decode(model.generate(ids, 420, generator=g, **kw)[0])
        print(f"    --- {name} ---")
        for line in text.splitlines()[:6]:
            print(f"      {line}")
        print()


if __name__ == "__main__":
    print(__doc__)
    model, vocab = a_trained_model()
    two_rows_one_certain_one_not(model, vocab)
    the_sweep(model, vocab)
    side_by_side(model, vocab)
