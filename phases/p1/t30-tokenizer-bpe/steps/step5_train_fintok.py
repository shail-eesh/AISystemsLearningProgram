#!/usr/bin/env python3
"""Step 5 — train FinTok, and measure what a domain vocabulary is actually worth.

Run:  python3 steps/step5_train_fintok.py

The comparison is controlled: the *same* trainer, the *same* requested vocabulary
size, the *same* pre-tokenizer, trained on financial text in one arm and general
English in the other, then both measured on held-out **financial** text. Only
the training domain varies.

Where the two arms end up with different achieved vocabulary sizes (they do —
corpora saturate at different points) both are truncated to the smaller merge
count, which is valid because any prefix of a rank-ordered merge list is itself
a complete BPE.
"""

import _bootstrap  # noqa: F401
from t30_fintok import (
    Tokenizer,
    compression,
    corpus_stats,
    financial_corpus,
    general_corpus,
    split,
)


def main() -> None:
    fin, gen = financial_corpus(docs=3000), general_corpus(docs=3000)
    fin_train, fin_hold = split(fin)
    gen_train, _ = split(gen)
    print(f"  financial corpus: {corpus_stats(fin_train)}")
    print(f"  general corpus:   {corpus_stats(gen_train)}")

    fintok, stats = Tokenizer.train(fin_train, 4096, pattern="finance", name="fintok")
    gentok, _ = Tokenizer.train(gen_train, 4096, pattern="finance", name="gentok")
    print(f"\n  requested 4096 merges -> FinTok learned {len(fintok.merges)}, "
          f"GeneralTok learned {len(gentok.merges)}")

    n = min(len(fintok.merges), len(gentok.merges))
    a = compression(fintok.truncate(n), fin_hold)
    b = compression(gentok.truncate(n), fin_hold)
    print(f"\n  matched at {n} merges, measured on held-out FINANCIAL text:")
    print(f"    FinTok      {a['bytes_per_token']:.3f} bytes/token")
    print(f"    GeneralTok  {b['bytes_per_token']:.3f} bytes/token")
    print("    raw bytes   1.000 bytes/token (the floor)")
    print(f"\n    -> FinTok is {a['bytes_per_token'] / b['bytes_per_token']:.2f}x more "
          "compact on the domain it was trained for.")

    print("\n  the first merges FinTok learns (the domain showing up in the vocabulary):")
    for h in stats.history[:14]:
        print(f"    rank {h['rank']:>3}  {h['piece']!r}")


if __name__ == "__main__":
    main()
