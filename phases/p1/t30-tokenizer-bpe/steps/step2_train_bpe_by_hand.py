#!/usr/bin/env python3
"""Step 2 — the merge loop, every merge shown, on a corpus you can read.

Run:  python3 steps/step2_train_bpe_by_hand.py

    while the vocabulary is too small:
        find the most frequent adjacent pair
        mint a new symbol for it
        replace every occurrence

That is the whole algorithm (Sennrich et al. 2015). Everything else is
efficiency. Watch it run on nine words.
"""

from collections import Counter

import _bootstrap  # noqa: F401
from t30_fintok import chunk_counts
from t30_fintok.bpe import _merge_in_word, _word_pairs, train_bpe

TINY = ["low low low low low lower lower newest newest newest newest newest newest widest widest widest"]


def by_hand() -> None:
    counts = chunk_counts(TINY, "gpt2")
    words = {tuple(w): c for w, c in counts.items()}
    print("  starting word types (chunk -> frequency):")
    for w, c in words.items():
        print(f"    {bytes(w)!r:<12} x{c}")

    vocab = {i: bytes([i]) for i in range(256)}
    for step in range(6):
        pairs = Counter()
        for w, c in words.items():
            for p in _word_pairs(w):
                pairs[p] += c
        if not pairs:
            break
        best, n = pairs.most_common(1)[0]
        new_id = len(vocab)
        vocab[new_id] = vocab[best[0]] + vocab[best[1]]
        print(f"\n  merge {step}: {vocab[best[0]]!r} + {vocab[best[1]]!r} "
              f"-> {vocab[new_id]!r}  (count {n})")
        words = {_merge_in_word(w, best, new_id): c for w, c in words.items()}
        print("    words now: " + ", ".join(
            "".join(vocab[i].decode("utf-8", "replace") for i in w) for w in words))


def the_real_trainer_agrees() -> None:
    merges, vocab, stats = train_bpe(TINY, 262, pattern="gpt2", min_frequency=2)
    print("\n  the production trainer, same corpus, first 6 merges:")
    for i, (a, b) in enumerate(merges[:6]):
        print(f"    {i}: {vocab[a]!r} + {vocab[b]!r} -> {vocab[256 + i]!r}")
    print("\n  (it keeps a pair->count table and a pair->words index, so a merge")
    print(f"   touches only the words that contain it -- {stats.merges_recorded} merges here,")
    print("   thousands on a real corpus, in about a second)")


if __name__ == "__main__":
    print("== six merges, by hand ==")
    by_hand()
    the_real_trainer_agrees()
