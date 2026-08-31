#!/usr/bin/env python3
"""Step 4 — the regex is a hard prior, and special tokens must never be learnable.

Run:  python3 steps/step4_pretokenizer_and_specials.py

Pre-tokenization decides what BPE is *allowed* to merge. GPT-2's regex is why
its tokens carry a leading space. A finance-aware regex adds three rules — whole
ISO dates, thousands-grouped numbers, all-caps ticker runs — and each one is
worth measuring rather than assuming.

Special tokens are the other half. `<|order|>` must be one id that no amount of
training can produce from ordinary text, because its whole job is to be
unforgeable by the data.
"""

import _bootstrap  # noqa: F401
from t30_fintok import SPECIAL_TOKENS, Tokenizer, financial_corpus, pretokenize, split


def compare_patterns() -> None:
    samples = [
        "BUY 1,000 RELIANCE @ 2,945.60 on 2024-03-31 at 09:15:23",
        "INE005A01049 EASTPOWER settled at Rs 1234567.89 (+2.15%)",
    ]
    for s in samples:
        print(f"  {s!r}")
        for name in ("gpt2", "finance"):
            chunks = pretokenize(s, name)
            print(f"    {name:<8} {len(chunks):>2} chunks: " + " | ".join(chunks))
        print()


def measure_the_difference() -> None:
    train, holdout = split(financial_corpus(docs=800))
    print("  same corpus, same requested vocab, two pre-tokenizers:")
    from t30_fintok import compression

    for name in ("gpt2", "finance"):
        tok, _ = Tokenizer.train(train, 2048, pattern=name)
        c = compression(tok, holdout)
        print(f"    {name:<8} vocab {tok.vocab_size:>5}  "
              f"{c['bytes_per_token']:.3f} bytes/token on held-out text")
    print("\n  The regex is not a detail. It is a prior on what tokens can exist.")


def special_tokens_are_unforgeable() -> None:
    train, _ = split(financial_corpus(docs=400))
    tok, _ = Tokenizer.train(train, 1024, pattern="finance", special_tokens=SPECIAL_TOKENS)
    text = "<|order|> BUY 100 ALPHAINFRA <|endoftext|>"
    ids = tok.encode(text)
    print(f"\n  with specials    : {ids[:4]} ...  ({len(ids)} tokens)")
    print(f"  without specials : {len(tok.encode(text, allowed_special=False))} tokens")
    print(f"  round trip: {tok.decode(ids) == text}")
    print("\n  ids of the special tokens (part of the model contract, never renumber):")
    for name, i in tok.special_tokens.items():
        print(f"    {name:<18} {i}")


if __name__ == "__main__":
    print("== two pre-tokenizers on the same text ==")
    compare_patterns()
    measure_the_difference()
    special_tokens_are_unforgeable()
