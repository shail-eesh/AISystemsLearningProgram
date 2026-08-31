#!/usr/bin/env python3
"""Step 6 — the property that must never break: decode(encode(s)) == s.

Run:  python3 steps/step6_fuzz_the_round_trip.py

Byte-level BPE makes this property *structural* rather than hopeful: the base
vocabulary is all 256 bytes, so every string encodes and every encoding decodes.
The fuzzer is there to catch the places where an implementation quietly breaks
it anyway — lone surrogates, unpaired combining marks, NUL bytes, text that
looks like a special token but is not.
"""

import random

import _bootstrap  # noqa: F401
from t30_fintok import SPECIAL_TOKENS, load_fintok

ALPHABETS = [
    "abcdefghijklmnopqrstuvwxyz ",
    "0123456789.,%-+/:()",
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "  \t\n\r",
    "éàüñçøßαβγдж中文日本語한국어हिन्दी",
    "📈📉💹🏦🧾",
    "".join(chr(i) for i in range(1, 128)),
]

ADVERSARIAL = [
    "", " ", "\n", "\x00", "\x00\x01\x02",
    "<|order|", "|order|>", "<|not_a_real_special|>", "<|order|><|order|>",
    "a" * 4000,
    "café", "café",
    "\U0001f469‍\U0001f4bb",
    "﷽", "​​​",
    "Rs 1,20,00,000.00", "2024-03-31T09:15:23+05:30",
]


def fuzz(tok, n: int, seed: int = 30) -> int:
    rng = random.Random(seed)
    for i in range(n):
        alphabet = rng.choice(ALPHABETS)
        s = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 60)))
        if tok.decode(tok.encode_ordinary(s)) != s:
            raise AssertionError(f"round trip failed at {i}: {s!r}")
    return n


def main() -> None:
    tok = load_fintok()
    print(f"  FinTok: {tok.vocab_size} tokens, {len(tok.merges)} merges")

    for s in ADVERSARIAL:
        assert tok.decode(tok.encode_ordinary(s)) == s, repr(s)
    print(f"  {len(ADVERSARIAL)} adversarial strings: ok")

    n = fuzz(tok, 100_000)
    print(f"  {n:,} fuzzed strings: ok")

    text = "<|filing|> Risk factors <|endoftext|>"
    assert tok.decode(tok.encode(text)) == text
    print(f"  special tokens round trip: ok ({SPECIAL_TOKENS[0]} -> "
          f"{tok.special_tokens[SPECIAL_TOKENS[0]]})")

    print("\n  A prefix of a token stream can end mid-character, which is what")
    print("  streaming generation does every step. decode() uses errors='replace'")
    print("  so a partial emoji renders as U+FFFD instead of raising:")
    ids = tok.encode_ordinary("📈")
    print(f"    partial decode: {tok.decode(ids[:1])!r}")


if __name__ == "__main__":
    main()
