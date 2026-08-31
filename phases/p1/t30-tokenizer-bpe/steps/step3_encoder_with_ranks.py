#!/usr/bin/env python3
"""Step 3 — encoding is *replaying* the merges, in the order they were learned.

Run:  python3 steps/step3_encoder_with_ranks.py

The decoder is trivial: concatenate the byte string of every id. The encoder is
where people go wrong. You must apply the merge with the **lowest rank** that is
currently applicable, repeatedly — not the longest match, not the first match
left to right. Any other order decodes to the same text but produces a different
segmentation from the one the model was trained on.
"""

import _bootstrap  # noqa: F401
from t30_fintok import Tokenizer, financial_corpus, split


def build():
    train, _ = split(financial_corpus(docs=400))
    tok, _ = Tokenizer.train(train, 1024, pattern="finance", name="demo")
    return tok


def show_encoding(tok, text: str) -> None:
    ids = tok.encode_ordinary(text)
    pieces = [tok.vocab[i].decode("utf-8", "replace") for i in ids]
    print(f"  {text!r}")
    print(f"    {len(text.encode('utf-8'))} bytes -> {len(ids)} tokens")
    print("    " + " | ".join(pieces))


def rank_order_matters(tok) -> None:
    text = "consolidated revenue"
    ids = list(text.encode("utf-8"))
    print(f"\n  merging {text!r} by rank, one step at a time:")
    for _ in range(6):
        best_rank, at = None, None
        for i in range(len(ids) - 1):
            r = tok.ranks.get((ids[i], ids[i + 1]))
            if r is not None and (best_rank is None or r < best_rank):
                best_rank, at = r, i
        if at is None:
            break
        merged = tok.vocab[256 + best_rank]
        ids[at : at + 2] = [256 + best_rank]
        print(f"    rank {best_rank:>4}: joined -> {merged!r}   ({len(ids)} tokens left)")


def round_trip_is_exact(tok) -> None:
    samples = [
        "BUY 1,000 RELIANCE @ 2,945.60",
        "NSE/CM/45012 dated 2024-03-31",
        "café 📈 ₹1,20,000",
        "",
        "\n\n   \t",
        "\x00\x01\xff",
    ]
    print("\n  encode -> decode, exactly:")
    for s in samples:
        ok = tok.decode(tok.encode_ordinary(s)) == s
        print(f"    {ok!s:<5} {s!r}")
        assert ok


if __name__ == "__main__":
    tok = build()
    print(f"== a {tok.vocab_size}-token demo tokenizer ==")
    for t in ("consolidated revenue of Rs 2170.0 crore",
              "the quick brown fox",
              "EASTPOWER INE005A01049"):
        show_encoding(tok, t)
    rank_order_matters(tok)
    round_trip_is_exact(tok)
