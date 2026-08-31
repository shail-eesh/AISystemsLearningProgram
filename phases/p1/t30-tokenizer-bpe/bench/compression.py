#!/usr/bin/env python3
"""Verification benchmark for T30.

    python3 phases/p1/t30-tokenizer-bpe/bench/compression.py

The capsule's two claims, measured:

1. **Round-trip holds on 100k fuzzed strings.** Byte-level BPE makes this
   structural; the fuzzer is there to catch implementations that break it
   anyway. (It already caught one: the finance pre-tokenizer silently deleted
   superscript digits, because they are `\\p{N}` but not `\\d`.)
2. **FinTok beats a general-domain vocabulary on financial text by a measured
   margin.** Controlled: same trainer, same requested vocabulary, same
   pre-tokenizer, both arms truncated to the same merge count; only the training
   domain differs.

Also measured, because it is the honest part:

3. **The saturation curve.** The plan asks for FinTok-16k. This corpus is
   templated synthetic text and runs out of pairs that occur twice long before
   that. The curve shows exactly where, which is a better lesson than a
   vocabulary padded with frequency-1 junk.
4. **The pre-tokenizer's own contribution**, isolated from the corpus.

`tiktoken`'s GPT-2 vocabulary is the external benchmark the plan names; it is
used when the package is installed *and* its vocabulary files are reachable,
and recorded as unavailable otherwise.
"""

from __future__ import annotations

import json
import pathlib
import platform
import random
import sys
from datetime import UTC, datetime

TOPIC = pathlib.Path(__file__).resolve().parent.parent
REPO = TOPIC.parents[2]
sys.path[:0] = [str(REPO), str(TOPIC / "src")]

from t30_fintok import (  # noqa: E402
    SPECIAL_TOKENS,
    Tokenizer,
    compression,
    corpus_stats,
    financial_corpus,
    general_corpus,
    load_fintok,
    split,
    train_bpe,
)

FUZZ_N = 100_000
ALPHABETS = [
    "abcdefghijklmnopqrstuvwxyz ",
    "0123456789.,%-+/:()",
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "  \t\n\r",
    "éàüñçøßαβγдж中文日本語한국어हिन्दी",
    "📈📉💹🏦🧾",
    "".join(chr(i) for i in range(1, 128)),
    bytes(range(256)).decode("latin-1"),
]


def fuzz_round_trip(tok: Tokenizer, n: int = FUZZ_N, seed: int = 30) -> dict:
    rng = random.Random(seed)
    failures = []
    for i in range(n):
        alphabet = rng.choice(ALPHABETS)
        s = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 60)))
        if tok.decode(tok.encode_ordinary(s)) != s:
            failures.append({"index": i, "string": repr(s)[:120]})
            if len(failures) >= 5:
                break
    return {"strings": n, "failures": failures, "passed": not failures}


def domain_comparison(docs: int = 3000, vocab_size: int = 4096) -> dict:
    fin_train, fin_hold = split(financial_corpus(docs=docs))
    gen_train, _ = split(general_corpus(docs=docs))
    fintok, _ = Tokenizer.train(fin_train, vocab_size, pattern="finance", name="fintok")
    gentok, _ = Tokenizer.train(gen_train, vocab_size, pattern="finance", name="gentok")
    n = min(len(fintok.merges), len(gentok.merges))

    fin_c = compression(fintok.truncate(n), fin_hold)
    gen_c = compression(gentok.truncate(n), fin_hold)
    return {
        "requested_vocab_size": vocab_size,
        "matched_merges": n,
        "financial_corpus": corpus_stats(fin_train),
        "general_corpus": corpus_stats(gen_train),
        "holdout": corpus_stats(fin_hold),
        "fintok": {"merges_learned": len(fintok.merges), **fin_c},
        "generaltok": {"merges_learned": len(gentok.merges), **gen_c},
        "byte_baseline_bytes_per_token": 1.0,
        "advantage": round(fin_c["bytes_per_token"] / gen_c["bytes_per_token"], 3),
    }


def pretokenizer_contribution(docs: int = 1500, vocab_size: int = 2048) -> dict:
    train, hold = split(financial_corpus(docs=docs))
    out = {}
    for pattern in ("gpt2", "finance"):
        tok, _ = Tokenizer.train(train, vocab_size, pattern=pattern)
        out[pattern] = {"merges_learned": len(tok.merges), **compression(tok, hold)}
    out["finance_over_gpt2"] = round(
        out["finance"]["bytes_per_token"] / out["gpt2"]["bytes_per_token"], 3
    )
    return out


def saturation_curve() -> dict:
    rows = []
    for docs in (400, 1200, 3000, 9000):
        train, _ = split(financial_corpus(docs=docs))
        merges, _vocab, _stats = train_bpe(train, 16384, pattern="finance", min_frequency=2)
        rows.append({
            "docs": docs,
            "corpus_kb": corpus_stats(train)["bytes"] // 1024,
            "merges_requested": 16128,
            "merges_learned": len(merges),
        })
    return {
        "rows": rows,
        "note": (
            "Requesting 16k merges from templated synthetic text yields a few "
            "thousand: training stops when no pair occurs twice, rather than "
            "minting frequency-1 junk. Vocabulary size is a property of the "
            "corpus, not a knob. Point the trainer at a real filings corpus "
            "(FORGE_ALLOW_NETWORK=1) and it scales."
        ),
    }


def gpt2_reference(hold) -> dict:
    try:
        import tiktoken

        enc = tiktoken.get_encoding("gpt2")
    except Exception as exc:  # noqa: BLE001 - the point is to record why
        return {"available": False, "reason": f"{type(exc).__name__}: {str(exc)[:160]}"}
    total_bytes = sum(len(t.encode("utf-8")) for t in hold)
    total_tokens = sum(len(enc.encode(t)) for t in hold)
    return {
        "available": True,
        "vocab_size": enc.n_vocab,
        "bytes_per_token": total_bytes / total_tokens,
    }


def main() -> int:
    fintok = load_fintok()
    _, hold = split(financial_corpus(docs=3000))

    fuzz = fuzz_round_trip(fintok)
    domain = domain_comparison()
    pretok = pretokenizer_contribution()
    curve = saturation_curve()
    gpt2 = gpt2_reference(hold)

    failures = []
    if not fuzz["passed"]:
        failures.append({"case": "round trip", "detail": fuzz["failures"]})
    if domain["advantage"] < 1.5:
        failures.append({"case": "domain advantage", "value": domain["advantage"]})

    report = {
        "topic": "T30",
        "benchmark": "byte-level BPE: round-trip fuzz, domain compression, pre-tokenizer, saturation",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "passed": not failures,
        "failures": failures,
        "artifact": {
            "name": fintok.name,
            "pattern": fintok.pattern,
            "merges": len(fintok.merges),
            "vocab_size": fintok.vocab_size,
            "special_tokens": fintok.special_tokens,
        },
        "round_trip_fuzz": fuzz,
        "domain_comparison": domain,
        "pretokenizer_contribution": pretok,
        "saturation_curve": curve,
        "gpt2_reference": gpt2,
        "environment": {"python": platform.python_version(), "machine": platform.machine()},
    }
    (TOPIC / "bench" / "results.json").write_text(json.dumps(report, indent=2) + "\n")

    print(f"artifact         : {fintok.name} — {fintok.vocab_size} tokens "
          f"({len(fintok.merges)} merges + {len(SPECIAL_TOKENS)} specials)")
    print(f"round-trip fuzz  : {fuzz['strings']:,} strings, "
          f"{'0 failures' if fuzz['passed'] else fuzz['failures']}")
    print(f"domain (matched at {domain['matched_merges']} merges, held-out financial text):")
    print(f"    FinTok       {domain['fintok']['bytes_per_token']:.3f} bytes/token")
    print(f"    GeneralTok   {domain['generaltok']['bytes_per_token']:.3f} bytes/token")
    print(f"    raw bytes    1.000 -> FinTok advantage {domain['advantage']}x")
    print(f"pre-tokenizer    : finance {pretok['finance']['bytes_per_token']:.3f} vs "
          f"gpt2 {pretok['gpt2']['bytes_per_token']:.3f} bytes/token "
          f"({pretok['finance_over_gpt2']}x)")
    print("saturation       : " + "  ".join(
        f"{r['corpus_kb']}KB->{r['merges_learned']}" for r in curve["rows"]))
    print(f"gpt2 reference   : {'available' if gpt2['available'] else 'unavailable — ' + gpt2['reason'][:70]}")
    print(f"\n-> {'PASS' if not failures else 'FAIL'}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
