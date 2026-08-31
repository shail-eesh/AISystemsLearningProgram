# T30 · Tokenizer (BPE) → FinTok

**Phase:** Foundations (P1) · **Generation day:** Day 2 · **Video episodes:** 3

> [← Back to course home](../../../index.html) · [Master plan](../../../MASTER_PLAN.md) · [Progress ledger](../../../EXECUTION/LEDGER.md)

## What you build

A **byte-level** BPE — trainer, encoder, decoder — and **FinTok**, the tokenizer
every later model in this course is trained on.

Byte-level is the load-bearing word: the base vocabulary is the 256 byte values,
not a character set, which deletes the entire "unknown token" problem. Every
string in every language has a UTF-8 encoding, and every byte of it is already a
token.

The pieces:

- `bpe.py` — the merge loop with a pair-count table and a pair→words index, so a
  merge touches only the words containing it. 3,233 merges on a 1 MB corpus in
  about a second.
- `pretokenize.py` — GPT-2's regex, plus a finance-aware variant that keeps ISO
  dates, times, thousands-grouped numbers and all-caps tickers whole.
- `fintok.py` — the trained artefact, six special tokens, save/load.
- `corpus.py` — the synthetic financial corpus, and the general-English control.

## Results

**Round trip on 100,000 fuzzed strings: 0 failures**, across ASCII, CJK,
Devanagari, emoji ZWJ sequences, control bytes and all 256 latin-1 bytes. The
fuzzer earned its keep immediately: it found that the finance pre-tokenizer was
*silently deleting* superscript digits (U+00B2, U+00B3 are `\p{N}` but not
`\d`, so no branch claimed them). That is now a permanent regression test.

**Does a domain vocabulary help?** Controlled experiment — same trainer, same
requested vocabulary, same pre-tokenizer, both arms truncated to the same merge
count, measured on held-out *financial* text:

| tokenizer | trained on | bytes/token on held-out financial text |
|:--|:--|--:|
| **FinTok** | financial corpus | **4.528** |
| GeneralTok | general English | 1.658 |
| raw bytes | — | 1.000 |

**2.73×** more compact, at identical vocabulary size. The pre-tokenizer accounts
for a further **1.08×** on its own (finance 4.788 vs gpt2 4.429 bytes/token, same
corpus, same vocabulary).

**The saturation curve** — and a deviation from the plan, stated plainly:

| corpus | 126 KB | 380 KB | 951 KB | 2.8 MB |
|:--|--:|--:|--:|--:|
| merges learned (16,128 requested) | 1,450 | 2,203 | **3,233** | 4,481 |

The plan asks for FinTok-16k. Asking this trainer for 16,384 merges on the
committed corpus yields 3,233: it is templated synthetic text and runs out of
pairs occurring twice long before then. Rather than pad the vocabulary with
frequency-1 junk, the shipped artefact is **FinTok-3.5k** (3,233 merges + 256
bytes + 6 specials = 3,495 tokens) and the curve above is the measurement that
explains why. Point the same trainer at a real filings corpus
(`FORGE_ALLOW_NETWORK=1`, see `common/data/loaders.py`) and it scales.

`tiktoken`'s GPT-2 vocabulary is the external benchmark the plan names. The
bench uses it when installed and reachable; in this sandbox its vocabulary host
is proxy-blocked, and the controlled two-corpus comparison is what runs. That is
arguably the better experiment: GPT-2 differs from FinTok in corpus, vocabulary
size *and* pre-tokenizer simultaneously.

## AlphaDesk hook

The one Phase-1 topic with a real product hook. Two components on the **models**
surface:

- `models.fintok` — the tokenizer AlphaSLM (T15) is pretrained on, and therefore
  the one every downstream topic inherits: embeddings (T43), LoRA (T17), DPO
  (T19), quantization (T8), serving (T3), structured output (T25).
- `models.bpe_trainer` — retrain it on your own corpus.

Special-token ids are part of the model contract and must never be renumbered:

| token | id | meaning |
|:--|--:|:--|
| `<\|endoftext\|>` | 3489 | document boundary / pretraining separator |
| `<\|pad\|>` | 3490 | batch padding, masked out of the loss |
| `<\|filing\|>` | 3491 | a regulatory filing section follows |
| `<\|commentary\|>` | 3492 | market commentary follows |
| `<\|announcement\|>` | 3493 | an exchange announcement follows |
| `<\|order\|>` | 3494 | a **simulated paper** order ticket follows |

AlphaDesk is a fictional educational simulation. The `<|order|>` token labels
simulated tickets over fictional issuers; nothing here reaches a real venue, and
the corpus is generated, not scraped.

## How to run

```bash
python3 phases/p1/t30-tokenizer-bpe/steps/step1_bytes_not_characters.py
python3 phases/p1/t30-tokenizer-bpe/steps/step2_train_bpe_by_hand.py
python3 phases/p1/t30-tokenizer-bpe/steps/step3_encoder_with_ranks.py
python3 phases/p1/t30-tokenizer-bpe/steps/step4_pretokenizer_and_specials.py
python3 phases/p1/t30-tokenizer-bpe/steps/step5_train_fintok.py
python3 phases/p1/t30-tokenizer-bpe/steps/step6_fuzz_the_round_trip.py

python3 -m pytest phases/p1/t30-tokenizer-bpe/tests -q
python3 phases/p1/t30-tokenizer-bpe/bench/compression.py

# regenerate the committed artefact (deterministic — the corpus is seeded)
python3 -m t30_fintok.fintok
```

Requires `regex` (the `\p{L}` classes the pre-tokenizer needs are not in `re`).

## Layout

- `src/t30_fintok/bpe.py` — trainer, `Tokenizer`, `truncate`, save/load
- `src/t30_fintok/pretokenize.py` — the two regexes
- `src/t30_fintok/corpus.py` — synthetic financial + general-English corpora
- `src/t30_fintok/fintok.py` — FinTok itself
- `src/t30_fintok/artifacts/fintok.json` — the committed vocabulary (merges only)
- `steps/` — the six checkpoints; step 2 shows every merge on nine words
- `tests/` — 54 tests including the 100k-string fuzz
- `bench/compression.py` + `results.json`

## Videos

Episode scripts live in [`video/topics/t30/`](../../../video/topics/t30/).
