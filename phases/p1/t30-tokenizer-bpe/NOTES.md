# T30 · BPE and FinTok — Notes

## Why bytes, not characters

"Length" is three different numbers. `"📈"` is 1 code point, 2 UTF-16 code units
(so `.Length` is 2 in C#), 4 UTF-8 bytes, and 1 thing a human sees.

Byte-level BPE picks bytes and never looks back. The base vocabulary is the 256
byte values, so:

- every string in every language encodes — there is no `<unk>` and no code path
  that needs one;
- `decode(encode(s)) == s` is *structural*, not aspirational;
- the tokenizer has no opinion about Unicode, which is the correct number of
  opinions to have about Unicode.

**Normalisation is still a hazard.** `café` typed as NFC is 5 bytes; typed as
NFD (e + combining acute) it is 6. Two strings a user cannot tell apart become
two different token sequences, and the model behaves differently for reasons
invisible in the logs. Fix a normalisation form at the door.

## The algorithm

    while the vocabulary is too small:
        find the most frequent adjacent pair
        mint a new symbol for it
        replace every occurrence

That is all of BPE (Sennrich et al. 2015; byte-level in GPT-2). Everything else
is efficiency:

- **Count types, not tokens.** Pre-tokenize into chunks and count *distinct*
  chunks with weights. A 1 MB corpus has ~200k chunk occurrences but ~20k
  distinct chunks.
- **Incremental updates.** Keep `pair → count` and `pair → {words containing
  it}`. After a merge, only the words in that set can change. This is the
  difference between one second and several minutes.

**The vocabulary is derived.** Only the merge list needs saving: token `256+k`
is the concatenation of the two pieces merge `k` joined. `rebuild_vocab` is four
lines and reconstructs everything.

## Encoding: rank order, not greedy matching

Decoding is trivial. Encoding is where implementations go wrong: repeatedly
apply the **lowest-ranked applicable merge**, which replays training order
exactly. Longest-match-first, or left-to-right-first, decodes to the same text
but produces a *different segmentation* — one the model was never trained on.
Silent, and it costs both compression and quality.

A useful corollary: any *prefix* of a rank-ordered merge list is itself a
complete BPE. That is what `truncate()` exploits to compare two tokenizers at
matched vocabulary size.

## The pre-tokenizer is a hard prior

The regex decides what BPE is *allowed* to merge. GPT-2's is why its tokens
carry a leading space (` the`, not `the`). The finance variant adds three rules:
whole ISO dates, whole times, thousands-grouped numbers, all-caps ticker runs.
Worth **1.08×** on its own, measured with the corpus and vocabulary held fixed.

> **The bug the fuzzer found.** The first finance pattern used `\d` for its
> numeric branches, while its punctuation fallback excluded `\p{N}`. Superscript
> digits (U+00B2, U+00B3) are `\p{N}` but not `\d`, so *no branch matched them
> and they were silently deleted* — the tokenizer round-tripped `café 📈` fine
> and quietly dropped `²`. **A pre-tokenizer must be total.** The regression test
> now runs every latin-1 byte and a 12k-codepoint sweep through both patterns.

## Special tokens must be unforgeable

`<|order|>` has one job: to mean something no amount of ordinary text can
counterfeit. So it is matched *before* BPE runs, gets a reserved id above the
learned vocabulary, and `encode(s, allowed_special=False)` provably does not
produce it. The tests check both directions, plus near-misses (`<|order|`,
`<|ORDER|>`, `<|orders|>`), because that is where a naive string replace breaks.

Their ids are part of the model contract. Renumbering them after AlphaSLM is
pretrained silently corrupts every downstream checkpoint.

## What a domain vocabulary is actually worth

Controlled: same trainer, same requested vocabulary, same pre-tokenizer, both
truncated to 997 merges, measured on held-out financial text.

| | bytes/token |
|:--|--:|
| FinTok (trained on financial text) | **4.528** |
| GeneralTok (trained on general English) | 1.658 |
| raw bytes | 1.000 |

**2.73×.** In context, that is 2.73× more text per unit of context window, 2.73×
fewer positions to attend over, and — since attention is quadratic in sequence
length — considerably more than 2.73× off the attention cost.

Note the control still beats raw bytes by 1.66×: even a mismatched vocabulary
learns English morphology that transfers.

## Vocabulary size is a property of the corpus

Asking for 16,128 merges gives you 1,450 at 126 KB and 3,233 at 951 KB. Training
stops when no pair occurs twice, because minting frequency-1 tokens produces
vocabulary entries the model will see once and never learn. The plan's
FinTok-16k is achievable — on a real filings corpus, not on 1 MB of templated
synthetic text. Ship what the data supports and publish the curve.

## Carry-forward

- **T15 (AlphaSLM):** FinTok is the tokenizer. Corpus → FinTok → packed `.bin`
  shards is the first step of that pipeline.
- **T25 (structured output):** constrained decoding operates on *token* ids, and
  a grammar over characters has to be compiled through the merge table. The
  leading-space convention is why this is harder than it sounds.
- **T12 (KV paging), T29 (prompt caching):** cache keys are token-id prefixes, so
  tokenizer stability is a correctness property of the cache, not a detail.
- Any time a model does arithmetic badly, look at how the tokenizer split the
  number. Half of "LLMs can't count" is a pre-tokenizer choice.
