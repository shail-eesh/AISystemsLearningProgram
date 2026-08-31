"""Step 4a: pre-tokenization — the regex that decides what BPE may never merge.

BPE will happily learn a token spanning ` the` and `\\n` if the data supports it.
GPT-2 stops that by first splitting the text into chunks with a regex, then
running BPE *inside each chunk only*. The regex is therefore not a detail: it is
a hard prior on the token inventory, and it is why GPT-2 tokens carry a leading
space (` the`, not `the`).

Two patterns live here:

* :data:`GPT2_PATTERN` — the original, reproduced so the comparison in step 5 is
  like-for-like.
* :data:`FINANCE_PATTERN` — the same idea with three finance-specific rules
  added, each of which is justified by a measurement in `bench/`.
"""

from __future__ import annotations

import regex as _regex_module  # type: ignore[import-untyped]

re = _regex_module

#: GPT-2's pre-tokenizer (Radford et al. 2019). Contractions, then letters,
#: then numbers, then punctuation, then whitespace runs.
GPT2_PATTERN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

#: The finance-aware variant. Three additions over GPT-2, in priority order:
#:
#: 1. **Numbers stay whole up to 3 digits, then group by thousands.** GPT-2
#:    splits ` 1234567` into arbitrary pieces; financial text is mostly numbers,
#:    and a consistent split is worth real compression.
#: 2. **ISO dates and times** (2024-03-31, 15:29:59) are one chunk each.
#: 3. **Ticker-like all-caps runs** (RELIANCE, NSE, INE002A01018) stay whole, so
#:    a symbol becomes one or two tokens rather than five.
FINANCE_PATTERN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d"""
    r"""| ?\d{4}-\d{2}-\d{2}"""              # ISO date
    r"""| ?\d{1,2}:\d{2}(?::\d{2})?"""       # time
    r"""| ?\p{Lu}{2,}\d*[A-Z0-9]*"""         # TICKER / ISIN-ish all-caps runs
    r"""| ?\p{L}+"""
    r"""| ?\d{1,3}(?:,\d{3})+"""             # 1,234,567 as one chunk
    r"""| ?\d{1,3}"""                        # otherwise up to three digits
    r"""| ?\p{N}+"""                         # any remaining numeric: superscripts,
                                             # fractions, non-ASCII digits. Without
                                             # this branch the pattern is not TOTAL
                                             # and silently deletes characters --
                                             # found by the byte fuzzer, not by eye.
    r"""| ?[^\s\p{L}\p{N}]+"""
    r"""|\s+(?!\S)|\s+"""
)

PATTERNS = {"gpt2": GPT2_PATTERN, "finance": FINANCE_PATTERN}


def pretokenize(text: str, pattern: str | object = "gpt2") -> list[str]:
    """Split `text` into the chunks BPE is allowed to work inside."""
    compiled = PATTERNS[pattern] if isinstance(pattern, str) else pattern
    return compiled.findall(text)


def chunk_counts(texts, pattern: str | object = "gpt2") -> dict[bytes, int]:
    """Frequency table over UTF-8-encoded chunks — the input to BPE training.

    Counting *types* rather than streaming *tokens* is what makes training
    tractable: a 1 MB corpus has ~200k chunk occurrences but only ~20k distinct
    chunks, and BPE only ever needs the distinct ones plus their weights.
    """
    counts: dict[bytes, int] = {}
    for text in texts:
        for chunk in pretokenize(text, pattern):
            key = chunk.encode("utf-8")
            counts[key] = counts.get(key, 0) + 1
    return counts
