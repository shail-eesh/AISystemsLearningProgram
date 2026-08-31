"""Steps 2–4: byte-level BPE — trainer, encoder, decoder.

**Byte-level** is the load-bearing word. The base vocabulary is the 256 possible
byte values, not a character set. That single choice removes the entire class of
"unknown token" bugs: every string in the universe encodes, because every string
has a UTF-8 encoding and every UTF-8 encoding is a sequence of bytes we already
have tokens for.

The algorithm (Sennrich et al. 2015, byte-level in GPT-2):

    while vocab is too small:
        find the most frequent adjacent pair of symbols
        mint a new symbol for it
        replace every occurrence

Done naively that is O(merges x corpus) and takes minutes. The implementation
below keeps a pair->count table and a pair->{words containing it} index, and
after each merge touches only the affected words. That is the difference
between a 12-second training run and a coffee break.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .pretokenize import PATTERNS, chunk_counts, pretokenize

Pair = tuple[int, int]


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------
def _word_pairs(word: tuple[int, ...]) -> list[Pair]:
    return list(zip(word[:-1], word[1:], strict=False))


def _merge_in_word(word: tuple[int, ...], pair: Pair, new_id: int) -> tuple[int, ...]:
    """Replace every non-overlapping occurrence of `pair` in `word`."""
    out: list[int] = []
    i = 0
    a, b = pair
    n = len(word)
    while i < n:
        if i < n - 1 and word[i] == a and word[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)


@dataclass
class TrainStats:
    """Enough history to draw the merge-by-merge chart in the video."""

    merges_recorded: int = 0
    history: list[dict] = field(default_factory=list)

    def record(self, rank: int, pair: Pair, count: int, piece: bytes, distinct_pairs: int) -> None:
        self.merges_recorded += 1
        if rank < 64 or rank % 256 == 0:
            self.history.append({
                "rank": rank,
                "count": count,
                "piece": piece.decode("utf-8", errors="replace"),
                "distinct_pairs_remaining": distinct_pairs,
            })


def train_bpe(
    texts,
    vocab_size: int,
    *,
    pattern: str = "gpt2",
    min_frequency: int = 2,
    verbose_every: int = 0,
) -> tuple[list[Pair], dict[int, bytes], TrainStats]:
    """Learn merges until the vocabulary reaches `vocab_size`.

    Returns `(merges, vocab, stats)` where `merges` is in rank order — rank is
    the whole point, because the encoder must apply merges in exactly the order
    they were learned or it produces a different (and worse) segmentation.
    """
    if vocab_size < 256:
        raise ValueError("vocab_size must be at least 256 (the byte alphabet)")

    counts = chunk_counts(texts, pattern)
    words: list[tuple[int, ...]] = [tuple(w) for w in counts]
    freqs: list[int] = list(counts.values())

    pair_counts: Counter[Pair] = Counter()
    pair_where: dict[Pair, set[int]] = defaultdict(set)
    for idx, (word, freq) in enumerate(zip(words, freqs, strict=True)):
        for p in _word_pairs(word):
            pair_counts[p] += freq
            pair_where[p].add(idx)

    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    merges: list[Pair] = []
    stats = TrainStats()

    while len(vocab) < vocab_size:
        if not pair_counts:
            break
        best, best_count = max(pair_counts.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))
        if best_count < min_frequency:
            break

        new_id = len(vocab)
        vocab[new_id] = vocab[best[0]] + vocab[best[1]]
        merges.append(best)
        stats.record(len(merges) - 1, best, best_count, vocab[new_id], len(pair_counts))
        if verbose_every and len(merges) % verbose_every == 0:
            print(f"    merge {len(merges):>6}  {vocab[new_id]!r:<24} count={best_count}")

        # Only the words that actually contain `best` can change.
        for idx in list(pair_where[best]):
            word, freq = words[idx], freqs[idx]
            if best not in set(_word_pairs(word)):
                continue
            for p in _word_pairs(word):          # retract the old contribution
                pair_counts[p] -= freq
                if pair_counts[p] <= 0:
                    del pair_counts[p]
                    pair_where.pop(p, None)
                else:
                    pair_where[p].discard(idx)
            new_word = _merge_in_word(word, best, new_id)
            words[idx] = new_word
            for p in _word_pairs(new_word):      # and post the new one
                pair_counts[p] += freq
                pair_where[p].add(idx)
        pair_counts.pop(best, None)
        pair_where.pop(best, None)

    return merges, vocab, stats


# --------------------------------------------------------------------------
# the tokenizer
# --------------------------------------------------------------------------
@dataclass
class Tokenizer:
    """Merges + vocab + special tokens = a tokenizer.

    `ranks` maps a pair to the order it was learned in. Encoding repeatedly
    applies the *lowest-ranked* applicable merge, which reproduces the training
    order exactly. Applying merges greedily by length, or in dictionary order,
    gives a different segmentation that decodes to the same text but compresses
    worse and does not match the model's training distribution.
    """

    merges: list[Pair]
    vocab: dict[int, bytes]
    pattern: str = "gpt2"
    special_tokens: dict[str, int] = field(default_factory=dict)
    name: str = "bpe"

    def __post_init__(self) -> None:
        self.ranks: dict[Pair, int] = {p: i for i, p in enumerate(self.merges)}
        self._inverse_special = {v: k for k, v in self.special_tokens.items()}
        if self.special_tokens:
            import regex as re

            self._special_re = re.compile(
                "(" + "|".join(re.escape(s) for s in sorted(self.special_tokens, key=len, reverse=True)) + ")"
            )
        else:
            self._special_re = None
        self._cache: dict[bytes, list[int]] = {}

    # -- size ------------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return len(self.vocab) + len(self.special_tokens)

    # -- encoding ---------------------------------------------------------
    def _encode_chunk(self, piece: bytes) -> list[int]:
        cached = self._cache.get(piece)
        if cached is not None:
            return cached
        ids = list(piece)
        while len(ids) >= 2:
            best_rank, best_at = None, None
            for i in range(len(ids) - 1):
                rank = self.ranks.get((ids[i], ids[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_at = rank, i
            if best_at is None:
                break
            a, b = ids[best_at], ids[best_at + 1]
            new_id = 256 + best_rank  # merge k mints token 256+k, by construction
            ids[best_at : best_at + 2] = [new_id]
            _ = (a, b)
        self._cache[piece] = ids
        return ids

    def encode_ordinary(self, text: str) -> list[int]:
        """Encode ignoring special tokens — the path used during training."""
        out: list[int] = []
        for chunk in pretokenize(text, self.pattern):
            out.extend(self._encode_chunk(chunk.encode("utf-8")))
        return out

    def encode(self, text: str, *, allowed_special: bool = True) -> list[int]:
        """Encode, honouring registered special tokens if `allowed_special`."""
        if not allowed_special or self._special_re is None:
            return self.encode_ordinary(text)
        out: list[int] = []
        for part in self._special_re.split(text):
            if not part:
                continue
            if part in self.special_tokens:
                out.append(self.special_tokens[part])
            else:
                out.extend(self.encode_ordinary(part))
        return out

    # -- decoding ---------------------------------------------------------
    def decode(self, ids, *, errors: str = "replace") -> str:
        """Bytes back to text.

        `errors="replace"` matters: a *prefix* of a token stream can end in the
        middle of a multi-byte UTF-8 character, which is exactly what streaming
        generation does on every step. Strict decoding would raise on every
        partial emoji.
        """
        buf = bytearray()
        for i in ids:
            special = self._inverse_special.get(i)
            if special is not None:
                buf.extend(special.encode("utf-8"))
            else:
                buf.extend(self.vocab[i])
        return buf.decode("utf-8", errors=errors)

    def decode_bytes(self, ids) -> bytes:
        buf = bytearray()
        for i in ids:
            special = self._inverse_special.get(i)
            buf.extend(special.encode("utf-8") if special is not None else self.vocab[i])
        return bytes(buf)

    # -- surgery -----------------------------------------------------------
    def truncate(self, n_merges: int) -> Tokenizer:
        """A smaller tokenizer built from the first `n_merges` merges.

        Valid because merges are rank-ordered and each one only ever depends on
        earlier ones: any prefix of the merge list is itself a complete BPE.
        This is what makes a *matched-vocabulary* comparison possible when two
        corpora support different numbers of merges.
        """
        merges = self.merges[:n_merges]
        base = 256 + len(merges)
        specials = {s: base + i for i, s in enumerate(self.special_tokens)}
        return Tokenizer(merges, rebuild_vocab(merges), self.pattern, specials,
                         f"{self.name}-{base}")

    # -- persistence -------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "name": self.name,
            "pattern": self.pattern,
            "vocab_size": self.vocab_size,
            "merges": [[int(a), int(b)] for a, b in self.merges],
            "special_tokens": self.special_tokens,
        }
        path.write_text(json.dumps(payload, indent=1))
        return path

    @classmethod
    def load(cls, path: str | Path) -> Tokenizer:
        payload = json.loads(Path(path).read_text())
        merges = [(int(a), int(b)) for a, b in payload["merges"]]
        return cls(
            merges=merges,
            vocab=rebuild_vocab(merges),
            pattern=payload.get("pattern", "gpt2"),
            special_tokens=payload.get("special_tokens", {}),
            name=payload.get("name", "bpe"),
        )

    @classmethod
    def train(
        cls,
        texts,
        vocab_size: int,
        *,
        pattern: str = "gpt2",
        special_tokens: list[str] | None = None,
        name: str = "bpe",
        min_frequency: int = 2,
        verbose_every: int = 0,
    ) -> tuple[Tokenizer, TrainStats]:
        specials = special_tokens or []
        merges, vocab, stats = train_bpe(
            texts, vocab_size - len(specials), pattern=pattern,
            min_frequency=min_frequency, verbose_every=verbose_every,
        )
        base = len(vocab)
        table = {s: base + i for i, s in enumerate(specials)}
        return cls(merges, vocab, pattern, table, name), stats


def rebuild_vocab(merges: list[Pair]) -> dict[int, bytes]:
    """Reconstruct the byte string for every token id from the merge list alone.

    This is why only the merges need saving: the vocabulary is a *derived*
    quantity. Token 256+k is the concatenation of the two pieces merge k joined.
    """
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for k, (a, b) in enumerate(merges):
        vocab[256 + k] = vocab[a] + vocab[b]
    return vocab


def compression(tokenizer: Tokenizer, texts) -> dict:
    """Bytes per token — the only compression number that is comparable."""
    total_bytes = sum(len(t.encode("utf-8")) for t in texts)
    total_tokens = sum(len(tokenizer.encode_ordinary(t)) for t in texts)
    total_chars = sum(len(t) for t in texts)
    return {
        "bytes": total_bytes,
        "chars": total_chars,
        "tokens": total_tokens,
        "bytes_per_token": total_bytes / max(total_tokens, 1),
        "chars_per_token": total_chars / max(total_tokens, 1),
    }


__all__ = [
    "PATTERNS", "Pair", "TrainStats", "Tokenizer", "compression",
    "rebuild_vocab", "train_bpe",
]
