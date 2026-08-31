"""A character-level corpus you can read, built from the committed samples.

Why character level for T4? Because the point of this topic is the
*architecture*, and a 96-symbol vocabulary means:

* the embedding table is 96 x d instead of 3,500 x d, so the model is mostly
  transformer rather than mostly lookup table — the loss curve is measuring
  what we are trying to teach;
* you can read the output. A char model that has learned structure prints
  ``2024-03-08 ALPHAINFRA CLOSE 121.44``; one that has not prints
  ``2 4-0 A LP HA 1..``. No metric needed for the first look.

T15 swaps this for FinTok (T30) and a real packed-shard pipeline. Same model,
better tokenizer, and the diff between the two loss curves is one of that
topic's lessons.

The corpus is a **market tape**: one line per trading day per symbol, plus
periodic filing sentences, in a fixed grammar. It is small (~200 KB), fully
synthetic/derived from the committed samples, and it has exactly the property
induction heads need — repeated symbols whose continuation is predictable from
an earlier occurrence.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass

import torch
from torch import Tensor

REPO = pathlib.Path(__file__).resolve().parents[5]
SAMPLES = REPO / "common" / "data" / "samples"


@dataclass(frozen=True)
class CharVocab:
    """The whole tokenizer, in two dicts. (T30 explains why real ones are harder.)"""

    chars: tuple[str, ...]

    @classmethod
    def from_text(cls, text: str) -> CharVocab:
        return cls(tuple(sorted(set(text))))

    @property
    def size(self) -> int:
        return len(self.chars)

    @property
    def stoi(self) -> dict[str, int]:
        return {c: i for i, c in enumerate(self.chars)}

    @property
    def itos(self) -> dict[int, str]:
        return dict(enumerate(self.chars))

    def encode(self, s: str) -> list[int]:
        stoi = self.stoi
        missing = sorted(set(s) - set(stoi))
        if missing:
            raise KeyError(f"characters absent from the vocabulary: {missing!r}")
        return [stoi[c] for c in s]

    def decode(self, ids) -> str:
        itos = self.itos
        return "".join(itos[int(i)] for i in ids)


def _read_ohlcv_rows() -> list[dict]:
    path = SAMPLES / "ohlcv_sample.csv"
    lines = path.read_text().strip().splitlines()
    head = lines[0].split(",")
    return [dict(zip(head, ln.split(","), strict=True)) for ln in lines[1:]]


def _read_filing_sentences(limit: int = 60) -> list[str]:
    path = SAMPLES / "filings_sample.jsonl"
    out: list[str] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        for sentence in rec["text"].split(". "):
            s = sentence.strip().rstrip(".")
            if 40 <= len(s) <= 160:
                out.append(f"{rec['issuer']} {rec['section'].upper()}: {s}.")
        if len(out) >= limit:
            break
    return out[:limit]


def build_corpus(*, filings_every: int = 25) -> str:
    """The tape, deterministic and derived only from committed samples.

    Line grammar (fixed on purpose — a regular language is something you can
    *check* the model has learned):

        YYYY-MM-DD SYMBOL O ppp.pp H ppp.pp L ppp.pp C ppp.pp V nnnnnn
    """
    rows = _read_ohlcv_rows()
    filings = _read_filing_sentences()
    lines: list[str] = []
    for i, r in enumerate(rows):
        lines.append(
            f"{r['date']} {r['symbol']} O {float(r['open']):.2f} H {float(r['high']):.2f} "
            f"L {float(r['low']):.2f} C {float(r['close']):.2f} V {int(r['volume'])}"
        )
        if filings and filings_every and (i + 1) % filings_every == 0:
            lines.append(filings[(i // filings_every) % len(filings)])
    return "\n".join(lines) + "\n"


def char_dataset(text: str | None = None, *, split: float = 0.9,
                 ) -> tuple[Tensor, Tensor, CharVocab]:
    """(train_ids, val_ids, vocab). The split is chronological, never shuffled —
    a language model evaluated on interleaved text has already seen the style,
    the symbols and half the sentences it is being tested on."""
    text = build_corpus() if text is None else text
    vocab = CharVocab.from_text(text)
    ids = torch.tensor(vocab.encode(text), dtype=torch.long)
    n = int(len(ids) * split)
    return ids[:n], ids[n:], vocab


def get_batch(data: Tensor, batch_size: int, block_size: int, *,
              generator: torch.Generator | None = None) -> tuple[Tensor, Tensor]:
    """x is ``data[i:i+T]``, y is ``data[i+1:i+T+1]`` — the targets are the
    inputs shifted by one, which is the entire supervision signal in language
    modelling. Every position in the block is a training example, which is why
    a transformer is so much more sample-efficient per step than an RNN."""
    if len(data) <= block_size:
        raise ValueError(f"need more than {block_size} tokens, have {len(data)}")
    ix = torch.randint(len(data) - block_size - 1, (batch_size,), generator=generator)
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x, y


# --------------------------------------------------------------------------
# The induction-head probe task
# --------------------------------------------------------------------------


def repeated_sequence_batch(batch: int, half: int, vocab_size: int, *,
                            generator: torch.Generator | None = None) -> Tensor:
    """``[random prefix of length `half`] + [the same prefix again]``.

    The simplest version of the copying task, and the one to look at first. Its
    flaw is instructive and is why ``variable_period_batch`` exists: because the
    period is *always* ``half``, a model with learned positional embeddings can
    solve it without ever comparing content — "attend to the slot `half` back"
    is a fixed positional rule. That is not an induction head, and a control
    experiment (see NOTES) catches it.
    """
    prefix = torch.randint(vocab_size, (batch, half), generator=generator)
    return torch.cat([prefix, prefix], dim=1)


def variable_period_batch(batch: int, length: int, vocab_size: int, *,
                          min_period: int = 6, max_period: int | None = None,
                          generator: torch.Generator | None = None,
                          ) -> tuple[Tensor, Tensor]:
    """Each row is a random sequence repeating with its *own* random period.

    Row ``b`` is ``base[i % p_b]`` for a fresh random ``base`` of length
    ``p_b``. Since the period differs per row and is never told to the model,
    no fixed positional rule can predict the continuation: the only way to know
    what follows position ``i`` is to find the earlier occurrence of the token
    at ``i`` and read off what came after it. That is the induction algorithm,
    and this task admits no cheaper solution.

    Returns ``(sequences (B, length), periods (B,))``.
    """
    max_period = max_period or length // 2
    if not 2 <= min_period <= max_period <= length:
        raise ValueError(f"need 2 <= min_period <= max_period <= length, got "
                         f"{min_period}, {max_period}, {length}")
    periods = torch.randint(min_period, max_period + 1, (batch,), generator=generator)
    base = torch.randint(vocab_size, (batch, max_period), generator=generator)
    pos = torch.arange(length)
    idx = pos[None, :] % periods[:, None]
    return base.gather(1, idx), periods
