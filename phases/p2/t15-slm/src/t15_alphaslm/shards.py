"""Corpus -> FinTok -> packed uint16 shards, read back by memory map.

Why not just tokenize in the training loop? Three reasons, all of which bite at
a scale you will reach on the 4070:

1. **Tokenizing is slower than a training step.** Doing it per batch means the
   GPU waits on Python. Pack once, read forever.
2. **A packed stream has no padding.** Documents are concatenated with an
   ``<|endoftext|>`` separator into one long array and windows are cut from it,
   so every position in every batch is a real training example. Padding to a
   fixed length wastes a quarter of the compute on a corpus like this one, where
   document lengths vary by 20x.
3. **A memory-mapped array does not have to fit in RAM.** ``np.memmap`` reads the
   windows the batch actually touches. This is the whole trick that lets a laptop
   train on a corpus bigger than its memory, and it is exactly what nanoGPT does.

The format is deliberately boring: a flat ``uint16`` array per split, plus a
``meta.json`` recording the tokenizer, the vocabulary size, the separator, the
token count and a checksum. uint16 because FinTok's vocabulary is 3,495 — a
uint32 array would double the file for no reason. ``pack_documents`` refuses to
write if the vocabulary would not fit, rather than wrapping around silently.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[5]
T30_SRC = REPO / "phases" / "p1" / "t30-tokenizer-bpe" / "src"
DTYPE = np.uint16
DEFAULT_DIR = pathlib.Path(__file__).resolve().parent / "artifacts" / "shards"


def load_tokenizer():
    """FinTok, the tokenizer T30 trained on this exact domain."""
    if str(T30_SRC) not in sys.path:
        sys.path.insert(0, str(T30_SRC))
    from t30_fintok import load_fintok

    return load_fintok()


def encode_documents(documents: list[str], tokenizer=None, *, separator: str = "<|endoftext|>",
                     ) -> np.ndarray:
    """One flat token array: doc, separator, doc, separator, ...

    The separator is a *learned* token, not a formatting convention. Without it
    the model sees the end of one filing running straight into the start of an
    unrelated one and has to infer the boundary from content; with it, "this
    document is over" is a thing it can predict, and generation has something to
    stop on.
    """
    tok = tokenizer or load_tokenizer()
    sep = tok.special_tokens[separator]
    if tok.vocab_size > np.iinfo(DTYPE).max + 1:
        raise ValueError(
            f"vocabulary of {tok.vocab_size} does not fit in {DTYPE.__name__}; "
            f"widen the dtype rather than letting ids wrap"
        )
    out: list[int] = []
    for doc in documents:
        out.extend(tok.encode(doc))
        out.append(sep)
    return np.asarray(out, dtype=DTYPE)


def pack_documents(train_docs: list[str], val_docs: list[str], *,
                   out_dir: pathlib.Path | str = DEFAULT_DIR, tokenizer=None,
                   separator: str = "<|endoftext|>") -> dict:
    """Write ``train.bin``, ``val.bin`` and ``meta.json``. Returns the metadata."""
    tok = tokenizer or load_tokenizer()
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta: dict = {
        "tokenizer": getattr(tok, "name", "fintok"),
        "vocab_size": tok.vocab_size,
        "dtype": DTYPE.__name__,
        "separator": separator,
        "separator_id": tok.special_tokens[separator],
        "splits": {},
    }
    for name, docs in (("train", train_docs), ("val", val_docs)):
        ids = encode_documents(docs, tok, separator=separator)
        path = out / f"{name}.bin"
        ids.tofile(path)
        meta["splits"][name] = {
            "documents": len(docs),
            "tokens": int(ids.size),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(ids.tobytes()).hexdigest()[:16],
            "max_id": int(ids.max()) if ids.size else 0,
        }
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


class ShardDataset:
    """A memory-mapped token stream you cut random windows out of.

    Deliberately *not* a ``torch.utils.data.Dataset``: there are no examples to
    index, only one long array and a window length. That is what a pretraining
    corpus is, and pretending otherwise adds a DataLoader, workers, collation
    and a shuffling buffer to a problem that needs a random integer.
    """

    def __init__(self, path: pathlib.Path | str, *, block_size: int) -> None:
        self.path = pathlib.Path(path)
        self.block_size = block_size
        if not self.path.exists():
            raise FileNotFoundError(
                f"{self.path} does not exist — run steps/step1_corpus_to_shards.py first"
            )
        self._data = np.memmap(self.path, dtype=DTYPE, mode="r")
        if len(self._data) <= block_size + 1:
            raise ValueError(
                f"{self.path} holds {len(self._data)} tokens, which is not enough "
                f"for a single {block_size}-token window"
            )

    def __len__(self) -> int:
        return len(self._data)

    @property
    def windows(self) -> int:
        """How many distinct starting positions exist."""
        return len(self._data) - self.block_size - 1

    def batch(self, batch_size: int, rng: np.random.Generator):
        """``(x, y)`` as int64 tensors, y being x shifted one token left.

        The copy out of the memmap is not optional: torch cannot own memory it
        did not allocate, and a tensor pointing into a memmap that gets garbage
        collected is a segfault waiting for a quiet afternoon.
        """
        import torch

        ix = rng.integers(0, self.windows, size=batch_size)
        x = np.stack([self._data[i : i + self.block_size] for i in ix]).astype(np.int64)
        y = np.stack([self._data[i + 1 : i + 1 + self.block_size] for i in ix]).astype(np.int64)
        return torch.from_numpy(x), torch.from_numpy(y)

    def sequential_batches(self, batch_size: int, limit: int | None = None):
        """Non-overlapping windows in order — what an *evaluation* wants.

        Random windows are right for training and wrong for a perplexity number:
        random draws overlap, so some tokens get counted several times and the
        result wobbles between runs. This walks the split once.
        """
        import torch

        n = self.windows // self.block_size
        if limit is not None:
            n = min(n, limit * batch_size)
        starts = [i * self.block_size for i in range(n)]
        for i in range(0, len(starts) - batch_size + 1, batch_size):
            chunk = starts[i : i + batch_size]
            x = np.stack([self._data[s : s + self.block_size] for s in chunk]).astype(np.int64)
            y = np.stack([self._data[s + 1 : s + 1 + self.block_size]
                          for s in chunk]).astype(np.int64)
            yield torch.from_numpy(x), torch.from_numpy(y)


def load_meta(out_dir: pathlib.Path | str = DEFAULT_DIR) -> dict:
    return json.loads((pathlib.Path(out_dir) / "meta.json").read_text())


def open_shards(out_dir: pathlib.Path | str = DEFAULT_DIR, *, block_size: int = 128,
                ) -> tuple[ShardDataset, ShardDataset, dict]:
    d = pathlib.Path(out_dir)
    return (ShardDataset(d / "train.bin", block_size=block_size),
            ShardDataset(d / "val.bin", block_size=block_size),
            load_meta(d))


def ensure_shards(out_dir: pathlib.Path | str = DEFAULT_DIR, *, block_size: int = 128,
                  docs: int = 30_000, rebuild: bool = False):
    """Open the shards, building them first if they are not there.

    ``*.bin`` is gitignored course-wide (weights and packed data are
    regenerable, and a repository is not a blob store), so a fresh clone has
    ``meta.json`` for the record and no arrays. Rebuilding takes about a second
    and is deterministic — the checksums in the committed ``meta.json`` are what
    a test compares against.
    """
    from .corpus import build_corpus, split_documents

    d = pathlib.Path(out_dir)
    if rebuild or not (d / "train.bin").exists() or not (d / "val.bin").exists():
        train_docs, val_docs = split_documents(build_corpus(docs=docs))
        pack_documents(train_docs, val_docs, out_dir=d)
    return open_shards(d, block_size=block_size)
