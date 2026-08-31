#!/usr/bin/env python3
"""Step 1 — corpus to packed shards, and why every choice in that sentence matters.

Run:  python3 steps/step1_corpus_to_shards.py      (~10 s)

A pretraining pipeline is four decisions, and three of them are about leakage
and waste rather than about text:

1. **What is a document?** Here: a filing section, a commentary line, an
   announcement, an order ticket, or one symbol's whole price tape — each
   prefixed with its FinTok special token, so register is something the model
   can condition on.
2. **Where does the split go?** Between *documents*, before packing. Split after
   packing and a document straddling the boundary puts its own first half in
   training and its second half in validation, which is the most common way a
   language-model eval quietly becomes meaningless.
3. **How is it stored?** One flat uint16 array per split with a learned
   separator token between documents. No padding, no per-example records,
   memory-mappable.
4. **What is recorded?** Token counts, checksums and the tokenizer identity, so
   a rebuild on another machine can be *proved* identical rather than assumed.
"""

import pathlib
import tempfile

import _bootstrap  # noqa: F401
import numpy as np
from t15_alphaslm import (
    build_corpus,
    corpus_stats,
    encode_documents,
    load_tokenizer,
    pack_documents,
    split_documents,
)
from t15_alphaslm.shards import DEFAULT_DIR, ShardDataset


def what_is_in_the_corpus():
    print("  building the corpus (deterministic, offline, fictional issuers):\n")
    docs = build_corpus()
    stats = corpus_stats(docs)
    print(f"    {stats['documents']:,} documents, {stats['chars']:,} characters, "
          f"{stats['distinct_words']:,} distinct whitespace words")
    print(f"    mean document {stats['mean_doc_chars']:.0f} characters\n")
    for tag, n in stats["by_tag"].items():
        print(f"      {tag:<18} {n:>6,}")
    print("\n  one document of each kind, truncated:\n")
    seen = set()
    for d in docs:
        tag = d.split(" ", 1)[0]
        if tag.startswith("<|") and tag not in seen:
            seen.add(tag)
            print(f"    {d[:150]}...\n")
    return docs


def the_split_goes_between_documents(docs):
    print("  splitting:\n")
    train_docs, val_docs = split_documents(docs)
    print(f"    train {len(train_docs):,} documents")
    print(f"    val   {len(val_docs):,} documents (filings only — that is what the")
    print("          perplexity claim is about; tape would flatter it)")
    overlap = set(train_docs) & set(val_docs)
    print(f"    documents in both splits: {len(overlap)}")
    print("\n    That zero is the point. Split a *packed* array instead and the")
    print("    boundary lands mid-document, so validation contains the continuation")
    print("    of something the model trained on. The number still goes down. It")
    print("    just stops meaning anything.")
    return train_docs, val_docs


def what_packing_does(train_docs, val_docs):
    print("\n  tokenizing with FinTok (T30) and packing:\n")
    tok = load_tokenizer()
    meta = pack_documents(train_docs, val_docs)
    print(f"    tokenizer {meta['tokenizer']}, vocab {meta['vocab_size']}, "
          f"dtype {meta['dtype']}, separator {meta['separator']} "
          f"= id {meta['separator_id']}\n")
    print(f"    {'split':<8} {'documents':>10} {'tokens':>12} {'bytes':>12} {'sha256':>18}")
    for name, s in meta["splits"].items():
        print(f"    {name:<8} {s['documents']:>10,} {s['tokens']:>12,} "
              f"{s['bytes']:>12,} {s['sha256']:>18}")
    chars = sum(len(d) for d in train_docs + val_docs)
    tokens = sum(s["tokens"] for s in meta["splits"].values())
    print(f"\n    {chars / tokens:.2f} characters per token — FinTok was trained on this")
    print("    domain, which is the entire reason T30 came before T15.")
    print(f"    uint16 because the vocabulary is {meta['vocab_size']}; uint32 would")
    print("    double the file to store the same numbers.")
    return meta


def a_memmap_is_not_a_dataset(meta):
    print("\n  reading it back:\n")
    shard = ShardDataset(DEFAULT_DIR / "train.bin", block_size=128)
    print(f"    {len(shard):,} tokens on disk, {shard.windows:,} distinct 128-token windows")
    rng = np.random.default_rng(0)
    x, y = shard.batch(4, rng)
    print(f"    one batch: x {tuple(x.shape)}, y {tuple(y.shape)}, "
          f"y is x shifted by one: {bool((x[:, 1:] == y[:, :-1]).all())}")
    tok = load_tokenizer()
    print("\n    what a training window actually looks like (decoded):\n")
    text = tok.decode(x[0][:60].tolist())
    for line in text.splitlines()[:5]:
        print(f"      {line}")
    print("\n    Note there is no padding anywhere. Every one of those "
          f"{shard.windows:,} windows")
    print("    is a full training example, and documents run into each other through")
    print("    the separator token. Padding to a fixed length on a corpus whose")
    print("    documents vary 20x in length would waste a quarter of the compute.")


def determinism_is_checkable(meta):
    print("\n  and rebuilding it somewhere else gives the same bytes:\n")
    with tempfile.TemporaryDirectory() as d:
        docs = build_corpus()
        tr, va = split_documents(docs)
        again = pack_documents(tr, va, out_dir=pathlib.Path(d))
        for split in ("train", "val"):
            same = again["splits"][split]["sha256"] == meta["splits"][split]["sha256"]
            print(f"    {split}: sha256 {again['splits'][split]['sha256']} "
                  f"{'==' if same else '!='} committed")
    print("\n    meta.json is committed; the .bin files are not (they are gitignored")
    print("    along with every other weight and blob in this course). A fresh clone")
    print("    rebuilds them in two seconds and the checksums prove it got the same")
    print("    corpus — which is a better guarantee than shipping the bytes.")


def the_separator_is_a_learned_token():
    print("\n  why the separator is a token and not a newline:\n")
    tok = load_tokenizer()
    ids = encode_documents(["First document.", "Second document."], tok)
    print(f"    two tiny documents -> {ids.tolist()}")
    print(f"    id {tok.special_tokens['<|endoftext|>']} appears at the end of each\n")
    print("    'This document is over' becomes a thing the model can predict, and")
    print("    therefore a thing generation can stop on. Encode a newline instead and")
    print("    the model has to infer the boundary from content — which it will do")
    print("    badly, because a filing that runs into an unrelated filing is exactly")
    print("    what half the training data looks like.")


if __name__ == "__main__":
    print(__doc__)
    docs = what_is_in_the_corpus()
    train_docs, val_docs = the_split_goes_between_documents(docs)
    meta = what_packing_does(train_docs, val_docs)
    a_memmap_is_not_a_dataset(meta)
    the_separator_is_a_learned_token()
    determinism_is_checkable(meta)
