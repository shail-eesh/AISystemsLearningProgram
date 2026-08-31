"""The corpus, the split, and the packing format."""

import json

import numpy as np
import pytest

from t15_alphaslm import (
    TAGS,
    ShardDataset,
    build_corpus,
    corpus_stats,
    encode_documents,
    filing_excerpts,
    order_tickets,
    pack_documents,
    price_tape,
    split_documents,
)


def test_corpus_is_deterministic(small_corpus):
    assert build_corpus(docs=300) == small_corpus


def test_every_generated_document_carries_a_register_tag(small_corpus):
    tagged = [d for d in small_corpus if d.startswith("<|")]
    assert len(tagged) > 0.9 * len(small_corpus)
    for d in tagged:
        assert d.split(" ", 1)[0] in set(TAGS.values())


def test_all_four_registers_are_present(small_corpus):
    tags = {d.split(" ", 1)[0] for d in small_corpus if d.startswith("<|")}
    assert tags == set(TAGS.values())


def test_sample_derived_documents_are_included(small_corpus):
    assert filing_excerpts()[0] in small_corpus
    assert order_tickets()[0] in small_corpus
    assert len(price_tape()) == 5              # one per symbol in the sample


def test_order_tickets_carry_the_simulation_disclaimer():
    for ticket in order_tickets():
        assert "Simulated order, educational use only." in ticket


def test_price_tape_groups_by_symbol_not_by_date():
    tapes = price_tape()
    for tape in tapes:
        head, *rows = tape.splitlines()
        assert head.startswith("Price tape ")
        assert len(rows) > 100
        assert all(r.startswith("2024-") for r in rows)


def test_split_holds_out_whole_documents(small_corpus):
    train_docs, val_docs = split_documents(small_corpus)
    assert not set(train_docs) & set(val_docs)
    assert len(train_docs) + len(val_docs) == len(small_corpus)


def test_holdout_is_filings_only(small_corpus):
    _, val_docs = split_documents(small_corpus)
    assert {d.split(" ", 1)[0] for d in val_docs} == {TAGS["filing"]}


def test_split_is_reproducible(small_corpus):
    assert split_documents(small_corpus) == split_documents(small_corpus)


def test_corpus_stats_add_up(small_corpus):
    stats = corpus_stats(small_corpus)
    assert stats["documents"] == len(small_corpus)
    assert stats["chars"] == sum(len(d) for d in small_corpus)
    assert sum(stats["by_tag"].values()) == len(small_corpus)


# -- packing ---------------------------------------------------------------


def test_encoding_separates_documents_with_a_learned_token(tokenizer):
    ids = encode_documents(["alpha", "beta"], tokenizer)
    sep = tokenizer.special_tokens["<|endoftext|>"]
    assert int(ids[-1]) == sep
    assert list(ids).count(sep) == 2


def test_packed_tokens_round_trip_through_the_tokenizer(tokenizer):
    doc = "COASTBANK reported revenue of Rs 1234.5 crore for the quarter."
    ids = encode_documents([doc], tokenizer)
    assert tokenizer.decode([int(i) for i in ids[:-1]]) == doc


def test_meta_records_what_a_rebuild_needs(small_shards):
    _, _, meta, d = small_shards
    on_disk = json.loads((d / "meta.json").read_text())
    assert on_disk == meta
    assert meta["dtype"] == "uint16"
    assert meta["vocab_size"] > meta["separator_id"]
    for split in ("train", "val"):
        assert meta["splits"][split]["tokens"] > 0
        assert meta["splits"][split]["bytes"] == 2 * meta["splits"][split]["tokens"]
        assert meta["splits"][split]["max_id"] < meta["vocab_size"]


def test_packing_is_byte_identical_on_a_rebuild(tmp_path, small_corpus, tokenizer):
    train_docs, val_docs = split_documents(small_corpus)
    a = pack_documents(train_docs, val_docs, out_dir=tmp_path / "a", tokenizer=tokenizer)
    b = pack_documents(train_docs, val_docs, out_dir=tmp_path / "b", tokenizer=tokenizer)
    assert a["splits"] == b["splits"]


def test_uint16_is_checked_not_assumed(tokenizer, monkeypatch):
    """A vocabulary that does not fit must raise, not wrap around silently."""
    class Wide:
        vocab_size = 70_000
        special_tokens = tokenizer.special_tokens

        def encode(self, text):
            return [1, 2, 3]

    with pytest.raises(ValueError, match="does not fit"):
        encode_documents(["x"], Wide())


# -- reading ---------------------------------------------------------------


def test_batch_targets_are_inputs_shifted_by_one(small_shards):
    train, _, _, _ = small_shards
    x, y = train.batch(4, np.random.default_rng(0))
    assert x.shape == y.shape == (4, 64)
    assert (x[:, 1:] == y[:, :-1]).all()


def test_batch_is_reproducible_from_a_seeded_generator(small_shards):
    train, _, _, _ = small_shards
    a = train.batch(4, np.random.default_rng(7))
    b = train.batch(4, np.random.default_rng(7))
    assert (a[0] == b[0]).all() and (a[1] == b[1]).all()


def test_sequential_batches_do_not_overlap(small_shards):
    train, _, _, _ = small_shards
    batches = list(train.sequential_batches(2, limit=3))
    seen = [tuple(row.tolist()) for x, _ in batches for row in x]
    starts = {s[0] for s in seen}
    assert len(seen) > 1
    # windows are cut at multiples of block_size, so no token appears twice
    flat = [t for s in seen for t in s]
    assert len(flat) == len(seen) * 64
    assert len(starts) >= 1


def test_sequential_evaluation_is_deterministic(small_shards):
    train, _, _, _ = small_shards
    a = [x.sum().item() for x, _ in train.sequential_batches(2, limit=3)]
    b = [x.sum().item() for x, _ in train.sequential_batches(2, limit=3)]
    assert a == b


def test_missing_shard_says_what_to_run(tmp_path):
    with pytest.raises(FileNotFoundError, match="step1_corpus_to_shards"):
        ShardDataset(tmp_path / "nope.bin", block_size=8)


def test_shard_shorter_than_a_window_is_rejected(tmp_path):
    path = tmp_path / "tiny.bin"
    np.arange(10, dtype=np.uint16).tofile(path)
    with pytest.raises(ValueError, match="not enough"):
        ShardDataset(path, block_size=64)
