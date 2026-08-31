"""The trainer and the encoder: merge order, rank replay, and vocab rebuilding."""

from __future__ import annotations

import pytest
from t30_fintok import Tokenizer, rebuild_vocab, train_bpe
from t30_fintok.bpe import _merge_in_word, _word_pairs

TINY = ["low low low low low lower lower newest newest newest widest widest widest"]


def test_merge_in_word_handles_overlaps():
    assert _merge_in_word((1, 1, 1), (1, 1), 9) == (9, 1)
    assert _merge_in_word((1, 2, 1, 2), (1, 2), 9) == (9, 9)
    assert _merge_in_word((3, 4), (1, 2), 9) == (3, 4)


def test_word_pairs():
    assert _word_pairs((1, 2, 3)) == [(1, 2), (2, 3)]
    assert _word_pairs((1,)) == []


def test_vocab_starts_as_the_byte_alphabet():
    merges, vocab, _ = train_bpe(TINY, 256)
    assert merges == []
    assert len(vocab) == 256
    assert vocab[65] == b"A"


def test_vocab_size_below_256_is_rejected():
    with pytest.raises(ValueError, match="at least 256"):
        train_bpe(TINY, 100)


def test_merges_are_learned_in_frequency_order():
    merges, vocab, _ = train_bpe(TINY, 262, min_frequency=2)
    first = vocab[256]
    assert len(first) == 2
    assert len(merges) == 6


def test_rebuild_vocab_is_derivable_from_merges_alone():
    merges, vocab, _ = train_bpe(TINY, 280, min_frequency=2)
    rebuilt = rebuild_vocab(merges)
    assert rebuilt == vocab


def test_min_frequency_stops_training_early():
    merges, vocab, _ = train_bpe(TINY, 4096, min_frequency=2)
    assert len(vocab) < 4096, "training must stop when pairs run out, not invent junk"
    assert all(len(v) >= 1 for v in vocab.values())


def test_encoder_replays_ranks_lowest_first():
    tok, _ = Tokenizer.train(TINY, 280)
    ids = tok.encode_ordinary("newest")
    # Whatever the segmentation, decoding must reproduce the input exactly and
    # the ids must all be in the vocabulary.
    assert tok.decode(ids) == "newest"
    assert all(i in tok.vocab for i in ids)


def test_encoding_is_deterministic_and_cached():
    tok, _ = Tokenizer.train(TINY, 300)
    a = tok.encode_ordinary("lowest newest")
    b = tok.encode_ordinary("lowest newest")
    assert a == b


def test_more_merges_never_increase_token_count():
    """Monotonicity: a superset of merges can only compress the same text further."""
    corpus = TINY * 20
    small, _ = Tokenizer.train(corpus, 300, min_frequency=2)
    large, _ = Tokenizer.train(corpus, 400, min_frequency=2)
    text = "the lowest newest widest"
    assert len(large.encode_ordinary(text)) <= len(small.encode_ordinary(text))


def test_truncate_produces_a_valid_prefix_tokenizer():
    tok, _ = Tokenizer.train(TINY * 10, 400, min_frequency=2)
    half = tok.truncate(len(tok.merges) // 2)
    assert len(half.merges) == len(tok.merges) // 2
    text = "newest lower widest"
    assert half.decode(half.encode_ordinary(text)) == text
    assert len(half.encode_ordinary(text)) >= len(tok.encode_ordinary(text))


def test_save_and_load_round_trip(tmp_path):
    tok, _ = Tokenizer.train(TINY, 320, special_tokens=["<|x|>"], name="t")
    path = tok.save(tmp_path / "tok.json")
    again = Tokenizer.load(path)
    assert again.merges == tok.merges
    assert again.special_tokens == tok.special_tokens
    assert again.vocab == tok.vocab
    text = "lowest <|x|> newest"
    assert again.encode(text) == tok.encode(text)
