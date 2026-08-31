"""FinTok: the artefact, the special-token contract, and the round-trip property."""

from __future__ import annotations

import random

import pytest
from t30_fintok import (
    SPECIAL_TOKENS,
    Tokenizer,
    compression,
    financial_corpus,
    general_corpus,
    load_fintok,
    split,
)
from t30_fintok.bytes_and_chars import normalisation_changes_the_answer, utf8_lengths


@pytest.fixture(scope="module")
def fintok() -> Tokenizer:
    return load_fintok()


def test_artifact_loads_and_has_the_expected_shape(fintok):
    assert fintok.name == "fintok"
    assert fintok.pattern == "finance"
    assert len(fintok.merges) > 1000
    assert fintok.vocab_size == 256 + len(fintok.merges) + len(SPECIAL_TOKENS)


def test_special_token_ids_are_contiguous_and_last(fintok):
    ids = [fintok.special_tokens[s] for s in SPECIAL_TOKENS]
    assert ids == sorted(ids)
    assert ids == list(range(ids[0], ids[0] + len(SPECIAL_TOKENS)))
    assert min(ids) == 256 + len(fintok.merges), "specials sit above the learned vocab"


def test_special_tokens_are_single_ids(fintok):
    for s in SPECIAL_TOKENS:
        assert fintok.encode(s) == [fintok.special_tokens[s]]


def test_special_tokens_are_unforgeable_from_text(fintok):
    """Encoded as ordinary text they must NOT collapse to the special id."""
    for s in SPECIAL_TOKENS:
        ordinary = fintok.encode(s, allowed_special=False)
        assert len(ordinary) > 1
        assert fintok.special_tokens[s] not in ordinary
        assert fintok.decode(ordinary) == s


def test_near_miss_special_tokens_stay_text(fintok):
    for s in ("<|order|", "|order|>", "<|orders|>", "<|ORDER|>"):
        assert fintok.decode(fintok.encode(s)) == s


@pytest.mark.parametrize(
    "text",
    [
        "", " ", "\n", "\x00", "\x00\x01\xff",
        "BUY 1,000 RELIANCE @ 2,945.60 on 2024-03-31",
        "café", "café", "रिलायंस", "東京証券取引所", "📈", "\U0001f469‍\U0001f4bb",
        "a" * 3000, "Rs 1,20,00,000.00", "2024-03-31T09:15:23+05:30",
    ],
)
def test_round_trip_on_adversarial_strings(fintok, text):
    assert fintok.decode(fintok.encode_ordinary(text)) == text


@pytest.mark.slow
def test_round_trip_on_100k_fuzzed_strings(fintok):
    """The capsule's 'done when'. Byte-level BPE makes this structural."""
    alphabets = [
        "abcdefghijklmnopqrstuvwxyz ",
        "0123456789.,%-+/:()",
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        "  \t\n\r",
        "éàüñçøßαβγдж中文日本語한국어हिन्दी",
        "📈📉💹🏦🧾",
        "".join(chr(i) for i in range(1, 128)),
    ]
    rng = random.Random(30)
    for i in range(100_000):
        alphabet = rng.choice(alphabets)
        s = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 60)))
        assert fintok.decode(fintok.encode_ordinary(s)) == s, f"failed at {i}: {s!r}"


def test_every_byte_is_encodable(fintok):
    raw = bytes(range(256))
    text = raw.decode("latin-1")
    assert fintok.decode_bytes(fintok.encode_ordinary(text)) == text.encode("utf-8")


def test_partial_decode_does_not_raise(fintok):
    ids = fintok.encode_ordinary("📈")
    assert fintok.decode(ids[:1]) == "�"


def test_domain_vocabulary_compresses_financial_text_better():
    """The capsule's other 'done when', as a controlled experiment.

    Same trainer, same requested vocab, same pre-tokenizer; only the training
    domain differs. Both arms are truncated to the same merge count, because a
    prefix of a rank-ordered merge list is itself a valid BPE.
    """
    fin_train, fin_hold = split(financial_corpus(docs=700))
    gen_train, _ = split(general_corpus(docs=700))
    a, _ = Tokenizer.train(fin_train, 2048, pattern="finance")
    b, _ = Tokenizer.train(gen_train, 2048, pattern="finance")
    n = min(len(a.merges), len(b.merges))
    fin_bpt = compression(a.truncate(n), fin_hold)["bytes_per_token"]
    gen_bpt = compression(b.truncate(n), fin_hold)["bytes_per_token"]
    assert fin_bpt > gen_bpt * 1.5, f"expected a wide margin, got {fin_bpt:.2f} vs {gen_bpt:.2f}"
    assert gen_bpt > 1.0, "even a mismatched vocabulary beats raw bytes"


def test_utf8_length_table():
    assert utf8_lengths("📈") == {
        "python_len_code_points": 1,
        "utf8_bytes": 4,
        "utf16_code_units": 2,
        "grapheme_clusters_approx": 1,
    }


def test_normalisation_is_a_real_hazard():
    r = normalisation_changes_the_answer()
    assert not r["same_code_points"]
    assert r["nfd_bytes"] > r["nfc_bytes"]
