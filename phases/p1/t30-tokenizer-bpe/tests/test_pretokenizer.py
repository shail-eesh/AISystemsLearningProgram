"""The regex that decides what BPE may never merge."""

from __future__ import annotations

import pytest
from t30_fintok import PATTERNS, pretokenize
from t30_fintok.pretokenize import chunk_counts


@pytest.mark.parametrize("pattern", list(PATTERNS))
def test_chunks_reassemble_into_the_original(pattern):
    for text in ["BUY 1,000 RELIANCE @ 2,945.60", "café 📈\n\n  tabs\there", "", "   "]:
        assert "".join(pretokenize(text, pattern)) == text


@pytest.mark.parametrize("pattern", list(PATTERNS))
def test_pattern_is_total_over_every_latin1_byte(pattern):
    """A pre-tokenizer that does not match a character silently deletes it.

    This is exactly how the finance pattern lost superscript digits (U+00B2,
    U+00B3): they are \\p{N} but not \\d, so no branch claimed them. Regression
    test, kept forever.
    """
    text = bytes(range(256)).decode("latin-1")
    assert "".join(pretokenize(text, pattern)) == text


@pytest.mark.parametrize("pattern", list(PATTERNS))
def test_pattern_is_total_over_a_unicode_sweep(pattern):
    text = "".join(chr(c) for c in range(0x20, 0x2FFF) if chr(c).isprintable())
    assert "".join(pretokenize(text, pattern)) == text


def test_gpt2_pattern_keeps_the_leading_space_with_the_word():
    assert pretokenize("the cat", "gpt2") == ["the", " cat"]


def test_finance_pattern_keeps_iso_dates_whole():
    assert " 2024-03-31" in pretokenize("settled 2024-03-31 close", "finance")


def test_finance_pattern_keeps_times_whole():
    chunks = pretokenize("at 09:15:23 sharp", "finance")
    assert " 09:15:23" in chunks


def test_finance_pattern_keeps_tickers_whole():
    chunks = pretokenize("buy EASTPOWER now", "finance")
    assert " EASTPOWER" in chunks


def test_finance_pattern_groups_thousands():
    chunks = pretokenize("Rs 1,234,567 total", "finance")
    assert " 1,234,567" in chunks


def test_gpt2_pattern_splits_what_finance_keeps():
    gpt2 = pretokenize("2024-03-31", "gpt2")
    finance = pretokenize("2024-03-31", "finance")
    assert len(gpt2) > len(finance)


def test_chunk_counts_weights_by_frequency():
    counts = chunk_counts(["a a a b"], "gpt2")
    assert counts[b"a"] == 1  # first occurrence has no leading space
    assert counts[b" a"] == 2
    assert counts[b" b"] == 1
