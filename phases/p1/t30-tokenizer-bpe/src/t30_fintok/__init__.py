"""T30 · byte-level BPE and FinTok, the tokenizer AlphaSLM is trained on.

    from t30_fintok import load_fintok
    tok = load_fintok()
    tok.decode(tok.encode("BUY 100 RELIANCE <|order|>")) == "BUY 100 RELIANCE <|order|>"
"""

from .bpe import Tokenizer, TrainStats, compression, rebuild_vocab, train_bpe
from .bytes_and_chars import byte_view, char_vs_byte_table, utf8_lengths
from .corpus import corpus_stats, financial_corpus, general_corpus, split
from .fintok import (
    DEFAULT_VOCAB_SIZE,
    FINTOK_PATH,
    SPECIAL_TOKENS,
    build_artifact,
    load_fintok,
    train_fintok,
)
from .pretokenize import FINANCE_PATTERN, GPT2_PATTERN, PATTERNS, chunk_counts, pretokenize

__all__ = [
    "DEFAULT_VOCAB_SIZE", "FINANCE_PATTERN", "FINTOK_PATH", "GPT2_PATTERN", "PATTERNS",
    "SPECIAL_TOKENS", "Tokenizer", "TrainStats", "build_artifact", "byte_view",
    "char_vs_byte_table", "chunk_counts", "compression", "corpus_stats",
    "financial_corpus", "general_corpus", "load_fintok", "pretokenize",
    "rebuild_vocab", "split", "train_bpe", "train_fintok", "utf8_lengths",
]
