"""T15 — AlphaSLM: the desk's own small language model."""

from .config import CPU_RUNGS, GPU_RUNGS, LADDER, Rung, describe_ladder
from .corpus import (
    TAGS,
    build_corpus,
    corpus_stats,
    filing_excerpts,
    order_tickets,
    price_tape,
    split_documents,
)
from .evaluate import (
    compare_by_token_class,
    compare_models,
    loss_by_token_class,
    perplexity_by_tag,
    perplexity_on_documents,
    sample_commentary,
)
from .harness import RunState, Trainer, TrainSpec, lr_at, train_rung
from .scaling import analyse, extrapolate, format_table, run_study
from .shards import (
    ShardDataset,
    encode_documents,
    ensure_shards,
    load_meta,
    load_tokenizer,
    open_shards,
    pack_documents,
)

__all__ = [
    "CPU_RUNGS", "GPU_RUNGS", "LADDER", "TAGS", "RunState", "Rung", "ShardDataset",
    "TrainSpec", "Trainer", "analyse", "build_corpus", "compare_models", "corpus_stats",
    "compare_by_token_class", "describe_ladder", "encode_documents", "ensure_shards",
    "extrapolate", "loss_by_token_class",
    "filing_excerpts", "format_table", "load_meta", "load_tokenizer", "lr_at",
    "open_shards", "order_tickets", "pack_documents", "perplexity_by_tag",
    "perplexity_on_documents", "price_tape", "run_study", "sample_commentary",
    "split_documents", "train_rung",
]
