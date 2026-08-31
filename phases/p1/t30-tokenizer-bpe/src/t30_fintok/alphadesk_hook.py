"""AlphaDesk wiring for T30 — the one Phase-1 topic with a real product hook.

FinTok is *the* tokenizer of the desk. AlphaSLM (T15) is pretrained on it, so
every downstream topic inherits it: the embedding model (T43), LoRA (T17), DPO
(T19), quantization (T8), the Rust server (T3), structured output (T25). Its
special-token ids are part of the model contract and must never be renumbered.

Two components are registered: the tokenizer itself, and the trainer, so a
learner can retrain FinTok on their own corpus without reaching into the topic
folder.

AlphaDesk is a fictional educational simulation. The `<|order|>` token labels
*simulated paper* order tickets; nothing here reaches a real venue.
"""

from __future__ import annotations

from typing import Any

from common.alphadesk import Surface, register


@register(
    topic="T30",
    name="fintok",
    surface=Surface.MODELS,
    summary=(
        "Byte-level BPE with a finance-aware pre-tokenizer and 6 special tokens -- "
        "the tokenizer AlphaSLM and every downstream model are trained on"
    ),
)
def build_fintok() -> Any:
    from .fintok import load_fintok

    return load_fintok()


@register(
    topic="T30",
    name="bpe_trainer",
    surface=Surface.MODELS,
    summary="Train a byte-level BPE on your own corpus (the same trainer that produced FinTok)",
)
def build_bpe_trainer() -> dict[str, Any]:
    from .bpe import Tokenizer, compression, train_bpe
    from .fintok import SPECIAL_TOKENS, train_fintok

    return {
        "Tokenizer": Tokenizer,
        "train_bpe": train_bpe,
        "train_fintok": train_fintok,
        "compression": compression,
        "special_tokens": list(SPECIAL_TOKENS),
    }
