"""T4 — a decoder-only transformer written from an empty file."""

from .attention import (
    CausalSelfAttention,
    KVCache,
    causal_mask,
    multi_head_attention_looped,
    scaled_dot_product,
    single_head_attention,
)
from .blocks import MLP, Block, LayerNorm, RMSNorm, build_norm
from .config import GPTConfig
from .data import CharVocab, build_corpus, char_dataset, get_batch, repeated_sequence_batch
from .interpret import attention_summary, induction_scores, train_induction_model
from .model import GPT
from .positions import (
    LearnedPositional,
    SinusoidalPositional,
    apply_rope,
    relative_phase_property,
    rope_angles,
    sinusoidal_table,
)
from .sampling import apply_temperature, greedy_next, sample_next, top_k_filter, top_p_filter
from .train import History, TrainConfig, estimate_loss, lr_at, smoothed, train

__all__ = [
    "GPT", "MLP", "Block", "CausalSelfAttention", "CharVocab", "GPTConfig", "History",
    "KVCache", "LayerNorm", "LearnedPositional", "RMSNorm", "SinusoidalPositional",
    "TrainConfig", "apply_rope", "apply_temperature", "attention_summary", "build_corpus",
    "build_norm", "causal_mask", "char_dataset", "estimate_loss", "get_batch", "greedy_next",
    "induction_scores", "lr_at", "multi_head_attention_looped", "relative_phase_property",
    "repeated_sequence_batch", "rope_angles", "sample_next", "scaled_dot_product",
    "single_head_attention", "sinusoidal_table", "smoothed", "top_k_filter", "top_p_filter",
    "train", "train_induction_model",
]
