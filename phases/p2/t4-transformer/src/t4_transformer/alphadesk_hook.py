"""AlphaDesk wiring for T4.

T4 does not ship a product surface — it ships the *architecture* every later
model on the desk is an instance of. Three things are registered:

* ``models.gpt_architecture`` — the factory T15 calls to build AlphaSLM, T9
  calls to swap in an MoE FFN, and T17 calls to attach LoRA adapters.
* ``models.kv_cache`` — the naive cache, registered so T12's paged replacement
  has something to be measured against rather than merely described.
* ``models.attention_inspector`` — attention maps + the induction probe, which
  the Phase-4 eval dashboard (T27) and the interpretability topic (T22) both
  read.

AlphaDesk is a fictional educational simulation: nothing here places an order,
touches money, or connects to a venue.
"""

from __future__ import annotations

from typing import Any

from common.alphadesk import Surface, register


@register(
    topic="T4",
    name="gpt_architecture",
    surface=Surface.MODELS,
    summary="Decoder-only transformer (RoPE/learned positions, RMS/LayerNorm, tied head)",
    requires=("T31", "T45A"),
)
def build_gpt(**overrides: Any) -> Any:
    """A configured, untrained GPT. Defaults are the char-level tape config."""
    from .config import GPTConfig
    from .model import GPT

    defaults = dict(vocab_size=66, block_size=128, n_layer=4, n_head=4, n_embd=128)
    return GPT(GPTConfig(**{**defaults, **overrides}))


@register(
    topic="T4",
    name="kv_cache",
    surface=Surface.MODELS,
    summary="Naive contiguous KV cache for incremental decoding (T12 replaces it with paging)",
)
def build_kv_cache(batch: int = 1, n_head: int = 4, block_size: int = 128,
                   head_dim: int = 32) -> Any:
    from .attention import KVCache

    return KVCache.empty(batch, n_head, block_size, head_dim)


@register(
    topic="T4",
    name="attention_inspector",
    surface=Surface.MODELS,
    summary="Attention maps + induction-head probe used by the eval dashboard",
)
def build_attention_inspector() -> dict[str, Any]:
    from .interpret import attention_summary, induction_scores

    return {"attention_summary": attention_summary, "induction_scores": induction_scores}
