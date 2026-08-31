"""One dataclass that decides everything about the model.

nanoGPT keeps its config next to the model; we keep it separate because three
other topics import it (T15 trains it, T9 swaps its MLP for a router, T12 pages
its KV cache) and none of them should have to import the model to describe one.

Two knobs here are unusual and deliberate:

* ``position`` selects between *learned*, *sinusoidal*, *rope* and *none*. The
  capsule asks for RoPE **and** learned positions "both, compared", so the
  choice has to be a config value rather than a rewrite.
* ``norm`` selects LayerNorm or RMSNorm. RMSNorm is what every model after
  LLaMA uses, and the diff is small enough to read in one sitting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

PositionKind = Literal["learned", "sinusoidal", "rope", "none"]
NormKind = Literal["layernorm", "rmsnorm"]


@dataclass
class GPTConfig:
    """Shape and behaviour of one decoder-only transformer."""

    vocab_size: int = 256
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    mlp_ratio: int = 4
    dropout: float = 0.0
    bias: bool = True
    position: PositionKind = "learned"
    norm: NormKind = "layernorm"
    tie_weights: bool = True
    rope_base: float = 10_000.0
    #: Filled in by ``GPT``; kept here so a checkpoint carries provenance.
    tags: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head:
            raise ValueError(
                f"n_embd={self.n_embd} must divide evenly into n_head={self.n_head}; "
                f"every head needs the same width"
            )
        if self.position == "rope" and self.head_dim % 2:
            raise ValueError(
                f"RoPE rotates coordinate *pairs*, so head_dim must be even, got {self.head_dim}"
            )
        if self.block_size < 1:
            raise ValueError("block_size must be >= 1")

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    def scaled(self, **overrides: object) -> GPTConfig:
        """A copy with fields replaced — the scaling study in T15 lives on this."""
        from dataclasses import replace

        return replace(self, **overrides)  # type: ignore[arg-type]

    def summary(self) -> str:
        return (
            f"GPT(L={self.n_layer}, H={self.n_head}, d={self.n_embd}, "
            f"ctx={self.block_size}, vocab={self.vocab_size}, "
            f"pos={self.position}, norm={self.norm}, tied={self.tie_weights})"
        )
