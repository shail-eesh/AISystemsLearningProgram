"""The decoder-only GPT: embeddings in, logits out, nothing hidden.

    idx (B,T) -> token embeddings [+ position embeddings]
              -> N x Block(attention, MLP)
              -> final norm
              -> lm_head -> logits (B, T, vocab)

Two details that look cosmetic and are not:

**Weight tying.** ``lm_head.weight`` *is* ``wte.weight``, the same tensor. The
embedding maps a token id to a direction in the residual stream; the unembedding
asks how much of each token's direction is present. Those are the same table
read in two directions. On a 3.5k-vocab model it saves ~450k parameters; on
GPT-2 it saved 38M and improved perplexity.

**Scaled residual init.** Every projection that writes *into* the residual
stream is initialised at ``0.02 / sqrt(2 * n_layer)``. Each of the 2N sublayers
adds its own variance to the stream; without the scaling, the activations at
layer N have grown by ~sqrt(2N) and the first few hundred steps are spent
undoing that instead of learning. GPT-2 does this in one line and never
explains it.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .attention import KVCache
from .blocks import Block, build_norm
from .config import GPTConfig
from .positions import build_positional
from .sampling import greedy_next, sample_next


class GPT(nn.Module):
    """A small decoder-only transformer, written to be read."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        c = config
        self.wte = nn.Embedding(c.vocab_size, c.n_embd)
        self.wpe = build_positional(c.position, c.block_size, c.n_embd)
        self.drop = nn.Dropout(c.dropout)
        self.blocks = nn.ModuleList([
            Block(n_embd=c.n_embd, n_head=c.n_head, block_size=c.block_size,
                  dropout=c.dropout, bias=c.bias, norm=c.norm,
                  use_rope=(c.position == "rope"), rope_base=c.rope_base,
                  mlp_ratio=c.mlp_ratio)
            for _ in range(c.n_layer)
        ])
        self.ln_f = build_norm(c.norm, c.n_embd, c.bias)
        self.lm_head = nn.Linear(c.n_embd, c.vocab_size, bias=False)
        if c.tie_weights:
            self.lm_head.weight = self.wte.weight

        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):     # attn.proj and mlp.proj
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * c.n_layer))

    # -- init -------------------------------------------------------------
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # -- shape bookkeeping ------------------------------------------------
    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.wte.weight.numel()
            if hasattr(self.wpe, "table") and isinstance(self.wpe.table, nn.Parameter):
                n -= self.wpe.table.numel()
        return n

    def parameter_report(self) -> dict[str, int]:
        """Where the parameters actually are — the answer surprises people the
        first time (the MLPs, by a factor of two)."""
        buckets = {"embedding": 0, "attention": 0, "mlp": 0, "norm": 0, "head": 0}
        seen: set[int] = set()
        for name, p in self.named_parameters():
            if id(p) in seen:      # tied head shares the embedding tensor
                continue
            seen.add(id(p))
            if name.startswith(("wte", "wpe")):
                buckets["embedding"] += p.numel()
            elif ".attn." in name:
                buckets["attention"] += p.numel()
            elif ".mlp." in name:
                buckets["mlp"] += p.numel()
            elif name.startswith("lm_head"):
                buckets["head"] += p.numel()
            else:
                buckets["norm"] += p.numel()
        buckets["total"] = sum(v for k, v in buckets.items() if k != "total")
        return buckets

    # -- forward ----------------------------------------------------------
    def forward(self, idx: Tensor, targets: Tensor | None = None, *,
                caches: list[KVCache] | None = None, need_weights: bool = False,
                ) -> tuple[Tensor, Tensor | None]:
        b, t = idx.shape
        offset = caches[0].length if caches else 0
        if offset + t > self.config.block_size:
            raise ValueError(
                f"sequence of {offset + t} exceeds block_size {self.config.block_size}"
            )
        x = self.wte(idx)
        if self.wpe is not None:
            x = x + self.wpe(t, offset=offset)
        x = self.drop(x)
        for i, block in enumerate(self.blocks):
            x = block(x, cache=caches[i] if caches else None, need_weights=need_weights)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        return logits, loss

    # -- interpretability -------------------------------------------------
    def attention_maps(self, idx: Tensor) -> list[Tensor]:
        """Run once and hand back every layer's attention weights ``(B,H,T,T)``."""
        was_training = self.training
        self.eval()
        with torch.no_grad():
            self(idx, need_weights=True)
        maps = [blk.attn.last_attn for blk in self.blocks]
        if was_training:
            self.train()
        if any(m is None for m in maps):
            raise RuntimeError("attention weights were not captured")
        return maps  # type: ignore[return-value]

    # -- generation -------------------------------------------------------
    def new_caches(self, batch: int, device=None) -> list[KVCache]:
        c = self.config
        return [KVCache.empty(batch, c.n_head, c.block_size, c.head_dim,
                              device=device or self.wte.weight.device)
                for _ in range(c.n_layer)]

    @torch.no_grad()
    def generate(self, idx: Tensor, max_new_tokens: int, *, temperature: float = 1.0,
                 top_k: int | None = None, top_p: float | None = None,
                 greedy: bool = False, use_cache: bool = False,
                 generator: torch.Generator | None = None) -> Tensor:
        """Autoregressive decoding. ``use_cache=True`` is the same arithmetic
        with the prefix computed once — the outputs must match token for token
        under greedy decoding, and a test pins exactly that."""
        self.eval()
        caches = self.new_caches(idx.shape[0], device=idx.device) if use_cache else None
        fed = 0
        for _ in range(max_new_tokens):
            if caches is None:
                window = idx[:, -self.config.block_size :]
                logits, _ = self(window)
            else:
                if idx.shape[1] > self.config.block_size:
                    raise ValueError(
                        "the naive KV cache cannot slide: once the context is full it is "
                        "full. Paged/rolling caches are T12."
                    )
                logits, _ = self(idx[:, fed:], caches=caches)
                fed = idx.shape[1]
            nxt = (greedy_next(logits[:, -1, :]) if greedy else
                   sample_next(logits[:, -1, :], temperature=temperature, top_k=top_k,
                               top_p=top_p, generator=generator))
            idx = torch.cat((idx, nxt), dim=1)
        return idx

    # -- optimiser --------------------------------------------------------
    def configure_optimizers(self, weight_decay: float = 0.1, lr: float = 3e-4,
                             betas: tuple[float, float] = (0.9, 0.95)) -> Any:
        """Decay matrices, not vectors.

        Weight decay on a LayerNorm gain or a bias is a slow instruction to
        delete the feature; on a weight matrix it is a useful prior. Splitting
        by ``dim >= 2`` is the whole rule.
        """
        decay, no_decay, seen = [], [], set()
        for p in self.parameters():
            if not p.requires_grad or id(p) in seen:
                continue
            seen.add(id(p))
            (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [{"params": decay, "weight_decay": weight_decay},
                  {"params": no_decay, "weight_decay": 0.0}]
        return torch.optim.AdamW(groups, lr=lr, betas=betas)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"GPT({self.config.summary()}, params={self.num_params():,})"
