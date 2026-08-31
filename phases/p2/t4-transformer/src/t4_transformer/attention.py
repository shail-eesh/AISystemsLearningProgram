"""Attention, from one head written the slow way to the batched real thing.

Three implementations live here on purpose:

* ``single_head_attention`` — a function over plain tensors, no modules, no
  batching, no heads. This is the one to read first and the one Episode 1
  walks through line by line.
* ``multi_head_attention_looped`` — heads as an explicit Python loop. Obviously
  correct, obviously slow. Kept as the *oracle* the fast path is tested against.
* ``CausalSelfAttention`` — the batched module the model actually uses: one
  fused qkv projection, heads as a reshaped dimension, optional RoPE, optional
  KV cache.

The pedagogy is the test suite: (2) and (3) must agree to floating-point noise,
so the fast version is never trusted, only checked.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .positions import apply_rope, rope_angles

NEG_INF = float("-inf")


def causal_mask(t: int, device=None) -> Tensor:
    """``True`` where a query may look. Lower triangular, including the diagonal.

    A token attends to itself and to its past. Not to its future — that would
    be reading the answer off the exam paper, and the loss would look wonderful
    right up until generation time.
    """
    return torch.ones(t, t, dtype=torch.bool, device=device).tril()


def scaled_dot_product(q: Tensor, k: Tensor, v: Tensor, *, causal: bool = True,
                       mask: Tensor | None = None,
                       dropout_p: float = 0.0, training: bool = False,
                       ) -> tuple[Tensor, Tensor]:
    """The four lines the whole paper is about, plus the mask.

        scores = q @ k.T / sqrt(d_k)
        scores = scores.masked_fill(~mask, -inf)
        attn   = softmax(scores, dim=-1)
        out    = attn @ v

    Why ``sqrt(d_k)``: q and k are roughly unit-variance per coordinate, so
    their dot product over ``d_k`` coordinates has variance ~``d_k`` and grows
    with width. Feed that into softmax and it saturates — one weight goes to 1,
    the rest to 0, and the gradient of a saturated softmax is ~0. Dividing by
    ``sqrt(d_k)`` puts the scores back at unit variance so the model can still
    learn. It is a variance fix, not a normalisation ritual.

    Returns ``(out, attn)`` — the weights come back because half of
    interpretability is looking at them.
    """
    d_k = q.shape[-1]
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is None and causal:
        mask = causal_mask(scores.shape[-2], device=q.device)
        if scores.shape[-1] != scores.shape[-2]:
            # cached generation: q is the tail of a longer key sequence
            full = torch.ones(scores.shape[-2], scores.shape[-1],
                              dtype=torch.bool, device=q.device)
            past = scores.shape[-1] - scores.shape[-2]
            mask = full.tril(diagonal=past)
    if mask is not None:
        scores = scores.masked_fill(~mask, NEG_INF)
    attn = F.softmax(scores, dim=-1)
    if dropout_p > 0.0 and training:
        attn = F.dropout(attn, p=dropout_p, training=True)
    return attn @ v, attn


def single_head_attention(x: Tensor, w_q: Tensor, w_k: Tensor, w_v: Tensor,
                          *, causal: bool = True) -> tuple[Tensor, Tensor]:
    """One head over one sequence: ``x`` is ``(T, d_model)``, weights ``(d_model, d_head)``.

    The retrieval metaphor, made literal:

    * every token emits a **query** — "what am I looking for?"
    * every token emits a **key** — "what am I, as a lookup target?"
    * every token emits a **value** — "what do I contribute if you pick me?"

    A query is compared against every key; the comparison scores become weights;
    the output is the weighted average of the values. That is the whole
    mechanism. Everything after this is batching, heads, and speed.
    """
    if x.ndim != 2:
        raise ValueError(f"single_head_attention takes one (T, d_model) sequence, got {tuple(x.shape)}")
    q, k, v = x @ w_q, x @ w_k, x @ w_v
    return scaled_dot_product(q, k, v, causal=causal)


def multi_head_attention_looped(x: Tensor, w_q: Tensor, w_k: Tensor, w_v: Tensor,
                                w_o: Tensor, n_head: int, *, causal: bool = True,
                                ) -> tuple[Tensor, Tensor]:
    """Heads as a Python ``for`` loop — the oracle, not the implementation.

    ``w_*`` are ``(d_model, d_model)`` and get *sliced* into per-head blocks,
    exactly the way the reshape in ``CausalSelfAttention`` slices them. If the
    two disagree, the reshape is wrong — which is the single most common bug in
    a hand-written transformer, because it produces a model that trains, just
    worse.
    """
    t, d_model = x.shape
    hd = d_model // n_head
    outs, attns = [], []
    for h in range(n_head):
        sl = slice(h * hd, (h + 1) * hd)
        out, attn = single_head_attention(x, w_q[:, sl], w_k[:, sl], w_v[:, sl], causal=causal)
        outs.append(out)
        attns.append(attn)
    return torch.cat(outs, dim=-1) @ w_o, torch.stack(attns)


@dataclass
class KVCache:
    """Per-layer key/value store for incremental decoding.

    Generation without a cache is quadratic *in wasted work*: to emit token
    ``t`` you re-run the whole prefix through every layer, recomputing keys and
    values that cannot have changed — the past is causally frozen. The cache
    keeps them.

    Deliberately the naive version: one preallocated ``(B, H, block, hd)``
    tensor per layer, filled left to right. T12 replaces it with paged blocks;
    seeing why *this* runs out of memory first is the point.
    """

    k: Tensor
    v: Tensor
    length: int = 0

    @classmethod
    def empty(cls, batch: int, n_head: int, block_size: int, head_dim: int,
              device=None, dtype=torch.float32) -> KVCache:
        shape = (batch, n_head, block_size, head_dim)
        return cls(torch.zeros(shape, device=device, dtype=dtype),
                   torch.zeros(shape, device=device, dtype=dtype), 0)

    def append(self, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Store the new keys/values, return everything seen so far."""
        t = k.shape[-2]
        if self.length + t > self.k.shape[-2]:
            raise ValueError(
                f"KV cache holds {self.k.shape[-2]} positions; asked to store "
                f"{self.length + t}. A fixed cache is a hard context limit — this "
                f"is exactly the wall T12's paged cache is built to remove."
            )
        self.k[:, :, self.length : self.length + t] = k
        self.v[:, :, self.length : self.length + t] = v
        self.length += t
        return self.k[:, :, : self.length], self.v[:, :, : self.length]

    def reset(self) -> None:
        self.length = 0

    @property
    def bytes(self) -> int:
        return self.k.numel() * self.k.element_size() * 2


class CausalSelfAttention(nn.Module):
    """Batched multi-head causal self-attention.

    One ``Linear(d, 3d)`` instead of three ``Linear(d, d)``: identical maths,
    one GEMM instead of three, and the reason every real implementation looks
    like this.
    """

    def __init__(self, n_embd: int, n_head: int, *, block_size: int, dropout: float = 0.0,
                 bias: bool = True, use_rope: bool = False, rope_base: float = 10_000.0) -> None:
        super().__init__()
        if n_embd % n_head:
            raise ValueError(f"n_embd={n_embd} not divisible by n_head={n_head}")
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        self.block_size = block_size
        self.dropout = dropout
        self.use_rope = use_rope
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.resid_dropout = nn.Dropout(dropout)
        self.register_buffer("_mask", causal_mask(block_size), persistent=False)
        if use_rope:
            cos, sin = rope_angles(self.head_dim, block_size, base=rope_base)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
        self.last_attn: Tensor | None = None

    def _split_heads(self, x: Tensor) -> Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.n_head, self.head_dim).transpose(1, 2)

    def forward(self, x: Tensor, *, cache: KVCache | None = None,
                need_weights: bool = False) -> Tensor:
        b, t, c = x.shape
        if c != self.n_embd:
            raise ValueError(f"expected width {self.n_embd}, got {c}")
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)
        q, k, v = self._split_heads(q), self._split_heads(k), self._split_heads(v)

        offset = cache.length if cache is not None else 0
        if self.use_rope:
            q = apply_rope(q, self.rope_cos, self.rope_sin, offset=offset)
            k = apply_rope(k, self.rope_cos, self.rope_sin, offset=offset)

        if cache is not None:
            k, v = cache.append(k, v)

        t_k = k.shape[-2]
        mask = self._mask[offset : offset + t, :t_k]
        out, attn = scaled_dot_product(q, k, v, mask=mask,
                                       dropout_p=self.dropout, training=self.training)
        self.last_attn = attn.detach() if need_weights else None
        out = out.transpose(1, 2).contiguous().view(b, t, c)
        return self.resid_dropout(self.proj(out))
