"""Norms, the MLP, and the block that wraps attention in both.

The block is where the *architecture* decisions live, and there are only three
of them:

1. **Pre-norm or post-norm.** Vaswani 2017 normalises *after* the residual add;
   every model since GPT-2 normalises *before* the sublayer. Pre-norm leaves a
   clean identity path from the loss to the embeddings, which is what lets deep
   stacks train without a warmup babysitter. We use pre-norm and keep post-norm
   available so the difference can be measured rather than believed.
2. **LayerNorm or RMSNorm.** RMSNorm drops the mean-subtraction and the bias:
   it rescales by the root-mean-square only. Same job, ~7 fewer ops per element,
   no measurable quality cost — hence LLaMA and everything downstream.
3. **What the MLP is for.** Attention *moves* information between positions;
   the MLP *transforms* it in place, and it holds roughly two thirds of the
   parameters. A transformer that only attends is a very expensive weighted
   average.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .attention import CausalSelfAttention, KVCache


class RMSNorm(nn.Module):
    """x / rms(x) * gain, with rms(x) = sqrt(mean(x^2) + eps).

    No mean subtraction, no bias. Zhang & Sennrich 2019 observed that the
    re-centering in LayerNorm does almost nothing and the re-scaling does
    almost everything.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x32 = x.float()
        rms = torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x32 * rms).to(dtype) * self.weight


class LayerNorm(nn.Module):
    """LayerNorm with an *optional* bias — nanoGPT's trick, and PyTorch cannot do it.

    ``nn.LayerNorm(bias=False)`` did not exist for most of PyTorch's life, and
    GPT-2-scale ablations show the bias buys nothing. Two lines to keep the
    choice.
    """

    def __init__(self, dim: int, bias: bool = True, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, self.eps)


def build_norm(kind: str, dim: int, bias: bool = True) -> nn.Module:
    if kind == "layernorm":
        return LayerNorm(dim, bias=bias)
    if kind == "rmsnorm":
        return RMSNorm(dim)
    raise ValueError(f"unknown norm {kind!r}")


class MLP(nn.Module):
    """Widen by 4x, apply a non-linearity, project back.

    GELU rather than ReLU: smooth, so its gradient does not vanish abruptly for
    slightly-negative inputs, and it is what GPT-2 used. The 4x is convention
    from the paper (d_ff = 2048 for d_model = 512), not a derived optimum.
    """

    def __init__(self, n_embd: int, ratio: int = 4, dropout: float = 0.0,
                 bias: bool = True) -> None:
        super().__init__()
        hidden = ratio * n_embd
        self.fc = nn.Linear(n_embd, hidden, bias=bias)
        self.proj = nn.Linear(hidden, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.proj(F.gelu(self.fc(x), approximate="tanh")))


class Block(nn.Module):
    """One transformer layer: ``x + attn(norm(x))`` then ``x + mlp(norm(x))``.

    Read the residual stream as a shared bus that every layer reads from and
    writes back into. Attention writes "here is what other positions know";
    the MLP writes "here is what that means". Nothing is ever overwritten —
    only added to — which is why you can delete a middle layer of a trained
    transformer and still get grammatical text out.
    """

    def __init__(self, *, n_embd: int, n_head: int, block_size: int, dropout: float = 0.0,
                 bias: bool = True, norm: str = "layernorm", use_rope: bool = False,
                 rope_base: float = 10_000.0, mlp_ratio: int = 4,
                 post_norm: bool = False) -> None:
        super().__init__()
        self.post_norm = post_norm
        self.ln1 = build_norm(norm, n_embd, bias)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size=block_size, dropout=dropout,
                                        bias=bias, use_rope=use_rope, rope_base=rope_base)
        self.ln2 = build_norm(norm, n_embd, bias)
        self.mlp = MLP(n_embd, ratio=mlp_ratio, dropout=dropout, bias=bias)

    def forward(self, x: Tensor, *, cache: KVCache | None = None,
                need_weights: bool = False) -> Tensor:
        if self.post_norm:  # Vaswani 2017 ordering, kept for the ablation
            x = self.ln1(x + self.attn(x, cache=cache, need_weights=need_weights))
            return self.ln2(x + self.mlp(x))
        x = x + self.attn(self.ln1(x), cache=cache, need_weights=need_weights)
        return x + self.mlp(self.ln2(x))
