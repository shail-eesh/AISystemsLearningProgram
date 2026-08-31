"""An independent GPT, written against PyTorch's own primitives.

The point of this file is to be *unlike* `src/t4_transformer/model.py` in every
way that does not matter and identical in every way that does:

* attention comes from ``F.scaled_dot_product_attention`` (PyTorch's fused
  kernel) rather than our explicit softmax;
* q, k and v come from three separate ``nn.Linear`` layers rather than one
  fused projection;
* norms are ``torch.nn.LayerNorm``;
* the block is written inline with no configuration options at all.

If our model and this one produce the same logits from the same weights, the
agreement is meaningful — two independent paths through the same mathematics.
This is the "nanoGPT reference config" the capsule asks the loss curve to be
compared against, made executable instead of quoted.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class RefBlock(nn.Module):
    def __init__(self, d: int, h: int) -> None:
        super().__init__()
        self.h = h
        self.ln1 = nn.LayerNorm(d)
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.fc = nn.Linear(d, 4 * d)
        self.proj = nn.Linear(4 * d, d)

    def _heads(self, x: Tensor) -> Tensor:
        b, t, d = x.shape
        return x.view(b, t, self.h, d // self.h).transpose(1, 2)

    def forward(self, x: Tensor) -> Tensor:
        b, t, d = x.shape
        n = self.ln1(x)
        y = F.scaled_dot_product_attention(self._heads(self.q(n)), self._heads(self.k(n)),
                                           self._heads(self.v(n)), is_causal=True)
        x = x + self.o(y.transpose(1, 2).contiguous().view(b, t, d))
        return x + self.proj(F.gelu(self.fc(self.ln2(x)), approximate="tanh"))


class RefGPT(nn.Module):
    """Learned positions, tied head — the GPT-2 recipe, plainly written."""

    def __init__(self, vocab: int, block: int, layers: int, heads: int, d: int) -> None:
        super().__init__()
        self.block = block
        self.wte = nn.Embedding(vocab, d)
        self.wpe = nn.Embedding(block, d)
        self.h = nn.ModuleList([RefBlock(d, heads) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.wte.weight

    def forward(self, idx: Tensor, targets: Tensor | None = None):
        b, t = idx.shape
        x = self.wte(idx) + self.wpe(torch.arange(t, device=idx.device))
        for blk in self.h:
            x = blk(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        return logits, loss


def init_like_gpt2(ref: RefGPT, n_layer: int) -> RefGPT:
    """Give the reference our initialisation, so a curve comparison isolates the
    *implementation* rather than re-measuring the init.

    PyTorch's defaults are not GPT-2's: ``nn.Embedding`` initialises at
    ``N(0, 1)`` and ``nn.Linear`` at a Kaiming-uniform bound, so the residual
    stream starts around fifty times louder than the 0.02-std recipe. That is a
    real difference, and the bench measures it separately as an ablation rather
    than letting it contaminate the implementation comparison.

    What the ablation actually says, on this 4-layer model at 400 steps, is
    worth reading before repeating folklore: the PyTorch default lands
    *slightly ahead*. Like pre-norm, the GPT-2 init recipe is a bet on depth
    and stability, not a free win at small scale — see ``bench/results.json``,
    ``init_ablation_penalty``.
    """
    for module in ref.modules():
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    for name, param in ref.named_parameters():
        if name.endswith(("o.weight", "proj.weight")):
            nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))
    ref.head.weight = ref.wte.weight     # re-tie after re-init
    return ref


def copy_weights_from(ours, ref: RefGPT) -> None:
    """Move our parameters into the reference model, unpacking the fused qkv.

    This is where a wrong head layout would show up: the reference splits the
    projection into three separate Linears, so if our ``view(B,T,H,hd)`` did not
    correspond to contiguous row blocks of the fused weight, the logits would
    diverge.
    """
    d = ours.config.n_embd
    ref.wte.weight.data.copy_(ours.wte.weight.data)
    ref.wpe.weight.data.copy_(ours.wpe.table.weight.data)
    for src, dst in zip(ours.blocks, ref.h, strict=True):
        dst.ln1.weight.data.copy_(src.ln1.weight.data)
        dst.ln1.bias.data.copy_(src.ln1.bias.data)
        wq, wk, wv = src.attn.qkv.weight.data.split(d, dim=0)
        bq, bk, bv = src.attn.qkv.bias.data.split(d, dim=0)
        for lin, w, b in ((dst.q, wq, bq), (dst.k, wk, bk), (dst.v, wv, bv)):
            lin.weight.data.copy_(w)
            lin.bias.data.copy_(b)
        dst.o.weight.data.copy_(src.attn.proj.weight.data)
        dst.o.bias.data.copy_(src.attn.proj.bias.data)
        dst.ln2.weight.data.copy_(src.ln2.weight.data)
        dst.ln2.bias.data.copy_(src.ln2.bias.data)
        dst.fc.weight.data.copy_(src.mlp.fc.weight.data)
        dst.fc.bias.data.copy_(src.mlp.fc.bias.data)
        dst.proj.weight.data.copy_(src.mlp.proj.weight.data)
        dst.proj.bias.data.copy_(src.mlp.proj.bias.data)
    ref.ln_f.weight.data.copy_(ours.ln_f.weight.data)
    ref.ln_f.bias.data.copy_(ours.ln_f.bias.data)


def attention_flops(layers: int, heads: int, d: int, t: int) -> int:
    """Forward FLOPs of the attention scores + weighted sum, for the record."""
    return layers * 2 * 2 * heads * t * t * (d // heads)


def theoretical_min_loss(vocab: int) -> float:
    return math.log(vocab)
