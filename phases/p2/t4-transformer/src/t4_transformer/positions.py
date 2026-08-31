"""Three ways to tell a transformer where a token is — and one way to say nothing.

Self-attention is *permutation-equivariant*: shuffle the tokens and the outputs
shuffle with them, unchanged. Nothing inside the attention arithmetic knows that
"BUY 100 INFY" differs from "100 INFY BUY". Position information is bolted on,
and the three standard bolts are:

1. **learned** — a lookup table of ``block_size`` vectors, added to the token
   embedding. GPT-2 does this. Simple, and hard-capped at ``block_size``.
2. **sinusoidal** — the fixed table from Vaswani 2017: sines and cosines at
   geometrically spaced frequencies, also added to the embedding. Free,
   extrapolates in principle, works less well than advertised in practice.
3. **RoPE** — rotate the query and key vectors themselves, by an angle
   proportional to position, *inside* every attention layer. Su et al. 2021.

The reason RoPE won is one identity, proved in ``relative_phase_property``:
after rotating q at position m and k at position n, their dot product depends
only on ``m - n``. Absolute position enters the model and only *relative*
position comes out of the score — which is what attention actually wanted.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class LearnedPositional(nn.Module):
    """GPT-2's table: one trained vector per absolute slot."""

    def __init__(self, block_size: int, n_embd: int) -> None:
        super().__init__()
        self.block_size = block_size
        self.table = nn.Embedding(block_size, n_embd)

    def forward(self, t: int, offset: int = 0) -> Tensor:
        if offset + t > self.block_size:
            raise ValueError(
                f"position {offset + t} is past the learned table ({self.block_size}); "
                f"a learned table cannot be asked about a slot it never saw in training"
            )
        pos = torch.arange(offset, offset + t, device=self.table.weight.device)
        return self.table(pos)


def sinusoidal_table(block_size: int, n_embd: int) -> Tensor:
    """The Vaswani 2017 table, built exactly as written in the paper.

        PE[pos, 2i]   = sin(pos / 10000^(2i/d))
        PE[pos, 2i+1] = cos(pos / 10000^(2i/d))

    Read it as a clock face per frequency: dimension pair *i* is a hand going
    round at its own rate, fast hands for small *i*, slow hands for large *i*.
    A position is the full set of hand angles — a binary counter in continuous
    clothing.
    """
    if n_embd % 2:
        raise ValueError("sinusoidal positions come in sin/cos pairs; n_embd must be even")
    pos = torch.arange(block_size, dtype=torch.float32).unsqueeze(1)
    i = torch.arange(0, n_embd, 2, dtype=torch.float32)
    inv = torch.exp(-math.log(10_000.0) * i / n_embd)
    table = torch.zeros(block_size, n_embd)
    table[:, 0::2] = torch.sin(pos * inv)
    table[:, 1::2] = torch.cos(pos * inv)
    return table


class SinusoidalPositional(nn.Module):
    """The fixed table, registered as a buffer so it rides along with the module.

    ``scale`` is not decoration. The raw table has entries in [-1, 1], while a
    GPT-2-style embedding is initialised at std 0.02 — so *adding* the raw table
    to the embeddings drowns the tokens under the positions by a factor of ~50,
    and the model spends its first thousand steps digging them back out. The
    paper hides this by multiplying the embeddings by ``sqrt(d_model)``
    instead; same ratio, opposite direction. We scale the table to match the
    embedding init and say so out loud. Measured in ``bench/``: initial loss
    with scale=1.0 sits *above* ln(V) because the logits are pure position.
    """

    def __init__(self, block_size: int, n_embd: int, scale: float = 0.02) -> None:
        super().__init__()
        self.block_size = block_size
        self.scale = scale
        self.register_buffer("table", scale * sinusoidal_table(block_size, n_embd),
                             persistent=False)

    def forward(self, t: int, offset: int = 0) -> Tensor:
        if offset + t > self.block_size:
            raise ValueError(f"position {offset + t} past table of {self.block_size}")
        return self.table[offset : offset + t]


# --------------------------------------------------------------------------
# RoPE
# --------------------------------------------------------------------------


def rope_angles(head_dim: int, seq_len: int, base: float = 10_000.0,
                device=None, dtype=torch.float32) -> tuple[Tensor, Tensor]:
    """cos/sin of the rotation angle for every (position, coordinate-pair).

    ``theta_i = base ** (-2i/head_dim)`` — the same geometric ladder of
    frequencies as the sinusoidal table, which is not a coincidence: RoPE is
    that table applied as a *rotation* instead of an *addition*.

    Returns two ``(seq_len, head_dim/2)`` tensors.
    """
    if head_dim % 2:
        raise ValueError("RoPE rotates pairs of coordinates; head_dim must be even")
    inv_freq = base ** (-torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim)
    pos = torch.arange(seq_len, device=device, dtype=dtype)
    ang = pos[:, None] * inv_freq[None, :]          # (T, head_dim/2)
    return torch.cos(ang), torch.sin(ang)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor, offset: int = 0) -> Tensor:
    """Rotate every coordinate pair of ``x`` by its position's angle.

    ``x`` is ``(B, n_head, T, head_dim)``. Pairing is *adjacent* — (0,1), (2,3),
    ... — which is the original paper's layout. (HuggingFace splits the vector
    in half instead and pairs (i, i+d/2). Both are valid rotations of the same
    space; a checkpoint trained under one convention is garbage under the other,
    and that is a real bug people ship.)

    ``offset`` is the absolute position of ``x[..., 0, :]`` — non-zero during
    cached generation, where the tensor holds only the newest token but that
    token sits at position ``t``.
    """
    *_, t, hd = x.shape
    if hd % 2:
        raise ValueError("head_dim must be even for RoPE")
    if offset < 0:
        # A position is never negative. Without this guard a negative offset
        # silently slices the *end* of the angle table (Python indexing), and
        # the rotation is wrong by an amount nothing downstream can detect.
        raise ValueError(f"offset must be >= 0, got {offset}")
    if offset + t > cos.shape[0]:
        raise ValueError(f"need angles for position {offset + t}, table has {cos.shape[0]}")
    # float32 minimum for stability, but never *downcast* a float64 input: the
    # relative-phase check below is a float64 identity and .float() would cap
    # its error at 1e-7 rather than 1e-16.
    work = torch.promote_types(x.dtype, torch.float32)
    c = cos[offset : offset + t].to(work)            # (T, hd/2)
    s = sin[offset : offset + t].to(work)
    x_pair = x.to(work).reshape(*x.shape[:-1], hd // 2, 2)
    x0, x1 = x_pair[..., 0], x_pair[..., 1]
    out0 = x0 * c - x1 * s
    out1 = x0 * s + x1 * c
    return torch.stack((out0, out1), dim=-1).reshape(x.shape).to(x.dtype)


def relative_phase_property(head_dim: int = 8, m: int = 7, n: int = 3,
                            trials: int = 32, base: float = 10_000.0) -> dict[str, float]:
    """Measure the identity that makes RoPE worth the trouble.

        <R_m q, R_n k>  ==  <R_{m-n} q, R_0 k>

    i.e. the attention score after rotation is a function of the *gap* only.
    We check it numerically on random vectors — the property the docs assert,
    turned into a number you can look at.
    """
    torch.manual_seed(0)
    q = torch.randn(trials, 1, 1, head_dim, dtype=torch.float64)
    k = torch.randn(trials, 1, 1, head_dim, dtype=torch.float64)
    seq = max(m, n, abs(m - n)) + 1
    cos, sin = rope_angles(head_dim, seq, base=base, dtype=torch.float64)
    lhs = (apply_rope(q, cos, sin, offset=m) * apply_rope(k, cos, sin, offset=n)).sum(-1)
    # A gap can be negative, and a *position* cannot. Rotating by the gap means
    # rotating whichever side keeps the offset non-negative:
    #     m >= n :  <R_(m-n) q, k>
    #     m <  n :  <q, R_(n-m) k>
    if m >= n:
        rhs = (apply_rope(q, cos, sin, offset=m - n) * k).sum(-1)
    else:
        rhs = (q * apply_rope(k, cos, sin, offset=n - m)).sum(-1)
    unrotated = (q * k).sum(-1)
    return {
        "max_abs_diff": float((lhs - rhs).abs().max()),
        "mean_abs_score": float(lhs.abs().mean()),
        "shift_changes_score_by": float((lhs - unrotated).abs().mean()),
    }


def build_positional(kind: str, block_size: int, n_embd: int) -> nn.Module | None:
    """The *additive* position modules only.

    RoPE returns None here and is handled inside attention instead — which is
    the whole structural difference between the two families: one edits the
    residual stream once, the other edits q and k in every layer.
    """
    if kind == "learned":
        return LearnedPositional(block_size, n_embd)
    if kind == "sinusoidal":
        return SinusoidalPositional(block_size, n_embd)
    if kind in ("rope", "none"):
        return None
    raise ValueError(f"unknown position kind {kind!r}")
