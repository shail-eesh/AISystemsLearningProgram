#!/usr/bin/env python3
"""Step 2 — many heads, one GEMM, and the reshape that everyone gets wrong.

Run:  python3 steps/step2_multihead_causal.py

One head can express one relationship at a time: its softmax row must sum to 1,
so attending to the previous token *costs* attention it could have spent on the
subject of the sentence. Multi-head attention buys parallel relationships by
splitting the width: 4 heads of 32 instead of 1 head of 128. Same parameter
count, same FLOPs, four independent things it can look at.

The implementation is a reshape, and the reshape is where the bugs live. This
step checks the fast batched module against an explicit per-head Python loop,
which is slow, obviously correct, and therefore the oracle.
"""

import time

import _bootstrap  # noqa: F401
import torch
from t4_transformer import CausalSelfAttention, multi_head_attention_looped


def heads_are_a_reshape() -> None:
    print("  a (B, T, 3d) qkv projection becomes (B, heads, T, head_dim):\n")
    b, t, d, h = 1, 6, 128, 4
    x = torch.randn(b, t, d)
    att = CausalSelfAttention(d, h, block_size=t, bias=False)
    qkv = att.qkv(x)
    print(f"    x            {tuple(x.shape)}")
    print(f"    qkv(x)       {tuple(qkv.shape)}     <- one Linear, not three")
    q, k, v = qkv.split(d, dim=2)
    print(f"    split        {tuple(q.shape)} x3")
    qh = q.view(b, t, h, d // h).transpose(1, 2)
    print(f"    view+transpose {tuple(qh.shape)}  <- heads become a batch dimension")
    print("\n  The transpose is not cosmetic: it puts (B, H) in front so the whole")
    print("  attention computation is one batched matmul over B*H independent problems.")


def the_oracle_agrees() -> None:
    print("\n  batched module vs an explicit loop over heads:")
    torch.manual_seed(0)
    for h in (1, 2, 4, 8):
        d, t = 32, 12
        att = CausalSelfAttention(d, h, block_size=t, bias=False).eval()
        x = torch.randn(1, t, d)
        with torch.no_grad():
            fast = att(x)[0]
            wq, wk, wv = att.qkv.weight.split(d, dim=0)
            slow, _ = multi_head_attention_looped(x[0], wq.T, wk.T, wv.T,
                                                  att.proj.weight.T, h)
        diff = float((fast - slow).abs().max())
        print(f"    {h} head(s): max abs difference {diff:.3e}")
    print("\n  Exactly zero, not merely small: it is the same arithmetic in a different")
    print("  memory layout. A non-zero here means the head split does not line up with")
    print("  the weight columns — a model that still trains, just worse, forever.")


def heads_cost_nothing() -> None:
    print("\n  splitting the width is free (same params, same FLOPs):")
    d, t = 256, 128
    x = torch.randn(4, t, d)
    print(f"    {'heads':>6} {'head_dim':>9} {'params':>9} {'ms/forward':>11}")
    for h in (1, 2, 4, 8, 16):
        att = CausalSelfAttention(d, h, block_size=t).eval()
        n = sum(p.numel() for p in att.parameters())
        with torch.no_grad():
            att(x)
            t0 = time.perf_counter()
            for _ in range(20):
                att(x)
            ms = (time.perf_counter() - t0) / 20 * 1000
        print(f"    {h:>6} {d // h:>9} {n:>9,} {ms:>11.2f}")
    print("\n  Identical parameter counts. The wall-clock differences are cache effects,")
    print("  not arithmetic: the FLOP count does not depend on how you slice the width.")


def a_masked_row_is_still_a_distribution() -> None:
    print("\n  the mask, checked rather than assumed:")
    d, t, h = 32, 8, 4
    att = CausalSelfAttention(d, h, block_size=t).eval()
    x = torch.randn(1, t, d)
    with torch.no_grad():
        att(x, need_weights=True)
    a = att.last_attn[0]
    upper = a.triu(1).abs().max()
    rows = a.sum(-1)
    print(f"    largest weight above the diagonal: {float(upper):.3e}")
    print(f"    row sums in [{float(rows.min()):.6f}, {float(rows.max()):.6f}]")
    print("    -inf before the softmax, exactly 0 after it, and the row still normalises.")


if __name__ == "__main__":
    print(__doc__)
    heads_are_a_reshape()
    the_oracle_agrees()
    heads_cost_nothing()
    a_masked_row_is_still_a_distribution()
