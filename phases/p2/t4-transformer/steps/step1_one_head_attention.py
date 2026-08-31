#!/usr/bin/env python3
"""Step 1 — one head, one sequence, and the attention matrix printed as numbers.

Run:  python3 steps/step1_one_head_attention.py

Before any of the machinery: attention is a *soft dictionary lookup*. A query
vector is compared against every key vector; the comparisons become weights that
sum to one; the output is that weighted average of the value vectors.

The sequence below is deliberately rigged so you can see it work. Tokens 0-2 are
random noise. Token 3 is a near-copy of token 1. Because the query and key
projections are the identity here, token 3's query lines up with token 1's key,
and the weight lands where you can point at it.
"""

import _bootstrap  # noqa: F401
import torch
from t4_transformer import causal_mask, scaled_dot_product, single_head_attention


def a_rigged_lookup() -> None:
    torch.manual_seed(0)
    d = 4
    x = torch.randn(4, d)
    x[3] = x[1] + 0.05 * torch.randn(d)      # token 3 "asks about" token 1
    eye = torch.eye(d)
    out, attn = single_head_attention(x, eye, eye, eye)

    print("  attention weights (row = querying token, column = attended token):")
    print(f"  {'':>6}" + "".join(f"{j:>8}" for j in range(4)))
    for i, row in enumerate(attn):
        cells = "".join(f"{w:>8.3f}" for w in row)
        print(f"  tok {i}:{cells}")
    print(f"\n  token 3 puts {attn[3, 1]:.3f} of its weight on token 1 "
          f"(uniform over its 4 visible slots would be 0.250)")
    print(f"  every row sums to {float(attn.sum(-1).mean()):.6f} — softmax guarantees it")
    print(f"  output shape {tuple(out.shape)}: one d-dimensional vector per position")


def the_mask_is_the_whole_causality_story() -> None:
    print("\n  the causal mask (True = allowed to look):")
    m = causal_mask(5)
    for i, row in enumerate(m):
        print(f"  tok {i}: " + " ".join("o" if v else "." for v in row))
    print("\n  Upper triangle blocked = a token can never see its own future.")
    print("  Remove those dots and validation loss looks superb and generation is gibberish,")
    print("  because at generation time the future does not exist yet.")


def scaling_is_a_variance_fix() -> None:
    print("\n  why divide by sqrt(d_k):  (64 keys per row, 200 random rows averaged)")
    torch.manual_seed(1)
    print(f"  {'d_k':>6} {'raw score var':>14} {'scaled var':>11} "
          f"{'max w, unscaled':>16} {'max w, scaled':>14}")
    for d_k in (4, 16, 64, 256, 1024):
        q = torch.randn(200, 64, d_k)
        k = torch.randn(200, 1, d_k)
        raw = (q * k).sum(-1)                      # (200, 64) scores
        scaled = raw / d_k**0.5
        w_un = torch.softmax(raw, -1).max(-1).values.mean()
        w_sc = torch.softmax(scaled, -1).max(-1).values.mean()
        print(f"  {d_k:>6} {float(raw.var()):>14.2f} {float(scaled.var()):>11.2f}"
              f" {float(w_un):>16.3f} {float(w_sc):>14.3f}")
    print("\n  Raw score variance tracks d_k exactly. Unscaled, the largest weight marches")
    print("  towards 1.0 as the model gets wider: softmax saturates, its gradient goes to")
    print("  zero, and the layer stops learning. Scaled, it stays put — the fix is a")
    print("  variance correction, not a normalisation ritual.")


def no_mask_no_causality() -> None:
    print("\n  the same head without the mask:")
    torch.manual_seed(0)
    x = torch.randn(4, 4)
    eye = torch.eye(4)
    _, attn = single_head_attention(x, eye, eye, eye, causal=False)
    print(f"  token 0 now places {float(attn[0, 3]):.3f} of its weight on token 3,")
    print("  which has not been generated yet. That is the bug, not a design choice.")
    _, causal = scaled_dot_product(x, x, x)
    print(f"  with the mask it is exactly {float(causal[0, 3]):.3f}.")


if __name__ == "__main__":
    print(__doc__)
    a_rigged_lookup()
    the_mask_is_the_whole_causality_story()
    scaling_is_a_variance_fix()
    no_mask_no_causality()
