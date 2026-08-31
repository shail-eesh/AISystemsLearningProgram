#!/usr/bin/env python3
"""Step 4 — RoPE, derived from the property it is designed to have.

Run:  python3 steps/step4_rope_from_scratch.py

Start from what attention *wants*. A score should depend on the relationship
between two positions, not their absolute indices: "the token three back" is a
useful thing to encode; "the token at index 1,047" is not.

Write that as a wish:

    <f(q, m), f(k, n)>  =  g(q, k, m - n)

Ask what f can be. In two dimensions the answer is immediate: a rotation.
Rotating q by angle m*theta and k by n*theta gives a dot product that depends on
(m - n)*theta, because rotations compose by adding angles and the inner product
is rotation-invariant. Stack d/2 independent 2-D rotations at geometrically
spaced frequencies and you have RoPE for a d-dimensional head.

Everything below measures that instead of asserting it.
"""

import math

import _bootstrap  # noqa: F401
import torch
from t4_transformer import GPT, GPTConfig, apply_rope, relative_phase_property, rope_angles
from t4_transformer.positions import sinusoidal_table


def two_dimensions_first() -> None:
    print("  the whole idea, in 2-D, by hand:\n")
    q = torch.tensor([1.0, 0.0])
    k = torch.tensor([0.6, 0.8])

    def rot(v, a):
        c, s = math.cos(a), math.sin(a)
        return torch.tensor([v[0] * c - v[1] * s, v[0] * s + v[1] * c])

    theta = 0.3
    print(f"    {'m':>3} {'n':>3} {'m-n':>5} {'<R_m q, R_n k>':>16}")
    for m, n in [(5, 2), (6, 3), (7, 4), (12, 9), (2, 5), (9, 12)]:
        score = float(rot(q, m * theta) @ rot(k, n * theta))
        print(f"    {m:>3} {n:>3} {m - n:>5} {score:>16.6f}")
    print("\n  Same gap, same score — and the sign of the gap matters, which is what")
    print("  lets a head distinguish 'three back' from 'three forward'.")


def the_property_measured() -> None:
    print("\n  the identity in d dimensions, numerically:")
    for hd in (4, 8, 32, 64):
        r = relative_phase_property(head_dim=hd, m=11, n=4)
        print(f"    head_dim {hd:>3}: max |<R_m q,R_n k> - <R_(m-n) q,R_0 k>| = "
              f"{r['max_abs_diff']:.2e}   (typical score magnitude {r['mean_abs_score']:.2f})")
    print("\n  Float64 round-off, nothing more. The rotation is exact by construction.")


def frequencies_are_a_ruler() -> None:
    print("\n  the frequency ladder (head_dim 16, base 10000):\n")
    cos, sin = rope_angles(16, 64)
    inv = 10_000.0 ** (-torch.arange(0, 16, 2).float() / 16)
    print(f"    {'pair i':>7} {'theta_i':>12} {'wavelength (positions)':>24}")
    for i, f in enumerate(inv):
        print(f"    {i:>7} {float(f):>12.6f} {2 * math.pi / float(f):>24.1f}")
    print("\n  Fast pairs resolve neighbours; slow pairs (wavelength >> context) act as a")
    print("  coarse 'roughly where in the document' signal. Same geometric ladder as the")
    print("  sinusoidal table — RoPE *rotates* by it instead of *adding* it.")


def decay_with_distance() -> None:
    """The honest version of RoPE's "long-term decay" claim.

    For two *independent* random vectors, rotating one changes nothing about the
    average magnitude of the dot product — a rotation is an isometry, and the
    first table below says so. The decay is real but it is about *matched
    content*: a query that would match a key perfectly at gap 0 matches it less
    and less as the gap grows, because
    ``<R_m q, R_0 q> = sum_i |q_i|^2 cos(m * theta_i)`` and those cosines fall
    out of phase with each other.
    """
    print("\n  RoPE and distance — two different questions:\n")
    hd = 64
    cos, sin = rope_angles(hd, 256, dtype=torch.float64)
    torch.manual_seed(0)
    q = torch.randn(4000, 1, 1, hd, dtype=torch.float64)
    k = torch.randn(4000, 1, 1, hd, dtype=torch.float64)

    print("    (a) independent q and k — mean |score| by gap:")
    base = float((q * k).sum(-1).abs().mean())
    cells = []
    for gap in (0, 1, 4, 16, 64, 192):
        s_ = (apply_rope(q, cos, sin, offset=gap) * apply_rope(k, cos, sin, offset=0)).sum(-1)
        cells.append(f"gap {gap:>3}: {float(s_.abs().mean()) / base:>5.3f}")
    print("      " + "   ".join(cells))
    print("      Flat, and it must be: rotation preserves lengths and angles, so for")
    print("      unrelated vectors it cannot change the typical score magnitude.")

    print("\n    (b) *matched* content — the same vector as query and key:")
    norm = float((q * q).sum(-1).mean())
    print(f"      {'gap':>5} {'mean score':>12} {'fraction of gap-0':>19}")
    for gap in (0, 1, 2, 4, 8, 16, 32, 64, 128, 192):
        s_ = (apply_rope(q, cos, sin, offset=gap) * apply_rope(q, cos, sin, offset=0)).sum(-1)
        m = float(s_.mean())
        print(f"      {gap:>5} {m:>12.3f} {m / norm:>19.3f}")
    print("\n      This is the decay the paper means. A perfect content match scores")
    print("      full marks next door and progressively less far away, because the")
    print("      per-pair cosines drift out of phase. Nobody wrote that rule down —")
    print("      it falls out of the rotation, and it is a *prior*, not a hard cutoff:")
    print("      strong enough content still wins at long range.")


def additive_versus_rotational() -> None:
    print("\n  the structural difference, in one table:\n")
    rows = [
        ("where it applies", "residual stream, once", "q and k, every layer"),
        ("parameters", "block_size x d (learned)", "none"),
        ("beyond block_size", "impossible (no row)", "defined (angles continue)"),
        ("what the score sees", "absolute positions", "the gap only"),
        ("KV cache", "cache k as stored", "cache k *after* rotating"),
    ]
    print(f"    {'':<20} {'learned / sinusoidal':<26} {'RoPE'}")
    for a, b, c in rows:
        print(f"    {a:<20} {b:<26} {c}")
    tbl = sinusoidal_table(8, 8)
    print(f"\n    (sinusoidal table row 0 = {[round(float(v), 3) for v in tbl[0][:4]]}...,"
          f" row 1 = {[round(float(v), 3) for v in tbl[1][:4]]}...)")
    print("\n  The last row is a real bug source: rotate keys before caching them, or")
    print("  every cached key silently carries the wrong position on the next step.")


def both_configs_train() -> None:
    print("\n  both position schemes build and run (same seed, one forward):")
    torch.manual_seed(0)
    idx = torch.randint(0, 66, (2, 32))
    for pos in ("learned", "sinusoidal", "rope", "none"):
        torch.manual_seed(0)
        m = GPT(GPTConfig(vocab_size=66, block_size=64, n_layer=2, n_head=4,
                          n_embd=64, position=pos))
        with torch.no_grad():
            _, loss = m(idx[:, :-1], idx[:, 1:])
        extra = m.num_params() - GPT(GPTConfig(vocab_size=66, block_size=64, n_layer=2,
                                               n_head=4, n_embd=64,
                                               position="none")).num_params()
        print(f"    {pos:<12} params +{extra:>6,}   init loss {float(loss):.4f} "
              f"(ln(66) = {math.log(66):.4f})")
    print("\n  Only the learned table costs parameters. Step 5 measures whether it earns them.")


if __name__ == "__main__":
    print(__doc__)
    two_dimensions_first()
    the_property_measured()
    frequencies_are_a_ruler()
    decay_with_distance()
    additive_versus_rotational()
    both_configs_train()
