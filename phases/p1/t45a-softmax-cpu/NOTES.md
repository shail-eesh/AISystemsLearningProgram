# T45A · Softmax numerics — Notes

## The shift identity, and why it is the *max*

    softmax(x)_i = exp(x_i − c) / Σ_j exp(x_j − c)     for any constant c

Proof: multiply numerator and denominator by `exp(−c)`. One line, no
approximation. Softmax is invariant to adding a constant to every input.

So any `c` fixes the scale problem. Only `c = max(x)` fixes it *unconditionally*:
it makes the largest exponent exactly `exp(0) = 1` and every other one smaller,
so overflow becomes impossible no matter how spread out the inputs are. Terms
that underflow to zero were negligible anyway — that is the right answer to
within the dtype.

Concretely, `x = [0, 1000]`: with `c = mean = 500` you compute `exp(500)`, which
is still inf in float64. With `c = max` you compute `exp(-1000)` and `exp(0)`.

## The online algorithm

The two-pass version still has to see the whole row before it can start summing,
because it needs the max first. The online version removes that dependency by
maintaining both quantities at once:

    on a new block B:
        m_new = max(m, max(B))
        d_new = d · exp(m − m_new) + Σ exp(B − m_new)

The invariant is `d = Σ exp(x_j − m)` over everything seen so far. When `m`
changes, every accumulated term was scaled by the *old* max, and one multiply by
`exp(m_old − m_new)` repairs all of them at once. That is the entire trick.

Worked by hand on `x = [3, 1, 9, 2]`:

| step | value | m | d | |
|--:|--:|--:|--:|:--|
| 0 | 3 | 3 | 1.000000 | |
| 1 | 1 | 3 | 1.135335 | |
| 2 | 9 | **9** | 1.002814 | max moved; d rescaled by exp(3−9) = 0.0025 |
| 3 | 2 | 9 | 1.003726 | |

and `Σ exp(x − max) = 1.0037260968`, identical.

**The merge is associative.** Two partial `(m, d)` states combine the same way,
so a row can be split across threads, blocks or tiles in any order and
recombined. Flash Attention's inner loop *is* this merge, carrying an output
accumulator alongside `(m, d)`.

## Measured, not asserted

- Online vs two-pass: **max absolute difference 0.0** in float64, float32 *and*
  float16. Not "close enough" — the same arithmetic reassociated.
- Chunk size 1 → 512 → whole row: **0.0** difference. Scheduling choice only.
- Naive softmax fails on **64 of 64** adversarial rows in every dtype.
- Online on a CPU is **1.8× slower** than two-pass. Say this out loud when
  someone claims online softmax is "faster"; it is not, on a CPU with NumPy.
  It is *tileable*, which is a different and much more valuable property.

## Gotchas

- **`exp(-inf − -inf)` is nan.** The empty state starts at `m = -inf`, so the
  first rescale must be guarded. This is the single bug everyone writes when
  implementing online softmax from the paper.
- **float16 cannot resolve neighbouring giant logits.** Near 1e4 the spacing
  between representable float16 values is 8, so `10000` and `9999` are the same
  number and softmax over them is 0.5/0.5. The algorithm is fine; the dtype is
  not. This is why attention logits are accumulated in fp32 even when the
  weights are fp16 — a fact Phase 7 depends on.
- **`log(softmax(x))` is not `log_softmax(x)`.** Any probability that underflowed
  to 0 becomes −inf, and the loss becomes inf. Compute `x − logsumexp(x)`
  instead: at `x = [0, −800]` it returns exactly −800 where the naive route
  returns −inf.
- **Fuse the loss, and the gradient gets simpler.** `−log softmax(x)_t =
  logsumexp(x) − x_t`, whose derivative is `softmax(x) − onehot(t)`. The fused
  backward pass needs no exponentials of the target at all. Every framework does
  this; now you know why.
- **A free invariant:** rows of `cross_entropy_grad` sum to exactly 0, because
  softmax sums to 1 and the one-hot sums to 1. Cheap to assert, and it catches a
  surprising number of implementation slips.

## Carry-forward

- **T4 (transformer):** attention is `softmax(QKᵀ/√d)V` — the stable version,
  applied row-wise, with the mask added as `-inf` before the max.
- **T7 (Flash Attention) and T45B:** the online normaliser is the whole reason
  those kernels exist. Everything derived here is reused unchanged; only the
  memory hierarchy changes.
- **T25 (structured output):** logit processors add `-inf` to forbidden tokens.
  Because of max-subtraction, `-inf` logits are exactly probability 0, with no
  special case anywhere.
