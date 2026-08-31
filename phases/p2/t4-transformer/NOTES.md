# T4 · Transformer from scratch — Notes

The intuition, the derivations, and the four bugs this build actually hit.
This file is the source for the five video episodes.

---

## 1. Attention is a soft dictionary lookup

A Python dict lookup is: compare the key you have against every key in the
table, take the one that matches, return its value. Attention is that with two
changes — the comparison is a dot product rather than equality, and instead of
taking the winner you take a *weighted average of everything*, weighted by how
well each key matched.

    query   "what am I looking for?"      one vector per token
    key     "what am I, as a target?"     one vector per token
    value   "what do I contribute?"       one vector per token

    scores = q @ k.T / sqrt(d_k)
    scores = scores.masked_fill(~causal, -inf)
    weights = softmax(scores)
    out = weights @ v

Four lines. Everything else in a transformer is batching, heads, speed, or
plumbing.

The reason it is a *soft* lookup and not a hard one is differentiability: argmax
has no useful gradient, softmax does. That is the whole reason this design could
be trained at all.

## 2. The √d_k is a variance fix, not a ritual

q and k have roughly unit-variance coordinates, so their dot product over `d_k`
coordinates has variance ≈ `d_k` and grows with model width. Feed that into
softmax and it saturates — one weight goes to 1, the others to 0, and the
gradient of a saturated softmax is ≈ 0. The layer stops learning, and it stops
learning *more* the wider you make it, which is the opposite of what you want
from a scaling knob.

Measured in `steps/step1`, averaging the largest softmax weight over 200 random
rows of 64 keys:

| d_k | raw score variance | max weight, unscaled | max weight, scaled |
|--:|--:|--:|--:|
| 4 | 3.84 | 0.273 | 0.104 |
| 16 | 16.27 | 0.584 | 0.110 |
| 64 | 63.83 | 0.810 | 0.109 |
| 256 | 266.08 | 0.900 | 0.109 |
| 1024 | 1012.50 | 0.959 | 0.105 |

Unscaled, the model marches towards a one-hot attention as it gets wider.
Scaled, it stays put.

## 3. Multi-head is a reshape, and the reshape is where the bugs are

One head must spend its softmax row on one relationship: attending to the
previous token *costs* attention it could have spent on the subject of the
sentence. Four heads of 32 instead of one head of 128 buys four parallel
relationships for the same parameters and the same FLOPs (measured — identical
counts, `steps/step2`).

The implementation is `view(B, T, H, hd).transpose(1, 2)`, which is correct only
if head *h* corresponds to a contiguous block of columns in the fused qkv
weight. Get that wrong and you have a model that still trains, just worse,
forever, with no error message. So the batched module is checked against an
explicit per-head Python loop:

    1, 2, 4, 8 heads: max abs difference 0.0

Exactly zero, not merely small — it is the same arithmetic in a different memory
layout. That test is worth more than any amount of staring at the code.

## 4. Pre-norm vs post-norm: the textbook answer is wrong at small scale

Everyone says pre-norm is better. Here is the 2×2, same seed, same data, no
warmup (`steps/step3`):

| depth | lr | pre-norm | post-norm |
|--:|--:|--:|--:|
| 6 | 3e-3 | 2.7891 | **1.7800** |
| 6 | 1e-2 | **2.4244** | 3.3014 (stalled) |
| 12 | 3e-3 | **2.6616** | 3.2993 (stalled) |
| 12 | 1e-2 | **2.5990** | 3.3013 (stalled) |

At 6 layers and a mild learning rate, post-norm — the original 2017 ordering —
is *better*. Push either knob and it stops training entirely, parking near a
degenerate solution, while pre-norm keeps descending.

**Pre-norm is not free quality. It is robustness.** It buys the right to stack 12,
48, 96 layers and raise the learning rate without a babysitter. That is the trade
every model after GPT-2 took, and stating it as "pre-norm is better" loses the
only part that matters.

The same shape of result appears in the init ablation (`bench/results.json`,
`init_ablation_penalty`): PyTorch's default initialisation beats the GPT-2 recipe
by 0.086 nats on this 4-layer model at 400 steps. The 0.02-std, depth-scaled
recipe is also a bet on depth, not a small-scale win.

## 5. RoPE, derived rather than presented

Start from what attention wants. A score should depend on the *relationship*
between two positions, not their absolute indices. Write the wish down:

    <f(q, m), f(k, n)>  =  g(q, k, m - n)

In two dimensions the answer falls out: a rotation. Rotations compose by adding
angles, and the inner product is rotation-invariant, so rotating q by `mθ` and k
by `nθ` leaves a dot product that depends only on `(m-n)θ`. Stack `d/2`
independent 2-D rotations at geometrically spaced frequencies and you have RoPE.

Measured to float64 round-off (`relative_phase_property`): `max |<R_m q, R_n k> −
<R_{m−n} q, R_0 k>| = 1.8e-15` across head dims 2…64 and both signs of the gap.

Two follow-ons worth keeping straight:

* **The decay claim is about matched content, not about everything.** For two
  *independent* random vectors, rotating one changes nothing about the average
  score magnitude — a rotation is an isometry (measured: flat to within 1% out to
  gap 192). The decay appears when the content matches: `<R_m q, R_0 q> = Σ_i
  |q_i|² cos(m θ_i)`, and those cosines fall out of phase. Measured: 1.00 at gap
  0, 0.75 at gap 4, 0.43 at gap 64. It is a soft prior, not a cutoff.
* **Adjacent pairing vs split-half pairing.** We pair coordinates (0,1), (2,3), …
  as the paper does; HuggingFace splits the vector in half and pairs (i, i+d/2).
  Both are valid rotations of the same space, and a checkpoint trained under one
  is garbage under the other. People ship this bug.

### The bake-off

| scheme | val loss | extra params | at 96 tokens | at 128 tokens |
|:--|--:|--:|--:|--:|
| RoPE | **1.0224** | 0 | 0.8120 | 0.9042 |
| learned | 1.1065 | 8,192 | n/a | n/a |
| none | 1.1156 | 0 | — | — |
| sinusoidal | 1.1178 | 0 | 1.2514 | 1.7236 |

Two things not to gloss over.

**"None" is not catastrophic — and it should not be.** A *causal* decoder leaks
position for free: the token at index *i* can see exactly *i+1* things, and the
model can count. Positional encoding sharpens a signal that already exists. In a
bidirectional encoder its absence really is fatal; here it costs 0.09 nats.

**The sinusoidal table can land behind no positions at all.** A fixed table added
to the residual stream competes with the token embeddings for the same
coordinates. At the wrong scale it is noise with a pattern — see the gotcha
below. RoPE never touches the residual stream, which is half of why it wins.

## 6. The KV cache is the removal of a mistake

Generating token *t* without a cache re-runs the entire prefix through every
layer, recomputing keys and values that *cannot* have changed — the mask
guarantees the past does not depend on the future. Counted, for 128 generated
tokens from an 11-token prompt: 9,536 position-passes without a cache, 139 with
one. 68× the work, all of it recomputing identical numbers.

Measured: token-identical output, **3.19×** wall-clock at 180 tokens (the ratio
grows with length; at 32 tokens it is 1.7× because per-step Python overhead is
being paid to save arithmetic there is not much of yet).

The cost is memory, linear in `context × batch × layers × heads × head_dim × 2 ×
4 bytes`. For this toy model that is 1 MB at 256 slots. For a 7B model at 4k
context it is gigabytes *per sequence* — bigger than the activations, and mostly
empty most of the time. That is the exact problem T12's paged cache solves, and
`generate(use_cache=True)` deliberately raises rather than sliding when the
context fills, so you hit the wall yourself.

## 7. Induction heads, and why the pretty picture lied

An **induction head** implements: "the pattern AB happened earlier, I have just
seen A, so B comes next." It needs two layers — one head to copy the previous
token's identity forward, a second to match on it — and it is the mechanism
behind in-context learning.

### The first version of this experiment was wrong

The obvious task is `prefix + prefix` with a fixed prefix length. Our first probe
used it and reported a 30× chance score with **100% accuracy from a one-layer
model** — which should have been impossible. It was: with a fixed period and
learned positional embeddings, "attend to the slot 24 back" is a *positional*
rule. No content matching required, no induction head involved, and a beautiful
attention heat-map to go with it.

The fix is `variable_period_batch`: each row repeats with its own random period
(10–14), so the induction target moves per row and no fixed-offset rule works.
The probe now checks the calibration of its own metric in two directions: a
*planted* head that looks exactly at the target scores 1.000, and a *fixed-offset*
head that always looks 11 back scores 0.128 (3.5× chance — it gets partial credit
on the rows whose period happens to be 11, which is itself worth knowing).

### What the corrected experiment says

| | 2 layers | 1 layer |
|:--|--:|--:|
| repeat-prediction accuracy | **94.0%** | 64.9% |
| best head's attention on the target | 0.347 (9.1×) | **0.438 (11.5×)** |

**The one-layer model has the higher attention score and the much worse
behaviour.** It hedges across the five possible offsets — attention mass lands
near the target often enough to look excellent on a heat-map — and cannot turn
that into a prediction. The emergence trace says the same thing over time: the
attention score plateaus at ~9.3× by step 750 and never moves again, while
accuracy climbs from 49% to 92% over the following 2,250 steps.

**An attention map that looks right is not a circuit that works.** Interpretability
claims need a behavioural control, every time. This is the single most useful
thing in the topic and it came out of a failed test, not a plan.

## 8. Gotchas this build hit

1. **`apply_rope` with a negative offset silently returned garbage.** Python
   indexing turned `cos[-8:-4]` into a slice from the *end* of the angle table,
   so the rotation was wrong by an arbitrary amount with no error. Found by
   parameterising the relative-phase test over `m < n`. Now `offset < 0` raises,
   and the identity check rotates whichever side keeps the offset non-negative.
2. **`apply_rope` capped its own precision at 1e-7.** It called `.float()` on the
   inputs for stability, which *downcast* float64 — so the float64 identity test
   could never do better than float32 round-off. Now it promotes to at least
   float32 and never demotes; the identity measures 1.8e-15.
3. **Sinusoidal positions drowned the token embeddings.** The raw table has
   entries in [−1, 1]; a GPT-2-style embedding is initialised at std 0.02. Adding
   them straight together buries the tokens by ~50×, and the initial loss came out
   *above* `ln(V)` because the logits were nearly pure position. The paper hides
   this by multiplying the embeddings by `sqrt(d_model)` instead — same ratio,
   opposite direction. We scale the table to the embedding init and say so.
4. **The induction probe measured a positional shortcut** — §7 above.

## 9. What this means for AlphaSLM (T15) and beyond

The config that won here is the config T15 scales: pre-norm, RoPE, RMSNorm
available, tied head, `4×` MLP. The KV cache is what T12 replaces with paging,
the softmax inside `scaled_dot_product` is what T45B fuses on the GPU, the whole
attention block is what T7's Flash Attention rewrites to never materialise the
`T×T` matrix, and `interpret.py` is where T22's sparse autoencoders will attach.

The one thing to carry forward from the *results* rather than the code: this
model learned the grammar of the tape perfectly (100% well-formed lines) and its
semantics barely at all (36% of lines have a coherent high/low). Language models
are extremely good at the shape of the thing. Every eval you write after this
should be built with that in mind.
