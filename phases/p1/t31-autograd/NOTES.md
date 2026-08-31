# T31 · Autograd engine — Notes

## The idea, in one paragraph

An expression is a DAG. Forward, you evaluate it. Backward, you want
d(output)/d(everything). The chain rule says you can get all of those in a single
reverse sweep, because each node only needs to know one local fact: how its own
output changes when each of its inputs changes. Nothing in `__mul__` knows what
the loss is. That locality is the entire design.

For a .NET reader: think `System.Linq.Expressions`, except every node also
carries a closure that *pushes* an accumulated derivative down into its
children, and `backward()` is a reverse topological walk calling those closures
once each.

## The two rules that make it correct

**1. Accumulate, never assign.** If a node feeds two consumers, it receives
gradient from both, and the total is the sum. `y = x*x + x` at x=3 must give 7,
not 6 and not 1. This is the bug that produces a network that trains *slightly*
worse, which is undebuggable.

**2. Reverse topological order.** A node may push into its children only after
every one of its own consumers has pushed into it. `step2` builds a diamond
graph and runs the wrong order deliberately so you can see the size of the
error: it is not a crash, it is a plausible number.

Both topological sorts here are **iterative**, not recursive. A 20,000-node
chain — trivially reachable once you unroll a network to scalars — blows
Python's 1000-frame recursion limit. The two-phase explicit stack (push node,
push children, emit node when children are done) is the standard fix.

## Broadcasting: the part that costs a day

Forward, `(32, 8) + (8,)` is free and silent. Backward it is a sum: the bias was
used by 32 rows, so its gradient is the sum of 32 upstream gradients.

> **forward broadcast == backward sum-and-reshape**

`unbroadcast(grad, shape)` is that sentence in six lines, and it does two
reductions in this order: sum away leading axes that did not exist in the target
shape, then sum (with `keepdims`) any axis that was length-1 in the target and
got stretched. Gradient is conserved through both — only regrouped, which is
what `test_unbroadcast_shapes_and_conservation` asserts.

Get it wrong and shapes still line up often enough that you will not notice.
That is why every op in `tensor.py` is gradchecked in `bench/`.

## Gotchas, each one paid for

- **`|z|` is not a constant.** In `bce_with_logits` the numerically-safe identity
  `log(1+e^z) = max(z,0) + log(1+e^-|z|)` needs `|z|` to stay differentiable.
  Writing `Tensor(np.abs(z.data))` computes the forward value correctly and
  silently zeroes that term's gradient. Build it as `z * sign(z)` with the
  *sign* detached instead. Finite differences catch this in a second; a loss
  curve never does.
- **`sum` backward is a broadcast.** The mirror image of the rule above. Every
  element that was added into the total gets the same gradient back.
- **`max` backward and ties.** NumPy's `argmax` picks the first maximum; this
  engine splits the gradient evenly across ties instead, so that a symmetric
  perturbation in gradcheck agrees with the analytic answer. Both conventions
  are defensible — the sin is not documenting which one you chose.
- **Fancy indexing needs `np.add.at`.** `g[idx] += v` silently drops repeated
  indices; the buffered version does not.
- **`backward()` zeroes first.** Calling it twice on the same graph must not
  double the gradients. PyTorch instead accumulates across calls and makes you
  call `zero_grad()`; this engine takes the other choice and the test pins it.
- **Adam's bias correction is not cosmetic.** Without dividing by `1 - beta^t`,
  the first ~10 steps are scaled down by up to 1000×. A model that "needs a
  warmup" is sometimes just missing this.

## Initialisation, measured rather than recited

The folklore says big weights ⇒ vanishing gradients. The measurement in `step4`
is more careful: scaling weights up by 5.6× saturates ~57% of tanh units at ±1,
but the gradient reaching layer 1 does *not* simply collapse, because the
backward pass multiplies by the same enlarged `Wᵀ` that caused the saturation.
What actually breaks is learning, and it breaks unevenly:

| init | saturated units | SGD(0.1), 300 steps | Adam(0.01), 300 steps |
|:--|--:|--:|--:|
| N(0,1) × 5.6 | ~57% | MSE 1.72 → **0.0925** | 1.72 → 0.0004 |
| 1/√fan_in (Xavier) | ~0% | 0.82 → **0.0023** | 0.82 → 0.0001 |

So: with plain SGD the bad init is 40× worse, and with Adam it is nearly
invisible. Adam's per-parameter step size papers over initialisation problems —
useful in practice, and a good reason not to conclude "the init is fine" from an
Adam run.

## Why the scalar engine is a teaching tool and nothing more

One 64×8 → 16 layer, unrolled to scalars, is **27,777 graph nodes** and runs
**498× slower** than the four-line tensor version. Every node is a Python object
with a closure. This is the concrete reason frameworks are array-first, and it
is worth measuring once so the number is yours.

## What to carry into the rest of the course

- Phase 7's Flash Attention kernel is this backward pass, written so that it
  never materialises the attention matrix. Same chain rule, different memory
  budget.
- The max-subtraction identity used in `bce_with_logits` is the *same* trick
  T45A derives properly for softmax. That is not a coincidence; it is one idea
  appearing twice.
- Once you have written `unbroadcast`, every "shape mismatch in backward" error
  message in PyTorch becomes readable.
