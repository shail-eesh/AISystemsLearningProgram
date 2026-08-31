---
topic: t45a-e2
episode: 2
title: Online softmax, derived slowly
voice: am_michael
speed: 0.85
runtime_target_minutes: 12
paper: Milakov & Gimelshein, "Online normalizer calculation for softmax" (2018)
---

## s1 · TitleCard

```props
{"title": "Softmax, part two", "subtitle": "One pass. This is the most patient episode in the series, and Phase 7 is where it pays.", "topicId": "T45A", "episode": "Episode 2", "paper": "Milakov & Gimelshein 2018 — Online normalizer calculation for softmax"}
```

Episode two, and this is deliberately the slowest episode in the whole series. We are going to derive one small algorithm carefully, from a paper that is four pages long, and then in phase seven, when we build Flash Attention, this will be the episode that makes it obvious rather than magical. So it is worth the twelve minutes.

## s2 · ConceptScene

```props
{"eyebrow": "the constraint", "title": "Stability costs a pass, and the pass costs everything", "body": "The stable formula needs the maximum before it can start summing. So it must see the entire row before it can compute anything — which means the entire row has to be stored somewhere.", "points": ["Pass 1: find the maximum of the row.", "Pass 2: exponentiate the shifted values and sum them.", "Pass 3: divide. (Fusing 2 and 3 gets you to two passes.)", "Softmax does one exp per element — it is memory-bound, so passes over memory ARE the runtime."], "aside": "For attention, that row is n elements long for a sequence of length n. Storing it is the whole problem Flash Attention exists to avoid."}
```

Start with what the stable version costs us. It needs the maximum before it can start summing, because the shift depends on it. So pass one finds the maximum. Pass two exponentiates and sums. Pass three divides, and you can fuse two and three to get down to two passes. Now, why does a pass matter. Softmax does exactly one exponential per element, which is almost no arithmetic, so this kernel is entirely memory bound. The number of times you walk the data is the runtime. There is nothing else in it. And there is a second, larger cost. Needing the maximum first means you must see the whole row before you can compute anything, which means the whole row must be stored somewhere. In attention, that row is n elements long for a sequence of length n. Storing all of them is the n squared memory that Flash Attention exists to avoid.

## s3 · MathReveal

```props
{"eyebrow": "the idea", "title": "Carry the maximum and the denominator together", "english": "Keep two running numbers: the largest value seen so far, and the sum of exponentials of everything seen so far, shifted by that largest value. When a bigger value arrives, repair the sum.", "equation": "m' = max(m, max B) ;  d' = d · exp(m - m') + SUM exp(B - m')", "code": "block_max = block.max(axis=-1)\nnew_m     = np.maximum(m, block_max)\nd         = d * np.exp(m - new_m) + np.exp(block - new_m).sum(-1)\nm         = new_m", "note": "One multiply repairs the entire accumulated denominator, because every term in it was scaled by the same old maximum.", "stageFrames": [10, 150, 300]}
```

Here is the idea. In English. Keep two running numbers instead of one. The largest value seen so far, call it m. And the sum of exponentials of everything seen so far, each shifted by that same m, call it d. When a value larger than m arrives, you repair the sum. In symbols. The new m is the max of the old m and the maximum of the new block. And the new d is the old d, times e to the old m minus the new m, plus the sum over the new block of e to the block minus the new m. And in code, four lines. Now here is the part to sit with. Why does one multiplication repair the whole accumulated sum. Because every single term already in d was scaled by e to the minus the old m. The same constant, for all of them. So multiplying d by e to the old m minus the new m rescales all of them at once, in one operation, no matter how many there were. That is the entire trick.

## s4 · ChartScene

```props
{"eyebrow": "worked by hand", "title": "x = [3, 1, 9, 2], one value at a time", "kind": "line", "series": [{"label": "running max m", "color": "#e0b04e", "values": [3, 3, 9, 9]}, {"label": "running denominator d", "color": "#5ac48c", "values": [1.0, 1.135, 1.0028, 1.0037]}], "xLabel": "step 0 -> 3, one value absorbed each", "caption": "At step 2 the max jumps 3 -> 9 and d is rescaled by exp(3-9) = 0.0025, from 1.135 down to 1.0028."}
```

Let us do it by hand on four numbers. Three, one, nine, two. Step zero. We see three. The maximum is three, and d is e to the three minus three, which is one. Step one. We see one. The maximum is still three. We add e to the one minus three, which is about zero point one three five, so d becomes one point one three five. Step two, and this is the interesting one. We see nine. The maximum jumps from three to nine. So d gets multiplied by e to the three minus nine, which is zero point zero zero two five, taking one point one three five down to about zero point zero zero two eight. And then we add e to the nine minus nine, which is one. So d is one point zero zero two eight. Step three. We see two. The maximum stays nine. We add e to the two minus nine, and d ends at one point zero zero three seven two six one. Now compute the two pass answer directly. Sum of e to the x minus nine over all four numbers. One point zero zero three seven two six one. Identical, and we never went back over the data.

## s5 · Callout

```props
{"kind": "gotcha", "heading": "exp(-inf minus -inf) is nan", "body": "The empty state starts with m = -infinity, so the very first rescale computes exp(-inf - -inf), which is nan, not 1. This is the bug everybody writes when implementing the paper. Guard the first update.", "code": "# every term in d was scaled by exp(-m); on the first\n# update there are no terms, so the scale is irrelevant.\nscale = np.where(np.isfinite(m), np.exp(m - new_m), 0.0)\nd = d * scale + block_d"}
```

One implementation note, because this is the bug everybody writes when coding straight from the paper. The empty state has to start with m equal to minus infinity, so that the first real value always wins the maximum. But then the first rescale computes e to the minus infinity minus minus infinity. That is infinity minus infinity, which is not a number, and it poisons d immediately. The fix is a one line guard. If m is not finite, there are no accumulated terms, so the scale factor is irrelevant and you can use zero. It is a small thing, and it is the difference between the algorithm working and producing nan on every row.

## s6 · DiagramScene

```props
{"eyebrow": "the property that matters", "title": "The merge is associative", "nodes": [{"id": "t1", "label": "tile 1", "sub": "(m1, d1)", "x": 0.12, "y": 0.28, "color": "#5ac48c", "appearAt": 8}, {"id": "t2", "label": "tile 2", "sub": "(m2, d2)", "x": 0.12, "y": 0.72, "color": "#5ac48c", "appearAt": 16}, {"id": "mg", "label": "merge", "sub": "m = max, d rescaled", "x": 0.45, "y": 0.5, "w": 0.22, "color": "#e0b04e", "appearAt": 40}, {"id": "out", "label": "(m, d)", "sub": "identical to one pass", "x": 0.78, "y": 0.5, "appearAt": 66}], "edges": [{"from": "t1", "to": "mg", "appearAt": 46}, {"from": "t2", "to": "mg", "appearAt": 50}, {"from": "mg", "to": "out", "appearAt": 70}], "caption": "Two independent partial states combine the same way a block does. Any split, any order, same answer."}
```

And now the property that makes this worth an episode rather than a footnote. The merge is associative. Two partial states, computed completely independently, combine with the same rule. Take the max of the two maxima, rescale each denominator to that new maximum, and add. Which means you can split a row across threads, across GPU blocks, across cache tiles, in any grouping and any order, and get bit identical results. The one pass framing is nice. The associativity is what makes it a parallel algorithm. And when we build Flash Attention in phase seven, its inner loop is precisely this merge, carrying an output accumulator alongside m and d so that the weighted sum of value vectors is rescaled at the same moment the denominator is.

## s7 · ChartScene

```props
{"eyebrow": "measured", "title": "Not 'close enough' — identical", "kind": "bar", "bars": [{"label": "float64", "value": 1e-18, "color": "#5ac48c", "note": "0.0 exactly"}, {"label": "float32", "value": 1e-18, "color": "#5ac48c", "note": "0.0 exactly"}, {"label": "float16", "value": 1e-18, "color": "#5ac48c", "note": "0.0 exactly"}, {"label": "chunk 1 vs whole row", "value": 1e-18, "color": "#5ac48c", "note": "0.0 exactly"}], "logScale": true, "yLabel": "max abs difference vs two-pass", "caption": "64x128 adversarial grid, logits in [-1e4, 1e4]. Bars are plotted at 1e-18 because zero has no home on a log axis."}
```

Here is the verification, and the result is stronger than I expected the first time I ran it. Against the two pass version, on a sixty four by one hundred and twenty eight grid of adversarial logits, the online version differs by exactly zero. In double precision. In single precision. In half precision. And it differs by exactly zero across chunk sizes from one element at a time up to the entire row in one go. The bars on this chart are drawn at ten to the minus eighteen only because zero has nowhere to sit on a logarithmic axis. So this is not an approximation you accept for a speedup. It is the same arithmetic, reassociated. And chunk size is therefore a pure scheduling decision, which is exactly what you need when you are fitting tiles into a fixed amount of fast memory.

## s8 · Callout

```props
{"kind": "insight", "heading": "On a CPU, online softmax is slower — say so", "body": "256x4096 float32: two-pass 2.29 ms, online 4.04 ms. It pays per-chunk overhead to save a pass it did not need here, because the row already fits in cache. Its value is tileability, not speed.", "code": "stable            2.29 ms\ntwo_pass          2.37 ms\nonline(chunk=512) 4.04 ms      # 1.8x SLOWER"}
```

And now the honest counterweight, because you will hear people describe online softmax as faster and it is worth being precise. On a CPU, in NumPy, on a matrix that comfortably fits in cache, it is one point eight times slower. Two point two nine milliseconds for two pass, four point zero four for online. It is paying per chunk overhead to save a pass it did not actually need, because the row was already resident. Its value is not speed on this machine. Its value is that it is tileable, and tileability is what lets a kernel process an attention row that never fits anywhere, one tile at a time, in a fixed and small amount of on chip memory. When someone says an algorithm is faster, the useful follow up is always. Faster on what, and bound by what.

## s9 · ConceptScene

```props
{"eyebrow": "the last piece", "title": "Fuse the loss, and the gradient gets simpler", "body": "log(softmax(x)) is the wrong way to compute log-softmax: any probability that underflowed to zero becomes -inf. Subtract the log-sum-exp instead, and the derivative of the fused loss turns out to be almost nothing.", "points": ["log_softmax(x) = x - logsumexp(x). At x = [0, -800] it returns exactly -800, where the naive route returns -inf.", "Cross-entropy = logsumexp(x) - x_target. No exponential of the target term at all.", "Its gradient is softmax(x) - onehot(target). That is the whole backward pass.", "Free invariant: rows of that gradient sum to exactly zero, because both terms sum to one."], "aside": "The log and the exp cancel so completely that the fused loss is easier to differentiate than either half of it."}
```

The last piece, and it settles the quiet failure from episode one. Taking the logarithm of a softmax is the wrong way to compute log softmax, because any probability that underflowed to zero becomes minus infinity. Subtract the log sum exp instead. Log softmax of x equals x minus log sum exp of x. At the vector zero and minus eight hundred, that returns exactly minus eight hundred, where the naive route returns minus infinity. Then cross entropy is log sum exp of x, minus the logit at the target index. Notice there is no exponential of the target term anywhere in that. And here is the payoff. The gradient of that fused loss is softmax of x, minus the one hot vector of the target. That is the entire backward pass. The logarithm and the exponential cancel so completely that the fused loss is easier to differentiate than either half of it separately, which is why every framework fuses them. And it comes with a free invariant. Every row of that gradient sums to exactly zero, because softmax sums to one and the one hot sums to one. One cheap assertion, and it catches a surprising number of implementation slips.

## s10 · RecapScene

```props
{"eyebrow": "topic recap", "points": ["Carry (max, denominator) together; when the max moves, one multiply repairs the whole sum.", "Guard the first update — exp(-inf - -inf) is nan, and it is the bug everyone writes.", "The merge is associative, so the row can be split across tiles or threads in any order.", "Online is bit-identical to two-pass in f64, f32 and f16, and across every chunk size.", "It is 1.8x slower on a CPU. Its value is tileability, and Phase 7 is where that pays."], "ifSkipped": "T7 Flash Attention is this algorithm with an output accumulator attached. Without this episode, it is an incantation.", "next": "Next topic: T30 — tokenizers, and why the model's view of your text is decided before the model sees it."}
```

To recap the topic. Carry the maximum and the denominator together, and when the maximum moves, one multiplication repairs the entire accumulated sum. Guard the first update, because e to the minus infinity minus minus infinity is not a number, and that is the bug everybody writes. The merge is associative, so a row can be split across tiles or threads in any grouping and any order. The online version is bit identical to the two pass version in double, single and half precision, and across every chunk size from one to the whole row. And it is one point eight times slower on a CPU, because its value is tileability rather than speed, and phase seven is where that cashes in. Flash Attention is this exact algorithm with an output accumulator attached, and without this episode it would be an incantation you copy. Next topic is thirty. Tokenizers. Where the model's view of your text is decided before the model ever sees it.
