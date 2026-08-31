---
topic: t4-e5
episode: 5
title: Sampling, and the cache that removes a mistake
voice: am_michael
speed: 0.85
runtime_target_minutes: 9
paper: Holtzman et al. 2020 (nucleus sampling)
---

## s1 · TitleCard

```props
{"title": "Sampling and the KV cache", "subtitle": "Turning logits into text, and stopping the model doing the same work n times.", "topicId": "T4", "episode": "Episode 5"}
```

Welcome back. The last episode of the transformer series. Two topics that both look like implementation details and are not. Sampling, which is where "the model is bad" usually turns out to mean "the sampler is wrong". And the key-value cache, which is not an optimisation so much as the removal of an obvious mistake.

## s2 · ConceptScene

```props
{"eyebrow": "the setup", "title": "The model gives you a distribution. Turning it into text is a separate design problem.", "body": "A trained language model hands back a probability for every token in the vocabulary. Choosing one is not part of the model, and the choice has more effect on perceived quality than most architecture decisions.", "points": ["Greedy: always take the argmax. Deterministic, and it loops.", "Temperature: divide the logits before the softmax. Sharpens or flattens; never removes an option.", "Top-k: keep the k most likely tokens. A fixed budget regardless of confidence.", "Top-p / nucleus: keep the smallest set whose probability mass reaches p. The budget adapts."], "aside": "All four are pure functions of a logit vector, which means they can be tested by construction rather than by squinting at generated text."}
```

Here is the setup. A trained model hands you a probability for every token in the vocabulary. Choosing one of them is not part of the model. It is a separate design problem, and it affects perceived quality more than most architecture decisions do. Four options. Greedy, always take the most likely token: deterministic, and it loops. Temperature, divide the logits before the softmax: below one sharpens, above one flattens, and it never removes an option, only reweights. Top-k, keep the k most likely: a fixed budget regardless of how confident the model is. And top-p, also called nucleus: keep the smallest set of tokens whose probability adds up to p. All four are pure functions of a logit vector, which is worth noticing, because it means you can test them properly instead of squinting at generated text and forming an impression.

## s3 · CodeWalkthrough

```props
{"eyebrow": "the off-by-one everyone writes", "title": "Nucleus filtering, and the token that crosses the threshold", "filename": "phases/p2/t4-transformer/src/t4_transformer/sampling.py", "code": "def top_p_filter(logits, p):\n    sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)\n    probs = sorted_logits.softmax(-1).cumsum(-1)\n    drop = probs - sorted_logits.softmax(-1) >= p   # mass BEFORE this token\n    drop[..., 0] = False                            # always keep the argmax\n    mask = torch.zeros_like(drop).scatter(-1, sorted_idx, drop)\n    return logits.masked_fill(mask, float('-inf'))", "highlights": [{"at": 20, "lines": [2, 3], "caption": "Sort descending, then take the cumulative probability."}, {"at": 130, "lines": [4], "caption": "The subtraction is the whole trick: compare the mass BEFORE this token, so the token that CROSSES p is kept."}, {"at": 250, "lines": [5], "caption": "Without this, top_p(0.0) would leave zero tokens alive and multinomial would throw."}, {"at": 340, "lines": [6], "caption": "Scatter the mask back into the original vocabulary order — the filter must not reorder the logits."}]}
```

Here is nucleus sampling in seven lines, with the mistake everyone makes marked. Sort descending, take the cumulative probability. Now line four. We subtract this token's own probability from the cumulative sum, so what we compare against p is the mass *before* this token. That means the token which *crosses* the threshold is kept, not dropped. Get that wrong — compare the cumulative sum directly — and you systematically drop the token that was about to complete the nucleus, which quietly makes the model more conservative than you asked for. Line five keeps the argmax alive unconditionally, which is what makes top p of zero point zero one still leave exactly one token rather than none. And line six scatters the mask back into vocabulary order, because a filter that reorders your logits is a filter that samples the wrong token.

## s4 · ChartScene

```props
{"eyebrow": "the case for nucleus", "title": "Same model, two positions: confident, and unsure", "kind": "bar", "bars": [{"label": "confident\ntop-k 20", "value": 20, "color": "#d98b4a"}, {"label": "confident\ntop-p 0.9", "value": 1, "color": "#5ac48c"}, {"label": "unsure\ntop-k 20", "value": 20, "color": "#d98b4a"}, {"label": "unsure\ntop-p 0.9", "value": 5, "color": "#5ac48c"}], "yLabel": "tokens still drawable", "caption": "After '...ALPHAINF' the model is 99.8% sure the next character is R. After a date, it is choosing between five tickers."}
```

And this is the argument for nucleus sampling, on two real rows from one real model. On the left, a confident position: the model has just seen A-L-P-H-A-I-N-F and is ninety nine point eight percent sure the next character is R. Top-k with k of twenty keeps twenty candidates — nineteen of which the model has already ruled out, every one of them a typo waiting to be drawn. Top-p with p of zero point nine keeps one. On the right, an genuinely uncertain position: the model has just emitted a date and is choosing between five ticker symbols, with probabilities zero point three one, zero point two one, zero point one nine, zero point one eight, zero point one. Top-k still keeps twenty. Top-p opens up to five. The budget follows the model's own confidence, which is a thing a constant k cannot do, and that is why nucleus sampling is the default in every serving stack you will meet.

## s5 · ChartScene

```props
{"eyebrow": "measured", "title": "Eleven samplers, scored by the tape grammar", "kind": "bar", "bars": [{"label": "greedy", "value": 0.25, "note": "distinct lines"}, {"label": "T=0.5", "value": 1.0, "color": "#5ac48c"}, {"label": "T=0.8", "value": 1.0, "color": "#5ac48c"}, {"label": "T=1.5", "value": 1.0}, {"label": "T=1.0\ntop-k 5", "value": 1.0}, {"label": "T=0.8\ntop-p 0.9", "value": 1.0, "color": "#5ac48c"}], "yLabel": "fraction of generated lines that are distinct", "caption": "Greedy scores 100% on grammar and 25% on distinctness — it emits the same safe line over and over."}
```

We swept eleven settings and scored each on two things: how many generated lines match the tape grammar exactly, and how many are distinct. Greedy is the interesting one. It scores one hundred percent on grammar and twenty five percent on distinctness. It produces the same safe line over and over, because the argmax of a language model is not fluent text — it is the least risky token at every step, forever, and least-risky compounds into a loop. At the other end, temperature one point five scores zero percent on grammar: variety bought by breaking the structure. The useful settings sit in the middle, and the two columns together catch both failure modes, which one column alone would not.

## s6 · ConceptScene

```props
{"eyebrow": "the cache", "title": "Generating token t re-runs a prefix that cannot have changed", "body": "The causal mask guarantees the past does not depend on the future. So the keys and values for positions 0 to t-1 are bit-for-bit identical to what they were on the previous step — and without a cache you compute them again anyway.", "points": ["Counted, for 128 generated tokens from an 11-token prompt:", "without a cache: 9,536 position-passes through every layer", "with a cache: 139", "68x the work, all of it recomputing identical numbers."], "aside": "This is not an optimisation trick. It is the removal of an obvious mistake, and it is quadratic."}
```

Now the cache. Think about what generating token t costs without one. You take the whole sequence so far, run it through every layer, and read the logits off the last position. Next step, you do it again with one more token. But the causal mask guarantees that nothing in the past depends on the future, so the keys and values for every position before the newest one are bit-for-bit what they were a moment ago. You computed them again anyway. Counted for one hundred and twenty eight generated tokens from an eleven-token prompt: nine thousand five hundred and thirty six position-passes through every layer without a cache, one hundred and thirty nine with one. Sixty eight times the work, every bit of it reproducing numbers you already had. That is not a missed optimisation. That is a mistake, and it is quadratic.

## s7 · ChartScene

```props
{"eyebrow": "measured", "title": "The speedup grows with length, because the waste does", "kind": "line", "series": [{"label": "no cache (seconds)", "values": [0.07, 0.17, 0.42, 0.92]}, {"label": "cached (seconds)", "values": [0.04, 0.09, 0.15, 0.23], "color": "#5ac48c"}], "xLabel": "tokens generated: 32, 64, 128, 200", "yLabel": "seconds", "caption": "1.73x at 32 tokens, 3.99x at 200. Identical output at every length — the cache changes nothing but the arithmetic done."}
```

Measured on two CPU cores, and the shape is the point. At thirty two tokens the cache is one point seven times faster. At sixty four, one point nine. At one hundred and twenty eight, two point eight. At two hundred, four times. The speedup grows with length because the thing being avoided grows with length. And at very short lengths a cache can actually lose, because it pays fixed per-step overhead to save arithmetic there is not much of yet. The output is token-identical at every length under greedy decoding, and there is a test that asserts exactly that — plus a stronger one that compares the *logits*, not just their argmax, because a subtly wrong cache can survive greedy decoding for quite a while before it shows.

## s8 · Callout

```props
{"kind": "gotcha", "heading": "With RoPE, rotate the key BEFORE you cache it", "body": "A cached key carries the position it had when it was stored. If you rotate after reading from the cache, every cached key is rotated by the current position instead of its own, and the model is quietly wrong in a way no crash will tell you about.", "code": "if self.use_rope:\n    q = apply_rope(q, cos, sin, offset=offset)\n    k = apply_rope(k, cos, sin, offset=offset)   # BEFORE the append\nif cache is not None:\n    k, v = cache.append(k, v)"}
```

One gotcha, and it is specific to the combination of the last two episodes. With rotary embeddings, the key is rotated by its position. A cached key was stored at some earlier position, and it has to carry that rotation with it. So you rotate before you write into the cache, never after you read out of it. Get the order wrong and every cached key is rotated by the *current* position instead of its own, the relative-phase property silently stops holding, and the model is wrong in a way that produces no crash and no obvious garbage. Our test for this checks the offset path against a contiguous forward pass, and it is one of the few tests in the repository I would call load-bearing.

## s9 · ConceptScene

```props
{"eyebrow": "the wall", "title": "And then the cache runs out", "body": "The naive cache is one preallocated tensor per layer, filled left to right. Its size is linear in context x batch x layers x heads x head_dim x 2 x 4 bytes — and when it is full, it is full.", "points": ["1 MB here, for a 256-token context on a 4-layer model.", "For a 7B model at 4k context: gigabytes, per sequence.", "Bigger than the model's activations, and mostly empty most of the time.", "generate(use_cache=True) raises rather than sliding — on purpose."], "aside": "That is exactly the problem T12's paged KV cache solves, and you cannot appreciate the design until you have hit this wall with your own model."}
```

And then the cache runs out, which is the last thing this topic has to teach. Ours is the naive version: one preallocated tensor per layer, filled left to right, and when it is full it is full. The size is linear in context times batch times layers times heads times head dimension, times two for keys and values, times four bytes. Here that is one megabyte for a two hundred and fifty six token context on a four layer model, which is nothing. For a seven billion parameter model at four thousand tokens of context it is gigabytes — per sequence. Larger than the model's activations, and mostly empty most of the time, because most requests are shorter than the maximum. Our generate function raises an error rather than quietly sliding the window, deliberately, so that you meet the wall yourself. That wall is the entire motivation for T12, the paged key-value cache, and the design will make a great deal more sense to you having hit it.

## s10 · RecapScene

```props
{"eyebrow": "recap", "title": "The series, in five sentences", "points": ["Attention is a differentiable dictionary lookup, and the sqrt(d_k) is a variance fix.", "The block is two writes to a shared residual bus; the MLP holds two thirds of the parameters.", "RoPE falls out of one wish about relative position, and is the only scheme with an answer past the training length.", "The model learned the grammar completely and the semantics barely — and our first interpretability experiment measured a positional shortcut.", "The KV cache is the removal of quadratic waste, and its memory is the reason paged caches exist."], "ifSkipped": "This architecture is what AlphaSLM, the embedding model, the ViT and the speech model are all instances of. Everything after this is a variation on what you have just built.", "next": "T15 — AlphaSLM: this architecture, FinTok, and a real pretraining pipeline."}
```

That is the series. Five things to carry forward. Attention is a differentiable dictionary lookup, and the square root is a variance fix. The block is two writes to a shared residual bus, and the MLP holds two thirds of the parameters. RoPE falls out of one wish about relative position and is the only scheme that has an answer past the training length. The model learned the grammar of the tape completely and its meaning barely, and our first interpretability experiment measured a positional shortcut rather than the circuit we were looking for. And the key-value cache removes quadratic waste, at a memory cost that is the whole reason paged caches exist. What you have built is the architecture that AlphaSLM, the embedding model, the vision transformer and the speech model in this course are all instances of. Next topic, T15: this architecture, plus the tokenizer you built in phase one, plus a real pretraining pipeline. See you there.
