---
topic: t4-e3
episode: 3
title: Positions, and the rotation that fixed them
voice: am_michael
speed: 0.85
runtime_target_minutes: 9
paper: Vaswani et al. 2017 §3.5; Su et al. 2021 (RoFormer / RoPE)
---

## s1 · TitleCard

```props
{"title": "Positions", "subtitle": "Self-attention cannot tell word order. Three fixes, and why the third one won.", "topicId": "T4", "episode": "Episode 3", "paper": "Su et al. 2021, RoFormer"}
```

Welcome back. Episode three. Today, a fact about attention that should bother you more than it usually does: self-attention has no idea what order the tokens are in. None. And then the three standard ways of telling it, and the one that won, derived from the property it was designed to have rather than presented as a formula to memorise.

## s2 · ConceptScene

```props
{"eyebrow": "the problem", "title": "Shuffle the tokens and the outputs shuffle with them, unchanged", "body": "Attention is permutation-equivariant. Every operation in it — the dot products, the softmax, the weighted sum — treats the sequence as a bag. Nothing inside the arithmetic knows that 'BUY 100 INFY' differs from '100 INFY BUY'.", "points": ["Position is bolted on. It is not emergent and it is not free.", "Fix 1 — learned: a trained vector per absolute slot, added to the embedding. GPT-2 does this.", "Fix 2 — sinusoidal: a fixed table of sines and cosines, also added. Vaswani 2017.", "Fix 3 — RoPE: rotate the query and key vectors themselves, inside every attention layer."], "aside": "The three differ in where they act. The first two edit the residual stream once. The third never touches it."}
```

Start with the fact. Attention is permutation-equivariant. Shuffle the input tokens and the outputs come out shuffled the same way, otherwise unchanged. Every operation inside it — the dot products, the softmax, the weighted sum — treats the sequence as a bag of vectors. Nothing in that arithmetic distinguishes buy one hundred INFY from one hundred INFY buy. So position has to be bolted on, and there are three standard bolts. One, learned: a trained vector for each absolute slot, added to the token embedding. That is GPT-2. Two, sinusoidal: a fixed table of sines and cosines at geometrically spaced frequencies, also added. That is the original paper. Three, rotary — RoPE — which rotates the query and key vectors themselves, inside every attention layer. And the structural difference to hold on to is *where* they act. The first two edit the residual stream once, at the bottom. The third never touches the residual stream at all.

## s3 · ConceptScene

```props
{"eyebrow": "deriving it", "title": "Start from what the score actually wants", "body": "A score should depend on the relationship between two positions, not on their absolute indices. 'The token three back' is a useful thing to encode. 'The token at index 1,047' is not.", "points": ["Write the wish as an equation: <f(q,m), f(k,n)> = g(q, k, m-n).", "Ask what f can be. In two dimensions the answer is immediate: a rotation.", "Rotations compose by adding angles, and the inner product is rotation-invariant.", "So rotating q by m-theta and k by n-theta leaves a score that depends only on (m-n)-theta."], "aside": "That is the entire derivation. RoPE is not a trick someone found; it is the unique simple answer to a question written down properly."}
```

Now instead of presenting RoPE, let us derive it, because it is one of those ideas that is obvious once the question is written down correctly. Start from what a score actually wants. It should depend on the *relationship* between two positions, not their absolute indices. The token three back is a useful thing to encode. The token at index one thousand and forty seven is not. So write that wish as an equation: the inner product of f of q at position m with f of k at position n should equal some function g of q, k, and m minus n. Only the gap. Now ask what f could possibly be. In two dimensions the answer arrives immediately: a rotation. Rotations compose by adding angles, and the inner product is invariant under rotating both vectors together. So rotate q by m theta and k by n theta, and their dot product depends only on m minus n times theta. That is the whole derivation. RoPE is not a trick somebody found by trial and error. It is the unique simple answer to a question that was written down properly.

## s4 · MathReveal

```props
{"eyebrow": "the identity", "title": "The relative-phase property", "english": "Rotate the query by its position and the key by its position, and the score you get depends only on the distance between them — not on where either one is.", "equation": "<R_m q, R_n k>  =  <R_(m-n) q, R_0 k>", "code": "lhs = (apply_rope(q, cos, sin, offset=m) * apply_rope(k, cos, sin, offset=n)).sum(-1)\nrhs = (apply_rope(q, cos, sin, offset=m - n) * k).sum(-1)\n# max |lhs - rhs| over head dims 2..64, both signs of the gap:  1.8e-15", "note": "Float64 round-off. Not an approximation that holds well — an identity that holds exactly."}
```

Here is that identity written three ways. In English: rotate the query by its position and the key by its position, and the score depends only on the distance between them, not on where either one sits. In symbols: the inner product of R m q with R n k equals the inner product of R m minus n q with R zero k. And in code, which is how you check it rather than believe it. We evaluate both sides on random vectors in float sixty four, over head dimensions from two to sixty four, for positive and negative gaps. The largest disagreement is one point eight times ten to the minus fifteen. That is float sixty four round-off. It is not an approximation that happens to hold well. It is an identity, and it holds exactly.

## s5 · DiagramScene

```props
{"eyebrow": "the frequency ladder", "title": "Each coordinate pair is a clock hand at its own rate", "nodes": [{"id": "p0", "label": "pair 0", "sub": "wavelength 6.3", "x": 0.14, "y": 0.2, "appearAt": 10, "color": "#5ac48c"}, {"id": "p1", "label": "pair 1", "sub": "wavelength 19.9", "x": 0.38, "y": 0.2, "appearAt": 22}, {"id": "p2", "label": "pair 2", "sub": "wavelength 62.8", "x": 0.62, "y": 0.2, "appearAt": 34}, {"id": "p3", "label": "pair 3", "sub": "wavelength 199", "x": 0.86, "y": 0.2, "appearAt": 46}, {"id": "fast", "label": "fast hands", "sub": "resolve neighbours", "x": 0.26, "y": 0.72, "w": 0.3, "appearAt": 60}, {"id": "slow", "label": "slow hands", "sub": "roughly where in the document", "x": 0.74, "y": 0.72, "w": 0.36, "appearAt": 76}], "edges": [{"from": "p0", "to": "fast", "appearAt": 66}, {"from": "p1", "to": "fast", "appearAt": 68}, {"from": "p2", "to": "slow", "appearAt": 80}, {"from": "p3", "to": "slow", "appearAt": 82}], "caption": "head_dim 16, base 10000: theta_i = 10000^(-2i/d). Same geometric ladder as the sinusoidal table."}
```

A d-dimensional head is d over two independent two-dimensional rotations, each at its own angular rate. Here is the ladder for a head of width sixteen. Pair zero turns fastest: its wavelength is six point three positions, so it comes back round to where it started every six tokens. Pair one, twenty positions. Pair two, sixty three. Pair three, one hundred and ninety nine. Think of them as clock hands at different speeds, and a position as the full set of hand angles — which is a binary counter wearing continuous clothing. The fast hands resolve neighbours: they tell one token from the next. The slow hands have wavelengths far longer than the context, so within a document they behave as a coarse "roughly where am I" signal. And that ladder, ten thousand to the minus two i over d, is exactly the ladder in the original sinusoidal table. RoPE is the same frequency scheme applied as a rotation instead of as an addition.

## s6 · ChartScene

```props
{"eyebrow": "the decay, honestly", "title": "Distance only weakens *matched* content", "kind": "line", "series": [{"label": "matched content <R_m q, R_0 q>", "values": [1.0, 0.966, 0.882, 0.745, 0.699, 0.603, 0.612, 0.433, 0.282], "color": "#5ac48c"}, {"label": "independent q, k", "values": [1.0, 1.004, 0.999, 1.01, 1.009, 1.011, 1.008, 1.009, 1.011]}], "xLabel": "gap: 0, 1, 2, 4, 8, 16, 32, 64, 128", "yLabel": "score as a fraction of gap 0", "caption": "A rotation is an isometry, so for unrelated vectors it changes nothing. The decay is about content that matches."}
```

You will read that RoPE has a long-term decay, that scores fall off with distance. That claim is true and it is usually stated wrongly, so here is the measurement. The flat line is two *independent* random vectors, and it is flat because it has to be: a rotation is an isometry, it preserves lengths and angles, so it cannot change the typical magnitude of a dot product between unrelated things. Out to a gap of one hundred and twenty eight it is within one percent of where it started. The green line is *matched* content — the same vector used as query and key, which is what a perfect content match looks like. That one decays: one point zero at gap zero, zero point seven four at gap four, zero point four three at gap sixty four. And you can see exactly why from the formula: the score is a sum of cosines of m theta i, and as the gap grows those cosines drift out of phase with each other and stop reinforcing. So the decay is real, and it is a soft prior on the *matched* case — not a cutoff, and not a general shrinking of every score.

## s7 · ChartScene

```props
{"eyebrow": "the bake-off", "title": "Four position schemes, identical seed, data and schedule", "kind": "bar", "bars": [{"label": "RoPE", "value": 1.0224, "color": "#5ac48c"}, {"label": "learned", "value": 1.1065}, {"label": "none", "value": 1.1156}, {"label": "sinusoidal", "value": 1.1178}], "yLabel": "validation loss (lower is better)", "caption": "500 steps on the market tape. RoPE wins on loss and costs zero parameters; the learned table costs 8,192."}
```

So we raced them. One model configuration, four position schemes, identical seed, identical data, identical schedule. RoPE wins at one point zero two, and it costs zero parameters. Learned positions, one point one one, and eight thousand parameters. No positions at all, one point one two. And the sinusoidal table, one point one two — behind doing nothing. Two things in that chart deserve saying out loud. First: no positions at all is not catastrophic, and it should not be, because a *causal* decoder leaks position for free. The token at index i can see exactly i plus one things, and the model can count them. Positional encoding sharpens a signal that already exists. In a bidirectional encoder its absence really is fatal. Second: the sinusoidal table landing behind nothing is not a bug, it is a warning about adding things to the residual stream, and that is the next scene.

## s8 · Callout

```props
{"kind": "gotcha", "heading": "The sinusoidal table drowns your token embeddings", "body": "The raw table has entries in [-1, 1]. A GPT-2-style embedding is initialised at standard deviation 0.02. Add them together and the positions are fifty times louder than the tokens — our initial loss came out ABOVE ln(V) because the logits were nearly pure position. The paper hides this by multiplying the embeddings by sqrt(d_model) instead. Same ratio, opposite direction.", "code": "self.register_buffer('table', scale * sinusoidal_table(block, d))  # scale = 0.02"}
```

Here is a bug we hit while building this, and I think it explains why the sinusoidal table has a worse reputation than the idea deserves. The raw table has entries between minus one and one. A GPT-2 style token embedding is initialised at a standard deviation of zero point zero two. Add them straight together and the position signal is roughly fifty times louder than the token signal. Our very first measurement showed an initial loss *above* log of the vocabulary size, which is worse than guessing uniformly — because the logits were almost pure position with a whisper of token underneath. The original paper hides this by multiplying the embeddings by root d model instead of scaling the table down. Same ratio, opposite direction. We scale the table to match the embedding initialisation and we say so in the code. And notice what RoPE avoids by construction: it never touches the residual stream, so it can never compete with the token embeddings for the same coordinates. That is half of why it wins.

## s9 · ChartScene

```props
{"eyebrow": "past the training length", "title": "The test a loss table cannot show", "kind": "bar", "bars": [{"label": "RoPE\n64", "value": 0.8783, "color": "#5ac48c"}, {"label": "RoPE\n96", "value": 0.812, "color": "#5ac48c"}, {"label": "RoPE\n128", "value": 0.9042, "color": "#5ac48c"}, {"label": "sinus\n64", "value": 0.9901}, {"label": "sinus\n96", "value": 1.2514}, {"label": "sinus\n128", "value": 1.7236}, {"label": "learned\n96+", "value": 0}], "yLabel": "loss on a window of that length", "caption": "Trained at 64 tokens. The learned table has a bar of zero because the question cannot be asked at all."}
```

And here is the test a loss table cannot show you. All three models were trained at a context of sixty four tokens. Now feed them longer windows. RoPE at sixty four, ninety six and one hundred and twenty eight: zero point eight eight, zero point eight one, zero point nine. It holds. The sinusoidal model: zero point nine nine, then one point two five, then one point seven two. It degrades badly, but it *answers* — the table is defined for any position, so there is a number. And the learned model has a bar of zero because the question cannot be asked. There is no row sixty five in its table. It is not that it does badly past its training length; it has no representation for a position it never saw. That is the practical difference, and it is why every long-context technique you will meet — position interpolation, NTK scaling, YaRN — is built on top of RoPE. They are all ways of stretching a rotation, and you cannot stretch a lookup table.

## s10 · RecapScene

```props
{"eyebrow": "recap", "title": "Positions, in three sentences", "points": ["Self-attention is permutation-equivariant: without help it cannot tell 'BUY 100 INFY' from '100 INFY BUY'.", "RoPE falls out of one wish — that the score depend only on the gap — and the identity holds to 1.8e-15, exactly, not approximately.", "It wins on loss, costs zero parameters, and is the only scheme that still has an answer past the training length."], "ifSkipped": "Skip this and every long-context paper you read afterwards — interpolation, NTK scaling, YaRN — is a stack of adjustments to a mechanism you never saw derived.", "next": "Episode 4 — training the thing, and finding out what it learned. Including an experiment that lied to us."}
```

Three things. One. Self-attention is permutation-equivariant, so word order has to be supplied from outside, and a causal decoder gets a weak version of it for free by counting. Two. RoPE falls out of a single wish — that the score should depend only on the gap — and the resulting identity holds to one point eight times ten to the minus fifteen, which is exactly, not approximately. Three. It wins on loss, costs nothing in parameters, and is the only one of the three that still has an answer past the training length. Next time, we train the whole model and then go looking for what it learned. Including an experiment that gave us a beautiful result which turned out to be completely wrong.
