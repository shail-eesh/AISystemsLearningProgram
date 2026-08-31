---
topic: t4-e1
episode: 1
title: Attention, slowly
voice: am_michael
speed: 0.85
runtime_target_minutes: 10
paper: Vaswani et al. 2017, Attention Is All You Need
---

## s1 · TitleCard

```props
{"title": "Attention, slowly", "subtitle": "One query, five keys, and the four lines the whole paper is about.", "topicId": "T4", "episode": "Episode 1", "paper": "Vaswani et al. 2017"}
```

Welcome back to the AI Systems Forge. This is phase two, topic four. The transformer, written from an empty file. This is the flagship series of the course, five episodes, and we are going to go slowly, because everything after this depends on it. Today, one head of attention. Not multi-head, not the block, not the model. One head, on one sequence, with real numbers you can point at.

## s2 · ArchitectureMap

```props
{"eyebrow": "where we are", "highlight": ["models"], "caption": "T4 is the architecture every later model on the desk is an instance of."}
```

Here is where we are. AlphaDesk is the fictional trading desk this course builds one topic at a time. The lit block is models. In phase one you built the autograd engine underneath it, the matmul kernel it runs on, the softmax it normalises with, and the tokenizer that feeds it. Every one of those was groundwork for this. Today the pieces come together into the architecture that the desk's own language model, its embedding model, its vision encoder and its speech model are all variations of.

## s3 · ConceptScene

```props
{"eyebrow": "the premise", "title": "Attention is a dictionary lookup that has been made differentiable", "body": "A dictionary lookup compares the key you hold against every key in the table, finds the one that matches, and returns its value. Attention does exactly that, with two changes — and those two changes are the entire invention.", "points": ["The comparison is a dot product instead of an equality test.", "Instead of taking the winner, you take a weighted average of every value.", "The weights come from a softmax, so they are positive and sum to one.", "Because it is an average and not a choice, it has a gradient."], "aside": "That last point is the whole reason this design could be trained at all. An argmax has no useful derivative. A softmax does."}
```

Before any code, here is the idea in one sentence. Attention is a dictionary lookup that has been made differentiable. Think about what an ordinary lookup does. You hold a key. It gets compared against every key in the table. The one that matches wins, and you get back its value. Attention does exactly that with two changes. First, the comparison is a dot product rather than an equality test, so instead of matching or not matching you get a score, a number saying how well those two things line up. Second, instead of taking the winner you take a weighted average of every single value, with the weights coming from a softmax over the scores. And now think about why. A hard lookup, an argmax, has no useful derivative. Nudge the query slightly and either nothing changes at all, or the answer jumps to a completely different entry. There is nothing for gradient descent to hold on to. Soften it into a weighted average and suddenly every weight moves smoothly with every input. That is not a detail. That is the reason this architecture exists.

## s4 · MathReveal

```props
{"eyebrow": "the three projections", "title": "Query, key, value", "english": "Every token emits three vectors: what it is looking for, what it is as a lookup target, and what it contributes if it is chosen.", "equation": "Q = X W_q      K = X W_k      V = X W_v", "code": "q, k, v = x @ w_q, x @ w_k, x @ w_v", "note": "Three learned matrices, three different views of the same token. Nothing else in attention has any parameters."}
```

So where do the queries and keys and values come from. Every token emits three vectors, and each has a job you can say in English. The query is what this token is looking for. The key is what this token is, considered as a lookup target. The value is what this token contributes if it gets chosen. Now the symbols. Q equals X times W q, K equals X times W k, V equals X times W v. Three learned matrices. And now the code, which is one line. Three matrix multiplies. That is it. Those three matrices are the only parameters in the entire attention mechanism. Everything else you are about to see is arithmetic with no weights in it at all.

## s5 · DiagramScene

```props
{"eyebrow": "the mechanism", "title": "One query, five keys", "nodes": [{"id": "q", "label": "query", "sub": "token 5 asks", "x": 0.1, "y": 0.5, "appearAt": 10, "color": "#5ac48c"}, {"id": "k0", "label": "key 0", "sub": "0.02", "x": 0.42, "y": 0.14, "appearAt": 40}, {"id": "k1", "label": "key 1", "sub": "0.61", "x": 0.42, "y": 0.32, "appearAt": 48}, {"id": "k2", "label": "key 2", "sub": "0.05", "x": 0.42, "y": 0.5, "appearAt": 56}, {"id": "k3", "label": "key 3", "sub": "0.09", "x": 0.42, "y": 0.68, "appearAt": 64}, {"id": "k4", "label": "key 4", "sub": "0.23", "x": 0.42, "y": 0.86, "appearAt": 72}, {"id": "out", "label": "output", "sub": "weighted average of values", "x": 0.82, "y": 0.5, "w": 0.28, "appearAt": 110}], "edges": [{"from": "q", "to": "k0", "appearAt": 44}, {"from": "q", "to": "k1", "label": "0.61", "appearAt": 52}, {"from": "q", "to": "k2", "appearAt": 60}, {"from": "q", "to": "k3", "appearAt": 68}, {"from": "q", "to": "k4", "label": "0.23", "appearAt": 76}, {"from": "k1", "to": "out", "appearAt": 114}, {"from": "k4", "to": "out", "appearAt": 120}], "caption": "The five weights sum to exactly 1.0 — softmax guarantees it."}
```

Now watch one query do its job. Token five is asking a question. Its query vector gets compared against the key of every token it is allowed to see, which is tokens zero through five. Five dot products, five scores, and then a softmax turns those scores into weights. Look at the numbers on the keys. Zero point zero two. Zero point six one. Zero point zero five. Zero point zero nine. Zero point two three. Add those up and you get one, exactly, because that is what a softmax is for. So this query has decided that token one is what it was looking for, with a bit of interest in token four, and essentially no interest in the rest. The output is the values of those five tokens, averaged with those five weights. Sixty one percent of token one's value, twenty three percent of token four's, and the remainder split between three tokens it has mostly ignored. That is one head of attention producing one output vector. Nothing else happened.

## s6 · MathReveal

```props
{"eyebrow": "the formula", "title": "Scaled dot-product attention", "english": "Score every query against every key, divide by the square root of the head width, mask out the future, softmax, and use the result to average the values.", "equation": "Attention(Q,K,V) = softmax( Q K^T / sqrt(d_k) ) V", "code": "scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)\nscores = scores.masked_fill(~mask, -inf)\nattn   = F.softmax(scores, dim=-1)\nout    = attn @ v", "note": "Four lines. The paper is 15 pages; this is the load-bearing part of it."}
```

Here is the whole thing written down. In English first. Score every query against every key. Divide by the square root of the head width. Mask out the future. Softmax. Use the result to average the values. Now the equation, and this is the one line people put on slides. Attention of Q, K and V equals softmax of Q K transpose over root d k, times V. And now the code, which is four lines, and which you should read as the definition rather than as an implementation of the equation. Line one, the scores. Line two, the mask. Line three, the softmax. Line four, the weighted average. The paper is fifteen pages. This is the load-bearing part of it, and it fits on a business card.

## s7 · ChartScene

```props
{"eyebrow": "why the square root", "title": "Score variance grows with head width — and softmax saturates", "kind": "bar", "bars": [{"label": "d=4", "value": 0.273, "note": "unscaled"}, {"label": "d=16", "value": 0.584, "note": "unscaled"}, {"label": "d=64", "value": 0.81, "note": "unscaled"}, {"label": "d=256", "value": 0.9, "note": "unscaled"}, {"label": "d=1024", "value": 0.959, "note": "unscaled"}, {"label": "d=1024 scaled", "value": 0.105, "color": "#5ac48c"}], "yLabel": "largest softmax weight", "caption": "200 random rows of 64 keys each. Unscaled, one weight marches to 1.0 as the model gets wider."}
```

Now the divide by root d k, because it is the one part of that formula that looks arbitrary and is not. Queries and keys have roughly unit variance in each coordinate. Their dot product sums over d k coordinates, so its variance grows with d k. Wider model, bigger scores. And here is what bigger scores do to a softmax. This chart is a real measurement: two hundred random rows of sixty four keys each, and the bar height is the largest weight the softmax produced. At width four it is zero point two seven. At sixty four, zero point eight one. At one thousand and twenty four, zero point nine six. The softmax has collapsed onto a single key. And a saturated softmax has almost no gradient, so that layer has stopped learning, and it stops learning more the wider you build it, which is the exact opposite of what you want from a scaling knob. Look at the green bar. Same width, one thousand and twenty four, with the division by root d k. Zero point one. Back where it started. The scaling is a variance correction. It is not a normalisation ritual and it is not there for numerical stability.

## s8 · CodeWalkthrough

```props
{"eyebrow": "the code", "title": "One head, on one sequence", "filename": "phases/p2/t4-transformer/src/t4_transformer/attention.py", "code": "def single_head_attention(x, w_q, w_k, w_v, *, causal=True):\n    q, k, v = x @ w_q, x @ w_k, x @ w_v\n    return scaled_dot_product(q, k, v, causal=causal)\n\n\ndef scaled_dot_product(q, k, v, *, causal=True, mask=None):\n    d_k = q.shape[-1]\n    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)\n    if mask is None and causal:\n        mask = causal_mask(scores.shape[-2])\n    scores = scores.masked_fill(~mask, float('-inf'))\n    attn = F.softmax(scores, dim=-1)\n    return attn @ v, attn", "highlights": [{"at": 20, "lines": [1, 2], "caption": "x is (T, d_model); the weights are (d_model, d_head). One sequence, no batching, no heads."}, {"at": 120, "lines": [8], "caption": "Every query against every key, in one matmul. The sqrt is the variance fix from the last scene."}, {"at": 230, "lines": [11], "caption": "-inf, not a large negative number. In float16, -1e9 leaks a non-zero weight through the softmax."}, {"at": 330, "lines": [12], "caption": "After the softmax, every masked position is exactly 0.0 and every row still sums to 1."}, {"at": 420, "lines": [13], "caption": "The weighted average — and the weights come back too, because half of interpretability is looking at them."}]}
```

And here is that written as code you can run. The top function takes one sequence, x, of shape T by d model, and three weight matrices. Three matmuls, and hand off. The second function is the four lines. Notice the mask fill uses negative infinity rather than a large negative number like minus one e nine. That matters more than it looks. In float sixteen, minus one e nine is not far enough from zero, and a small but non-zero weight leaks through the softmax onto a future token. Your model then trains slightly on information it will not have at generation time, and the failure looks like a model that validates beautifully and generates nonsense. Use negative infinity. After the softmax every masked entry is exactly zero and every row still sums to one. And notice the last line returns the weights as well as the output. That is deliberate. Half of interpretability is being able to look at the attention matrix, and episode four is going to do exactly that.

## s9 · Callout

```props
{"kind": "gotcha", "heading": "The mask is causality, and removing it feels like success", "body": "Delete the mask and validation loss drops immediately, because every token can see the answer that comes after it. Then generation produces gibberish, because at generation time the future does not exist yet. This is the single easiest way to build a language model that scores well and cannot write.", "code": "_, attn = single_head_attention(x, eye, eye, eye, causal=False)\n# token 0 now places 0.062 of its weight on token 3 — a token\n# that has not been generated yet."}
```

One gotcha before we stop, and it is the one that costs people days. The causal mask is not a performance detail. It is the entire reason a decoder can generate. Take it away and every token can see the tokens after it, which includes the one it is being asked to predict. Loss falls off a cliff, and it looks like you have done something right. Then you sample from the model and it produces nothing coherent, because at generation time there is no future to look at. In our own run, without the mask, token zero placed six percent of its attention on token three, a token that will not exist for another three steps. With the mask, that weight is exactly zero.

## s10 · RecapScene

```props
{"eyebrow": "recap", "title": "One head, in three sentences", "points": ["Attention is a differentiable dictionary lookup: dot-product scores, softmax weights, weighted average of values.", "The division by root d_k is a variance fix — without it the softmax saturates and the layer stops learning as you widen it.", "The causal mask is what makes the model able to generate, and removing it improves your loss and destroys your model."], "ifSkipped": "Skip this and multi-head in episode two is just an unexplained reshape — because multi-head is nothing but this, four times in parallel.", "next": "Episode 2 — the block: many heads, the MLP, and what the residual stream is for."}
```

Three things to take away. One. Attention is a differentiable dictionary lookup. Dot products become scores, softmax turns scores into weights, and the output is a weighted average of the values. Two. The division by root d k is a variance fix, and without it the softmax saturates and the layer stops learning precisely as you make the model bigger. Three. The causal mask is what lets the model generate at all, and taking it out will improve your loss while destroying your model. If you skip this episode, episode two is just an unexplained reshape, because multi-head attention is nothing but what you have just seen, four times in parallel. Next time, the block. Many heads, the multi-layer perceptron that holds two thirds of the parameters, and what the residual stream is actually for.
