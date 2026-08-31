---
topic: t4-e2
episode: 2
title: The block — many heads, one bus
voice: am_michael
speed: 0.85
runtime_target_minutes: 9
paper: Vaswani et al. 2017; Xiong et al. 2020 (pre-LN); Zhang & Sennrich 2019 (RMSNorm)
---

## s1 · TitleCard

```props
{"title": "The block", "subtitle": "Many heads, the MLP that holds the parameters, and the bus everything writes to.", "topicId": "T4", "episode": "Episode 2"}
```

Welcome back. Episode two of the transformer series. Last time, one head of attention on one sequence. Today we turn that into the unit a real model is a stack of. Three things: why there are many heads and what they cost, what the multi-layer perceptron is actually for, and what the residual stream is — which is the idea that makes the whole stack trainable and which almost nobody explains.

## s2 · ConceptScene

```props
{"eyebrow": "the problem", "title": "One head can express one relationship at a time", "body": "A softmax row sums to one. That is a budget. Attention spent on the previous token is attention not spent on the subject of the sentence, and a single head has to choose.", "points": ["Four heads of width 32 instead of one head of width 128.", "Same parameters. Same FLOPs. Four independent softmax budgets.", "Each head can specialise: one tracks position, one tracks the ticker symbol, one tracks the last number seen.", "Nobody assigns those jobs. They fall out of training."], "aside": "This is why you widen by adding heads rather than by making one head bigger — a wider head does not get a second opinion, it gets a longer vector."}
```

Here is the problem with what we built last time. A softmax row sums to one. That is not a nice property, it is a budget. If a head spends sixty percent of its attention on the token immediately behind it, that is sixty percent it cannot spend on the subject of the sentence. One head has to choose what kind of relationship it is in the business of noticing. Multi-head attention buys parallel budgets by splitting the width. Four heads of thirty two, instead of one head of one hundred and twenty eight. Same parameter count, same floating point operations, four independent softmax budgets. And in a trained model they visibly specialise. One head tracks position. One tracks which ticker symbol is under discussion. One tracks the last number that appeared. Nobody assigned those jobs; they fall out of training, and in episode four we are going to look at one of them.

## s3 · CodeWalkthrough

```props
{"eyebrow": "the implementation", "title": "Heads are a reshape, and the reshape is where the bugs are", "filename": "phases/p2/t4-transformer/src/t4_transformer/attention.py", "code": "self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=bias)\n\ndef _split_heads(self, x):\n    b, t, _ = x.shape\n    return x.view(b, t, self.n_head, self.head_dim).transpose(1, 2)\n\ndef forward(self, x, *, cache=None, need_weights=False):\n    b, t, c = x.shape\n    q, k, v = self.qkv(x).split(self.n_embd, dim=2)\n    q, k, v = self._split_heads(q), self._split_heads(k), self._split_heads(v)\n    out, attn = scaled_dot_product(q, k, v, mask=mask)\n    out = out.transpose(1, 2).contiguous().view(b, t, c)\n    return self.resid_dropout(self.proj(out))", "highlights": [{"at": 20, "lines": [1], "caption": "One Linear producing 3d, not three Linears producing d. Same maths, one GEMM instead of three."}, {"at": 120, "lines": [5], "caption": "view then transpose: heads become a batch dimension, so B*H attention problems run as one batched matmul."}, {"at": 240, "lines": [9, 10], "caption": "Split the fused projection back into q, k, v — then slice each into heads."}, {"at": 340, "lines": [12], "caption": "Transpose back and flatten: the heads' outputs are concatenated, then mixed by one more projection."}]}
```

And here is the implementation, which is a reshape. First thing to notice: one Linear layer producing three times the width, not three Linear layers producing the width. Identical mathematics, one matrix multiply instead of three, and this is why every real implementation looks like this. Then split heads. View, then transpose. The view chops the width into head count by head dimension. The transpose moves the head axis in front of the sequence axis, so that batch and head sit together at the front and the whole attention computation becomes one batched matrix multiply over batch times heads independent problems. At the end, transpose back, flatten the heads into one vector, and pass it through one more projection which is what lets the heads' outputs mix with each other.

## s4 · Callout

```props
{"kind": "gotcha", "heading": "Check the reshape against a loop, or it will be wrong forever", "body": "The view is correct only if head h corresponds to a contiguous block of columns in the fused qkv weight. Get it wrong and the model still trains — just worse, permanently, with no error message. So the fast path is tested against an explicit per-head Python loop that is obviously correct and obviously slow.", "code": "for 1, 2, 4, 8 heads:\n    max abs difference between batched and looped:  0.0"}
```

Now a gotcha, and it is the most valuable habit in this whole episode. That view and transpose are correct only if head h corresponds to a contiguous block of columns in the fused weight matrix. If it does not, nothing crashes. There is no error message. The model trains. It is simply worse than it should be, forever. So in the repository there is a second implementation, multi head attention looped, which does the heads as an explicit Python for loop. It is slow and it is obviously correct, and it exists purely as an oracle. The test asserts the two agree, and the number that comes back is exactly zero for one, two, four and eight heads. Not close to zero. Zero, because it is the same arithmetic in a different memory layout. When you write a fast path, write the slow path first and never delete it.

## s5 · ChartScene

```props
{"eyebrow": "what heads cost", "title": "Splitting the width is free", "kind": "bar", "bars": [{"label": "1 head", "value": 263168, "note": "d=256"}, {"label": "2 heads", "value": 263168}, {"label": "4 heads", "value": 263168}, {"label": "8 heads", "value": 263168}, {"label": "16 heads", "value": 263168}], "yLabel": "parameters", "caption": "Measured, not asserted: identical parameter counts at every head count. The FLOP count is identical too."}
```

And this is what heads cost, measured rather than asserted. Five configurations, one head through sixteen heads, at a fixed model width of two hundred and fifty six. Two hundred and sixty three thousand, one hundred and sixty eight parameters. Every single one. The floating point operation count is identical as well, because splitting a width does not change how much arithmetic there is. The wall clock differs a little — more heads means smaller, less cache-friendly matrices — but that is a memory effect, not an arithmetic one. Heads are close to free, and you should think of head count as a knob about *expressiveness* with no price tag on the other side.

## s6 · ConceptScene

```props
{"eyebrow": "the other half", "title": "Attention moves information; the MLP decides what it means", "body": "A transformer that only attends is a very expensive weighted average. The multi-layer perceptron is where the model does its thinking — and where roughly two thirds of the parameters live.", "points": ["Widen by four, apply GELU, project back. That is the whole layer.", "It is applied to every position independently — no mixing between tokens.", "Measured on a 4-layer model: attention 33k parameters, MLP 66k. A 1 to 2 ratio.", "The 4x is convention from the paper, not a derived optimum."], "aside": "GELU rather than ReLU: smooth, so its gradient does not vanish abruptly for slightly-negative inputs. GPT-2 used it and everyone copied."}
```

Now the other half of the block, and the half people underrate. Attention *moves* information between positions. It does not transform it. A stack of pure attention layers is a very expensive weighted average. The multi-layer perceptron is where the model thinks. Widen by a factor of four, apply a non-linearity, project back down. That is the whole layer, and it is applied to each position independently — there is no communication between tokens inside it at all. And look at the parameter split we measured on a four-layer model: attention thirty three thousand, the MLP sixty six thousand. One to two. Two thirds of the model is the part that has never heard of the sequence. The four times widening is convention from the paper, not a derived optimum, and the non-linearity is GELU rather than ReLU because it is smooth, so its gradient does not fall off a cliff for slightly negative inputs.

## s7 · MathReveal

```props
{"eyebrow": "the block", "title": "Two writes to a shared bus", "english": "Normalise, attend, add the result back. Normalise again, run the MLP, add that back too. Nothing is ever overwritten.", "equation": "x = x + Attn(Norm(x))\nx = x + MLP(Norm(x))", "code": "x = x + self.attn(self.ln1(x))\nreturn x + self.mlp(self.ln2(x))", "note": "Read the residual stream as a bus every layer reads from and adds to — which is why you can delete a middle layer of a trained transformer and still get grammatical text."}
```

And here is the block itself. In English: normalise, attend, add the result back onto what you had. Normalise again, run the MLP, add that back too. In symbols, two lines, both of the form x equals x plus something. In code, the same two lines. Now the thing worth sitting with. Nothing is ever overwritten. Each sublayer *adds* to a stream that runs unbroken from the embeddings all the way to the final norm. The right mental model is a shared bus: every layer reads the bus, computes a contribution, and writes it back on. That is why you can delete a middle layer from a trained transformer and still get grammatical text out — you have removed one contribution from a sum, not cut a wire. And it is why gradients reach layer zero at all: there is a path from the loss to the embeddings that passes through nothing but additions.

## s8 · ChartScene

```props
{"eyebrow": "pre-norm or post-norm", "title": "The textbook answer is wrong at small scale", "kind": "bar", "bars": [{"label": "6L, lr 3e-3\npre", "value": 2.7891}, {"label": "6L, lr 3e-3\npost", "value": 1.78, "color": "#5ac48c"}, {"label": "6L, lr 1e-2\npre", "value": 2.4244, "color": "#5ac48c"}, {"label": "6L, lr 1e-2\npost", "value": 3.3014}, {"label": "12L, lr 3e-3\npre", "value": 2.6616, "color": "#5ac48c"}, {"label": "12L, lr 3e-3\npost", "value": 3.2993}, {"label": "12L, lr 1e-2\npre", "value": 2.599, "color": "#5ac48c"}, {"label": "12L, lr 1e-2\npost", "value": 3.3013}], "yLabel": "validation loss (lower is better)", "caption": "Green is the winner in each pair. Same seed, same data, 150 steps, no warmup."}
```

Which brings us to where the normalisation goes, and to a result I did not expect. Everyone says pre-norm — normalise before the sublayer — is better than the original paper's post-norm. So we measured it. Four pairs: six layers and twelve layers, at two learning rates, same seed, same data, no warmup. Look at the first pair. Six layers, mild learning rate. Post-norm wins, and not narrowly: one point seven eight against two point seven nine. The textbook answer is wrong here. Now look at the other three pairs. Raise the learning rate, or go to twelve layers, and post-norm parks at three point three and stops. It has not slowed down, it has stopped training. Pre-norm carries on in all four. So the honest statement is not "pre-norm is better". It is that pre-norm is *robust*. It buys you the right to stack twelve, forty eight, ninety six layers and to turn the learning rate up, without a babysitter. That is the trade every model after GPT-2 took, and saying "pre-norm is better" throws away the only part of it that matters.

## s9 · Callout

```props
{"kind": "insight", "heading": "RMSNorm is LayerNorm with the useless half removed", "body": "LayerNorm subtracts the mean and divides by the standard deviation. RMSNorm skips the mean subtraction and divides by the root-mean-square. Same job, about seven fewer operations per element, no measurable quality cost — which is why LLaMA and everything downstream uses it.", "code": "x / sqrt(mean(x^2) + eps) * gain      # no mean, no bias"}
```

One more component and then the recap. RMSNorm. LayerNorm subtracts the mean and divides by the standard deviation. Zhang and Sennrich noticed in twenty nineteen that the re-centering does approximately nothing and the re-scaling does approximately everything. So RMSNorm keeps the rescaling and drops the mean subtraction and the bias term. Divide by the root mean square, multiply by a learned gain, done. About seven fewer operations per element, no measurable quality cost, which is why LLaMA used it and why everything downstream of LLaMA uses it. Both are in the repository and the config picks between them, so you can measure the difference yourself instead of taking my word for it.

## s10 · RecapScene

```props
{"eyebrow": "recap", "title": "The block, in three sentences", "points": ["Multi-head attention buys parallel softmax budgets for free — same parameters, same FLOPs — and the reshape that implements it must be checked against a slow loop.", "The MLP holds two thirds of the parameters and is where the model thinks; attention only moves information between positions.", "Pre-norm is not better quality, it is robustness: post-norm wins at 6 layers and stops training at 12."], "ifSkipped": "Skip this and the model in episode four is a black box you are watching train, rather than a stack of contributions you can reason about one at a time.", "next": "Episode 3 — positions: why attention cannot tell word order, and how RoPE fixes it with a rotation."}
```

Three things. One. Multi-head attention buys parallel relationships for free, and the reshape that implements it must be checked against an obviously-correct slow version, because getting it wrong produces a model that trains, just worse, forever. Two. The MLP holds two thirds of the parameters and is where the thinking happens; attention only moves information around. Three. Pre-norm is not better quality, it is robustness — post-norm actually won at six layers, and stopped training entirely at twelve. Next time, positions. Self-attention has no idea what order the tokens are in, which is a strange thing to discover about a language model, and rotary embeddings fix it with one very elegant idea.
