---
topic: t45a-e1
episode: 1
title: The overflow story
voice: am_michael
speed: 0.85
runtime_target_minutes: 9
paper: the max-subtraction identity; IEEE 754 ranges
---

## s1 · TitleCard

```props
{"title": "Softmax, part one", "subtitle": "A formula that is correct on paper and unusable on a computer — and the one-line identity that fixes it.", "topicId": "T45A", "episode": "Episode 1"}
```

Phase one, topic forty five A. Softmax. This looks like the smallest topic in the phase and it is not, because episode two contains the single algorithmic trick that Flash Attention is built on. But we have to earn that, and we earn it by watching the textbook formula die.

## s2 · MathReveal

```props
{"eyebrow": "the formula", "title": "Softmax, as everyone writes it", "english": "Turn a vector of arbitrary real numbers into a probability distribution: exponentiate everything, then divide by the total.", "equation": "softmax(x)_i = exp(x_i) / SUM_j exp(x_j)", "code": "def naive_softmax(x):\n    e = np.exp(x)\n    return e / e.sum(axis=-1, keepdims=True)", "note": "Every symbol is correct. Two lines of NumPy. It is also unusable, and the reason has nothing to do with the mathematics.", "stageFrames": [10, 120, 230]}
```

Here is the function, three times. In English. Take a vector of arbitrary real numbers and turn it into a probability distribution. Exponentiate everything, then divide by the total. In symbols. Softmax of x, component i, is e to the x i over the sum over j of e to the x j. And in code, two lines of NumPy that are a direct transcription. Every symbol in that middle line is correct. The code is a faithful implementation of it. And it is unusable, for a reason that has nothing at all to do with the mathematics.

## s3 · ChartScene

```props
{"eyebrow": "the reason", "title": "Where exp dies, by dtype", "kind": "bar", "bars": [{"label": "float16", "value": 11, "color": "#e05e6b", "note": "eleven"}, {"label": "float32", "value": 88, "color": "#e0b04e"}, {"label": "float64", "value": 709, "color": "#5ac48c"}], "yLabel": "largest x with finite exp(x)", "caption": "Attention logits routinely leave these ranges. Half precision gives up at eleven."}
```

Here is the reason. Exponentiation has a finite range in every floating point format, because the result has to fit. In double precision, the largest input with a finite exponential is about seven hundred and nine. In single precision it is eighty eight. And in half precision, which is what a great deal of modern inference actually runs in, it is eleven. Not eleven thousand. Eleven. Attention logits leave that range routinely and without warning. So this is not an edge case you handle for robustness. It is the normal operating condition.

## s4 · Callout

```props
{"kind": "warning", "heading": "The failure has two faces, and only one of them is loud", "body": "Overflow gives you inf, then inf over inf, then nan, and the nan spreads through the whole model in one step. Underflow gives you a clean, plausible zero — and only bites later, when you take its logarithm.", "code": "naive_softmax([1e4, -1e4, 0.0])\n  -> [nan,  0.,  0.]        # loud\n\nstable_softmax([0.0, -800.0])\n  -> [1., 0.]               # quiet, and correct to the dtype\nnp.log(_)\n  -> [0., -inf]             # the bill arrives here"}
```

And the failure has two faces, which is worth separating because only one of them is loud. Overflow is the loud one. Exponentiate ten thousand and you get infinity. Divide infinity by infinity and you get not a number. And a single nan spreads through an entire model in one forward pass. You will notice. Underflow is the quiet one. Softmax of zero and minus eight hundred gives you one and zero. That zero is not an error. It is the correct answer to within the precision of the dtype, because the true value fell below the smallest representable number. It sits there behaving perfectly until somebody takes its logarithm, and then you get minus infinity in a loss function, and the bill arrives in a completely different part of the codebase. Hold onto that one. Episode two settles it.

## s5 · MathReveal

```props
{"eyebrow": "the fix", "title": "The shift identity", "english": "Softmax does not care if you add the same constant to every input. Not approximately — exactly. So subtract the largest one, and every exponent becomes zero or negative.", "equation": "exp(x_i - c) / SUM_j exp(x_j - c)  ==  softmax(x)_i    for ANY c", "code": "c = x.max(axis=-1, keepdims=True)\ne = np.exp(x - c)          # largest term is exactly exp(0) = 1\nreturn e / e.sum(axis=-1, keepdims=True)", "note": "Proof: multiply top and bottom by exp(-c). One line. Nothing is approximated anywhere.", "stageFrames": [10, 140, 270]}
```

The fix, and it is a one line proof. In English. Softmax does not care if you add the same constant to every input. Not approximately. Exactly. In symbols. Exponent of x i minus c, over the sum of exponent of x j minus c, equals softmax of x at i, for any constant c whatsoever. The proof is to multiply the numerator and the denominator by e to the minus c, which cancels. That is the entire derivation. And in code. Take c to be the row maximum. Now the largest exponent is exactly e to the zero, which is one, and every other exponent is smaller. Overflow has become structurally impossible rather than unlikely.

## s6 · ConceptScene

```props
{"eyebrow": "the choice", "title": "Why the max, and not the mean", "body": "Any constant removes the scale problem in principle. Only the maximum removes it unconditionally, for every input, with no assumption about the spread.", "points": ["x = [0, 1000], c = mean = 500: you still compute exp(500), which is still inf.", "x = [0, 1000], c = max = 1000: you compute exp(-1000) and exp(0). Both fine.", "The terms that underflow to zero were negligible anyway — correct to the dtype.", "So max-subtraction costs one extra pass over the data and cannot fail."], "aside": "This identity reappears in log-sum-exp, in log-softmax, in cross-entropy, and in the softplus form of binary cross-entropy. Learn it once."}
```

But why the maximum specifically. Any constant fixes the scale in principle, so why not the mean, which is a natural choice. Take the vector zero and one thousand. Its mean is five hundred. Subtract that and you are computing e to the five hundred, which is still infinity in double precision. You have not fixed anything, you have moved it. Take the maximum instead, which is one thousand. Now you compute e to the minus one thousand and e to the zero. The first underflows to zero and the second is one, and both are fine. And that underflow is not a loss. Those terms were negligible relative to the largest one, which is what the arithmetic just told us. So max subtraction costs one extra pass over the data and, in exchange, cannot fail for any input at all. Learn this identity properly, because it comes back constantly. Log sum exp is this. Log softmax is this. Cross entropy is this. The softplus form of binary cross entropy is this.

## s7 · ChartScene

```props
{"eyebrow": "measured", "title": "64 adversarial rows, logits in [-10000, +10000]", "kind": "bar", "bars": [{"label": "naive, float64", "value": 64, "color": "#e05e6b", "note": "inf or nan"}, {"label": "naive, float32", "value": 64, "color": "#e05e6b"}, {"label": "naive, float16", "value": 64, "color": "#e05e6b"}, {"label": "stable, any dtype", "value": 0, "color": "#5ac48c", "note": "0 failures"}], "yLabel": "rows producing inf or nan (of 64)", "caption": "The benchmark keeps the broken version and runs it, so the failure is something you have seen."}
```

Here is the measurement from the topic's benchmark. Sixty four rows of adversarial logits, ranging from minus ten thousand to plus ten thousand, including one row that is entirely eight hundred and one that is entirely minus ten thousand. The naive implementation produces infinity or not a number on sixty four out of sixty four rows, in every dtype. The stable version produces zero failures in every dtype. And note that the broken implementation is kept in the codebase and executed by the benchmark on purpose, rather than deleted with a comment saying do not do this. A failure you have watched happen is worth more than a warning you have read.

## s8 · RecapScene

```props
{"eyebrow": "episode one", "points": ["exp overflows at 709 in float64, 88 in float32, and 11 in float16 — attention logits leave that range routinely.", "Overflow is loud (nan spreads in one pass); underflow is quiet, and bills you later at the logarithm.", "Softmax is exactly invariant to adding a constant to every input. One-line proof.", "Subtract the maximum: the largest exponent becomes exp(0), so overflow is structurally impossible.", "Only the max works unconditionally — the mean still overflows on [0, 1000]."], "ifSkipped": "Every masked attention, every logit processor, every cross-entropy in the rest of this course assumes this identity.", "next": "Episode 2: computing the normaliser in ONE pass — the trick Flash Attention rests on."}
```

To recap. Exponentiation overflows at seven hundred and nine in double precision, eighty eight in single, and eleven in half, and attention logits leave those ranges routinely. Overflow is loud, because one nan spreads through a model in a single pass. Underflow is quiet, and sends you the bill later when something takes a logarithm. Softmax is exactly invariant to adding a constant to every input, with a one line proof. Subtract the maximum and the largest exponent becomes e to the zero, so overflow stops being unlikely and becomes impossible. And only the maximum works unconditionally, because the mean still overflows on a vector as simple as zero and one thousand. Every masked attention, every logit processor and every cross entropy in the rest of this course assumes this identity holds. Next episode is the one I have been promising. Computing the normaliser in a single pass, which is the trick Flash Attention is built on, derived slowly on one whiteboard.
