---
topic: t31-e3
episode: 3
title: Broadcasting gradients, slowly
voice: am_michael
speed: 0.85
runtime_target_minutes: 12
paper: CS231n backpropagation notes; NumPy broadcasting rules
---

## s1 · TitleCard

```props
{"title": "Autograd, part three", "subtitle": "Forward broadcast equals backward sum. This episode is deliberately slow, because this is the one that costs people a day.", "topicId": "T31", "episode": "Episode 3"}
```

Episode three, and the reason the scalar engine was never the destination. Today we move to arrays, and we meet the single subtlety that separates an autograd engine that works from one that appears to. Broadcasting. I am going to take this slowly, because in my experience this is where the day goes.

## s2 · ChartScene

```props
{"eyebrow": "the motivation", "title": "Why the scalar engine cannot be the destination", "kind": "bar", "bars": [{"label": "scalar engine", "value": 498, "color": "#e05e6b", "note": "27,777 Python objects"}, {"label": "tensor engine", "value": 1, "color": "#5ac48c", "note": "4 lines, one node"}], "logScale": true, "caption": "One 64x8 -> 16 layer, forward and backward. Relative time, log scale. Measured, not estimated."}
```

First, the motivation, measured rather than asserted. Take one small layer. Sixty four samples, eight features in, sixteen units out. Unrolled to scalars, that graph is twenty seven thousand seven hundred and seventy seven nodes, each one a Python object holding a closure. It runs four hundred and ninety eight times slower than the same computation as four lines of array code. That factor is not a tuning problem. It is the interpreter, and it is the concrete reason every real framework is array first. The structure we built in the last two episodes survives completely. What changes is that one node now holds an entire matrix, and the inner loops happen in C.

## s3 · ConceptScene

```props
{"eyebrow": "the rule", "title": "Forward broadcast equals backward sum", "body": "NumPy silently stretches a (1, 8) bias across 32 rows so it can add it to a (32, 8) activation. Forward, that is free and invisible. Backward, it is a summation — and it is invisible there too, which is the problem.", "points": ["The bias contributed to all 32 rows, so it receives gradient from all 32.", "Its gradient is therefore the SUM of 32 upstream gradients, not one of them.", "Axes NumPy invented (rank promotion) must be summed away entirely.", "Axes NumPy stretched (length-1) must be summed with keepdims, to preserve rank."], "aside": "Say it once more: forward broadcast == backward sum-and-reshape. Everything in unbroadcast() is those five words."}
```

Here is the rule, and I want to state it before any code, because it is the whole episode. NumPy will silently stretch a bias of shape one by eight across thirty two rows so it can be added to an activation of shape thirty two by eight. Forwards, that stretch is free and completely invisible. Backwards, it is a summation. And it is invisible there too, which is exactly the problem. Think about why. The bias contributed to all thirty two rows. So on the way back, it receives gradient from all thirty two rows. Its gradient is the sum of thirty two upstream gradients, not one of them, and not the average of them. There are two flavours of this. Axes that NumPy invented, because your operand had fewer dimensions than the result, get summed away entirely. Axes that NumPy stretched, because your operand had length one there, get summed with keep dimensions set, so the result keeps its rank. Say it once more and then we will look at the code. Forward broadcast equals backward sum and reshape.

## s4 · CodeWalkthrough

```props
{"eyebrow": "the code", "title": "unbroadcast, in six lines", "filename": "phases/p1/t31-autograd/src/t31_autograd/tensor.py", "code": "def unbroadcast(grad, shape):\n    while grad.ndim > len(shape):\n        grad = grad.sum(axis=0)\n    for axis, size in enumerate(shape):\n        if size == 1 and grad.shape[axis] != 1:\n            grad = grad.sum(axis=axis, keepdims=True)\n    return grad.reshape(shape)", "highlights": [{"at": 20, "lines": [2, 3], "caption": "Rank promotion: leading axes the operand never had are summed away, one at a time."}, {"at": 170, "lines": [4, 5, 6], "caption": "Stretched axes: length 1 in the target but not in the gradient — sum with keepdims to hold the rank."}, {"at": 330, "lines": [7], "caption": "The reshape is a no-op assertion: if the two loops were right, this cannot fail."}, {"at": 420, "lines": [1], "caption": "Every single op's backward funnels through this. Get it right once."}]}
```

And here is the whole thing. Six lines. The first loop handles rank promotion. While the gradient has more dimensions than the target shape, sum away the leading axis. One at a time, because each sum removes exactly one. The second loop handles stretching. For every axis where the target has length one but the gradient does not, sum along that axis with keep dimensions set, so the rank does not collapse. And then the reshape at the end. That reshape should always be a no operation. It is there as an assertion. If the two loops did their job the shape already matches, and if they did not, this line raises immediately rather than letting a wrong shaped gradient escape into the rest of the graph. Every operator's backward pass funnels through this function. Which is the good news. You have to get it right exactly once.

## s5 · MathReveal

```props
{"eyebrow": "the invariant", "title": "Gradient is conserved, only regrouped", "english": "Reducing a gradient to a smaller shape must not change its total. The 256 numbers in a (32,8) gradient become 8 numbers in a (1,8) gradient, and both sum to the same thing.", "equation": "sum( unbroadcast(g, s) )  ==  sum(g)     for every valid s", "code": "grad = np.ones((32, 8))          # total 256\nunbroadcast(grad, (1, 8)).sum()  # 256\nunbroadcast(grad, (8,)).sum()    # 256\nunbroadcast(grad, ()).sum()      # 256", "note": "A cheap property test that catches mean-instead-of-sum, and the wrong axis, in one line.", "stageFrames": [10, 130, 250]}
```

There is a property here that is worth having as a test, and I will show it three ways. In English. Reducing a gradient to a smaller shape must not change its total. The two hundred and fifty six numbers in a thirty two by eight gradient become eight numbers in a one by eight gradient, and both collections sum to the same value. In symbols. The sum of unbroadcast of g to shape s equals the sum of g, for every valid s. And in code. A gradient of all ones with shape thirty two by eight totals two hundred and fifty six. Reduce it to one by eight and it still totals two hundred and fifty six. Reduce it to a bare vector of eight, still two hundred and fifty six. Reduce it all the way to a scalar, still two hundred and fifty six. Gradient is conserved. It is only ever regrouped. And that single assertion catches two of the three ways people get this wrong. Using mean instead of sum, and summing over the wrong axis.

## s6 · CodeWalkthrough

```props
{"eyebrow": "the payoff", "title": "Matmul: derive it once with indices", "filename": "src/t31_autograd/tensor.py", "code": "def __matmul__(self, other):\n    out = Tensor(self.data @ other.data, (self, other), \"@\")\n\n    def _backward():\n        self._accumulate(out.grad @ swapaxes(other.data, -1, -2))\n        other._accumulate(swapaxes(self.data, -1, -2) @ out.grad)\n\n    out._backward = _backward\n    return out", "highlights": [{"at": 20, "lines": [1, 2], "caption": "C = A @ B. Forward is one line, as always."}, {"at": 140, "lines": [5], "caption": "dL/dA = dL/dC @ B^T. Derive it from C_ij = sum_k A_ik B_kj: dC_ij/dA_ik = B_kj."}, {"at": 280, "lines": [6], "caption": "dL/dB = A^T @ dL/dC. Same derivation, other index."}, {"at": 380, "lines": [5, 6], "caption": "Shape check as a sanity rule: the answer must come out the shape of the input it is a gradient for."}]}
```

Now the operator everybody looks up instead of deriving. Matrix multiply. The forward is one line as usual. The backward is two, and they are worth deriving once so you never look them up again. Write out the definition with indices. C i j equals the sum over k of A i k times B k j. So the derivative of C i j with respect to A i k is simply B k j. Collect those into matrix form and you get. The gradient of A is the upstream gradient times B transposed. And symmetrically, the gradient of B is A transposed times the upstream gradient. If you ever forget which transpose goes where, there is a sanity rule that never fails. The gradient of a thing has the shape of that thing. There is usually only one arrangement of the transposes that produces the right shape, and it is the correct one.

## s7 · Callout

```props
{"kind": "gotcha", "heading": "The bug this engine actually shipped with", "body": "Numerically safe binary cross-entropy needs log(1 + e^-|z|). Building |z| as a detached constant computes the right forward value and silently zeroes that term's gradient. Gradcheck caught it in one second. The loss curve never would have.", "code": "# wrong: forward correct, gradient half missing\nabs_z = Tensor(np.abs(z.data), requires_grad=False)\n\n# right: d|z|/dz = sign(z), so build it as z * sign\nsign  = Tensor(np.sign(z.data), requires_grad=False)\nabs_z = z * sign"}
```

I want to show you a real bug, because this one shipped in a draft of the engine you are building. Numerically safe binary cross entropy uses the identity log of one plus e to the z equals max of z and zero, plus log of one plus e to the minus absolute z. That absolute value has to stay differentiable. The first version built it as a detached constant. Take the data, call numpy absolute, wrap it as a tensor with gradients off. The forward value was exactly right. Every test on the loss value passed. And the gradient of that entire term was zero, because no gradient can flow through a detached node. The fix is to remember that the derivative of absolute z is the sign of z, so you build it as z times sign, with the sign detached and the z not. Here is the part worth internalising. Gradcheck found this in one second. The loss curve would never have found it, because a loss with half a gradient still goes down.

## s8 · ChartScene

```props
{"eyebrow": "verification", "title": "Two implementations, one trajectory", "kind": "line", "series": [{"label": "your engine", "color": "#5ac48c", "values": [0.7873, 0.0856, 0.0624, 0.0422, 0.0281, 0.0192, 0.0135, 0.0097, 0.0074]}, {"label": "hand-derived reference", "color": "#7a8aa0", "values": [0.7873, 0.0856, 0.0624, 0.0422, 0.0281, 0.0192, 0.0135, 0.0097, 0.0074]}], "xLabel": "Adam step, sampled every 25 (0 -> 199)", "yLabel": "BCE loss", "caption": "Identical init, identical optimiser, gradients derived two independent ways. Max divergence over 200 steps: 6e-17."}
```

And here is the verification for the whole topic. Two networks. Same architecture, same initialisation, same Adam hyper parameters, same data. One is differentiated by the closures we have spent three episodes writing. The other has its gradients derived by hand on paper, written out in plain NumPy, sharing no code with the engine at all. Train both for two hundred steps and plot the loss. You cannot see two lines because there are not two lines. The maximum divergence across all two hundred steps is six times ten to the minus seventeen, which is machine epsilon. And that is the sentence I want you to leave with. Automatic differentiation is not an approximation of the derivative. It is the derivative, evaluated in a different order.

## s9 · ConceptScene

```props
{"eyebrow": "the cost", "title": "What the engine costs you", "body": "The same benchmark also times both. Your engine is 3.7x slower than the hand-written gradients it matches exactly. That is the price of generality, and it is worth understanding rather than hiding.", "points": ["Every op allocates a node and a closure — Python object overhead per operation.", "Nothing is fused: exp then sum then log is three full passes over memory.", "No graph is compiled, so no algebraic simplification happens.", "All three are exactly what torch.compile, XLA and TorchScript exist to fix."], "aside": "Knowing the shape of that 3.7x is what makes the compiler chapters later in the course make sense."}
```

The same benchmark also times both, and honesty is worth more here than a flattering number. Your engine is three point seven times slower than the hand written gradients it matches exactly. Where does that go. Every operation allocates a node and a closure, which is Python object overhead per operation rather than per batch. Nothing is fused, so exponent then sum then logarithm is three separate full passes over memory instead of one. And nothing is compiled, so no algebraic simplification ever happens. Those three sentences are, almost word for word, the problems that torch compile, XLA and TorchScript exist to solve. Understanding the shape of this three point seven times is what makes those tools legible later, instead of magic with a decorator.

## s10 · RecapScene

```props
{"eyebrow": "topic recap", "points": ["Forward broadcast equals backward sum-and-reshape. Six lines, and every operator funnels through them.", "Gradient is conserved through unbroadcast — a free property test.", "Derive matmul's backward from indices once; check it with 'a gradient has the shape of its thing'.", "Detaching a value that should be differentiable is the bug that ships. Gradcheck finds it; loss curves do not.", "Your engine matches a hand-derived reference to 6e-17, at 3.7x the cost."], "ifSkipped": "Phase 7 rebuilds this same backward pass to fit in GPU cache. Without this topic, that is an incantation.", "next": "Next topic: T16A — matrix multiplication, and why the same arithmetic can run 36x apart."}
```

To recap the whole topic. Forward broadcast equals backward sum and reshape, six lines, and every operator funnels through them. Gradient is conserved through that reduction, which gives you a free property test. Derive matmul's backward from index notation once, and check yourself with the rule that a gradient has the shape of the thing it is a gradient for. Detaching a value that should have stayed differentiable is the bug that actually ships, and gradcheck finds it while loss curves never will. And your engine matches an independently hand derived reference to six times ten to the minus seventeen, at three point seven times the cost. Phase seven will rebuild this exact backward pass so that it fits in a GPU's cache, and without this topic that would be an incantation. Next topic is sixteen A. Matrix multiplication, where the same arithmetic, compiled with the same flags, runs thirty six times apart depending only on the order it touches memory.
