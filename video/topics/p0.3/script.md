---
topic: p0.3
episode: 1
title: The training loop
voice: am_michael
speed: 0.85
runtime_target_minutes: 11
paper: none — the humility lesson is the point
---

## s1 · TitleCard

```props
{"title": "PyTorch tensors & the training loop", "subtitle": "Five lines to learn. One habit that takes longer, and is worth more.", "topicId": "P0.3", "episode": "Episode 1"}
```

Phase zero, topic three. PyTorch tensors and the training loop. Two halves to this episode, and the second one is why the topic exists. The first half is mechanical. Five lines that you must be able to type from an empty file without looking anything up. The second half is judgement. We take the model we just trained, evaluate it honestly, and watch it score exactly chance on a task it appeared to be learning. In finance that second half is the part that costs money, so we front load it.

## s2 · ArchitectureMap

```props
{"eyebrow": "where we are", "highlight": ["training"], "caption": "Every training topic from T15 onwards reuses this loop verbatim."}
```

Where this lands on the desk. The training block. What we build today is reused verbatim by every training topic that follows. Topic fifteen trains a forty million parameter language model with this loop. Topic seventeen adds low rank adaptation on top of it. Topic nineteen swaps the loss for direct preference optimisation. The scaffolding does not change. Which means a mistake we make here, we make forty million parameters later, on a run that takes a night instead of a second.

## s3 · ConceptScene

```props
{"eyebrow": "the tensor", "title": "An ndarray that remembers what happened to it", "body": "Everything from the last episode transfers unchanged: shapes, broadcasting, reductions, einsum. A tensor adds exactly three things.", "points": ["A device the storage lives on — cpu, or cuda:0.", "Optional gradient tracking: the tensor records the operations applied to it so they can be replayed backwards.", "In-place operations, spelled with a trailing underscore, which autograd sometimes refuses to differentiate through.", "Two spelling traps: torch says dim and keepdim where NumPy says axis and keepdims, and torch defaults to float32 where NumPy defaults to float64."], "aside": "from_numpy shares memory. torch.tensor copies."}
```

Start with what a tensor is, because it is less than you might think. Everything from the previous episode transfers unchanged. Shapes, broadcasting, reductions, einsum, all identical. A torch tensor adds exactly three things. A device that the storage lives on, cpu or cuda zero. Optional gradient tracking, which means the tensor records the operations applied to it so they can be replayed in reverse. And in place operations, spelled with a trailing underscore, which are fast and which autograd will sometimes refuse to differentiate through. Two spelling traps to note now and save an hour later. Torch says dim and keep dim where NumPy says axis and keep dims. And torch defaults to float thirty two where NumPy defaults to float sixty four. That default is not carelessness. It halves memory and bandwidth, and gradient noise dwarfs the extra precision. Phase five goes further still, into brain float sixteen.

## s4 · CodeWalkthrough

```props
{"eyebrow": "autograd", "title": "backward() ADDS into .grad", "filename": "steps/step2_autograd_as_a_user.py", "code": "x = torch.tensor(2.0, requires_grad=True)\n\nfor i in range(1, 4):\n    (x ** 2).backward()\n    print(x.grad.item())\n\n# 4.0\n# 8.0\n# 12.0", "highlights": [{"at": 30, "lines": [1], "caption": "requires_grad makes x a leaf of the graph. Operations on it record nodes."}, {"at": 180, "lines": [4], "caption": "backward() walks the graph in reverse, applying the chain rule."}, {"at": 320, "lines": [7, 8, 9], "caption": "Four, eight, twelve. It adds. It does not set."}]}
```

Here is the single most important fact about autograd, and it is on screen. A tensor created with requires grad equals true is a leaf of a computation graph. Operations on it record nodes. Calling backward on a scalar walks that graph in reverse, applying the chain rule, and puts the result into each leaf's grad attribute. Now look at the output. We call backward three times on the same leaf. Four. Eight. Twelve. It adds. It does not set. This is not a wart. It is exactly what makes gradient accumulation across micro batches work, which is how you train a model that does not fit in memory. But it means every single training loop must clear the gradients before it starts. And omitting that is the most common PyTorch bug there is, because the loss still goes down. It just goes down using the sum of every gradient computed so far, and absolutely nothing tells you.

## s5 · ConceptScene

```props
{"eyebrow": "autograd", "title": "Three more facts, then you have the user's model", "points": ["backward() on a non-scalar raises. A loss is scalar precisely so the upstream gradient is implicitly one.", "The graph is freed after backward(). Every forward pass builds a fresh one; retain_graph is nearly always a sign you meant to restructure something.", "Intermediates get no .grad — storing one per node would explode memory. retain_grad() opts one in.", "no_grad() stops recording; detach() cuts one tensor out. Confusing them costs memory, not correctness."], "aside": "You build this machinery yourself in T31. Today you only need to drive it."}
```

Three more facts and you have the complete user's model of autograd. First, calling backward on something that is not a scalar raises an error. A loss is a scalar precisely so that the upstream gradient is implicitly the number one. For a vector you have to supply that gradient yourself. Second, the graph is freed after backward runs. Every forward pass builds a fresh one, and reaching for retain graph equals true is nearly always a sign that you meant to restructure something. Third, intermediate tensors get no gradient attribute at all, because storing one per node would explode memory. Retain grad opts a specific one in, which is useful for debugging and for the interpretability work in topic twenty two. And finally, no grad stops the recording entirely, while detach cuts a single tensor out of the graph. Use no grad for evaluation and detach when you are storing a value for logging. Confusing the two costs memory, not correctness.

## s6 · MathReveal

```props
{"eyebrow": "the loss", "title": "Why logits, not probabilities", "english": "The sigmoid's derivative cancels against the logarithm in the cross-entropy, leaving a gradient that is simply prediction minus target.", "equation": "dL/dz = (σ(z) − y) / n", "code": "loss = nn.BCEWithLogitsLoss()(logits, y)   # fused, stable\n# NOT: BCELoss()(torch.sigmoid(logits), y)", "note": "Hand-written and autograd agree to 2.8e-17 in step 3.", "stageFrames": [10, 130, 250]}
```

The model returns raw logits and never probabilities, and there is a reason worth stating three ways. In words. The sigmoid's derivative cancels against the logarithm in the cross entropy, leaving a gradient that is simply the prediction minus the target. In symbols. The derivative of the loss with respect to the pre activation z is sigma of z, minus y, divided by n. That is it. No sigmoid derivative term survives. And in code, you get that cancellation by using the fused loss, binary cross entropy with logits. Applying a sigmoid yourself and then using plain binary cross entropy is mathematically identical and numerically worse. The fused version uses the log sum exp trick internally. The split version overflows once a logit reaches about plus or minus forty, and hands you a not a number. The step three script writes this derivative out by hand in NumPy and matches PyTorch's autograd to two point eight times ten to the minus seventeen. Doing that once is what stops autograd from feeling like magic.

## s7 · CodeWalkthrough

```props
{"eyebrow": "the loop", "title": "The five lines", "filename": "src/p0_3_training/loop.py", "code": "for epoch in range(cfg.epochs):\n    model.train()\n    for xb, yb in loader:\n        optimiser.zero_grad()          # gradients accumulate\n        logits = model(xb)             # forward\n        loss = criterion(logits, yb)   # a scalar\n        loss.backward()                # reverse-mode autodiff\n        optimiser.step()               # p -= lr * p.grad", "highlights": [{"at": 40, "lines": [4], "caption": "Clear first. Always. This is the line people delete by accident and never notice."}, {"at": 150, "lines": [5, 6], "caption": "Forward, then a scalar loss — scalar so backward needs no upstream gradient."}, {"at": 260, "lines": [7], "caption": "One call fills every leaf's .grad."}, {"at": 360, "lines": [8], "caption": "For plain SGD this is literally p -= lr * p.grad under no_grad. Proven to 7e-09 in step 3."}]}
```

And here is the whole thing. Five lines inside two loops. Clear the gradients. Forward pass. Compute a scalar loss. Call backward. Step the optimiser. Everything else in that file, the seeding, the batching, the evaluation mode, the best checkpoint restore, is bookkeeping you can look up. These five lines you type from memory. It is worth demystifying the last one. For plain stochastic gradient descent, optimiser dot step is literally, for each parameter, subtract the learning rate times its gradient, inside a no grad block. The step three script runs the hand written version alongside the real optimiser for five steps and they agree to seven times ten to the minus nine. Adam adds two exponential moving averages per parameter, which is why the optimiser state costs roughly twice the model in memory. At forty million parameters that is the difference between fitting on a twelve gigabyte card and not.

## s8 · Callout

```props
{"kind": "insight", "heading": "Overfit a tiny batch. Always. First.", "body": "Before any real run: take sixty-four rows and drive training accuracy to one hundred per cent. It takes seconds, and it separates 'my model cannot learn' from 'my data has no signal' — two completely different bugs with completely different fixes. Here: eight hundred parameters, one hundred per cent by epoch sixty-three.", "code": "A. overfit 64 rows: train acc 1.000 (100% at epoch 63)  -> PASS"}
```

Before any serious training run, do this. Take thirty two to a hundred and twenty eight examples, and drive the training accuracy to one hundred per cent. It takes seconds. And it separates two bugs that look identical from the outside and have nothing in common. My model cannot learn, versus my data has no signal. Here, eight hundred parameters memorise sixty four rows completely by epoch sixty three. What that proves is that the loop is wired correctly. Forward, loss, backward, step, the clearing, the data loader, the shapes. All of it. What it proves about the data is nothing whatsoever. The same network would have memorised those sixty four rows just as happily if we had shuffled the labels first.

## s9 · ChartScene

```props
{"eyebrow": "the honest number", "title": "Same model, honest protocol", "kind": "bar", "bars": [{"label": "memorised 64", "value": 100.0, "color": "#5ac48c", "note": "training set"}, {"label": "test", "value": 51.1, "color": "#e05e6b", "note": "z = +0.30"}, {"label": "always 'up'", "value": 55.0, "color": "#e0b04e", "note": "the baseline"}, {"label": "coin flip", "value": 50.0, "color": "#5c6b7c", "note": "± 3.7 points"}], "caption": "Accuracy, per cent. The model is worse than a constant — which is the correct answer here."}
```

And now the number that matters. Same model, same data, chronological split, scaler fitted on the training window only. On the memorised batch, one hundred per cent. On the held out test window, fifty one point one per cent. Predicting up every single day scores fifty five. A coin flip scores fifty, with a standard error of three point seven points at this sample size. So the model is worse than a constant, and its result sits well inside the error bar of pure chance. The z score against a coin flip is plus zero point three. The model is not broken. The prices are a synthetic geometric random walk, so the true predictability of tomorrow's direction is exactly zero, and anything above chance here would be a bug in the protocol rather than a discovery. Three habits follow from this slide. Always report a baseline computed on the same window. Always report the loss as well as the accuracy, because a loss of zero point six nine six five against a natural logarithm of two says nothing was learned more clearly than the accuracy does. And always ask what your sample size can actually support.

## s10 · Callout

```props
{"kind": "warning", "heading": "Leakage does not look like cheating. It looks like a good result.", "body": "One keyword takes this model from chance to eighty-one per cent. Nobody writes center=True intending to cheat — it is a smoothing default that makes the feature look nicer on a chart, and it hands the model tomorrow's return.", "code": "ret.rolling(3).mean()                # t-2, t-1, t     causal\nret.rolling(3, center=True).mean()   # t-1, t,   t+1   the label\n\nhonest      0.511\ncentred     0.811   (+0.300)"}
```

Here is how that fifty one becomes something publishable. We added exactly one feature. A three period rolling mean of the daily return, with centre equals true. Accuracy went from fifty one point one to eighty one point one. Thirty points, from one keyword. And look at what that keyword does. A centred three period window averages t minus one, t, and t plus one. It contains tomorrow's return, which is the label. Nobody types centre equals true intending to cheat. It is a smoothing default. It makes the line look nicer on a chart. The benchmark also measures two other leaks, a shuffled train test split and fitting the scaler before splitting, and on this dataset both of them are nearly silent. That asymmetry is the real lesson. They are quiet because this series is a random walk with no autocorrelation to exploit. Change the data and they wake up. It did not change my number is a statement about your dataset, not about your methodology.

## s11 · ConceptScene

```props
{"eyebrow": "protocol", "title": "What would actually convince you", "points": ["A baseline — majority class, buy-and-hold, last value — on the same window.", "A walk-forward evaluation, not a single split. T48 makes that point-in-time correct.", "Costs: spread, slippage, borrow. A fifty-two per cent hit rate does not survive them.", "The same protocol run on shuffled labels. It must score chance, or your protocol is broken.", "Out-of-sample data the model was never tuned against, used exactly once."], "aside": "Phase 4's eval harness, T27, turns this checklist into code."}
```

So what would convince you, if not accuracy. Five things. A baseline on the same window, whether that is the majority class, buy and hold, or simply predicting the last value. A walk forward evaluation rather than a single split, which topic forty eight makes point in time correct. Costs, meaning spread, slippage and borrow, because a fifty two per cent hit rate does not survive them. The same protocol run on deliberately shuffled labels, which must come back at chance, or your protocol itself is broken. And out of sample data the model was never tuned against, used exactly once. Phase four's evaluation harness, topic twenty seven, turns that checklist into code and runs it against everything the desk produces.

## s12 · RecapScene

```props
{"eyebrow": "P0.3", "title": "Recap", "points": ["Five lines: zero the gradients, forward, scalar loss, backward, step. Backward adds, which is why the clearing comes first.", "Logits with a fused loss, evaluation under eval() and no_grad, and nn.Module is only a parameter registry.", "Overfit a tiny batch before trusting any run — it validates the loop and tells you nothing about the data.", "A falling loss is evidence that gradient descent works. The baseline, the error bar and the causality test are the evidence about your problem."], "ifSkipped": "T31 is this episode inverted: you build the machinery whose interface you just learned. T15 reuses loop.py at forty million parameters, where a missing zero_grad costs a night, not a second.", "next": "Phase 1 · T31 · Building autograd from scratch"}
```

Four takeaways. The five lines. Zero the gradients, forward, scalar loss, backward, step. Backward adds rather than sets, which is exactly why the clearing comes first. Return logits and use a fused loss, evaluate under eval mode and no grad, and remember that a module is only a parameter registry, so a bare tensor attribute is invisible to the optimiser. Overfit a tiny batch before trusting any run, because it validates the loop and tells you nothing at all about the data. And a falling loss is evidence that gradient descent works. The baseline, the error bar, and the causality test are the evidence about your actual problem. What breaks if we skip this. Topic thirty one is this episode inverted. You build the machinery whose user interface you just learned. Topic fifteen reuses this loop file almost verbatim at forty million parameters, where a missing zero grad costs a night of compute instead of a second. And every evaluation claim from phase four onwards rests on the protocol discipline in the second half of this episode. Next up, phase one. Building autograd from scratch.
