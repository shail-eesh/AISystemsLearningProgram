---
topic: t15-e2
episode: 2
title: The harness, and what a checkpoint has to contain
voice: am_michael
speed: 0.85
runtime_target_minutes: 8
paper: nanoGPT training loop; Loshchilov & Hutter 2017 (AdamW, cosine schedule)
---

## s1 · TitleCard

```props
{"title": "The harness", "subtitle": "The difference between a teaching loop and a loop you leave running overnight.", "topicId": "T15", "episode": "Episode 2"}
```

Welcome back. Episode two. In topic four we wrote a training loop in forty lines: run it, read the number, move on. Today we write the version you leave running overnight on a GPU, and the entire difference between the two is *what happens when it stops*. Four things that loop does not do, and one test that decides whether we got it right.

## s2 · ChartScene

```props
{"eyebrow": "the schedule", "title": "Warmup, then cosine decay to a tenth", "kind": "line", "series": [{"label": "learning rate", "values": [0.0000075, 0.00075, 0.0015, 0.00297, 0.003, 0.00266, 0.00165, 0.00064, 0.0003], "color": "#5ac48c"}], "xLabel": "step: 0, 25, 50, 99, 100, 300, 600, 900, 1199", "yLabel": "learning rate", "caption": "Warmup for 100 steps, then a cosine from the peak down to 10% of it."}
```

Start with the schedule, because it explains two decisions people copy without knowing why. The first hundred steps are a linear warmup from almost nothing up to the peak. The reason is Adam. Adam divides the gradient by a running estimate of its second moment, and at step zero that estimate has seen essentially no gradients, so the update is roughly the sign of the gradient times the learning rate — a confident, full-size step in a direction the optimiser has no evidence about. Warmup is the fix. Then the cosine decay down to a tenth of the peak. Late in training you want small, careful steps; a constant learning rate bounces around the minimum instead of settling into it. Neither of these is a magic recipe: both are answers to a specific problem, and if you can say what the problem is you can decide when the answer does not apply.

## s3 · ConceptScene

```props
{"eyebrow": "the point", "title": "A checkpoint is not the weights", "body": "A checkpoint is everything the loop needs so that stopping and restarting is invisible. Save only the weights and your 'resume' silently restarts the optimiser and the data sampler from scratch.", "points": ["model — the weights. The easy part.", "optimizer — Adam's two moment tensors per parameter. Roughly two thirds of the file.", "state — step, tokens seen, the history so far.", "numpy_rng — the data sampler's exact position in its stream.", "torch_rng — dropout draws and anything else stochastic."], "aside": "The optimiser state is the part people omit, because a model that loads and generates *looks* like a successful resume."}
```

Now the point of the episode. A checkpoint is not the weights. A checkpoint is everything the loop needs in order for stopping and restarting to be invisible. Five things. The model, which is the easy part. The optimiser — Adam's two moment tensors for every parameter, which is about two thirds of the file. The run state: which step we are on, how many tokens have been seen. The data sampler's random number generator state, so the resumed run draws the windows it would have drawn. And torch's own generator, for dropout and anything else stochastic. The optimiser state is the one people omit, and I think I know why: a model that loads and generates text *looks* like a successful resume. The failure is silent. Your moment estimates restart at zero, warmup does not happen because you are at step six thousand, and the first few dozen steps after the resume take large, badly-scaled updates into a model that was nearly converged.

## s4 · CodeWalkthrough

```props
{"eyebrow": "the test", "title": "120 steps straight, versus 60 + resume + 60", "filename": "phases/p2/t15-slm/bench/verify.py", "code": "straight = build()\nTrainer(straight, train, val, spec(), run_dir=root/'a').train()\n\npart = build()\nTrainer(part, train, val, spec(), run_dir=root/'b').train(until=60)\n\ntorch.manual_seed(999)          # deliberately poison the global RNG\nresumed = build()\nt = Trainer(resumed, train, val, spec(), run_dir=root/'b')\nt.load()\nt.train()\n\nworst = max((a - b).abs().max() for a, b in zip(\n    straight.state_dict().values(), resumed.state_dict().values()))\n# 0.0", "highlights": [{"at": 20, "lines": [1, 2], "caption": "One run, straight through, 120 steps. This is the reference."}, {"at": 140, "lines": [4, 5], "caption": "The same run, stopped halfway and checkpointed."}, {"at": 250, "lines": [7], "caption": "Poison the global RNG on purpose — a resume that only works if nothing else touched torch is not a resume."}, {"at": 360, "lines": [13, 14, 15], "caption": "Exactly 0.0. Not 'close'. Drop any one of the five checkpoint keys and this stops being zero."}]}
```

So here is the test, and it is the strong form. Train one hundred and twenty steps straight through; that is the reference. Then train sixty, checkpoint, throw the object away, build a fresh model, load, and train sixty more. In between the two halves we deliberately poison torch's global random number generator, because a resume that only works when nothing else in the process touched torch is not a resume, it is a coincidence. Then compare every parameter. The answer is exactly zero point zero. Not close. The resumed run computed identical arithmetic, because the sampler's position, the optimiser moments and the schedule position all came back. Drop any one of those five keys from the checkpoint and this number stops being zero, which is exactly what makes it a useful test.

## s5 · ConceptScene

```props
{"eyebrow": "the memory trick", "title": "Gradient accumulation decouples the batch the optimiser sees from the batch memory holds", "body": "Run the batch as several micro-batches, sum their gradients, then take one optimiser step. Mathematically the same update; arbitrarily less memory.", "points": ["Divide each micro-batch's loss by the number of micro-batches.", "Otherwise the accumulated gradient is a SUM where it should be a MEAN.", "Forget that and your effective learning rate is multiplied by the accumulation factor.", "Which people then report as 'accumulation makes training unstable'."], "aside": "This is what buys you a 40M model on 12 GB of VRAM. The optimiser's batch size stops being a memory constraint."}
```

Next: gradient accumulation, which is what will let a forty million parameter model train on twelve gigabytes. Instead of one batch of twenty four, run four micro-batches of six, sum their gradients, then take one optimiser step. Mathematically the same update, arbitrarily less memory. There is exactly one thing to get right, and it is a division. Each micro-batch's loss must be divided by the number of micro-batches, so that what accumulates is the *mean* gradient over the full batch rather than the sum. Forget it, and your effective learning rate is silently multiplied by the accumulation factor, training gets unstable, and you conclude that accumulation is unstable rather than that you scaled your learning rate by four without meaning to.

## s6 · Callout

```props
{"kind": "insight", "heading": "And it is NOT bit-identical — which is the interesting part", "body": "Both runs draw the same sixteen windows in the same order. But summing four partial gradients is a different order of float additions than summing sixteen at once, and float addition is not associative — the same fact T45A spent two episodes on. A few parts in a million after thirty steps is reassociation noise, not a bug.", "code": "batch 16 x micro 1  ->  loss 4.243864\nbatch  4 x micro 4  ->  loss 4.243864\nlargest parameter difference: 6.3e-06"}
```

And here is a detail I did not expect and am glad we measured. The two runs draw the same sixteen windows in the same order, so you might expect the parameters to match to the bit. They do not. The difference after thirty steps is six parts in ten million. The reason is that summing four partial gradients is a different order of floating-point additions than summing sixteen at once, and floating-point addition is not associative — which is the same fact topic forty five A spent two episodes on. So this is reassociation noise, not a bug. It is also why a resume test must not change the accumulation factor half way through, and why "my accumulated run does not exactly reproduce my unaccumulated one" is an expected observation rather than a defect report.

## s7 · ConceptScene

```props
{"eyebrow": "the log", "title": "One JSON object per line, no service, no account", "body": "Everything a hosted experiment tracker would have charted, in a file you can grep, diff between runs, and read on a machine with no network.", "points": ["step, loss, learning rate, gradient norm, tokens seen, wall-clock seconds", "eval records interleaved, tagged, in the same file", "appended, so a crashed run still has everything up to the crash", "and a wall-clock budget that stops cleanly at a checkpoint rather than being killed"], "aside": "The gradient-norm column is the one people forget to log and the first one they want when a run goes strange."}
```

Two smaller things and then the recap. First, the log: one JSON object per line. Step, loss, learning rate, gradient norm, tokens seen, seconds elapsed, with evaluation records interleaved in the same file. It is appended as it goes, so a run that dies still has everything up to the moment it died. No service, no account, no network. And I want to single out the gradient norm column, because it is the one people forget to log and the first one they want when a run starts behaving oddly — a loss spike with a normal gradient norm is a different problem from a loss spike with a gradient norm of two hundred. Second, a wall-clock budget: the loop can be told to stop after so many seconds, and it stops *at a checkpoint*, cleanly, rather than being killed at an arbitrary point by whatever is enforcing your time limit.

## s8 · RecapScene

```props
{"eyebrow": "recap", "title": "The harness, in three sentences", "points": ["A checkpoint is model, optimiser, run state and BOTH random number generator states — and the test is that 60 + resume + 60 equals 120 exactly.", "Gradient accumulation is the same update with less memory, provided you divide the loss by the number of micro-batches.", "Log the gradient norm. It is the column you will want first when something goes wrong."], "ifSkipped": "Skip this and the overnight 4070 run in episode four is a thing you start and hope about, rather than a thing you can interrupt.", "next": "Episode 3 — the scaling mini-study: three model sizes, one corpus, and a result I did not expect."}
```

Three things. One. A checkpoint is the model, the optimiser, the run state and both random number generator states, and the test that decides whether you got it right is that sixty plus a resume plus sixty equals one hundred and twenty, exactly. Two. Gradient accumulation gives you the same update with less memory, provided you divide by the number of micro-batches — and it is not bit-identical, for reasons that are about floating point and not about you. Three. Log the gradient norm. Next time, the scaling mini-study: three model sizes, one corpus, a matched schedule, and a result that turned out to be about the *corpus* rather than about the models.
