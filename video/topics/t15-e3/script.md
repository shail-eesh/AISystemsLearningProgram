---
topic: t15-e3
episode: 3
title: The scaling study, and the floor underneath it
voice: am_michael
speed: 0.85
runtime_target_minutes: 9
paper: Kaplan et al. 2020; Hoffmann et al. 2022 (Chinchilla)
---

## s1 · TitleCard

```props
{"title": "The scaling study", "subtitle": "Three sizes, one corpus, and a result that turned out to be about the data.", "topicId": "T15", "episode": "Episode 3", "paper": "Hoffmann et al. 2022"}
```

Welcome back. Episode three. Three models, zero point six million, one point eight million and five point nine million parameters. Identical corpus, identical schedule, identical seed. Nothing varies but size. This is the least glamorous experiment in machine learning and the one that decides whether anyone spends a month of GPU time, so it is worth doing carefully — and ours produced a result I want to walk through honestly, because the honest version is more useful than the tidy one.

## s2 · ConceptScene

```props
{"eyebrow": "the setup", "title": "The only fair scaling study is a boring one", "body": "Everything except parameter count is held fixed: same packed shards, same tokenizer, same context, same optimiser, same learning rate, same warmup, same number of steps, same seed.", "points": ["0.6M — 2 layers, width 96", "1.8M — 4 layers, width 160", "5.9M — 6 layers, width 264", "Width and depth grow together, roughly d ~ 64 sqrt(L), the way real model families grow."], "aside": "Growing depth alone gives you a slow model. Growing width alone gives you a shallow one. Real ladders move both."}
```

The setup first, because a scaling study that varies two things measures nothing. Three rungs. Zero point six million parameters: two layers, width ninety six. One point eight million: four layers, width one hundred and sixty. Five point nine million: six layers, width two hundred and sixty four. Width and depth grow together, roughly as width proportional to the square root of depth, which is how real model families are built — growing depth alone gives you a slow model, growing width alone gives you a shallow one. And everything else is nailed down: same shards, same tokenizer, same context, same optimiser, same learning rate, same warmup, same number of steps, same seed. The parameter count is the only thing that moves.

## s3 · ChartScene

```props
{"eyebrow": "the result", "title": "The ordering holds — and the margins are tiny", "kind": "bar", "bars": [{"label": "0.6M", "value": 0.8129}, {"label": "1.8M", "value": 0.8017}, {"label": "5.9M", "value": 0.7955, "color": "#5ac48c"}], "yLabel": "held-out loss (nats per token)", "caption": "Bigger is better at every rung. But the total improvement across a 10.7x parameter increase is 0.017 nats."}
```

Here is the result. Zero point eight one three, zero point eight zero two, zero point seven nine six. The ordering holds: every larger model reaches a lower held-out loss than the one below it, under a matched schedule, which is the claim the study exists to test. And the margins are tiny. Ten point seven times the parameters buys seventeen thousandths of a nat. If you have seen scaling charts in papers, where a ten-fold increase moves the loss by a substantial fraction, this looks broken. It is not broken. It is telling us something specific about our corpus, and finding out what took one more measurement.

## s4 · ChartScene

```props
{"eyebrow": "the diagnosis", "title": "Split the loss by what kind of token it is", "kind": "bar", "bars": [{"label": "numeric tokens\n19% of corpus", "value": 3.563, "color": "#d98b4a"}, {"label": "prose tokens\n81% of corpus", "value": 0.086, "color": "#5ac48c"}], "yLabel": "loss (nats per token)", "caption": "3.563 nats on digits is close to uniform over a large digit alphabet. 0.086 on prose is close to deterministic."}
```

So we split the per-token loss into two classes: tokens containing digits, and everything else. Look at the two numbers. Numeric tokens are nineteen percent of the corpus and cost three point five six nats each — that is close to uniformly random over a large alphabet of digit pieces, which makes sense, because our corpus generator draws prices and volumes and percentages from continuous ranges. Nobody can predict those. And prose tokens, eighty one percent of the corpus, cost zero point zero eight six nats. Zero point zero eight six. That is a perplexity of about one point zero nine. The prose is *nearly deterministic*, because it is generated from a fixed set of templates.

## s5 · ConceptScene

```props
{"eyebrow": "the finding", "title": "The corpus, not the model, is the binding constraint", "body": "Put the two halves together and the whole shape of the scaling result falls out — and it is a statement about the data, not about transformers.", "points": ["Nineteen percent of the loss budget is dice rolls no model can reduce.", "The remaining eighty one percent was already exhausted by the SMALLEST model.", "So there is almost nothing left for extra parameters to win.", "The measured 0.017-nat margin is not a weak result. It is close to all the headroom there was."], "aside": "The fix is a harder corpus, not a bigger model — which is precisely what T23 (synthetic data) and T38 (curation) are for."}
```

And now the two halves go together and the whole shape of the result falls out. Nineteen percent of our loss budget is dice rolls that no model of any size can reduce. The other eighty one percent was already exhausted by the *smallest* model. So there is essentially nothing left for extra parameters to win, and the seventeen-thousandths-of-a-nat margin is not a weak result — it is close to all the headroom that existed. This is a statement about our data, not about transformers. And it is the most useful thing this study produced, because it says what to do next: the fix is a harder corpus, not a bigger model. Which is exactly what topic twenty three, synthetic data, and topic thirty eight, curation, are for. Those two topics usually read as chores. After this measurement they read as the bottleneck.

## s6 · ChartScene

```props
{"eyebrow": "the other constraint", "title": "How far under Chinchilla each rung is", "kind": "bar", "bars": [{"label": "0.6M", "value": 22.1, "note": "% of budget"}, {"label": "1.8M", "value": 6.9}, {"label": "5.9M", "value": 2.1, "color": "#d98b4a"}], "yLabel": "percent of the compute-optimal token budget", "caption": "Chinchilla's rule of thumb is ~20 tokens per parameter. The corpus has 2.4M tokens."}
```

There is a second constraint on top of the first, and it points the same way. Chinchilla's rule of thumb says a compute-optimal run trains on roughly twenty tokens per parameter. Our corpus has two point four million tokens. So the smallest rung sees about twenty two percent of its compute-optimal budget, the middle one about seven percent, and the largest one about two percent. Every model in this study is deep in the data-limited regime, which is precisely the regime where extra parameters stop paying. Almost no published scaling chart is drawn from here — and almost every hobby project lives here. That mismatch is worth carrying around: the charts you have seen were made in a place you are not standing.

## s7 · Callout

```props
{"kind": "warning", "heading": "The power-law fit is three points, two of which drew the line", "body": "We fit L(N) = a N^-b on the smallest and largest rungs only, then *check* it on the middle one — because a fit through all three points hides its own error. And then we extrapolate it two orders of magnitude, which is exactly the number people commit a month of GPU time to.", "code": "L(N) = 0.9168 * N^-0.00910   (anchored on 0.6M and 5.9M)\nmiddle rung, predicted 0.8043 vs actual 0.8017   -> +0.33% error\nextrapolated to 40M:  0.7819       <- a two-point fit, stretched 7x"}
```

One methodological note, because this is where scaling studies get oversold. We fit a power law, loss equals a times parameters to the minus b, using only the smallest and largest rungs — and then we check it against the middle rung, which was not used to fit it. A curve fitted through all three points would pass through all three points and tell you nothing about whether the relationship is real. Ours predicts zero point eight zero four for the middle rung against an actual of zero point eight zero two, an error of a third of a percent — which is genuine evidence, because that point did not draw the line. And then we extrapolate out to the forty million parameter model, which gives zero point seven eight. Look at what that number actually is: a two-point fit, stretched by nearly an order of magnitude, on a corpus none of those models could saturate. It is exactly the kind of number people commit a month of GPU time to. Report it, extrapolate with it if you must, and never forget how it was made.

## s8 · ChartScene

```props
{"eyebrow": "overfitting?", "title": "The generalisation gap, which is the other thing to watch", "kind": "bar", "bars": [{"label": "0.6M", "value": 0.004}, {"label": "1.8M", "value": 0.0039}, {"label": "5.9M", "value": 0.0068, "color": "#d98b4a"}], "yLabel": "val loss minus train loss (nats)", "caption": "Small, but growing faster at the top rung than the val loss is falling. That is the signal to add data, not parameters."}
```

Last chart. The gap between training loss and held-out loss, per rung. Four thousandths, four thousandths, seven thousandths. All small — nobody here is badly overfitting after one pass over the corpus. But watch the direction. Between the middle and top rungs the gap grew by nearly three thousandths while the held-out loss fell by six. That ratio is the thing to watch. When the gap starts growing faster than the loss is falling, you have stopped buying generalisation and started buying memorisation, and the correct response is more data rather than fewer parameters. On this corpus, at these sizes, we are right at the turn.

## s9 · RecapScene

```props
{"eyebrow": "recap", "title": "The study, in three sentences", "points": ["The ordering holds: every larger rung reaches a lower held-out loss under a matched schedule.", "The margin is tiny because 19% of the corpus is irreducible digits and the remaining 81% was already solved by the smallest model.", "Both constraints — the entropy floor and the Chinchilla budget — say the same thing: the data is the bottleneck, not the model."], "ifSkipped": "Skip this and you would read the small margins as 'scaling does not work at small scale', which is the wrong lesson from the right numbers.", "next": "Episode 4 — reading your first pretrain: four ways to be wrong about a perplexity number."}
```

Three things. One. The ordering holds — every larger model reaches a lower held-out loss under a matched schedule, which is the claim the study was built to test. Two. The margin is small because nineteen percent of the corpus is irreducible digits and the other eighty one percent was already solved by the smallest model, so there was very little headroom to compete for. Three. Both constraints, the entropy floor and the Chinchilla token budget, point at the same conclusion: the data is the bottleneck here, not the model. If you take away only one thing, take away that the interesting question was not "did bigger win" but "why was the margin that size", and that the second question is answerable with one extra measurement. Next time: reading a pretrain, and four different ways to be wrong about a perplexity number.
