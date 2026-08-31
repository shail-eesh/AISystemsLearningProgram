---
topic: t4-e4
episode: 4
title: Training it, and the experiment that lied
voice: am_michael
speed: 0.85
runtime_target_minutes: 10
paper: Olsson et al. 2022, In-context Learning and Induction Heads
---

## s1 · TitleCard

```props
{"title": "Training it, and the experiment that lied", "subtitle": "A loss curve, an attention map, and why the second one nearly fooled us.", "topicId": "T4", "episode": "Episode 4", "paper": "Olsson et al. 2022"}
```

Welcome back. Episode four. The model is built. Today we train it, look at what it learned, and then I am going to walk you through an experiment we ran that produced a beautiful, convincing, completely wrong result — and how we caught it. That last part is the most useful thing in this series, so stay to the end.

## s2 · ChartScene

```props
{"eyebrow": "the loss curve", "title": "Reading a loss curve like an ECG", "kind": "line", "series": [{"label": "validation loss", "values": [4.0862, 1.6298, 1.2908, 1.1878, 1.142, 1.1002], "color": "#5ac48c"}], "xLabel": "step: 0, 100, 200, 300, 400, 499", "yLabel": "loss (nats per character)", "reference": {"value": 4.19, "label": "ln(66) — uniform guessing"}, "caption": "Four layers, 810k parameters, a 92,743-character market tape. 29 seconds on two CPU cores."}
```

Here is the first thing you should do with any new model: check that step zero is where it must be. The reference line is log of sixty six, the vocabulary size, which is four point one nine. That is the loss of a model that guesses uniformly. Our step-zero measurement is four point zero nine. Not lower, which would mean something is leaking the answer, and not higher, which would mean the initialisation is broken. Then it falls: one point six three at a hundred steps, one point two nine at two hundred, one point one at five hundred. Four layers, eight hundred and ten thousand parameters, twenty nine seconds on two CPU cores. The shape matters more than the endpoint. A steep initial drop is the model learning character frequencies, which is cheap. The long shallow tail after it is the model learning structure, which is not.

## s3 · ConceptScene

```props
{"eyebrow": "read the output", "title": "The model learned the grammar of the tape perfectly and its meaning barely at all", "body": "Generated lines, scored automatically against the exact tape grammar and against basic market arithmetic:", "points": ["100% of generated lines match the field grammar exactly — date, symbol, O/H/L/C/V, two decimal places.", "100% are distinct — it is not reciting one memorised line.", "36% have a coherent high and low — that is, H >= max(O,C) and L <= min(O,C).", "It learned the language. It did not learn what 'high' means."], "aside": "Every eval you write after this should be built with that gap in mind. Language models are extremely good at the shape of a thing."}
```

Now read what it writes, because no aggregate will tell you this. We scored eight hundred generated characters two ways. First against the exact grammar of the tape: date, symbol, the letters O H L C V in order, two decimal places on every price. One hundred percent of the generated lines match. And they are all distinct, so it is not reciting one memorised line. Then we scored them against basic market arithmetic: is the high at least as large as the open and the close, is the low at most as small. Thirty six percent. So the model learned the *language* of the tape completely and learned what high and low *mean* barely at all. That gap is the single most useful thing on this slide, and it generalises. Language models are extremely good at the shape of a thing. Every eval you write for the rest of this course should be built by someone who knows that.

## s4 · ConceptScene

```props
{"eyebrow": "interpretability", "title": "An induction head is the first real algorithm a transformer invents", "body": "It implements: 'the pattern A-B happened earlier, I have just seen A, so B comes next.' It needs two layers — one head to carry the previous token's identity forward, a second to match on it — and it is the mechanism behind in-context learning.", "points": ["At position i, find an earlier occurrence of the current token.", "Attend to the token that came AFTER it.", "Copy that token's identity into the prediction.", "Nobody trains it to do this. It appears on its own from ordinary next-token training."], "aside": "Olsson et al. traced in-context learning in real models to exactly this circuit, and watched it appear as a visible bend in the loss curve."}
```

Now let us go looking inside. The thing worth looking for is called an induction head, and it implements one sentence: the pattern A B happened earlier, I have just seen A, so B comes next. Mechanically, at position i, find an earlier occurrence of the token you are currently at, attend to the token that came *after* that occurrence, and copy it. It needs two layers, because one head has to carry the previous token's identity forward before a second head can match on it. And nobody trains it to do this. It emerges from ordinary next-token prediction, and Olsson and colleagues traced in-context learning in real production models back to exactly this circuit, appearing as a visible bend in the loss curve.

## s5 · DiagramScene

```props
{"eyebrow": "the first experiment", "title": "prefix + prefix — and the shortcut hiding inside it", "nodes": [{"id": "a", "label": "random prefix", "sub": "24 tokens", "x": 0.22, "y": 0.3, "w": 0.34, "appearAt": 8}, {"id": "b", "label": "the same prefix again", "sub": "24 tokens", "x": 0.68, "y": 0.3, "w": 0.34, "appearAt": 20}, {"id": "ind", "label": "induction", "sub": "match on content", "x": 0.28, "y": 0.76, "w": 0.28, "appearAt": 48, "color": "#5ac48c"}, {"id": "pos", "label": "positional", "sub": "'look 24 back'", "x": 0.7, "y": 0.76, "w": 0.28, "appearAt": 62, "color": "#d98b4a"}], "edges": [{"from": "a", "to": "b", "label": "repeat", "appearAt": 30}, {"from": "b", "to": "ind", "appearAt": 54}, {"from": "b", "to": "pos", "appearAt": 68}], "caption": "Two ways to solve the same task. Only one of them is an induction head."}
```

So we built the obvious experiment. Take a random prefix of twenty four tokens and repeat it. The second half is perfectly predictable, but only by something that can say "I have seen this token before, what followed it last time". We trained a two-layer model on it, measured how much attention the best head put on the induction target, and got thirty times chance with one hundred percent accuracy on the second half. Beautiful heat map. And then, because the repository has a control for everything, we ran the same experiment with a *one-layer* model. Which should be impossible. One layer cannot form an induction circuit; there is no second layer to do the matching. The one-layer model scored one hundred percent as well.

## s6 · Callout

```props
{"kind": "warning", "heading": "A fixed period is a positional shortcut, not induction", "body": "With a prefix length that never changes, 'attend to the slot 24 back' solves the task perfectly — and a learned positional embedding can express that rule in one layer. No content matching required, no induction head involved, and a beautiful attention map to go with it. We had measured the wrong thing.", "code": "# the fix: every row repeats with its OWN random period\nseq[b, i] = base[b, i % period[b]]     # period ~ U[10, 14]"}
```

Here is what had gone wrong, and it is worth sitting with because the failure mode is so plausible. Our prefix length was always twenty four. So "attend to the slot twenty four back" solves the task perfectly, and a learned positional embedding can express that rule in a single layer. No content matching. No induction circuit. And a gorgeous attention heat map showing enormous weight in exactly the right place. We had measured a positional shortcut and called it an induction head. The fix is one line: give every row in the batch its own random period, drawn between ten and fourteen. Now the induction target moves from row to row, no fixed-offset rule works, and the task can only be solved by matching on content.

## s7 · ChartScene

```props
{"eyebrow": "the corrected experiment", "title": "Two layers learn it. One layer plateaus.", "kind": "line", "series": [{"label": "2-layer: repeat accuracy", "values": [0.014, 0.285, 0.401, 0.493, 0.502, 0.503, 0.63, 0.687, 0.72, 0.78, 0.851, 0.894, 0.92], "color": "#5ac48c"}, {"label": "2-layer: best head attention score x2", "values": [0.076, 0.414, 0.601, 0.68, 0.704, 0.689, 0.708, 0.71, 0.699, 0.715, 0.718, 0.719, 0.703], "color": "#d98b4a"}], "xLabel": "training step (0 to 3000, every 250)", "yLabel": "fraction", "caption": "The attention score plateaus by step 750. Accuracy keeps climbing for the next 2,250 steps."}
```

And here is the corrected experiment, which tells a much more interesting story than the wrong one did. The green line is behaviour: how often the two-layer model correctly predicts the repeated token. It climbs from one percent to ninety two percent over three thousand steps. The orange line is the attention score of the best head — how much weight it places on the induction target. Watch what it does. It rises fast, and then by step seven hundred and fifty it is *flat*. It never moves again. And accuracy climbs for another two thousand two hundred and fifty steps after that. The attention map stopped changing long before the model stopped learning.

## s8 · ChartScene

```props
{"eyebrow": "the control", "title": "The one-layer model has the BETTER attention score and much worse behaviour", "kind": "bar", "bars": [{"label": "2 layers\nrepeat accuracy", "value": 0.94, "color": "#5ac48c"}, {"label": "1 layer\nrepeat accuracy", "value": 0.649}, {"label": "2 layers\nattention score", "value": 0.347}, {"label": "1 layer\nattention score", "value": 0.438, "color": "#d98b4a"}], "yLabel": "score", "caption": "Chance for the attention score is 0.038. The one-layer model hedges across five offsets and looks excellent."}
```

And now the control, which is the punchline of the whole episode. Two layers: ninety four percent accuracy, attention score zero point three five. One layer: sixty five percent accuracy — and an attention score of zero point four four, which is *higher*. The one-layer model looks better on the attention map and performs far worse. What it is doing is hedging: there are only five possible periods, so it spreads weight across the five candidate offsets, and enough of that lands on the right one to score well on the metric while never being able to commit to a prediction. We verified that directly by planting a synthetic head that always looks exactly eleven positions back — a rule with no content matching in it at all — and it scores three and a half times chance. An attention map that looks right is not a circuit that works.

## s9 · Callout

```props
{"kind": "insight", "heading": "Calibrate your interpretability metric in both directions", "body": "Our probe now has two synthetic controls, and every interpretability metric you write should have the same pair: one input that must score maximum, and one plausible-but-wrong mechanism that must score low.", "code": "planted head, attends exactly at the target   -> 1.000\nfixed-offset head, always 11 back             -> 0.128  (3.5x chance)\nuntrained model                               -> 1.0x chance"}
```

So the probe now has controls in both directions, and I would push this as a general rule. Any interpretability metric you write needs two synthetic inputs. One that must score the maximum — here, a planted head that attends exactly at the target, which scores one point zero zero zero. And one plausible-but-wrong mechanism that must score low — here, a fixed-offset head, which scores zero point one three. Without the first, you do not know what a good score looks like. Without the second, you do not know what your metric will accept. And the whole thing sits on top of a behavioural measure, because behaviour is the only thing that cannot be faked by a picture.

## s10 · RecapScene

```props
{"eyebrow": "recap", "title": "What we found, and how we nearly did not", "points": ["The model learned the tape's grammar completely (100% well-formed) and its semantics barely (36% coherent high/low).", "Induction heads emerge from ordinary next-token training — but only if the task cannot be solved positionally.", "The one-layer control had the better attention map and the worse behaviour: an attention map is not a circuit."], "ifSkipped": "Skip this and you will believe the first beautiful heat map you produce, which is how most bad interpretability claims are made.", "next": "Episode 5 — sampling and the KV cache: turning logits into text, and stopping the model doing the same work n times."}
```

Three things. One. The model learned the grammar of the tape completely and its semantics barely at all, and you only find that out by reading the output. Two. Induction heads do emerge on their own from ordinary next-token training — but only when the task genuinely cannot be solved by a positional rule, and our first version of the task could. Three. The one-layer control had a better attention map and much worse behaviour, which means an attention map is evidence about attention and not evidence about a circuit. Next time, the last episode: sampling, and the KV cache. How to turn a logit vector into text that is not the word "the" forever, and how to stop the model redoing the same arithmetic on every step.
