---
topic: t15-e4
episode: 4
title: Reading your first pretrain
voice: am_michael
speed: 0.85
runtime_target_minutes: 8
paper: —
---

## s1 · TitleCard

```props
{"title": "Reading your first pretrain", "subtitle": "Four ways to be wrong about a perplexity number, and what to do instead.", "topicId": "T15", "episode": "Episode 4"}
```

Welcome back. The last episode of topic fifteen. The models are trained. Today: how to evaluate a language model without fooling yourself, which is four specific mistakes, each of which produces a number that looks fine. And then the 4070 run — what to plan for, and what our own planning arithmetic says about whether it is even worth doing.

## s2 · ConceptScene

```props
{"eyebrow": "mistake 1", "title": "Reporting one number for a mixed corpus", "body": "Perplexity on a mixed corpus is dominated by whatever register is most predictable. Our price tape is nearly deterministic; our filing prose is not. One number averages them and tells you about neither.", "points": ["Break it out: filing, commentary, announcement, order, tape.", "Then decide which one your claim is actually about.", "Ours is about filings, so the held-out split is filings only — chosen at packing time, not at reporting time.", "A validation set diluted with tape would measure something easier and different."], "aside": "The habit generalises: any aggregate over heterogeneous data is a weighted average whose weights you did not choose deliberately."}
```

Mistake one: reporting a single number for a mixed corpus. Perplexity is an average over tokens, so it is dominated by whatever register is most predictable — and in our corpus that is the price tape, which is nearly deterministic. Average the tape together with filing prose and you get a number that describes neither. So we break perplexity out by register: filing, commentary, announcement, order, tape. And then, crucially, we decided *at packing time* which register the claim is about. This topic's claim is about held-out filings, so the held-out split is filings only. That decision belongs in the data pipeline, not in the reporting code, because by the time you are writing the report the temptation to pick the flattering split is very strong.

## s3 · ChartScene

```props
{"eyebrow": "mistake 2", "title": "Evaluating on overlapping random windows", "kind": "bar", "bars": [{"label": "random\ndraw 1", "value": 0.9312}, {"label": "random\ndraw 2", "value": 0.9455}, {"label": "random\ndraw 3", "value": 0.9187}, {"label": "random\ndraw 4", "value": 0.9401}, {"label": "sequential\n(any run)", "value": 0.9336, "color": "#5ac48c"}], "yLabel": "held-out loss", "caption": "Same model, same data. Random windows overlap and skip; sequential windows walk the split once, in order."}
```

Mistake two: evaluating on randomly sampled windows. Random draws overlap, so some tokens get counted several times and others not at all, and the number wobbles between runs of the same model on the same data. Here are four random draws from one model: zero point nine three, zero point nine five, zero point nine two, zero point nine four. A spread of nearly three hundredths, purely from which windows happened to come up. Now the green bar: sequential evaluation, walking the split once in non-overlapping windows. It is deterministic — run it five times and get the same number five times. Random windows are the right thing for *training*, where you want variety. They are the wrong thing for a number you are going to put in a table and compare against another number.

## s4 · Callout

```props
{"kind": "gotcha", "heading": "Mistake 3: comparing perplexities across tokenizers", "body": "Perplexity is per token. A model with a 50,000-token vocabulary gets a better-looking number than one with a 3,500-token vocabulary on identical text, because each of its tokens carries more characters. Bits per character is the comparable unit, and it is one extra line to compute.", "code": "loss              -> per token   (not comparable across tokenizers)\nbits_per_char     -> per character (comparable)\n# 4.82 characters per FinTok token on this corpus"}
```

Mistake three, and this one shows up in published comparisons. Perplexity is per *token*. A model with a fifty thousand token vocabulary needs fewer tokens to express the same text than one with a three and a half thousand token vocabulary, so it gets a better perplexity for free, on identical content, without being a better model. The comparable unit is bits per character, and it is one extra line to compute: total negative log likelihood, divided by log two, divided by the number of characters. Every one of our models shares FinTok so the two columns tell the same story here, but the habit is the point. If you ever compare your model to somebody else's, and the tokenizers differ, perplexity is not a comparison.

## s5 · ChartScene

```props
{"eyebrow": "the ladder, on held-out filings", "title": "Document-by-document perplexity across the three rungs", "kind": "bar", "bars": [{"label": "0.6M", "value": 2.2469}, {"label": "1.8M", "value": 2.2247}, {"label": "5.9M", "value": 2.2158, "color": "#5ac48c"}], "yLabel": "perplexity on held-out filings", "caption": "Scored document by document, not on sampled windows. 1.014x from the smallest rung to the largest."}
```

With those three fixed, here is the actual comparison. Scored document by document over held-out filings the models have never seen, sequentially, with a tokenizer they all share. Two point two five, two point two two, two point two two. The largest rung is one point four percent better than the smallest. And you already know from episode three why that margin is what it is: nineteen percent of the tokens are irreducible digits and the prose was solved by the smallest model. So the number is small, it is real, it is in the right direction, and we can say exactly why it is not bigger. That last part is what separates an evaluation from a leaderboard entry.

## s6 · ConceptScene

```props
{"eyebrow": "mistake 4", "title": "Stopping at the number", "body": "Read what the model writes. Prompt it with each register token and look at the output — an aggregate cannot show you a model that is fluent and wrong.", "points": ["Is it the right register? Prompt with <|filing|>, get filing prose.", "Are the fields in the right order and the format right?", "Are the numbers plausible for that issuer, or does it quote a bank at a chemical company's price?", "Does it contradict itself inside one sentence?"], "aside": "In T4 the model produced 100% grammatically perfect tape lines of which only 36% had a coherent high and low. No aggregate showed that."}
```

Mistake four, and it is the one nobody admits to: stopping at the number. Read what the model writes. Prompt it with each register token and look at the output. Is it the right register — does the filing token actually produce filing prose. Are the fields in the right order. Are the numbers plausible for that issuer, or is it quoting a bank at a chemical company's price. Does it contradict itself inside a single sentence. Remember what happened in topic four: the model produced perfectly-formed tape lines, one hundred percent of them matching the grammar exactly, of which thirty six percent had a coherent high and low. No aggregate metric showed that. Ten seconds of reading did.

## s7 · ConceptScene

```props
{"eyebrow": "the 4070 lane", "title": "Plan the overnight run before you start it", "body": "The gpu-runner script prints its plan and refuses to start if it does not fit. Four copies of every parameter exist before a single activation does: the weight, its gradient, and Adam's two moments.", "points": ["40M parameters -> 633 MB of weights, gradients and optimiser state.", "Activations at batch 24, context 512: 5.4 GB — dominated by the attention matrix, which is quadratic in context.", "Estimated total with headroom: 7.9 GB. It fits in 12 GB.", "And 491 million tokens against a 2.4 million token corpus is 207 epochs."], "aside": "The script prints that epoch count with a warning, because it is the number that decides whether the run is worth the electricity."}
```

And then the 4070 run, which is mostly planning. The runner prints its plan and refuses to start if the plan does not fit in the device. Four copies of every parameter exist before a single activation does — the weight, its gradient, and Adam's two moment estimates — so forty million parameters is six hundred and thirty three megabytes before you have run anything. Activations at a batch of twenty four and a context of five hundred and twelve add five point four gigabytes, dominated by the attention matrix, which is quadratic in context and is exactly the term topic seven's Flash Attention exists to delete. Total, with headroom, seven point nine gigabytes. It fits. And then the number that actually matters: four hundred and ninety one million training tokens against a two point four million token corpus is two hundred and seven passes over the same data. The script prints that with a warning, because it is the number that decides whether the night is worth the electricity.

## s8 · Callout

```props
{"kind": "insight", "heading": "Run it anyway — to watch the divergence", "body": "Two hundred epochs on this corpus will memorise it. Training loss will keep falling and held-out perplexity will flatten and then rise. That divergence, seen once on your own model with your own eyes, is worth more than reading about overfitting ten times.", "code": "python3 gpu-runner/t15_alphaslm_40m.py --dry-run\npython3 gpu-runner/t15_alphaslm_40m.py --rung alphaslm-40m --hours 8\npython3 gpu-runner/t15_alphaslm_40m.py --rung alphaslm-40m --resume"}
```

My recommendation is to run it anyway. Two hundred epochs on this corpus will memorise it, and that is the point: training loss will keep falling while held-out perplexity flattens and then turns upward, and the eval curve in the results file will show you the exact step where it turned. Seeing that divergence once, on your own model, on data you generated, is worth more than reading about overfitting ten times. And when it has happened, you will know precisely why topic twenty three, synthetic data, and topic thirty eight, curation, are the next interesting things to build — because your compute stopped being the constraint some hours ago.

## s9 · RecapScene

```props
{"eyebrow": "recap", "title": "AlphaSLM, in four sentences", "points": ["Evaluate by register, sequentially, in bits per character — and then read the output anyway.", "The largest CPU rung beats the smallest on held-out filings by 1.4%, and we can say exactly why it is not more.", "A checkpoint that resumes invisibly is five things, and the test is that 60 + resume + 60 equals 120 exactly.", "The 4070 run is planned before it is started, and its most important output is the epoch count in the plan."], "ifSkipped": "AlphaSLM is the model every later topic modifies: LoRA in T17, DPO in T19, quantization in T8, served by tickerd in T3. This is where its weights come from.", "next": "T43 — the embedding model, and what 'similar' means geometrically."}
```

Four things to close the topic. One. Evaluate by register, sequentially, in bits per character — and then read the output anyway, because the metric cannot see fluent-and-wrong. Two. The largest CPU rung beats the smallest on held-out filings by one point four percent, and the value of the study is that we can say exactly why it is not more. Three. A checkpoint that resumes invisibly is five things, and the test is that sixty plus a resume plus sixty equals one hundred and twenty, exactly. Four. The overnight run gets planned before it gets started, and the most important line in that plan is the epoch count. AlphaSLM is the model that every later topic in this course modifies rather than replaces — LoRA tunes it, DPO aligns it, quantization shrinks it, and a Rust server you will write serves it. This is where its weights come from. Next topic, T43: embedding models, and what similarity means geometrically. See you there.
