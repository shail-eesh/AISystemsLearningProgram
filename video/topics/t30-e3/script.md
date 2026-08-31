---
topic: t30-e3
episode: 3
title: Building FinTok
voice: am_michael
speed: 0.85
runtime_target_minutes: 11
paper: GPT-2 pre-tokenizer regex; tiktoken as benchmark
---

## s1 · TitleCard

```props
{"title": "Tokenizers, part three", "subtitle": "The regex is a prior. Special tokens must be unforgeable. And a domain vocabulary is worth 2.73x, measured.", "topicId": "T30", "episode": "Episode 3"}
```

Episode three, and the last episode of phase one. Today we build FinTok, the tokenizer the rest of this course runs on. Three ideas. The pre tokenizer as a hard prior. Special tokens as unforgeable symbols. And an honest measurement of what training on your own domain actually buys you.

## s2 · ConceptScene

```props
{"eyebrow": "the prior", "title": "The regex decides what BPE may never merge", "body": "BPE is greedy about frequency. Left alone it will happily learn a token spanning ' the' and a newline, because that pair is common. Pre-tokenization splits the text first, and BPE runs inside each chunk only.", "points": ["No merge can ever cross a chunk boundary. That is a hard constraint, not a preference.", "GPT-2's regex is why its tokens carry a leading space: ' the', not 'the'.", "So the regex is a prior on the token inventory, chosen before any data is seen.", "Which makes it worth measuring rather than copying."], "aside": "It is the one part of a tokenizer that is designed rather than learned."}
```

Start with pre tokenization, which people treat as plumbing and is not. BPE is greedy about frequency and has no notion of linguistic sense. Left completely alone, it will cheerfully learn a token that spans the word the and a following newline, because that pair is genuinely frequent. Pre tokenization prevents that by splitting the text with a regular expression first, and running BPE strictly inside each chunk. No merge can ever cross a chunk boundary. That is a hard constraint, not a preference. GPT two's regex is the reason its tokens carry a leading space. Space the, rather than the. So the regex is a prior on which tokens can exist at all, chosen before a single byte of data is seen. It is the one part of a tokenizer that is designed rather than learned, which makes it worth measuring rather than copying.

## s3 · CodeWalkthrough

```props
{"eyebrow": "the design", "title": "Three finance-specific rules", "filename": "src/t30_fintok/pretokenize.py", "code": "FINANCE_PATTERN = re.compile(\n    r\"'s|'t|'re|'ve|'m|'ll|'d\"\n    r\"| ?\\d{4}-\\d{2}-\\d{2}\"          # ISO date, one chunk\n    r\"| ?\\d{1,2}:\\d{2}(?::\\d{2})?\"   # time, one chunk\n    r\"| ?\\p{Lu}{2,}\\d*[A-Z0-9]*\"     # TICKER / ISIN runs\n    r\"| ?\\p{L}+\"\n    r\"| ?\\d{1,3}(?:,\\d{3})+\"         # 1,234,567 as one chunk\n    r\"| ?\\d{1,3}\"\n    r\"| ?\\p{N}+\"                     # <- everything else numeric\n    r\"| ?[^\\s\\p{L}\\p{N}]+\"\n    r\"|\\s+(?!\\S)|\\s+\"\n)", "highlights": [{"at": 20, "lines": [3], "caption": "2024-03-31 as one chunk. GPT-2 splits it into five."}, {"at": 130, "lines": [5], "caption": "EASTPOWER, INE005A01049 — a symbol becomes one or two tokens, not five."}, {"at": 240, "lines": [7, 8], "caption": "Numbers group by thousands. Consistency matters more than the specific grouping."}, {"at": 350, "lines": [9], "caption": "This branch was added after a fuzzer found the pattern was silently deleting characters."}]}
```

Here is the finance variant, which is GPT two's pattern with three additions. ISO dates become one chunk, where GPT two splits them into about five. All caps runs, meaning tickers and ISINs, stay whole, so a symbol is one or two tokens rather than five. And numbers group by thousands, where the important property is consistency rather than the specific grouping. Now look at the highlighted line near the bottom. That branch, any remaining Unicode numeric, was not in the original design. It was added after the fuzzer found something, and that is the next scene.

## s4 · Callout

```props
{"kind": "warning", "heading": "A pre-tokenizer must be TOTAL", "body": "The first version used \\d for its numeric branches while the punctuation fallback excluded \\p{N}. Superscript digits are \\p{N} but not \\d — so no branch claimed them, and the pattern silently deleted them.", "code": "pretokenize('\\xb2\\xb3', 'finance')  ->  []      # characters gone\n\n# round-tripped 'caf\\u00e9 chart-emoji' perfectly the whole time.\n# regression test now: every latin-1 byte, and a 12k codepoint sweep."}
```

Here is the bug, and it is my favourite kind, because it is invisible until you look for it in exactly the right way. The first version of this pattern used backslash d for its numeric branches. And its fallback branch, the one that catches punctuation, excluded backslash p N, the Unicode numeric category. Superscript two and superscript three are p N but they are not backslash d. So no branch in the entire alternation matched them, findall skipped them, and the pre tokenizer silently deleted those characters from the text. The tokenizer round tripped café and emoji flawlessly the whole time. It just quietly dropped superscripts. The fuzzer found it on the run where I fed it all two hundred and fifty six latin one bytes. A pre tokenizer must be total. Every character must be claimed by some branch. There is now a regression test that runs every latin one byte and a twelve thousand code point sweep through both patterns.

## s5 · CodeWalkthrough

```props
{"eyebrow": "the contract", "title": "Special tokens must be unforgeable", "filename": "src/t30_fintok/fintok.py", "code": "SPECIAL_TOKENS = [\n    \"<|endoftext|>\",     # 3489  document boundary\n    \"<|pad|>\",           # 3490  masked out of the loss\n    \"<|filing|>\",        # 3491  a regulatory filing follows\n    \"<|commentary|>\",    # 3492  market commentary follows\n    \"<|announcement|>\",  # 3493  exchange announcement follows\n    \"<|order|>\",         # 3494  a SIMULATED paper ticket follows\n]\n\n# matched BEFORE bpe runs; ids reserved above the learned vocab\ntok.encode('<|order|>')                        # [3494]\ntok.encode('<|order|>', allowed_special=False) # [60, 124, ...] - 9 tokens", "highlights": [{"at": 20, "lines": [2, 3], "caption": "Structural tokens: where a document ends, what the loss should ignore."}, {"at": 150, "lines": [4, 5, 6, 7], "caption": "Semantic tokens: which kind of financial text follows. AlphaSLM conditions on these."}, {"at": 300, "lines": [11, 12], "caption": "The proof: as ordinary text the same string encodes to nine tokens, never to 3494."}, {"at": 420, "lines": [2], "caption": "These ids are part of the model contract. Renumber them and every checkpoint is silently corrupt."}]}
```

Special tokens next. Six of them. Two structural. End of text marks a document boundary and separates documents during pretraining. Pad is masked out of the loss. And four semantic ones, saying which kind of financial text follows, which AlphaSLM will condition on. The requirement is that these must be unforgeable. Their entire job is to mean something that no amount of ordinary text can counterfeit. So they are matched before BPE runs, and their ids are reserved above the learned vocabulary where no merge can ever produce them. The bottom two lines are the proof, and it is a test in the suite. Encode the string with specials enabled and you get one token, three thousand four hundred and ninety four. Encode the same string as ordinary text and you get nine tokens, none of which is three four nine four. And these ids are part of the model contract. Renumber them after AlphaSLM is pretrained and every downstream checkpoint is silently corrupt.

## s6 · ChartScene

```props
{"eyebrow": "measured", "title": "What a domain vocabulary is worth", "kind": "bar", "bars": [{"label": "FinTok", "value": 4.528, "color": "#5ac48c", "note": "trained on financial text"}, {"label": "GeneralTok", "value": 1.658, "color": "#e0b04e", "note": "trained on general English"}, {"label": "raw bytes", "value": 1.0, "color": "#7a8aa0", "note": "the floor"}], "yLabel": "bytes per token (higher is better)", "caption": "Same trainer, same pre-tokenizer, both truncated to 997 merges, measured on held-out FINANCIAL text."}
```

And here is the measurement, set up as carefully as I could make it. Same trainer, same requested vocabulary size, same pre tokenizer, both truncated to exactly nine hundred and ninety seven merges so the vocabularies are identical in size. The only thing that differs is the corpus each was trained on. Both are then measured on held out financial text that neither saw. FinTok gets four point five two eight bytes per token. The general English vocabulary gets one point six five eight. Raw bytes, the floor, is one. So the domain vocabulary is two point seven three times more compact on the domain it was built for. And notice the middle bar is still well above the floor. Even a completely mismatched vocabulary learns English morphology that transfers.

## s7 · ConceptScene

```props
{"eyebrow": "what 2.73x buys", "title": "Compression is not a vanity metric", "body": "Bytes per token is the exchange rate between your text and everything the model costs. Improving it improves four things at once.", "points": ["2.73x more text fits in the same context window.", "2.73x fewer positions to attend over — and attention is quadratic, so the saving is larger than the ratio.", "2.73x fewer forward passes to generate the same amount of text.", "Rarer tokens: 'consolidated revenue' is a few tokens instead of a dozen fragments, so the model has less to learn."], "aside": "This is the highest-leverage 200 lines of code in the whole pipeline, and it runs once."}
```

It is worth being precise about what that two point seven three actually buys, because compression sounds like a vanity metric and is not. Bytes per token is the exchange rate between your text and everything the model costs. Two point seven three times more text fits in the same context window. Two point seven three times fewer positions to attend over, and since attention cost grows with the square of length, the saving there is considerably larger than the ratio. Two point seven three times fewer forward passes to generate the same amount of output text, because generation is one forward pass per token. And a subtler one. Domain phrases become single tokens instead of fragments, so the model has less structure to learn from scratch. Two hundred lines of code, run once, before any training starts.

## s8 · ChartScene

```props
{"eyebrow": "the honest part", "title": "Vocabulary size is a property of the corpus", "kind": "line", "series": [{"label": "merges learned (16,128 requested)", "color": "#e0b04e", "values": [1450, 2203, 3233, 4481]}], "xLabel": "corpus: 126 KB -> 380 KB -> 951 KB -> 2.8 MB", "yLabel": "merges learned", "caption": "The plan asked for FinTok-16k. This corpus supports 3,233. Shipping the curve beats padding the vocabulary."}
```

And now a deviation from the plan, which I want to state plainly rather than quietly ship past. The master plan for this course asks for FinTok sixteen K. Ask this trainer for sixteen thousand one hundred and twenty eight merges and, on the committed corpus, you get three thousand two hundred and thirty three. Look at the curve. A hundred and twenty six kilobytes supports fourteen hundred and fifty merges. Nine hundred and fifty one kilobytes supports three thousand two hundred. Two point eight megabytes supports four thousand five hundred. Training stops when no pair occurs even twice, because minting a token the model will see exactly once is worse than not minting it. So vocabulary size is a property of your corpus, not a knob you turn. The shipped artefact is FinTok three point five K, the curve is published, and pointing the same trainer at a real filings corpus scales it. Ship what the data supports, and show the measurement that explains why.

## s9 · ChartScene

```props
{"eyebrow": "verification", "title": "100,000 fuzzed strings, zero failures", "kind": "bar", "bars": [{"label": "round-trip failures", "value": 0, "color": "#5ac48c", "note": "0 of 100,000"}, {"label": "adversarial strings", "value": 0, "color": "#5ac48c", "note": "0 of 17"}, {"label": "bugs the fuzzer found", "value": 1, "color": "#e0b04e", "note": "the deleted superscripts"}], "caption": "ASCII, CJK, Devanagari, emoji ZWJ sequences, control bytes, and all 256 latin-1 bytes."}
```

The verification for the topic. One hundred thousand randomly generated strings, drawn from seven different alphabets including CJK, Devanagari, emoji with zero width joiners, control characters and all two hundred and fifty six latin one bytes. Zero round trip failures. Seventeen hand picked adversarial strings, including empty, whitespace only, a null byte, near miss special tokens and a four thousand character run. Zero failures. And one real bug found, which is the deleted superscripts. That is the honest scorecard for a fuzzer. It found the thing that no amount of staring at the regex was going to find.

## s10 · RecapScene

```props
{"eyebrow": "topic recap", "points": ["The pre-tokenizer is a hard prior on the token inventory, and it must be TOTAL — every character claimed.", "Special tokens are matched before BPE and reserved above the learned vocab; their ids are part of the model contract.", "A domain vocabulary is worth 2.73x on its own domain, at matched vocabulary size.", "That 2.73x is context window, attention cost, generation steps and learnability, all at once.", "Vocabulary size is a property of the corpus. Ship what the data supports and publish the curve."], "ifSkipped": "FinTok is frozen into AlphaSLM in Phase 2. Everything downstream inherits whatever it decided.", "next": "Phase 1 complete. Next: T4, the transformer from scratch — five episodes, and the flagship of the series."}
```

To recap the topic, and phase one. The pre tokenizer is a hard prior on which tokens can exist, and it must be total, with every character claimed by some branch, or it will delete text without telling you. Special tokens are matched before BPE and reserved above the learned vocabulary, and their ids are part of the model contract for the life of every checkpoint. A domain vocabulary is worth two point seven three times on its own domain at matched vocabulary size. That number is simultaneously context window, attention cost, generation steps and learnability. And vocabulary size is a property of your corpus rather than a knob, so ship what the data supports and publish the curve that explains it. FinTok gets frozen into AlphaSLM in phase two, and everything downstream inherits whatever it decided today. That is phase one complete. Autograd, matrix multiplication, softmax and tokenization. Four topics that every remaining phase quietly assumes. Next up is topic four, the transformer from scratch, five episodes, and the flagship series of the whole course.
