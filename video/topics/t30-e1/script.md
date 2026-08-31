---
topic: t30-e1
episode: 1
title: Why tokenization exists
voice: am_michael
speed: 0.85
runtime_target_minutes: 10
paper: Sennrich, Haddow & Birch 2015 (BPE for NMT); GPT-2 byte-level BPE
---

## s1 · TitleCard

```props
{"title": "Tokenizers, part one", "subtitle": "The model never sees your text. It sees whatever the tokenizer decided your text was.", "topicId": "T30", "episode": "Episode 1"}
```

Phase one, topic thirty. Tokenizers. This is the last topic of the foundations phase and it is the one with a real product hook, because the tokenizer we build here is the one AlphaSLM is pretrained on, which means every model in the rest of this course inherits it. Here is the framing I want you to carry through all three episodes. The model never sees your text. It sees whatever the tokenizer decided your text was.

## s2 · ArchitectureMap

```props
{"eyebrow": "where we are", "highlight": ["models"], "caption": "FinTok is the tokenizer AlphaSLM is pretrained on — and therefore the one every downstream topic inherits."}
```

Here is where this lands on the desk. The lit block is models. FinTok, the tokenizer we build in episode three, feeds AlphaSLM in phase two. And once a model is pretrained on a vocabulary, that vocabulary is frozen for the life of the model. LoRA in phase five inherits it. The embedding model inherits it. The quantised weights in phase six inherit it. The Rust inference server inherits it. Changing a tokenizer after pretraining does not mean retraining a tokenizer. It means retraining everything.

## s3 · ConceptScene

```props
{"eyebrow": "the problem", "title": "Three bad answers, and then the good one", "body": "A neural network needs a finite set of integer symbols. Text is not that. Every obvious way of bridging the gap fails in a specific, instructive way.", "points": ["Characters: tiny vocabulary, but sequences get 4-5x longer, and attention is quadratic in length.", "Words: short sequences, but the vocabulary is unbounded and every unseen word becomes <unk>.", "Bytes: no unknowns ever, but the sequences are the longest of all.", "Subwords: learn the units from the data. Common words stay whole, rare ones decompose."], "aside": "BPE is the answer to 'what if the vocabulary were a hyperparameter you fit to the corpus?'"}
```

The problem is that a neural network needs a finite set of integer symbols, and text is not that. There are three obvious bridges and each fails instructively. Characters. The vocabulary is tiny and there are no unknowns, but sequences get four or five times longer, and attention cost grows with the square of length, so you pay dearly. Words. Sequences are short, but the vocabulary is unbounded. New tickers appear, people misspell things, and everything you did not see in training collapses to a single unknown token, which is a hole in the model's perception. Bytes. No unknowns ever, guaranteed, but the sequences are the longest of the three. Subwords are the answer that won, and the framing worth having is this. What if the vocabulary were a hyperparameter you fit to the corpus. Common words stay whole because they are frequent. Rare words decompose into pieces the model has seen. That is byte pair encoding.

## s4 · ChartScene

```props
{"eyebrow": "the three lengths", "title": "'length' is three different numbers", "kind": "bar", "bars": [{"label": "Python len()", "value": 1, "color": "#7a8aa0", "note": "code points"}, {"label": "C# .Length", "value": 2, "color": "#e0b04e", "note": "UTF-16 units"}, {"label": "UTF-8 bytes", "value": 4, "color": "#5ac48c", "note": "what a tokenizer sees"}, {"label": "what a human sees", "value": 1, "color": "#7a8aa0", "note": "one chart emoji"}], "caption": "The string is a single chart emoji. Four defensible answers to 'how long is it'."}
```

Before the algorithm, a detour that is not pedantry. Take a single emoji, the upward chart. How long is it. Python says one, because a Python string is a sequence of code points. C sharp says two, because a dot NET string is UTF sixteen and this character needs a surrogate pair. UTF eight says four bytes. And a human says one thing. Four defensible answers, and if you are coming from the dot NET side, the second one is the one your instincts are calibrated to. The number a tokenizer cares about is the third one. Bytes.

## s5 · CodeWalkthrough

```props
{"eyebrow": "the choice", "title": "Byte-level: the vocabulary starts complete", "filename": "phases/p1/t30-tokenizer-bpe/src/t30_fintok/bpe.py", "code": "vocab = {i: bytes([i]) for i in range(256)}\n\n# and therefore, for every string in the universe:\n#   'hello'    ->  5 base tokens, 0 unknowns\n#   'RELIANCE' ->  8 base tokens, 0 unknowns\n#   emoji      ->  4 base tokens, 0 unknowns\n#   b'\\x00\\xff' -> 2 base tokens, 0 unknowns\n\n# there is no <unk> token, and no code path that needs one", "highlights": [{"at": 20, "lines": [1], "caption": "The entire base vocabulary. 256 entries, and it is finished before training starts."}, {"at": 160, "lines": [3, 4, 5, 6, 7], "caption": "Every string has a UTF-8 encoding, and every byte of it is already a token."}, {"at": 320, "lines": [9], "caption": "decode(encode(s)) == s becomes structural, not aspirational."}]}
```

Here is the decision that makes everything downstream simple. The base vocabulary is the two hundred and fifty six byte values. That is the whole line. One dictionary comprehension, and the base vocabulary is finished before training even begins. And look at what falls out. Every string in the universe has a UTF eight encoding, and every byte of that encoding is already a token. So hello is five base tokens. A ticker symbol is eight. An emoji is four. Two arbitrary control bytes are two. And there is no unknown token anywhere in the design, and no code path that needs one. Round tripping becomes structural rather than aspirational. Decode of encode of s equals s, for every s, because the identity holds at the byte level and merging is reversible.

## s6 · Callout

```props
{"kind": "gotcha", "heading": "Normalisation is still your problem", "body": "'café' typed as one precomposed character is 5 bytes. Typed as 'e' plus a combining acute it is 6. Two strings a user cannot tell apart become two different token sequences — and the model behaves differently for reasons invisible in the logs.", "code": "len('caf\\u00e9'.encode())   # 5   NFC, precomposed\nlen('cafe\\u0301'.encode())  # 6   NFD, combining acute\n\n# identical on screen. different token ids. different output."}
```

Byte level does remove the unknown token problem. It does not remove Unicode. Here is the one that will bite you in production. The word café, typed with a precomposed e acute, is five bytes. The same word typed as a plain e followed by a combining acute accent is six bytes. They are pixel identical on screen. They are different byte sequences, so they are different token sequences, so the model produces different output. And nothing in your logs will show you the difference, because your log viewer renders them identically too. Fix a normalisation form at the door, on the way in, once. This is a five line decision that saves a very confusing Friday.

## s7 · MathReveal

```props
{"eyebrow": "the algorithm", "title": "Byte pair encoding, in three lines", "english": "Repeatedly find the most frequent adjacent pair of symbols in the corpus, mint a new symbol for it, and replace every occurrence. Stop when the vocabulary is big enough.", "equation": "merge_k = argmax over pairs (a,b) of  count(a,b)     ;     vocab[256+k] = vocab[a] + vocab[b]", "code": "while len(vocab) < vocab_size:\n    best = max(pair_counts, key=pair_counts.get)\n    vocab[len(vocab)] = vocab[best[0]] + vocab[best[1]]\n    merges.append(best)\n    replace_everywhere(best)", "note": "Sennrich et al. 2015. That is the entire algorithm; everything else in the file is efficiency.", "stageFrames": [10, 130, 260]}
```

And now the algorithm, which is genuinely three lines. In English. Repeatedly find the most frequent adjacent pair of symbols in the corpus, mint a new symbol for that pair, and replace every occurrence of it. Stop when the vocabulary is large enough. In symbols. Merge k is the argmax over pairs of their count, and the new vocabulary entry at index two fifty six plus k is the concatenation of the two pieces. And in code, five lines. That is byte pair encoding, from the Sennrich paper in twenty fifteen, originally proposed for machine translation. Everything else in the implementation file is efficiency, and none of it changes the answer.

## s8 · ConceptScene

```props
{"eyebrow": "the consequence", "title": "The vocabulary is derived, not stored", "body": "Token 256+k is, by construction, the concatenation of the two pieces that merge k joined. So the merge list is the entire artefact — the vocabulary rebuilds from it in four lines.", "points": ["Saving a tokenizer means saving an ordered list of integer pairs. Nothing else.", "Rank — the position in that list — is load-bearing, and episode 2 is about why.", "A prefix of the merge list is itself a complete, valid, smaller tokenizer.", "That last property is what lets you compare two tokenizers at matched vocabulary size."], "aside": "FinTok's committed artefact is 3,233 pairs of integers. The vocabulary is reconstructed at load time."}
```

One consequence worth pulling out, because it surprises people. The vocabulary is derived, not stored. Token two fifty six plus k is, by construction, the concatenation of the two pieces that merge k joined together. So saving a tokenizer means saving an ordered list of integer pairs and nothing else. The vocabulary rebuilds from it in four lines. Three things follow. First, the artefact is tiny. Second, the rank, meaning the position in that list, is load bearing, and episode two is entirely about why. And third, a prefix of the merge list is itself a complete and valid smaller tokenizer, which turns out to be exactly the tool we need in episode three to compare two tokenizers fairly.

## s9 · RecapScene

```props
{"eyebrow": "episode one", "points": ["Characters make sequences too long, words make the vocabulary unbounded; subwords are fitted to the corpus.", "'Length' is three different numbers. A tokenizer cares about UTF-8 bytes.", "A byte-level base vocabulary is 256 entries and is complete before training — no <unk>, ever.", "Normalisation is still yours to fix: NFC and NFD are the same word and different tokens.", "BPE: most frequent adjacent pair, mint a symbol, replace, repeat. The merge list IS the tokenizer."], "ifSkipped": "Half of 'LLMs cannot do arithmetic' is a pre-tokenizer choice about how numbers get split.", "next": "Episode 2: training it, with every merge shown on a corpus of nine words."}
```

To recap. Characters make sequences too long, words make the vocabulary unbounded, and subwords are the compromise that is fitted to your corpus rather than chosen in advance. Length is three different numbers and a tokenizer cares about the UTF eight one. A byte level base vocabulary is two hundred and fifty six entries, complete before training starts, which eliminates the unknown token entirely. Normalisation is still your problem, because the same word in two normal forms is two token sequences. And the algorithm is. Most frequent adjacent pair, mint a symbol, replace everywhere, repeat. The merge list is the tokenizer. One last thought before the next episode. A good fraction of the complaint that language models cannot do arithmetic is really a claim about how the pre tokenizer chose to split numbers. Episode three has the measurements. Next episode. Training, with every merge shown, on a corpus of nine words.
