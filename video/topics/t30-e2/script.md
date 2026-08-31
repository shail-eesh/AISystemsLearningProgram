---
topic: t30-e2
episode: 2
title: Training BPE, every merge shown
voice: am_michael
speed: 0.85
runtime_target_minutes: 10
paper: Sennrich, Haddow & Birch 2015
---

## s1 · TitleCard

```props
{"title": "Tokenizers, part two", "subtitle": "Nine words, six merges, shown one at a time — then the two ideas that make it fast.", "topicId": "T30", "episode": "Episode 2"}
```

Episode two. We are going to run the merge loop by hand on a corpus small enough to read, then look at the two ideas that take training from minutes to about a second, and then the encoder, where the interesting bug lives.

## s2 · CodeWalkthrough

```props
{"eyebrow": "the corpus", "title": "Nine words, and their frequencies", "filename": "steps/step2_train_bpe_by_hand.py", "code": "\"low low low low low lower lower newest newest newest widest widest widest\"\n\n# after pre-tokenization, as chunk -> frequency:\n#   b'low'    x1        <- no leading space, first token\n#   b' low'   x4\n#   b' lower' x2\n#   b' newest' x3\n#   b' widest' x3", "highlights": [{"at": 20, "lines": [1], "caption": "The classic BPE demonstration corpus, from the original paper."}, {"at": 140, "lines": [3, 4, 5, 6, 7, 8], "caption": "Count TYPES with weights, not tokens. 15 words in the text; 5 distinct chunks to work on."}, {"at": 300, "lines": [4, 5], "caption": "GPT-2's regex attaches the leading space to the word, so ' low' and 'low' are different chunks."}]}
```

Here is the corpus, which is the one from the original paper. Low, five times. Lower, twice. Newest, three times. Widest, three times. And the first idea is already visible in how we have written it down. We do not process fifteen words. We process five distinct chunks with weights attached. Count types, not tokens. A one megabyte corpus has about two hundred thousand chunk occurrences and only about twenty thousand distinct chunks, and BPE only ever needs the distinct ones plus their frequencies. That alone is a factor of ten. And notice the leading spaces. The pre tokenizer attaches a space to the front of a word, so low at the start of the text and space low in the middle are two different chunks. Episode three is about why that choice exists.

## s3 · DiagramScene

```props
{"eyebrow": "merge by merge", "title": "The first four merges", "nodes": [{"id": "m0", "label": "merge 0", "sub": "'e' + 's' -> 'es'", "x": 0.14, "y": 0.3, "w": 0.22, "color": "#5ac48c", "appearAt": 10}, {"id": "m1", "label": "merge 1", "sub": "'es' + 't' -> 'est'", "x": 0.38, "y": 0.3, "w": 0.22, "color": "#5ac48c", "appearAt": 34}, {"id": "m2", "label": "merge 2", "sub": "'l' + 'o' -> 'lo'", "x": 0.62, "y": 0.3, "w": 0.22, "appearAt": 58}, {"id": "m3", "label": "merge 3", "sub": "'lo' + 'w' -> 'low'", "x": 0.86, "y": 0.3, "w": 0.22, "appearAt": 82}, {"id": "note", "label": "merges compose", "sub": "later merges consume earlier ones", "x": 0.5, "y": 0.78, "w": 0.5, "color": "#e0b04e", "appearAt": 100}], "edges": [{"from": "m0", "to": "m1", "appearAt": 40}, {"from": "m2", "to": "m3", "appearAt": 88}], "caption": "'est' exists only because 'es' already did. The list is ordered, and the order is a dependency chain."}
```

Now watch the merges. Merge zero joins e and s, because e s appears in both newest and widest, six times in total. Merge one joins the new symbol e s with t, giving e s t, again six times. Merge two joins l and o. Merge three joins the new symbol l o with w, giving low. And here is the structural point. Merge one could not have happened before merge zero, because the symbol e s did not exist yet. Later merges consume the output of earlier ones. So the merge list is not a set. It is an ordered dependency chain, and that ordering is going to matter enormously in two scenes' time.

## s4 · ConceptScene

```props
{"eyebrow": "efficiency", "title": "The second idea: touch only what changed", "body": "Recounting every pair after every merge is O(merges x corpus). With a pair-count table and an index from pair to the words containing it, a merge only touches the words that actually contain that pair.", "points": ["Keep pair_counts: how often each adjacent pair occurs, weighted by word frequency.", "Keep pair_where: for each pair, the set of word indices containing it.", "On a merge: for each affected word, retract its old pair contributions, merge, post the new ones.", "3,233 merges over a 1 MB corpus in about one second."], "aside": "This is an inverted index and incremental maintenance — the same shape as a search engine's posting lists."}
```

The second efficiency idea is where the real speedup is. The naive loop recounts every pair in the whole corpus after every merge, which is merges times corpus, and takes minutes. Instead, keep two structures. A pair count table, saying how often each adjacent pair occurs, weighted by the word frequencies. And an index from each pair to the set of words containing it. Now when you merge a pair, only the words in that pair's set can possibly change. For each of those, retract its old pair contributions from the counts, apply the merge, and post the new ones back. Everything else in the corpus is untouched. Three thousand two hundred and thirty three merges over a one megabyte corpus, in about one second. If this shape feels familiar, it should. It is an inverted index with incremental maintenance, which is what a search engine does to its posting lists.

## s5 · CodeWalkthrough

```props
{"eyebrow": "the encoder", "title": "Encoding is replaying the merges, by rank", "filename": "src/t30_fintok/bpe.py", "code": "def _encode_chunk(self, piece: bytes) -> list[int]:\n    ids = list(piece)\n    while len(ids) >= 2:\n        best_rank, best_at = None, None\n        for i in range(len(ids) - 1):\n            rank = self.ranks.get((ids[i], ids[i + 1]))\n            if rank is not None and (best_rank is None or rank < best_rank):\n                best_rank, best_at = rank, i\n        if best_at is None:\n            break\n        ids[best_at:best_at + 2] = [256 + best_rank]\n    return ids", "highlights": [{"at": 20, "lines": [2], "caption": "Start from raw bytes. Always valid, by construction."}, {"at": 130, "lines": [5, 6, 7, 8], "caption": "Scan for the applicable merge with the LOWEST rank — earliest learned wins."}, {"at": 280, "lines": [9, 10], "caption": "No applicable merge anywhere: this chunk is fully encoded."}, {"at": 380, "lines": [11], "caption": "Merge k mints token 256+k, so the new id is derivable from the rank alone."}]}
```

And here is the encoder. Start from the raw bytes of the chunk, which is always a valid starting state. Then repeatedly scan for the applicable merge with the lowest rank, and apply it. Lowest rank, meaning the one learned earliest during training. When no merge in the table applies anywhere in the chunk, you are done. And note line eleven. Because merge k always mints token two fifty six plus k, the new token id is derivable from the rank alone. You never look anything up. The vocabulary dictionary is only needed for decoding.

## s6 · Callout

```props
{"kind": "gotcha", "heading": "Longest-match-first is the wrong algorithm", "body": "It is the natural instinct, it decodes to the same text, and it is wrong. Rank order is what the model was trained on; any other segmentation puts the model on a distribution it never saw — quietly, with no error.", "code": "# rank order (correct): replays training exactly\n# longest match: different segmentation, same text\n# left-to-right first: different again\n#\n# all three decode(encode(s)) == s.\n# only one matches the model's training distribution."}
```

This is the bug I promised. The natural instinct when encoding is to take the longest match, the way a lexer would. Or to take the first applicable merge scanning left to right, which is faster. Both of those produce valid encodings. Both decode back to the original text perfectly. Both will pass a round trip test suite. And both are wrong, because they produce a different segmentation from the one the training data was tokenised with, which puts the model on a distribution it has never seen. It costs compression, it costs quality, and absolutely nothing raises an error. Rank order, lowest first, always.

## s7 · ConceptScene

```props
{"eyebrow": "an aside", "title": "A prefix of the merge list is a smaller tokenizer", "body": "Because merge k only ever depends on merges before it, truncating the list at any point leaves a complete and valid BPE — one with a smaller vocabulary.", "points": ["truncate(n) = keep the first n merges, rebuild the vocab, renumber the specials.", "It is exact, not approximate: the first n merges are the same merges either way.", "Which gives you the tool for a fair comparison between two tokenizers.", "Two corpora saturate at different merge counts; matching them is the only honest test."], "aside": "Episode 3 uses this to compare a finance vocabulary against a general one at exactly equal size."}
```

One aside that becomes a tool in the next episode. Because merge k only ever depends on merges that came before it, you can truncate the list at any point and what remains is a complete and valid tokenizer with a smaller vocabulary. Not an approximation of one. The same first n merges you would have got if you had asked for n in the first place. That gives you something you need for honest measurement. Two different corpora will saturate at different numbers of merges, so comparing a tokenizer trained on one against a tokenizer trained on the other means comparing different vocabulary sizes, which is not a controlled experiment. Truncate both to the smaller count and it becomes one.

## s8 · RecapScene

```props
{"eyebrow": "episode two", "points": ["Count chunk TYPES with weights, not token occurrences — that alone is about 10x.", "Keep pair counts plus a pair-to-words index, and a merge touches only what changed.", "Merges compose: the list is an ordered dependency chain, not a set.", "Encoding replays the merges by RANK, lowest first. Longest-match-first is a silent, plausible error.", "A prefix of the merge list is itself a valid smaller tokenizer — the tool for matched comparisons."], "ifSkipped": "A tokenizer whose encoder disagrees with its trainer produces text the model was never trained on, and reports no error at all.", "next": "Episode 3: FinTok — the pre-tokenizer as a prior, special tokens, and what a domain vocabulary is worth."}
```

To recap. Count chunk types with weights rather than token occurrences, which is about a factor of ten on its own. Keep a pair count table and a pair to words index, so a merge touches only the words that actually changed. Merges compose, so the list is an ordered dependency chain rather than a set. Encoding replays those merges by rank, lowest first, and longest match first is a silent plausible error that will pass your tests. And a prefix of the merge list is a valid smaller tokenizer, which is the tool for comparing two tokenizers honestly. Next episode. FinTok itself. The pre tokenizer as a hard prior on what tokens can exist, special tokens that have to be unforgeable, and a measured answer to what a domain vocabulary is actually worth.
