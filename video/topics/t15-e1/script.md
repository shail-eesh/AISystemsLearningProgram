---
topic: t15-e1
episode: 1
title: From corpus to shards
voice: am_michael
speed: 0.85
runtime_target_minutes: 8
paper: nanoGPT data pipeline; Chinchilla (Hoffmann et al. 2022) for the token budget
---

## s1 · TitleCard

```props
{"title": "From corpus to shards", "subtitle": "Four decisions, three of which are about leakage and waste rather than about text.", "topicId": "T15", "episode": "Episode 1"}
```

Welcome back to the AI Systems Forge. Phase two, topic fifteen. AlphaSLM — the desk's own small language model. You have the architecture from T4 and the tokenizer from T30. This topic is everything around them: the data pipeline, the training harness, and how to tell whether the result is any good. Today, the data pipeline, which is four decisions, and three of them are about leakage and waste rather than about text.

## s2 · ArchitectureMap

```props
{"eyebrow": "where we are", "highlight": ["models", "data"], "caption": "AlphaSLM is the desk's local model — later LoRA-tuned, DPO-aligned, distilled, quantized and served."}
```

Here is where we are. Two blocks lit this time: data and models. AlphaSLM is the model the rest of this course modifies rather than replaces. In phase five it gets LoRA-tuned and DPO-aligned. In phase six it gets quantized and served by a Rust inference server you will also write. Everything downstream assumes there is a local model with weights you own. This topic is where those weights come from.

## s3 · ConceptScene

```props
{"eyebrow": "decision 1", "title": "What is a document?", "body": "The corpus is 12 MB of deterministically generated financial text plus everything already committed in the repository's sample files. Every document carries a FinTok special token naming its register.", "points": ["<|filing|> — section-structured filing prose with numbers", "<|commentary|> — daily market commentary", "<|announcement|> — exchange announcements", "<|order|> — order tickets from the committed sample, every one labelled as simulated", "plus the price tape, grouped by symbol rather than by date"], "aside": "Register as a token means it is something the model can condition on. Prompt with <|filing|> and you get filing prose, not a price line."}
```

Decision one: what is a document? Our corpus is about twelve megabytes, generated deterministically from templates with fictional issuers, plus everything already committed in the repository's sample files. And every document is prefixed with a FinTok special token naming its register. Filing. Commentary. Announcement. Order. Plus the price tape, which is grouped by symbol rather than by date, so a single document contains one symbol's whole year and the model has some chance of seeing a series rather than a shuffle. Making register a *token* rather than a formatting convention means it is something the model can condition on. Prompt with the filing token and you get filing prose; prompt with the commentary token and you get a market comment. That is a free capability and it costs five entries in the vocabulary.

## s4 · Callout

```props
{"kind": "warning", "heading": "Decision 2: the split goes between documents, before packing", "body": "Split a packed token array instead and the boundary lands mid-document, so your validation set contains the continuation of something the model trained on. The number still goes down. It just stops meaning anything.", "code": "train_docs, val_docs = split_documents(build_corpus())   # whole documents\nmeta = pack_documents(train_docs, val_docs)              # then pack\n# documents in both splits: 0"}
```

Decision two, and this is the one that quietly ruins evaluations. The split goes between *documents*, before anything is packed. If you pack first and then slice the token array at ninety percent, the boundary lands in the middle of some document, and your validation set now contains the second half of a filing whose first half is in your training set. The model has seen the style, the issuer, the numbers, and half the sentences. Validation loss still goes down over training, the chart still looks right, and the number has stopped measuring generalisation. Our split is by document and there is a test asserting that zero documents appear in both. And we go one step further: the held-out set is *filings only*, because the claim this topic makes is about perplexity on held-out filings, and a validation set diluted with near-deterministic price tape would measure something easier and different.

## s5 · CodeWalkthrough

```props
{"eyebrow": "decision 3", "title": "How it is stored: one flat uint16 array per split", "filename": "phases/p2/t15-slm/src/t15_alphaslm/shards.py", "code": "def encode_documents(documents, tokenizer, *, separator='<|endoftext|>'):\n    sep = tokenizer.special_tokens[separator]\n    if tokenizer.vocab_size > np.iinfo(DTYPE).max + 1:\n        raise ValueError('vocabulary does not fit; widen the dtype')\n    out = []\n    for doc in documents:\n        out.extend(tokenizer.encode(doc))\n        out.append(sep)\n    return np.asarray(out, dtype=np.uint16)", "highlights": [{"at": 20, "lines": [2], "caption": "The separator is a learned token, not a newline — so 'this document is over' becomes predictable."}, {"at": 130, "lines": [3, 4], "caption": "uint16 because FinTok's vocabulary is 3,495. Checked, not assumed: a wider vocab raises instead of wrapping."}, {"at": 250, "lines": [6, 7, 8], "caption": "One flat array. No padding, no per-example records — documents just run into each other."}]}
```

Decision three: how it is stored. One flat unsigned sixteen-bit array per split, with a separator token between documents. Three things in that sentence. The separator is a learned token rather than a newline, so "this document is over" becomes something the model can predict, and therefore something generation can stop on. Sixteen bits because FinTok's vocabulary is three thousand four hundred and ninety five — a thirty-two-bit array would double the file to store the same numbers — and it is *checked* rather than assumed: hand this function a wider tokenizer and it raises instead of silently wrapping ids around. And it is one flat array with no padding anywhere. Documents run into each other through the separator. Our corpus has documents that vary twenty-fold in length; padding them all to a fixed size would throw away roughly a quarter of the compute on nothing.

## s6 · ConceptScene

```props
{"eyebrow": "why a memmap", "title": "There are no examples. There is one long array and a window length.", "body": "A pretraining corpus is not a list of examples, and modelling it as one adds a DataLoader, workers, collation and a shuffling buffer to a problem whose answer is a random integer.", "points": ["2,371,748 tokens on disk -> 2,371,619 distinct 128-token windows.", "x is data[i : i+T]; y is data[i+1 : i+T+1]. The targets are the inputs shifted by one.", "Every position in every window is a training example — that is why transformers are sample-efficient per step.", "np.memmap reads only the windows the batch touches, so the corpus does not have to fit in RAM."], "aside": "The copy out of the memmap into a tensor is not optional: torch cannot own memory it did not allocate."}
```

And then reading it back, where there is a design point worth stating plainly. A pretraining corpus is not a list of examples. It is one long array and a window length. Modelling it as a list of examples means a DataLoader, worker processes, a collate function and a shuffle buffer, all to solve a problem whose actual answer is "draw a random integer". Two point three seven million tokens gives two point three seven million distinct windows of one hundred and twenty eight. The inputs are a slice; the targets are the same slice shifted one token to the left. And note that every position inside the window is a supervised example, not just the last one — that is why a transformer is so much more sample-efficient per step than a recurrent model. The memmap means only the windows a batch actually touches are read from disk, which is the trick that lets a laptop train on a corpus larger than its memory.

## s7 · ChartScene

```props
{"eyebrow": "decision 4", "title": "What gets recorded, so a rebuild can be proved identical", "kind": "bar", "bars": [{"label": "train\ndocuments", "value": 28886}, {"label": "val\ndocuments", "value": 1374, "color": "#5ac48c"}, {"label": "train tokens\n(thousands)", "value": 2372}, {"label": "val tokens\n(thousands)", "value": 117, "color": "#5ac48c"}], "yLabel": "count", "caption": "Plus a sha256 per split. meta.json is committed; the .bin files are not — they rebuild in two seconds."}
```

Decision four: what gets recorded. Token counts, document counts, the tokenizer's identity, the separator id, and a checksum per split. Twenty eight thousand eight hundred and eighty six training documents, one thousand three hundred and seventy four held-out. Two point three seven million training tokens, one hundred and seventeen thousand held out. And here is the thing the checksum buys. The binary files are gitignored, along with every other weight and blob in this course. A fresh clone has the metadata and no arrays. It rebuilds them in two seconds, and then it can *prove* it got the same corpus by comparing checksums, rather than assuming. That is a better guarantee than shipping the bytes, and it is four megabytes lighter.

## s8 · Callout

```props
{"kind": "insight", "heading": "4.82 characters per token — which is why T30 came first", "body": "FinTok was trained on this domain, so it packs it densely. A general-purpose vocabulary of the same size splits 'consolidated', 'EBITDA' and 'INE005A01049' into many more pieces, and every extra piece is context you paid for and compute you spent.", "code": "12.0 MB of text  ->  2,488,398 FinTok tokens  ->  4.98 MB of uint16"}
```

One number to end on. Four point eight two characters per token. FinTok was trained on exactly this domain in topic thirty, which is why it packs it this densely — a general vocabulary of the same size splits words like consolidated and EBITDA and ISIN codes into many more pieces, and every extra piece is context window you paid for and compute you spent. This is the concrete reason the tokenizer topic came before the language model topic rather than after it, and in a later episode we will see that same choice show up as a straight discount on training cost.

## s9 · RecapScene

```props
{"eyebrow": "recap", "title": "The pipeline, in three sentences", "points": ["Documents carry their register as a token, so register becomes something the model can condition on.", "The split is between whole documents and happens before packing — anything else leaks, and the leak is invisible in the chart.", "The packed format is one flat uint16 array with a learned separator: no padding, memory-mappable, and checksummed so a rebuild is provable."], "ifSkipped": "Skip this and every number in the next three episodes is measured against a validation set that may contain the training set.", "next": "Episode 2 — the harness: what a checkpoint has to contain for a resume to be invisible."}
```

Three things. One. Documents carry their register as a token, which makes register something the model can be prompted with. Two. The split is between whole documents and it happens before packing, because anything else leaks, and the leak is completely invisible in your loss chart. Three. The packed format is one flat sixteen-bit array with a learned separator: no padding, memory-mappable, and checksummed so that a rebuild on another machine is provably the same corpus. Next time, the training harness — and specifically, what a checkpoint has to contain for stopping and restarting to be genuinely invisible.
