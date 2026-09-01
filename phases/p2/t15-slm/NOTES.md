# T15 · AlphaSLM — Notes

What the pipeline is for, what the study actually found, and the four things
that would have quietly invalidated the numbers.

---

## 1. Three decisions that are about leakage and waste, not about text

A pretraining pipeline looks like plumbing. Almost every decision in it is
actually a decision about whether your evaluation means anything.

**The split goes between documents, before packing.** Pack first and slice the
token array at 90% and the boundary lands mid-document: your validation set now
holds the continuation of a filing whose first half is in the training set. The
loss still falls. The chart still looks right. The number has stopped measuring
generalisation, and nothing anywhere will tell you. A test asserts that zero
documents appear in both splits.

**The held-out set is filings only, chosen at packing time.** This topic's claim
is "perplexity on held-out filings". A validation set diluted with price tape
would measure something easier and different. Deciding that at packing time
rather than at reporting time matters, because by the time you are writing the
report the temptation to pick the flattering split is very strong.

**No padding, ever.** Documents are concatenated into one flat array with a
learned `<|endoftext|>` separator, and windows are cut out of it. Our documents
vary twenty-fold in length; padding them to a fixed size would spend about a
quarter of the compute on nothing. And the separator being a *token* rather than
a newline means "this document is over" is something the model can predict, and
therefore something generation can stop on.

## 2. A checkpoint is five things

Not the weights. Everything the loop needs so that stopping and restarting is
invisible:

| key | what breaks without it |
|:--|:--|
| `model` | obvious |
| `optimizer` | Adam's moments restart at zero — the first steps after a "resume" take large, badly-scaled updates into a nearly-converged model |
| `state` | the schedule restarts, so you get warmup at step 6,000 |
| `numpy_rng` | the data sampler restarts from the top of the corpus |
| `torch_rng` | dropout draws differ, so the run is no longer reproducible |

The optimiser state is roughly two thirds of the file, and it is the part people
omit — because a model that loads and generates text *looks* like a successful
resume. The failure is silent.

The test is the strong form: 160 steps straight, versus 80 + checkpoint + 80,
with `torch.manual_seed(999)` deliberately called in between (a resume that only
works when nothing else touched torch is a coincidence, not a resume). The
largest parameter difference is **0.0**. Drop any one of the five keys and it
stops being zero, which is what makes it a test rather than a ritual.

## 3. Gradient accumulation, and the division everybody forgets

Run a batch as *k* micro-batches, sum the gradients, take one optimiser step.
Same update, arbitrarily less memory. There is exactly one thing to get right:

    (loss / micro_batches).backward()

so that what accumulates is the *mean* gradient over the full batch rather than
the sum. Forget it and your effective learning rate is multiplied by *k*,
training gets unstable, and you conclude that accumulation is unstable rather
than that you quietly scaled your learning rate by four.

Measured: batch 16×1 versus 4×4, same seed, same sampler, therefore the same
sixteen windows in the same order. Loss difference 1.2e-07. Largest parameter
difference after 30 steps **6.3e-06** — *not* zero. Summing four partial
gradients is a different order of float32 additions than summing sixteen at
once, and float addition is not associative (T45A spent two episodes on exactly
this). So "my accumulated run does not exactly reproduce my unaccumulated one"
is an expected observation, and a resume test must not change the accumulation
factor half way through.

## 4. The scaling study, and what it actually found

Three rungs, identical corpus, schedule and seed:

| rung | params | val loss |
|:--|--:|--:|
| 0.6M | 557,184 | 0.8129 |
| 1.8M | 1,789,440 | 0.8017 |
| 5.9M | 5,944,224 | 0.7955 |

The ordering holds. And 10.7× the parameters bought **0.017 nats**, which next
to any published scaling chart looks broken.

It is not broken. One extra measurement explains the whole shape — split the
per-token loss by whether the token contains a digit:

| token class | share | loss (nats) | perplexity |
|:--|--:|--:|--:|
| numeric | 19.5% | 3.563 | 35.3 |
| prose | 80.5% | 0.086 | 1.09 |

Our corpus is template-generated. Prices, volumes, percentages and dates are
drawn from continuous ranges, so their digits are close to uniform over a large
alphabet — 3.56 nats is very nearly "no information available". And the prose is
close to *deterministic*, at 0.086 nats, because there are only so many
templates. The smallest rung already reaches that floor: prose loss is 0.085 at
0.6M and 0.086 at 5.9M.

So: about 90% of the remaining loss budget is dice rolls no model can reduce,
and the rest was solved by the smallest model. **The margin is small by
construction.** This is a fact about the data, not about transformers, and it is
the most useful thing the study produced, because it says what to do next: build
a harder corpus. Which is precisely what T23 (synthetic data) and T38 (curation)
are for — two topics that read as chores until you have this measurement in
front of you, and then read as the bottleneck.

The Chinchilla arithmetic points the same way. The rule of thumb is ~20 tokens
per parameter; our corpus is 2.4M tokens, so at 1,200 steps each rung sees
22%, 6.9% and 2.1% of its compute-optimal budget. Every model in the study is
deep in the data-limited regime — which is where almost no published scaling
chart is drawn from, and where almost every hobby project lives.

### The power-law fit, done so it can be wrong

`L(N) = a·N^-b` fitted on the **two end rungs only**, then checked on the middle
one, which did not draw the line:

    predicted 0.8043   actual 0.8017   +0.33%

A curve fitted through all three points would pass through all three points and
tell you nothing. And the extrapolation to 40M — 0.7819 — is reported with the
warning it deserves: a two-point fit stretched sevenfold, on a corpus none of
those models could saturate. That is exactly the kind of number people commit a
month of GPU time to.

## 5. Four ways to be wrong about a perplexity number

1. **One number for a mixed corpus.** Perplexity is dominated by whichever
   register is most predictable. Break it out.
2. **Random windows.** Overlapping draws re-count some tokens and skip others.
   Measured on one model, five random draws gave 1.0395, 1.0724, 0.9903, 1.0658,
   1.0344 — a spread of **0.082 nats, five times the entire margin between our
   smallest and largest model.** Sequential non-overlapping windows gave
   1.0649 five times out of five. Random is right for training and wrong for a
   table.
3. **Comparing across tokenizers.** Perplexity is per *token*, so a 50k
   vocabulary beats a 3.5k vocabulary on identical text for free. `bits_per_char`
   is the comparable unit and it is one extra line.
4. **Stopping at the number.** Ours writes a clean `<|announcement|>`
   continuation and also this, prompted with `<|filing|>`:

       O 2024-05-26.2 percent an average of Rs 611.2 crore, of lareducational only.

   Fluent-looking and meaningless. Ten seconds of reading found it; no aggregate
   would have.

## 6. The 4070 plan, and the number that decides it

Four copies of every parameter exist before a single activation does — the
weight, its gradient, and Adam's two moments. For the 40M rung that is 633 MB.
Activations at batch 24 × context 512 add 5.4 GB, dominated by the attention
matrix, which is quadratic in context and is the exact term T7's Flash Attention
removes. Total with headroom: 7.9 GB, so it fits in 12 GB.

And then the number the runner prints with a warning: **207 epochs**. 491M
training tokens against a 2.4M-token corpus. It will memorise. Run it anyway —
the divergence between a falling training loss and a flattening held-out
perplexity, watched once on your own model, is worth more than reading about
overfitting ten times, and the eval curve in `gpu_results.json` records the step
where it turned.

## 7. Gotchas

1. **The corpus contains visible near-duplicate spans.** T30's generator picks
   risk clauses with replacement, so a document can contain the same clause
   twice in a row. We left it: it is exactly the kind of thing T38's MinHash
   deduplication exists to find, and having a corpus with a known defect to run
   it against later is worth more than a clean one now.
2. **`np.memmap` slices must be copied before becoming tensors.** Torch cannot
   own memory it did not allocate, and a tensor pointing into a memmap that gets
   garbage collected is a segfault waiting for a quiet afternoon.
3. **uint16 is checked, not assumed.** Hand `encode_documents` a tokenizer with a
   70,000-token vocabulary and it raises. Without that check, ids would wrap
   silently and the corpus would be quietly corrupt.
4. **The parameter counts in `config.py` are a closed-form formula**, and a test
   asserts each one against an actually-built model. Otherwise the 40M rung's
   VRAM "plan" is a guess printed in a nice table.

## 8. What this hands to the rest of the course

AlphaSLM is the model every later topic *modifies* rather than replaces:
LoRA-tuned in T17, DPO-aligned in T19, distilled in T47, quantized in T8, served
by the Rust `tickerd` in T3, and evaluated by the harness in T27. The packed
shards are what T38 and T23 improve. The training harness is the one those
topics reuse.

And the finding to carry: on this corpus, compute stopped being the constraint
some time ago. The next useful thing to build is better data.
