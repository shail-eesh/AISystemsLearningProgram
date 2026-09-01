# T15 · Small Language Model — AlphaSLM

**Phase:** Transformers & Small Models (P2) · **Generation day:** Day 3 · **Video episodes:** 4

> [← Back to course home](../../../index.html) · [Master plan](../../../MASTER_PLAN.md) · [Progress ledger](../../../EXECUTION/LEDGER.md)

## What you build

**AlphaSLM** — the desk's own pretrained language model. T4's architecture, T30's
tokenizer, and everything that has to exist around them before you can call it a
model you own:

| piece | what is in the repo |
|:--|:--|
| corpus | 12 MB of deterministically generated financial text plus every committed sample, tagged by register (`<\|filing\|>`, `<\|commentary\|>`, `<\|announcement\|>`, `<\|order\|>`) |
| shards | corpus → FinTok → one flat `uint16` array per split, memory-mapped, checksummed; **split by document, before packing** |
| harness | warmup + cosine, gradient clipping, gradient accumulation, checkpoints that resume *exactly*, JSONL metrics, a wall-clock budget |
| ladder | five rungs from 0.6M to 40M parameters, with closed-form parameter counts asserted against built models |
| study | three CPU rungs trained under a matched schedule, with a power-law fit checked on a held-out rung |
| evaluation | per-register perplexity, bits per character, and a token-class decomposition that explains the result |
| GPU lane | `gpu-runner/t15_alphaslm_40m.py` — the 15M and 40M rungs, planned before they are started |

*Ref: nanoGPT's data and training pipeline; Hoffmann et al. 2022 (Chinchilla) for
the token budget; Loshchilov & Hutter 2017 for AdamW and the cosine schedule.*

## Results

All six checks in `bench/verify.py` pass on 2 CPU cores in **20 minutes**
(`bench/results.json`).

**1 · The data pipeline is provable.** Rebuilding the corpus on another machine
reproduces both splits byte for byte (`sha256` recorded in the committed
`meta.json`), **0** documents appear in both splits, and the held-out set
contains filings only — by construction, not by filtering at report time.
2,371,748 training tokens, 116,650 held out, **4.82 characters per token**
(FinTok was trained on this domain; that density is what T30 bought).

**2 · A resume is invisible.** 160 steps straight through, versus 80 + a
checkpoint + 80 with torch's global RNG deliberately poisoned in between:

| | |
|:--|--:|
| largest parameter difference | **0.0** |

Exactly zero, because the checkpoint carries the optimiser moments, the data
sampler's RNG state and the schedule position, not just the weights.

**3 · Gradient accumulation is the same update.** Batch 16 × 1 versus 4 × 4:

| | |
|:--|--:|
| loss difference | 1.2e-07 |
| largest parameter difference after 30 steps | **6.3e-06** |

Not bit-identical, and that is the interesting part: summing four partial
gradients reassociates float32 additions (T45A, at length). A few parts per
million is reassociation noise, not a bug.

**4 · Bigger wins — and the margin is small for a reason we can name.**
Three rungs, identical corpus, schedule and seed, 1,200 steps each:

| rung | params | val loss | perplexity | train−val gap | tokens/param |
|:--|--:|--:|--:|--:|--:|
| alphaslm-0.6m | 557,184 | 0.8129 | 2.25 | +0.0040 | 4.41 |
| alphaslm-1.8m | 1,789,440 | 0.8017 | 2.23 | +0.0039 | 1.37 |
| **alphaslm-5m** | 5,944,224 | **0.7955** | **2.22** | +0.0068 | 0.41 |

The ordering holds at every rung. A power law fitted on the two end rungs
(`L(N) = 0.9168 · N^-0.00910`) predicts **0.8043** for the middle rung against an
actual **0.8017** — a 0.33% error on a point that did not draw the line.

**5 · Held-out filings, scored document by document.**

| rung | loss | perplexity | bits/char |
|:--|--:|--:|--:|
| alphaslm-0.6m | 0.8323 | 2.2986 | 0.2487 |
| alphaslm-1.8m | 0.8269 | 2.2862 | 0.2471 |
| **alphaslm-5m** | **0.8183** | **2.2667** | **0.2445** |

**1.014×** from smallest to largest. And here is why it is not more — the finding
this topic actually produced:

| token class | share of corpus | loss (nats) |
|:--|--:|--:|
| numeric | 19.5% | **3.563** |
| prose | 80.5% | **0.086** |

The corpus is template-generated. Its numbers are drawn from continuous ranges,
so they are close to uniform noise that no model of any size can predict; its
prose is close to deterministic, and the **smallest** rung already reaches its
floor. Roughly 90% of the remaining loss is dice rolls. The measurable margin
between rungs is small *by construction*, and the fix is a harder corpus — T23
(synthetic data) and T38 (curation) — not a bigger model. That is the result,
not a failed check.

**6 · It writes in the register it is prompted with.** 83% of the
register-and-issuer checks hit. Prompted with `<|announcement|>`:

```
to consider unaudited financial results. Symbol: ALPHAINFRA. ISIN: INE001A01011.
Series: EQ. The company has informed the exchange that the board will meet on
2024-01-24 to consider and approve the unaudited financial results...
```

And prompted with `<|filing|>`, the honest counter-example, kept because it is
what a small model at this scale actually does:

```
O 2024-05-26.2 percent an average of Rs 611.2 crore, of lareducational only.
```

Fluent-looking and meaningless. No aggregate metric shows you that.

## The 4070 lane

The 15M and 40M rungs need a CUDA device, so their bench row is
🖥️ **awaiting-4070**. Everything else here — the pipeline, the harness, the
resume guarantee, the CPU ladder — is verified in the cloud.

```bash
python3 gpu-runner/t15_alphaslm_40m.py --dry-run              # plan only
python3 gpu-runner/t15_alphaslm_40m.py --rung alphaslm-15m --hours 1
python3 gpu-runner/t15_alphaslm_40m.py --rung alphaslm-40m --hours 8
python3 gpu-runner/t15_alphaslm_40m.py --rung alphaslm-40m --resume
```

The plan it prints is the point. For the 40M rung at batch 24 × 2 micro-batches:
633 MB of weights, gradients and Adam moments; 5.4 GB of activations (dominated
by the attention matrix, which is quadratic in context — the term T7 deletes);
7.9 GB with headroom, so it fits in 12 GB. And **207 epochs** over a 2.4M-token
corpus, which the runner prints with a warning, because that is the number that
decides whether the night is worth the electricity. Run it anyway: watching
held-out perplexity flatten while training loss keeps falling, once, on your own
model, is worth more than reading about overfitting ten times.

## How to run

```bash
pip install -e '.[torch,dev]'

python3 steps/step1_corpus_to_shards.py       # corpus -> shards           (~10 s)
python3 steps/step2_harness_and_resume.py     # the resume test            (~3 min)
python3 steps/step3_scaling_ministudy.py      # three rungs, one schedule  (~22 min)
python3 steps/step4_the_4070_run.py           # plan the overnight run     (~15 s)
python3 steps/step5_evaluate.py               # four ways to be wrong      (~9 min)

pytest phases/p2/t15-slm -q                   # 68 tests
python3 bench/verify.py --quick               # smoke test (~5 min)
python3 bench/verify.py                       # the full run (~20 min)
```

`*.bin` is gitignored course-wide. `meta.json` is committed, and a rebuild is
checked against its checksums — a better guarantee than shipping five megabytes
of packed tokens.

## AlphaDesk hook

Three components:

- `models.alphaslm` — the desk's local model. Takes a rung name, loads a
  checkpoint if one exists, and returns the untrained model if not, because a
  desk that boots without weights is a desk you can still demo.
- `data.pretrain_shards` — the packed corpus, so T38 (curation) and T23
  (synthetic data) have something concrete to improve.
- `models.alphaslm_eval` — per-register perplexity and the model comparison.

AlphaSLM is what T17 LoRA-tunes, T19 aligns with DPO, T47 distils, T8 quantizes
and T3 serves. AlphaDesk is a fictional educational simulation: generated
commentary is a language-model artefact about invented issuers — never a market
view, never advice, and never routed to anything that could act on it.

## Layout

- `src/t15_alphaslm/` — `corpus`, `shards`, `config` (the ladder), `harness`,
  `scaling`, `evaluate`, `alphadesk_hook`
- `steps/` — the five-step ladder, each runnable on its own
- `tests/` — 68 pytest tests (skipped wholesale if `torch` is not installed)
- `bench/` — `verify.py`, `results.json`, `scaling_study.json`
- `NOTES.md` — the derivations, the entropy-floor finding, and the gotchas

## Videos

Episode scripts live in [`video/topics/t15-e1`](../../../video/topics/t15-e1/)
through `t15-e4`. Rendered `.mp4`s are delivered in chat (not committed).
