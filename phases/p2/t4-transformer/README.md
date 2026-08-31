# T4 · Transformer from scratch (Attention Is All You Need)

**Phase:** Transformers & Small Models (P2) · **Generation day:** Day 3 · **Video episodes:** 5

> [← Back to course home](../../../index.html) · [Master plan](../../../MASTER_PLAN.md) · [Progress ledger](../../../EXECUTION/LEDGER.md)

## What you build

A decoder-only GPT in PyTorch from an empty file — embeddings, causal multi-head
self-attention, MLP, pre-norm residual blocks, weight tying, and KV-cached
generation — with the pieces the paper leaves implicit made explicit and
measurable:

| piece | what is in the repo |
|:--|:--|
| attention | a single-head function you can read, a looped multi-head **oracle**, and the batched module the model uses — the last two agree to `0.0` |
| positions | learned, sinusoidal, RoPE, and none, all four selectable by config and raced against each other |
| norms | LayerNorm (with optional bias) and RMSNorm, plus a pre-norm/post-norm switch for the ablation |
| sampling | greedy, temperature, top-k, top-p — pure functions of logits, tested as such |
| KV cache | a naive contiguous cache, proved token-identical to the uncached path and timed |
| interpretability | attention maps, and an induction-head probe with two controls |

*Ref: Vaswani et al. 2017 (Attention Is All You Need); Radford et al. 2019 (GPT-2);
Su et al. 2021 (RoFormer / RoPE); Olsson et al. 2022 (In-context Learning and
Induction Heads).*

## Results

All six checks in `bench/verify.py` pass on 2 CPU cores in **7m18s**
(`bench/results.json`). Corpus: a 92,743-character market tape, 66-symbol
vocabulary, built from the committed samples.

**1 · Does it compute the same thing as a reference implementation?**
`bench/reference.py` is an independently written GPT built on
`F.scaled_dot_product_attention`, three separate q/k/v `Linear`s and
`torch.nn.LayerNorm` — it shares no code with `src/`. Loading our weights into it:

| | |
|:--|--:|
| max relative logit difference | **3.8e-07** |
| loss difference | **0.0** |

**2 · Does it *learn* the same?** Same architecture, same init recipe, same batch
stream, three seeds each:

| | |
|:--|--:|
| max gap between the mean curves (after warm-up) | **0.0400** |
| seed-to-seed noise band | 0.2304 |
| gap ÷ noise | **0.17×** |

The two implementations are closer to each other than one implementation is to
itself under a different seed, which is what "matches within noise" has to mean
to be worth saying.

**3 · Is the generated text coherent?** A 4-layer RoPE model, 800 steps:

| | |
|:--|--:|
| validation loss | **0.9047** (1.305 bits/char) |
| generated lines matching the tape grammar exactly | **100%** |
| distinct lines | 100% |
| lines where H ≥ max(O,C) and L ≤ min(O,C) | **36%** |

That last row is the honest one. The model learned the *language* of the tape —
field order, symbol names, two decimal places, plausible magnitudes per symbol —
and did not learn what high and low *mean*. Grammar is cheap; semantics is not.

```
2024-09-24 BHARATCHEM O 259.82 H 267.26 L 264.15 C 263.60 V 339348
2024-02-08 EASTPOWER O 626.13 H 585.41 L 594.52 C 593.03 V 429301
2024-01-17 DECCANMOT O 471.68 H 485.09 L 477.98 C 471.21 V 364326
```

**4 · Are there induction heads?** The probe uses sequences that repeat with a
*random per-row period* (10–14), so no fixed-offset rule can solve them:

| | |
|:--|--:|
| best head's attention on the induction target | **0.347** (9.1× chance) |
| 2-layer model's repeat-prediction accuracy | **94.0%** |
| 1-layer control | 64.9% |
| depth gap | **+29.1 points** |

Both controls matter and one of them is a warning: the *one-layer* model's best
head scores **0.438** — higher than the two-layer model's — while predicting far
worse. An attention map that looks right is not a circuit that works. See
`NOTES.md`.

**5 · Positions, raced.** Identical seed, data and steps:

| scheme | val loss | extra params |
|:--|--:|--:|
| RoPE | **1.0224** | 0 |
| learned | 1.1065 | 8,192 |
| none | 1.1156 | 0 |
| sinusoidal | 1.1178 | 0 |

RoPE wins on loss *and* is the only scheme that still answers past the training
length (step 5 measures 0.90 at 128 tokens for a model trained at 64; the
sinusoidal model degrades to 1.72; the learned table cannot be asked).

**6 · KV cache.** 180 greedy tokens, identical output, **3.19× faster**
(0.84 s → 0.26 s), 1 MB of cache for a 256-slot context.

## How to run

```bash
pip install -e '.[torch,dev]'          # torch is an optional extra

python3 steps/step1_one_head_attention.py     # one head, printed as numbers
python3 steps/step2_multihead_causal.py       # the reshape, vs a looped oracle
python3 steps/step3_block_and_first_train.py  # the block; first tape training  (~3 min)
python3 steps/step4_rope_from_scratch.py      # RoPE derived from its wish
python3 steps/step5_train_the_gpt.py          # the four-way position bake-off (~6 min)
python3 steps/step6_sampling.py               # greedy/temperature/top-k/top-p  (~1 min)
python3 steps/step7_kv_cache.py               # exact, faster, and its wall     (~2 min)

pytest phases/p2/t4-transformer -q            # 121 tests
python3 bench/verify.py --quick               # smoke test (~45 s)
python3 bench/verify.py                       # the full run (~7 min)
```

## AlphaDesk hook

Three components on the `models` surface:

- `models.gpt_architecture` — the factory T15 calls to build AlphaSLM, T9 calls
  to swap in an MoE FFN, and T17 calls to attach LoRA adapters.
- `models.kv_cache` — the naive cache, registered so T12's paged replacement has
  something measured to beat rather than merely described.
- `models.attention_inspector` — attention maps and the induction probe, read by
  the eval dashboard (T27) and the interpretability topic (T22).

AlphaDesk is a fictional educational simulation — no real orders, money, venues,
or redistributed market data anywhere in this repository.

## Layout

- `src/t4_transformer/` — `config`, `positions`, `attention`, `blocks`, `model`,
  `sampling`, `data`, `train`, `interpret`, `alphadesk_hook`
- `steps/` — the seven-step ladder, each runnable on its own
- `tests/` — 121 pytest tests (skipped wholesale if `torch` is not installed)
- `bench/` — `reference.py` (the independent model), `verify.py`, `results.json`
- `NOTES.md` — the intuition, the derivations, and the four bugs this build found

## Videos

Episode scripts live in [`video/topics/t4/`](../../../video/topics/t4/).
Rendered `.mp4`s are delivered in chat (not committed).
