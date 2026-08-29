# AI Systems Forge — Master Plan

**A build-it-yourself mastery program: 51 "Build Your Own X" topics · Remotion learning videos narrated by Kokoro · step-laddered practice code · one capital-markets capstone (AlphaDesk)**

*Plan authored with Claude Fable 5 for review · Execution in cloud Cowork sessions on Claude Opus · v1.0 · 29 Aug 2026*

---

## 1. What this program is

You learn each AI system by building it from primary sources (the original papers), at small scale, with tests that prove it works — then you wire it into one continuously growing capital-markets platform, **AlphaDesk**, so every component earns its place in a realistic system you can demo. For every topic, the Cowork session also produces **slow-paced, detailed Remotion video lessons** (narrated with Kokoro-82M TTS) that teach the concept and walk the code line by line. By the end you have three assets:

1. **Deep skill** — you have personally implemented the full modern AI stack: autograd → transformer → training/alignment → inference serving → GPU kernels → retrieval → agents → multimodal.
2. **A portfolio capstone** — AlphaDesk, a fictional AI-native trading desk (research copilot + OMS/EMS-style paper-order workflow + compliance guardrails) that maps directly onto your LPL/Fidelity background and your Director/VP–Technology job search.
3. **A video library** — 120–160 rendered lesson episodes you can review for spaced repetition, and optionally publish (your existing YouTube upload pipeline can be reused).

**The learning loop for every topic** (this is the method; every work package follows it):

> **Explain** (video E1: concept, paper, intuition) → **Derive** (math shown three ways: words, symbols, code) → **Build** (step ladder: 5–8 runnable steps, each with tests) → **Verify** (compare against a reference implementation on accuracy/speed) → **Integrate** (wire into AlphaDesk) → **Teach back** (final recap video — if you can't follow it at 1× speed without pausing, the episode failed its job).

### Assumptions this plan is built on (from your answers)

| Question | Your answer | What it changes |
|---|---|---|
| Python/PyTorch depth | Mostly .NET; Python casual | Phase 0 is a dedicated 2-week Python-for-ML ramp, written for a senior .NET engineer (C# ↔ Python idiom tables, not beginner material) |
| Pace | 20+ h/week (sabbatical) | Full program ≈ 26 weeks (~6 months); accelerated "core path" option ≈ 16 weeks |
| GPU topics | RTX 4070 local lane | CUDA/Triton code is written + CPU-verified in cloud sessions; you run real benchmarks on your 4070 via a provided runner script; Colab notebooks included as backup |
| Capstone | Full AI trading desk | AlphaDesk = research copilot + order workflow + compliance, paper-trading only |

### Ground rules

- **From scratch means from scratch.** NumPy/PyTorch tensor ops are allowed as the "assembly language"; the thing being learned (attention, HNSW, PPO, paging…) is never imported. Reference libraries (tiktoken, hnswlib, vLLM, TRL) appear only as *benchmarks to beat or match*.
- **Small scale, real behavior.** Every model trains in minutes-to-hours on CPU or your 4070, never days. A 4-layer transformer that you built teaches more than a 70B you downloaded.
- **Tests define done.** Each step ships with pytest/cargo tests; a topic is complete only when the final step passes its verification benchmark.
- **Everything is fictional and paper-only.** AlphaDesk never touches real orders or real money; it is an educational simulation with explicit disclaimers. No real brokerage branding.

---

## 2. Environment & feasibility (read this first — it shapes everything)

**Cloud Cowork sandbox (where Opus works):** Linux container, CPU-only, no GPU, allowlisted network (PyPI/npm/GitHub OK), Chromium preinstalled (Remotion renders work), ephemeral filesystem per session. Consequences:

1. **State lives in a PUBLIC GitHub repo; the proxy reads but can't write it.** This environment's git proxy can *clone* a public repo but refuses to push to it (embedded tokens are ignored). So each session `git clone`s to get the latest state, builds on top, commits locally, and delivers the day's work as a **git bundle** that you push. Videos (large mp4s) are gitignored and delivered in-chat, regenerable from committed scripts + audio manifests. One-time setup: make `shail-eesh/AISystemsLearningProgram` public and push the scaffold to `main`.
2. **Kokoro-82M runs fine here.** 82M params, CPU real-time+ via ONNX (`kokoro-onnx` or the `kokoro` pip package + espeak-ng). This matches your kids-pipeline experience — same stack, reused.
3. **Remotion renders fine here, at 720p.** Budget ~1–2× realtime per render on container CPUs; a topic's ~35–50 min of episodes renders within a session. 1280×720/30fps keeps files at ~5–8 MB/min for in-chat delivery.
4. **GPU truth comes from your 4070.** For kernel topics (matmul, softmax, Flash Attention), quantization, distributed training, and any "how fast is it really" claim, the cloud session produces: (a) the CUDA/Triton source, (b) a NumPy/PyTorch-CPU reference that proves correctness in-cloud (Triton's `TRITON_INTERPRET=1` mode runs kernels on CPU), and (c) `gpu-runner/` — a script you run on the 4070 (WSL2 or native Linux recommended for Triton/CUDA toolchain) that executes benchmarks and writes `results.json`, which you paste/commit back so the video can show real numbers.
5. **12 GB VRAM budget (4070).** Everything is sized for it: SLM ≤ 60M params trains comfortably; LoRA on a 1–3B open model fits with bf16 + gradient checkpointing; Int8/FP4 quantization experiments use 1–3B models. Nothing in this program needs more.
6. **Distributed training on one box.** FSDP/tensor-parallel concepts are learned with `torch.distributed` + gloo backend running 2–4 CPU processes in-cloud (correctness), plus a single-GPU + CPU-offload variant on the 4070. You learn the sharding math and collective ops for real; you just don't rent a cluster. The capsule notes what changes at 8×H100 scale.

**Data sources (all free/legit for personal education):** NSE bhavcopy & corporate announcements, yfinance for OHLCV, SEC EDGAR full-text filings (10-K/10-Q, fair-use excerpts), earnings-call transcripts from EDGAR 8-K exhibits + open datasets, FiQA/financial-phrasebank open datasets for sentiment, synthetic order/position data generated in-repo. The plan avoids anything requiring paid market-data licenses; NSE data is used per its personal-use terms.

**Licensing notes:** Remotion is free for individuals and companies of up to 3 people (company license needed beyond that — irrelevant for personal learning; revisit if this becomes company work). Kokoro-82M weights are Apache-2.0. All "build your own" code is yours; papers are cited in each video.

---

## 3. Curriculum map — 9 phases + capstone thread

Topics are ordered by dependency, and the application layer (retrieval, agents) comes *early* — right after you can train a small transformer — because those are the highest-leverage skills for your job search; the heavy training/inference/kernels depth follows once the payoff is visible.

```
Phase 0  Python & Tensor Ramp        (wk 1–2)    .NET → Python/NumPy/PyTorch
Phase 1  Foundations                 (wk 3–5)    autograd · matmul(CPU) · softmax(CPU) · tokenizer
Phase 2  Transformers & Small Models (wk 6–9)    transformer · SLM · embeddings · Mamba · MoE
Phase 3  Retrieval & Knowledge       (wk 10–12)  HNSW · vector driver · RAG · semantic router ·
                                                 KG builder · GraphRAG · text-to-SQL · feature store · two-tower
Phase 4  Agents & Safety             (wk 13–15)  CoT reasoner · ReAct · function router · sandbox ·
                                                 guardrails · eval harness · adversarial · SAE
Phase 5  Training & Alignment        (wk 16–19)  data curation · synthetic data · LoRA · PEFT ·
                                                 distillation · DPO · RLHF(PPO) · merging · FSDP/TP · NAS
Phase 6  Inference Systems           (wk 20–23)  Rust server · KV paging · logit processors · CFG outputs ·
                                                 speculative decoding · prompt caching · quantization · AI gateway
Phase 7  GPU Kernels (4070 lane)     (wk 24–25)  matmul(GPU) · softmax(GPU) · Flash Attention
Phase 8  Multimodal                  (wk 26–28)  ViT · CLIP projector · diffusion · audio spectrogram ·
                                                 Whisper-style ASR · TTS
Phase 9  Capstone Assembly           (wk 29–30)  AlphaDesk polish, end-to-end demo, portfolio video
```

Every phase ends with an **AlphaDesk milestone** (Section 6) — a working, demoable increment. The capstone is not a final project; it grows continuously.

**Topic count check:** 51 user-listed topics, all mapped exactly once (matmul and softmax kernels each have a CPU part in Phase 1 and a GPU part in Phase 7 — one topic, two passes). The full mapping table is in Section 6.4.

**Accelerated core path (optional, ~16 weeks):** if you ever need to compress, the core is Phases 0–4 + 6 with Phase 5 reduced to {LoRA, DPO, data curation} and Phase 8 reduced to {ASR, TTS}. The plan marks deferrable topics with ◇ in the capsules. Default remains the full program.

---
## 4. The curriculum, phase by phase

Each topic below is a **capsule**: what you build · the step ladder (each step is a runnable, tested checkpoint) · the "done when" verification · the video episodes · the AlphaDesk hook · effort (your hands-on hours; video production is Opus's job, not yours). Papers/references named in a capsule are the primary sources the videos teach from. ◇ marks topics deferrable on the accelerated path.

### Phase 0 · Python & Tensor Ramp (weeks 1–2) — for a senior .NET engineer

Not a beginner course. A translation layer: C#/.NET idioms → Python idioms, then straight into numerical computing.

- **P0.1 Python for the .NET veteran** — modules/packaging vs assemblies; `uv` vs NuGet; typing + dataclasses/pydantic vs POCOs; async/await differences; context managers vs `using`; magic methods vs operator overloading; pytest vs xUnit; debugging with `pdb`/VS Code. *Exercise:* port a small C#-style OMS domain model (Order, Fill, Position) to idiomatic Python with tests.
- **P0.2 NumPy as your new LINQ** — arrays, broadcasting, vectorization, einsum; why loops die. *Exercise:* compute SMA/EMA/RSI/Bollinger/VWAP over NSE bhavcopy data fully vectorized; match TA-Lib outputs.
- **P0.3 PyTorch tensors & the training loop skeleton** — tensors, autograd (as a user), `nn.Module`, optimizer, DataLoader; overfit a tiny MLP on a toy next-day-return classification task and understand *why that's expected and meaningless* (first lesson in financial ML humility).
- **Videos:** 3 episodes (~40 min total): "Python for .NET architects", "NumPy thinking", "The training loop".
- **Done when:** the indicator suite matches reference values to 1e-6 and you can write a training loop from an empty file without looking anything up.
- **Effort:** ~20 h.

### Phase 1 · Foundations (weeks 3–5) — how learning actually works

#### T31 · Autograd engine (micrograd-style)
**Build:** a scalar reverse-mode autodiff engine, then a tensor version on NumPy — `Value`/`Tensor` with `backward()`, topological sort, broadcasting-aware gradients, and a mini `nn` layer library on top. **Ladder:** (1) scalar Value + add/mul/tanh, gradcheck vs finite differences → (2) full op set + topo-sort backward → (3) tensor version with broadcasting & reduction grads → (4) `nn.Linear`/MLP + SGD/Adam → (5) train MLP on the Phase-0 task using *your* engine. **Done when:** gradcheck passes on 30 random graphs; your MLP matches PyTorch's loss curve within noise. **Videos:** 3 eps — chain rule to code; backward pass line-by-line; broadcasting gradients (the hard part, slowly). **AlphaDesk:** the engine trains AlphaDesk's first toy signal model; later phases switch to PyTorch but you now know what `.backward()` does. **Effort:** ~14 h. *Ref: Karpathy micrograd lineage, CS231n notes.*

#### T16 · Matrix multiplication kernel — Part A: CPU (GPU part in Phase 7)
**Build:** matmul from naive triple loop to cache-aware blocked/tiled + parallel implementation (C via ctypes or Rust via PyO3), understanding roofline thinking. **Ladder:** (1) naive Python (feel the pain) → (2) NumPy baseline + FLOPs accounting → (3) naive C/Rust → (4) loop-order + blocking for L1/L2 → (5) SIMD hints + threads → (6) benchmark table vs OpenBLAS. **Done when:** your blocked kernel beats naive C by ≥10× on 1024×1024 and you can explain the memory hierarchy chart from measurements, not folklore. **Videos:** 2 eps — why matmul is everything; cache lines make it fast (animated memory-access diagrams). **AlphaDesk:** none directly — this is the literacy every later kernel builds on. **Effort:** ~10 h.

#### T45 · Softmax kernel optimization — Part A: numerics & online softmax (GPU part in Phase 7)
**Build:** numerically safe softmax → fused softmax → **online (streaming) softmax**, the exact trick Flash Attention rests on. **Ladder:** (1) naive softmax, watch it overflow → (2) max-subtraction stability proof + tests → (3) two-pass fused version → (4) one-pass online softmax with running max/denominator, proved equivalent → (5) log-softmax + cross-entropy fusion. **Done when:** online version matches reference at fp32/fp16 on adversarial inputs (±1e4 logits). **Videos:** 2 eps — the overflow story; online softmax derived slowly on one whiteboard scene (this episode is deliberately the most patient in the whole series — it pays off in Phase 7). **AlphaDesk:** none directly; prerequisite literacy. **Effort:** ~6 h.

#### T30 · Tokenizer (BPE)
**Build:** byte-level BPE: trainer + encoder/decoder with special tokens, regex pre-tokenization (GPT-2 style), and a `FinTok` vocabulary trained on a financial corpus (filings + market news + NSE announcements). **Ladder:** (1) UTF-8 bytes and why chars lie → (2) BPE merge training loop → (3) encoder with merge ranks → (4) regex pre-splitting + special tokens (`<|order|>`, `<|filing|>`) → (5) train FinTok-16k; compare compression vs GPT-2 vocab on financial text → (6) property tests: encode∘decode = identity on fuzzer input. **Done when:** round-trip holds on 100k fuzzed strings; FinTok beats GPT-2 vocab compression on filings by a measured margin. **Videos:** 3 eps — why tokenization exists; training BPE step-by-step on a tiny corpus (every merge shown); building FinTok. **AlphaDesk:** FinTok is *the* tokenizer for AlphaSLM and every downstream model. **Effort:** ~10 h. *Ref: Sennrich 2015, GPT-2 tokenizer, tiktoken as benchmark.*

### Phase 2 · Transformers & Small Models (weeks 6–9)

#### T4 · Transformer from scratch (Attention Is All You Need)
**Build:** a decoder-only GPT in pure PyTorch from an empty file: embeddings, RoPE and learned positions (both, compared), multi-head causal self-attention, MLP, residuals + LayerNorm/RMSNorm, weight tying, KV-cached generation. **Ladder:** (1) single attention head on toy sequences, attention matrix visualized → (2) multi-head + causal mask → (3) full block; train on char-level ticker data → (4) RoPE derived and implemented → (5) full GPT trains on TinyStories-scale corpus in-cloud → (6) sampling: greedy/temperature/top-k/top-p → (7) naive KV cache for fast generation. **Done when:** loss curve matches a nanoGPT reference config within noise; generated text is coherent; attention maps show interpretable structure (induction heads found). **Videos:** 5 eps, the flagship series — attention derived slowly (Q/K/V as a retrieval metaphor, one full worked example with real numbers); the block; positions & RoPE; training dynamics; sampling & KV cache. **AlphaDesk:** the architecture AlphaSLM uses. **Effort:** ~22 h. *Ref: Vaswani 2017, GPT-2, RoPE (Su 2021).*

#### T15 · Small Language Model (SLM)
**Build:** **AlphaSLM-40M** — your Phase-2 transformer + FinTok, pretrained on a curated financial corpus (filings excerpts, market news, synthetic market commentary), with a proper training harness: LR warmup+cosine, grad clipping, checkpointing, wandb-style local logging, loss-scaling ablations. Pretrain a base in-cloud (CPU, small config) then the real 40M run on your 4070 overnight (~hours, not days). **Ladder:** (1) data pipeline: corpus → FinTok → packed .bin shards → (2) training harness with resume → (3) scaling mini-study: 5M vs 15M vs 40M loss curves (Chinchilla intuition at toy scale) → (4) 4070 run → (5) evaluate perplexity on held-out filings + qualitative generation. **Done when:** AlphaSLM-40M generates plausible market commentary and beats the 5M model on perplexity by the expected margin; you can read a loss curve like an ECG. **Videos:** 4 eps — data to shards; the harness; the scaling mini-study (slow, chart-heavy); reading your first pretrain. **AlphaDesk:** AlphaSLM is the desk's local model — later LoRA-tuned (T17), DPO-aligned (T19), distilled (T49), quantized (T8), served by tickerd (T3). **Effort:** ~18 h + one overnight 4070 run.

#### T43 · Embedding model
**Build:** a sentence-embedding model trained with contrastive learning: mean-pooled transformer encoder + InfoNCE/MultipleNegativesRanking loss on financial text pairs (filing section ↔ summary, question ↔ passage from FiQA), hard-negative mining. **Ladder:** (1) pooling + cosine similarity baseline from AlphaSLM activations → (2) InfoNCE loss from scratch, temperature ablation → (3) in-batch negatives training on FiQA pairs → (4) hard-negative mining round → (5) evaluate Recall@k / MRR on held-out finance retrieval set vs a MiniLM reference. **Done when:** your model beats the raw-activation baseline by a wide measured margin and lands within striking distance of MiniLM on the finance eval. **Videos:** 3 eps — what "similar" means geometrically; InfoNCE derived slowly; hard negatives. **AlphaDesk:** the embedding model powering the vector DB (T5), RAG (T6), semantic router (T36), two-tower recsys (T42). **Effort:** ~12 h. *Ref: SBERT, SimCSE, InfoNCE (Oord 2018).*

#### T13 · State Space Model (Mamba)
**Build:** a minimal Mamba block: discretized SSM (ZOH), selective scan implemented first as a sequential scan then as a parallel associative scan, gating/projections — trained on the same char-level task as T4 for a head-to-head. **Ladder:** (1) linear RNN/SSM recurrence + HiPPO intuition → (2) discretization (Δ, A, B) with tests vs continuous solution → (3) selectivity: input-dependent Δ/B/C → (4) parallel scan (Blelloch) with equivalence test vs sequential → (5) full block; train; compare vs transformer on speed & long-range recall (associative-recall synthetic task). **Done when:** parallel scan matches sequential to 1e-5; Mamba wins the long-sequence throughput benchmark and the transformer wins short-context quality — and you can explain both from the math. **Videos:** 3 eps — RNNs strike back (history, slowly); the selective scan; the showdown. **AlphaDesk:** experimental long-context branch for tick-sequence modeling. ◇ **Effort:** ~14 h. *Ref: Gu & Dao 2023, S4.*

#### T9 · Mixture of Experts (MoE) routing layer
**Build:** a top-k gated MoE FFN layer swapped into your transformer: router with noisy top-k, load-balancing auxiliary loss, capacity factor + token dropping, expert-utilization dashboards. **Ladder:** (1) dense FFN baseline metrics → (2) router + top-2 dispatch (gather/scatter, batched) → (3) load-balance loss; watch expert collapse happen *without* it (deliberately) → (4) capacity factor & dropped-token accounting → (5) train MoE-AlphaSLM vs dense at matched FLOPs; compare. **Done when:** expert utilization is balanced (entropy metric), and MoE matches dense quality at lower active-params — measured, not asserted. **Videos:** 3 eps — why sparsity (economics of inference); the router, slowly; the collapse failure and its fix. **AlphaDesk:** optional AlphaSLM-MoE variant; the routing telemetry feeds the eval dashboard. ◇ **Effort:** ~12 h. *Ref: Shazeer 2017, Switch Transformer, Mixtral.*

### Phase 3 · Retrieval & Knowledge (weeks 10–12) — the job-search payoff phase

This phase turns AlphaSLM into a *useful research tool*. It's the highest-signal phase for a Director/VP portfolio because it produces visible product behavior fast.

#### T5 · Vector database (HNSW index)
**Build:** Hierarchical Navigable Small World index from scratch: multi-layer graph, greedy search with `ef`, heuristic neighbor selection, insert/delete, persistence. **Ladder:** (1) brute-force kNN baseline + recall metric → (2) single-layer NSW greedy search → (3) hierarchical layers with probabilistic level assignment → (4) neighbor-selection heuristic (the quality lever) → (5) delete + rebuild + save/load → (6) benchmark recall@10 vs latency vs hnswlib on 100k finance embeddings. **Done when:** you match hnswlib recall within a few % at comparable latency and can draw the layer-descent search on a whiteboard. **Videos:** 4 eps — the ANN problem; small-world graphs (beautiful, slow visualization); HNSW search descent; the insertion heuristic. **AlphaDesk:** the vector store under RAG. **Effort:** ~16 h. *Ref: Malkov & Yashunin 2016.*

#### T50 · Database driver for vectors ◇
**Build:** a clean storage/query layer wrapping the HNSW index: a `VectorStore` with a driver interface (connection, batched upsert, metadata filtering, hybrid BM25+vector search, WAL for durability), so AlphaDesk talks to "a database," not a data structure. **Ladder:** (1) driver interface + in-memory impl → (2) metadata filter + pre/post-filtering tradeoffs → (3) BM25 lexical index + reciprocal-rank fusion → (4) write-ahead log + crash recovery → (5) simple query planner picking lexical vs vector vs hybrid. **Done when:** hybrid search beats pure-vector on a keyword-heavy finance query set (ticker symbols, exact phrases); crash-recovery test passes. **Videos:** 2 eps — what a DB driver actually is; hybrid retrieval. **AlphaDesk:** the persistence + query API the whole desk uses. **Effort:** ~12 h.

#### T6 · RAG pipeline
**Build:** the full retrieval-augmented generation loop over SEC filings + earnings transcripts: chunking strategies, embed→retrieve→rerank→pack→generate with citations, and an honest evaluation. **Ladder:** (1) document loaders + chunking (fixed/semantic/structural) ablation → (2) retrieve top-k with the T5 store → (3) cross-encoder reranker (fine-tuned from AlphaSLM) → (4) context packing + citation-forced prompt contract → (5) generate with AlphaSLM (and an API model for comparison) → (6) RAG eval: faithfulness, answer-relevance, context-precision on a hand-built finance QA set. **Done when:** answers cite the correct filing sections; measured faithfulness beats a no-rerank baseline. **Videos:** 4 eps — RAG anatomy; chunking is underrated (slow, with real failure examples); reranking; evaluating RAG honestly. **AlphaDesk:** the **Research Copilot's** core — "What did $TICKER say about margins last quarter?" **Effort:** ~14 h.

#### T36 · Semantic router
**Build:** an embedding-based intent router that sends a query to the right AlphaDesk tool (research / SQL / order / general) without an LLM call — utterance encoding, per-route centroids + thresholds, fallback to LLM classifier on low confidence. **Ladder:** (1) route definitions + example utterances → (2) centroid routing with cosine thresholds → (3) confidence calibration + reject option → (4) LLM fallback + logging misroutes → (5) eval on a labeled finance-query set. **Done when:** routing accuracy beats a keyword baseline and p50 latency stays sub-10ms. **Videos:** 2 eps — routing without an LLM; calibration & fallbacks. **AlphaDesk:** the front-door dispatcher for every user query. **Effort:** ~8 h.

#### T37 · Knowledge Graph builder
**Build:** an entity/relation extraction pipeline turning filings + news into a financial KG (companies, people, subsidiaries, supply-chain, ownership, events), stored in NetworkX/SQLite. **Ladder:** (1) NER for financial entities (AlphaSLM-tagger or spaCy baseline) → (2) relation extraction via prompted extraction with a JSON contract → (3) entity resolution/dedup (link "Apple"/"AAPL"/"Apple Inc.") → (4) graph store + schema → (5) graph queries (n-hop neighbors, shortest supply-chain path). **Done when:** the KG correctly links a query like "who supplies $TICKER" across 3 filings; entity-resolution precision measured. **Videos:** 3 eps — from text to triples; entity resolution (the unglamorous hard part); querying the graph. **AlphaDesk:** the KG powering GraphRAG and relationship-aware research. **Effort:** ~14 h.

#### T20 · Graph RAG system
**Build:** GraphRAG over the T37 knowledge graph: community detection + hierarchical summarization (Microsoft GraphRAG-style) for global "what are the themes across all my filings" questions that flat RAG can't answer. **Ladder:** (1) build KG from a filing corpus (reuse T37) → (2) Leiden/Louvain community detection → (3) per-community summaries via AlphaSLM → (4) global query: map over community summaries → reduce → (5) compare GraphRAG vs flat RAG on global vs local questions. **Done when:** GraphRAG wins measurably on "themes/trends across the corpus" questions while flat RAG wins on pinpoint lookups — you can articulate exactly when to use which. **Videos:** 3 eps — where flat RAG breaks; communities & summaries; map-reduce over a graph. **AlphaDesk:** "What themes are emerging across my whole watchlist?" — the Research Copilot's global mode. **Effort:** ~14 h. *Ref: Microsoft GraphRAG 2024.*

#### T40 · Text-to-SQL engine
**Build:** natural-language → SQL over a real market-data schema (prices, fundamentals, positions, orders), with schema-linking, a constrained-decoding SQL grammar (previews T25), execution guardrails (read-only, row limits), and self-correction on errors. **Ladder:** (1) DuckDB market schema + seed data → (2) schema-linking retrieval (which tables/columns matter) → (3) prompt→SQL with AlphaSLM/API → (4) grammar-constrained generation so output is always valid SQL → (5) execute, catch errors, self-repair loop → (6) eval on a hand-built NL-question set (execution accuracy). **Done when:** execution accuracy beats a naive prompt baseline; no query ever writes or scans unboundedly. **Videos:** 3 eps — text-to-SQL anatomy; schema linking; safe execution & self-repair. **AlphaDesk:** "Show me my worst 5 positions by unrealized P&L today" — the analytics query path. **Effort:** ~12 h.

#### T48 · Feature store ◇
**Build:** a point-in-time-correct feature store for market features (rolling vols, momentum, sentiment scores) with offline (batch) and online (serving) parity — the piece that prevents lookahead bias, which is *the* classic quant ML bug. **Ladder:** (1) feature definitions + batch materialization → (2) point-in-time joins (as-of correctness) with a test that catches lookahead → (3) online store (Redis-style, in-proc) → (4) offline/online parity test → (5) feature versioning + lineage. **Done when:** the lookahead-detection test fails on a deliberately buggy feature and passes on the correct one; offline and online serve identical values. **Videos:** 2 eps — why point-in-time correctness is everything in finance ML; offline/online parity. **AlphaDesk:** feeds the two-tower recsys and any signal model with leak-free features. **Effort:** ~10 h.

#### T42 · Recommendation system (two-tower)
**Build:** a two-tower retrieval model recommending relevant instruments/research to a user given their portfolio & behavior — user tower + item tower, trained with sampled-softmax, served via the T5 vector index. **Ladder:** (1) synthetic user-interaction + portfolio dataset → (2) two towers over T48 features → (3) sampled-softmax / in-batch negatives training → (4) index item embeddings into HNSW; retrieve → (5) offline eval (recall@k) + a simple counterfactual note on why online eval differs. **Done when:** recall@k beats a popularity baseline; retrieval runs through your own vector index end-to-end. **Videos:** 3 eps — retrieval vs ranking; the two-tower trick; serving recommendations. **AlphaDesk:** "Ideas for you" — research & watchlist suggestions. ◇ **Effort:** ~12 h. *Ref: YouTube two-tower, Google two-tower retrieval.*

### Phase 4 · Agents & Safety (weeks 13–15) — behavior you can demo, safety you can defend

#### T1 · Reasoner (Chain-of-Thought)
**Build:** a CoT reasoning wrapper with self-consistency and verification: structured reasoning prompts, multi-sample majority voting, a self-critique/verify pass, and a reasoning-trace logger — all measured on finance word problems (position sizing, P&L, options payoff arithmetic). **Ladder:** (1) baseline direct-answer accuracy → (2) zero-shot + few-shot CoT, measure the lift → (3) self-consistency (sample N, vote) → (4) verifier pass (check arithmetic / constraints) → (5) reasoning-trace storage for the eval harness. **Done when:** CoT + self-consistency beats direct answering on a finance-math set by a measured margin; traces are inspectable. **Videos:** 3 eps — why models reason better out loud; self-consistency; verify-then-trust. **AlphaDesk:** the reasoning layer for multi-step research questions. **Effort:** ~10 h. *Ref: Wei 2022 CoT, Wang 2022 self-consistency.*

#### T2 · Agent loop (ReAct)
**Build:** a ReAct agent from scratch: the thought→action→observation loop, tool-calling protocol, a tool registry (research, SQL, calculator, order-preview), memory/scratchpad, step limits, and error recovery — no agent framework, you write the loop. **Ladder:** (1) the bare loop with one tool → (2) tool registry + parsing/validation of tool calls → (3) multi-tool orchestration → (4) memory + reflection on failed steps → (5) guardrails: max steps, loop detection, timeout → (6) trace visualization. **Done when:** the agent answers a multi-hop finance question requiring ≥3 tools correctly and recovers from a deliberately failing tool call. **Videos:** 4 eps — ReAct explained (slowly, with a full annotated trace); the tool protocol; memory & reflection; making agents not spiral. **AlphaDesk:** the **orchestrator** that ties Research Copilot, SQL, and order workflow together. **Effort:** ~16 h. *Ref: Yao 2022 ReAct.*

#### T24 · Function-calling router
**Build:** the structured function-calling layer under the agent: tool schemas (JSON Schema), argument extraction with constrained decoding, parallel tool calls, validation & type coercion, and a dispatcher — the reliable plumbing beneath ReAct. **Ladder:** (1) tool schema definitions + registry → (2) prompt→function-call JSON with AlphaSLM/API → (3) grammar/schema-constrained argument generation (leans on T25) → (4) validation, coercion, error-return-to-model → (5) parallel calls + result aggregation. **Done when:** malformed tool calls are impossible by construction (constrained), and parallel calls aggregate correctly. **Videos:** 2 eps — function calling under the hood; making it un-break-able. **AlphaDesk:** the tool-invocation layer for the agent. **Effort:** ~9 h.

#### T17-sandbox · Code interpreter sandbox
**Build:** a sandboxed Python execution tool for the agent (run generated analysis code safely): subprocess isolation, resource limits (CPU/mem/time via rlimits), filesystem jail, import allowlist, and captured stdout/plots returned to the agent. **Ladder:** (1) subprocess exec + captured output → (2) rlimit CPU/mem/time caps with tests that trip them → (3) filesystem + network isolation → (4) import allowlist / AST screening → (5) return plots/dataframes to the agent as observations. **Done when:** a battery of hostile snippets (fork bombs, `os.system`, infinite loops, huge allocs) are all contained; legitimate pandas analysis runs and returns results. **Videos:** 3 eps — why you never `exec()` model output raw; building the jail; the escape-attempt test suite. **AlphaDesk:** lets the agent compute custom analytics ("plot my sector exposure") safely. **Effort:** ~12 h. *(Note: a learning-grade sandbox; the video is explicit that production isolation needs gVisor/Firecracker/containers.)*

#### T28-guardrails · Guardrails system (input/output filtering)
**Build:** an input/output guardrail layer: prompt-injection detection, PII detection/redaction, topic/compliance filters (no investment advice framed as a recommendation, mandatory disclaimers), jailbreak heuristics, and a policy engine with audit logging. **Ladder:** (1) policy schema + audit log → (2) input filters: injection & jailbreak detection (heuristic + classifier) → (3) PII detection/redaction → (4) output filters: disclaimer enforcement, advice-language detection → (5) red-team eval set + metrics (block rate vs false-positive rate). **Done when:** the guardrail blocks a curated injection/jailbreak/advice set while keeping false positives on benign finance queries low — both numbers reported. **Videos:** 3 eps — the threat model; input guardrails; output guardrails & compliance. **AlphaDesk:** the **compliance perimeter** — the piece that makes the demo defensible to a risk officer, and a genuine differentiator on your résumé. **Effort:** ~12 h.

#### T27 · LLM eval harness
**Build:** a proper eval harness: task/dataset abstractions, metrics (exact-match, F1, LLM-as-judge with bias controls, pass@k), a regression suite over all AlphaDesk components, and an HTML report. **Ladder:** (1) task+metric interfaces → (2) deterministic metrics + a finance QA task → (3) LLM-as-judge with position-bias mitigation → (4) full regression suite wired to every prior component → (5) report generation + CI-style pass/fail gates. **Done when:** one command evaluates the whole desk and produces a scored report; a deliberately regressed component is caught. **Videos:** 3 eps — you can't improve what you don't measure; LLM-as-judge (and its traps); building the regression gate. **AlphaDesk:** the quality gate every other component is scored against — used continuously from here on. **Effort:** ~12 h.

#### T44 · Adversarial attack generator ◇
**Build:** an adversarial testing tool: text attacks (typos, synonym swaps, prompt-injection payloads, unicode tricks) against AlphaDesk's classifiers/guardrails, plus a gradient-based embedding attack on the T43 embedding model — used to *harden*, not to harm. **Ladder:** (1) perturbation library (char/word/sentence) → (2) attack success metric vs the guardrail → (3) embedding-space adversarial examples (FGSM-style) → (4) transferability check → (5) adversarial-training loop that hardens the guardrail. **Done when:** attacks measurably degrade the un-hardened model and adversarial training recovers most of the gap. **Videos:** 2 eps — how text models get fooled; hardening by attacking yourself. **AlphaDesk:** continuous red-teaming of the guardrail layer. ◇ **Effort:** ~10 h.

#### T22 · Interpretability tool (Sparse Autoencoders)
**Build:** a Sparse Autoencoder trained on AlphaSLM's residual-stream activations to extract monosemantic features, with an auto-interp + feature-dashboard, and a steering demo (amplify a "bullish sentiment" feature and watch generation shift). **Ladder:** (1) activation capture harness → (2) SAE (tied weights, L1 sparsity) training → (3) feature analysis: max-activating examples per feature → (4) auto-interpretation labels → (5) activation steering: add a feature direction, observe behavior change. **Done when:** you find and name ≥5 interpretable finance features (e.g., a "risk/uncertainty" feature) and steering with them changes outputs predictably. **Videos:** 4 eps — superposition & why models are hard to read; SAEs explained slowly; finding features; steering the model live. **AlphaDesk:** an "explainability" panel — a standout portfolio talking point. **Effort:** ~16 h. *Ref: Anthropic SAE / dictionary-learning line of work.*

### Phase 5 · Training & Alignment (weeks 16–19) — how models are made and shaped

#### T38 · Data curation pipeline (MinHash / dedup)
**Build:** a corpus-cleaning pipeline: quality filtering, MinHash + LSH near-deduplication, PII scrubbing, decontamination against eval sets, and dataset cards. **Ladder:** (1) quality heuristics (length, symbol ratio, language) → (2) exact dedup (hashing) → (3) MinHash signatures + LSH banding for near-dup → (4) eval-set decontamination → (5) run it on your raw financial corpus; report before/after stats. **Done when:** near-dup detection catches paraphrased filings at a chosen Jaccard threshold; the cleaned corpus improves AlphaSLM perplexity vs the raw one (measured). **Videos:** 3 eps — garbage in, garbage out (with examples); MinHash & LSH derived slowly; decontamination. **AlphaDesk:** produces the clean corpus AlphaSLM and the embedding model train on. **Effort:** ~12 h. *Ref: MinHash/LSH, GKG/C4 cleaning practices.*

#### T23 · Synthetic data generator
**Build:** a synthetic-data engine generating instruction/preference data for finance: templated + LLM-generated Q&A over filings, self-instruct expansion, quality filtering, and a preference-pair generator (for DPO). **Ladder:** (1) templated generation from KG facts → (2) LLM self-instruct expansion with a JSON contract → (3) quality + diversity filtering → (4) preference-pair construction (chosen/rejected) → (5) dataset card + leakage check. **Done when:** the synthetic set trains a measurably better instruction-follower than templates alone; preference pairs are ready for T19. **Videos:** 3 eps — when synthetic data helps (and when it poisons); self-instruct; building preference pairs. **AlphaDesk:** the instruction + preference datasets for AlphaSLM alignment. **Effort:** ~12 h.

#### T17 · LoRA trainer
**Build:** Low-Rank Adaptation from scratch: `LoRALinear` (A·B low-rank deltas, scaling, merge), injection into a frozen open model (1–3B) or AlphaSLM, and instruction-tuning on your synthetic finance data — trained on the 4070. **Ladder:** (1) LoRA math + `LoRALinear` with a merge()→base-equivalence test → (2) inject into attention/MLP of a frozen model → (3) train on finance instructions (4070) → (4) adapter save/load/hot-swap → (5) compare against full fine-tune quality vs memory. **Done when:** LoRA reaches near-full-FT quality on your task at a fraction of trainable params, and merged weights are numerically identical to base+delta. **Videos:** 4 eps — the low-rank insight (slowly, with SVD intuition); implementing LoRA; injecting into a real model; adapters as swappable skills. **AlphaDesk:** `AlphaSLM-finance` adapter — the desk's tuned model. **Effort:** ~14 h. *Ref: Hu 2021 LoRA.*

#### T39 · Parameter-Efficient Fine-Tuning (PEFT) library ◇
**Build:** generalize T17 into a small PEFT library: LoRA + QLoRA (4-bit base, leans on T8) + prefix-tuning + (IA)³, a unified adapter API, and a comparison study across methods on the same task. **Ladder:** (1) refactor T17 into a method-agnostic adapter interface → (2) add prefix-tuning → (3) add (IA)³ → (4) QLoRA path over a 4-bit base → (5) benchmark: quality vs trainable-params vs memory across methods. **Done when:** all four methods train through one API; the comparison table is clear enough to guide a real method choice. **Videos:** 2 eps — the PEFT zoo; choosing a method from data. **AlphaDesk:** the tuning toolkit; QLoRA lets bigger bases fit the 4070. ◇ **Effort:** ~10 h.

#### T47 · Model distillation pipeline ◇
**Build:** knowledge distillation: a large teacher (API model or a bigger open model) → AlphaSLM student, via response distillation + logit/KD-loss distillation, with a quality/latency comparison. **Ladder:** (1) teacher response dataset generation → (2) SFT student on teacher outputs → (3) KD loss (soft targets, temperature) where logits are available → (4) evaluate student vs teacher vs base → (5) latency/cost comparison. **Done when:** the distilled student closes a measured fraction of the base→teacher quality gap at a fraction of teacher latency/cost. **Videos:** 2 eps — learning from a teacher; soft targets (why they carry more signal). **AlphaDesk:** a small fast desk model distilled from a strong teacher. ◇ **Effort:** ~10 h. *Ref: Hinton 2015.*

#### T19 · DPO loss function
**Build:** Direct Preference Optimization from scratch: the DPO loss + reference-model KL, a training loop over your synthetic preference pairs, and an alignment eval — the modern, RL-free alignment method. **Ladder:** (1) DPO loss derived and implemented with a unit test vs the closed-form → (2) reference-model setup (frozen SFT) → (3) training loop on preference pairs → (4) β sweep + reward-margin tracking → (5) eval: win-rate vs the SFT model (LLM-judge via T27). **Done when:** the DPO'd model wins a measured majority of head-to-heads vs SFT on finance-appropriateness (helpful + compliant tone). **Videos:** 3 eps — RLHF without the RL (the big idea, slowly); the DPO loss line-by-line; measuring alignment. **AlphaDesk:** aligns AlphaSLM toward compliant, disclaimer-aware, helpful behavior. **Effort:** ~12 h. *Ref: Rafailov 2023 DPO.*

#### T14 · RLHF pipeline (PPO)
**Build:** the classic RLHF stack: a reward model (from preference pairs) + PPO fine-tuning of AlphaSLM against it, with KL-to-reference penalty, advantage estimation, and reward-hacking detection — the harder, historically important counterpart to DPO. **Ladder:** (1) reward model training + accuracy eval → (2) PPO core: rollout, log-probs, advantages (GAE), clipped objective → (3) KL penalty to the reference → (4) training loop with reward + KL curves → (5) reward-hacking watch (length/format exploits) + comparison to the T19 DPO result. **Done when:** reward goes up while KL stays bounded and outputs don't degenerate; you can explain, from your own two implementations, exactly why the field largely moved to DPO. **Videos:** 5 eps — the RLHF picture; reward modeling; PPO explained slowly (the most involved derivation in Phase 5); KL as a leash; reward hacking & DPO-vs-PPO. **AlphaDesk:** an alternate alignment path; the DPO/PPO comparison is a strong interview story. **Effort:** ~20 h. *Ref: Ouyang 2022 InstructGPT, Schulman 2017 PPO.*

#### T21 · Model merger (Model Soups / SLERP)
**Build:** weight-space model merging: linear soups, SLERP, task-arithmetic (add/subtract task vectors), and TIES-merging — combine your finance adapter with a general model without retraining. **Ladder:** (1) linear averaging of two compatible checkpoints → (2) SLERP (spherical interpolation) with the geometry shown → (3) task vectors: `merged = base + Σ(task_i − base)` → (4) TIES (trim/elect-sign/merge) for conflict resolution → (5) evaluate merged vs individuals across tasks. **Done when:** a merged model retains finance skill *and* general ability better than either parent alone on a two-task eval. **Videos:** 3 eps — averaging models shouldn't work, but does; SLERP geometry; task arithmetic. **AlphaDesk:** ships a merged desk model combining tuned + general skills. ◇ **Effort:** ~10 h. *Ref: Model Soups 2022, Task Arithmetic 2022, TIES 2023.*

#### T11 · Distributed training loop (FSDP / Tensor Parallelism)
**Build:** the real sharding math: a hand-written data-parallel loop (all-reduce gradients), then FSDP-style parameter/grad/optimizer sharding, then tensor parallelism (column/row-parallel linears) — correctness on multi-process CPU in-cloud, plus a single-GPU + CPU-offload run on the 4070. **Ladder:** (1) DDP by hand: gradient all-reduce with gloo, verify equivalence to single-process → (2) ZeRO-1/2/3 intuition; shard optimizer, then grads, then params → (3) FSDP-style shard/all-gather/reduce-scatter around the forward/backward → (4) tensor parallel: split a linear across ranks, all-reduce the output, verify identical logits → (5) 4070 run with CPU offload; note what changes at cluster scale. **Done when:** every distributed variant produces logits/gradients identical (to tolerance) to the single-process baseline — you've proven you understand the collectives, not just called an API. **Videos:** 5 eps — data vs model parallelism; all-reduce & friends (the collectives, animated); ZeRO/FSDP sharding; tensor parallelism; what a real cluster adds. **AlphaDesk:** not a runtime component — this is the "can operate at scale" credential for VP-level conversations. **Effort:** ~20 h. *Ref: ZeRO 2019, Megatron-LM, PyTorch FSDP.*

#### T46 · Neural Architecture Search ◇
**Build:** a small NAS: a search space over tiny transformer/CNN configs, random + evolutionary search, and a weight-sharing (one-shot) supernet — run at a scale that finishes on the 4070. **Ladder:** (1) search space + a fast proxy task → (2) random search baseline → (3) evolutionary search (mutate/select) → (4) weight-sharing supernet + subnet sampling → (5) compare found architectures vs hand-designed. **Done when:** search finds a config beating the hand-picked baseline on the proxy task within a fixed compute budget. **Videos:** 2 eps — automating architecture design; weight sharing makes it affordable. **AlphaDesk:** tunes a small signal-model architecture. ◇ **Effort:** ~10 h.

### Phase 6 · Inference Systems (weeks 20–23) — making models fast and cheap to serve

#### T3 · Inference server (in Rust)
**Build:** `tickerd` — a Rust inference server: an Axum HTTP/OpenAI-compatible API, a request queue with continuous batching, a model backend (call into a small model via `candle`, or proxy to a Python worker for AlphaSLM), streaming SSE responses, and metrics. This is your headline systems piece and plays to your Rust interest. **Ladder:** (1) Axum server + `/v1/completions` echo → (2) load a small model with `candle`, single-request generation → (3) request queue + continuous batching (dynamic batch assembly) → (4) streaming tokens via SSE → (5) metrics (tokens/s, queue depth, p50/p99) + load test. **Done when:** the server sustains concurrent streaming clients with continuous batching measurably beating one-at-a-time throughput. **Videos:** 5 eps — why serving is its own discipline; Rust + Axum scaffold; continuous batching (the core idea, slowly); streaming; measuring an inference server. **AlphaDesk:** the **serving layer** for AlphaSLM behind the whole desk. **Effort:** ~22 h. *Ref: vLLM/TGI serving concepts, candle.*

#### T12 · KV-cache paging system (like vLLM)
**Build:** PagedAttention-style KV cache management: block-based KV allocation, a block table per sequence, copy-on-write for shared prefixes, and eviction — eliminating the memory fragmentation that kills naive KV caches. **Ladder:** (1) naive contiguous KV cache; measure fragmentation/waste → (2) fixed-size KV blocks + per-sequence block table → (3) dynamic allocation/free on sequence growth/finish → (4) copy-on-write prefix sharing (many prompts, one shared system prefix) → (5) eviction + benchmark memory utilization vs naive. **Done when:** paged KV serves more concurrent sequences in the same memory than contiguous, with prefix sharing demonstrated — numbers measured. **Videos:** 4 eps — the KV memory problem; paging (OS analogy, slowly); prefix sharing; the utilization win. **AlphaDesk:** upgrades `tickerd`'s memory management. **Effort:** ~16 h. *Ref: vLLM PagedAttention 2023.*

#### T25-logit · Logit processor *(ships inside the T25 folder)*
**Build:** a composable logit-processing pipeline applied at each decode step: temperature/top-k/top-p/min-p, repetition & presence penalties, logit bias, banned-token masks, and a JSON-schema-forcing processor (bridges to T25/T24). **Ladder:** (1) processor interface + temperature/top-k/top-p → (2) repetition/presence/frequency penalties → (3) logit bias + banned tokens → (4) a schema-enforcing processor (mask invalid next tokens) → (5) compose into a pipeline; unit-test each. **Done when:** each processor is individually tested and the schema-forcing one makes invalid JSON impossible token-by-token. **Videos:** 2 eps — sampling knobs demystified; constraining generation at the logit level. **AlphaDesk:** the control layer inside `tickerd`; underpins structured outputs & function calling. **Effort:** ~8 h.

#### T25 · Structured output parser (Context-Free Grammars)
**Build:** grammar-constrained decoding: a CFG/EBNF parser, an incremental parser that computes the valid next-token set at each step, and a token-mask generator — so the model *cannot* emit output that violates a grammar (JSON, SQL, a custom order-DSL). **Ladder:** (1) EBNF grammar representation + a recognizer → (2) incremental parse state → (3) valid-next-terminal computation → (4) map terminals to token masks over the vocab → (5) drive generation; grammars for JSON, a SQL subset, and an order DSL. **Done when:** generation is provably grammar-valid on adversarial prompts for all three grammars. **Videos:** 3 eps — parsing 101 for engineers; constrained decoding (the elegant core idea, slowly); grammars for JSON/SQL/orders. **AlphaDesk:** guarantees valid tool calls, SQL, and order tickets. **Effort:** ~14 h. *Ref: GBNF/llama.cpp grammars, Outlines.*

#### T12-spec · Speculative decoding
**Build:** speculative decoding: a small draft model proposes k tokens, the target model verifies them in one pass, accepted tokens are kept — 2–3× latency wins with identical output distribution. **Ladder:** (1) draft (tiny) + target (AlphaSLM) setup → (2) draft k tokens → (3) target parallel verification + the accept/reject rule (proved distribution-preserving) → (4) rollback + resample on rejection → (5) measure speedup vs acceptance rate across draft sizes. **Done when:** output distribution matches plain decoding (statistical test) while wall-clock latency drops measurably. **Videos:** 3 eps — the speculative idea (slowly — the accept/reject proof is the star); implementing verify; tuning the draft. **AlphaDesk:** a `tickerd` latency mode. **Effort:** ~12 h. *Ref: Leviathan 2023, Chen 2023.*

#### T29 · Prompt caching mechanism
**Build:** a prefix KV-cache reuse system: hash-based prefix matching, a radix-tree prefix cache (RadixAttention-style), cache eviction, and hit-rate metrics — reuse computed KV for shared system prompts / few-shot prefixes. **Ladder:** (1) exact-prefix hash cache → (2) radix tree for longest-shared-prefix reuse → (3) integrate with the T12 paged KV (COW blocks) → (4) eviction (LRU) + memory accounting → (5) measure TTFT improvement + hit rate on repeated-prefix workloads. **Done when:** repeated system-prefix requests show a large measured drop in time-to-first-token via cache hits. **Videos:** 2 eps — paying once for a shared prefix; radix-tree prefix caching. **AlphaDesk:** speeds every desk request that shares the compliance/system prefix. **Effort:** ~10 h. *Ref: SGLang RadixAttention 2023.*

#### T8 · Quantization library (Int8 / FP4)
**Build:** post-training quantization from scratch: symmetric/asymmetric Int8 (per-tensor & per-channel), an INT8 matmul path, GPTQ-style and abs-max calibration, and an FP4/NF4 4-bit path — measured accuracy/size/speed tradeoffs on a small open model, on the 4070. **Ladder:** (1) quant/dequant math + error metrics → (2) per-tensor vs per-channel Int8 weight quant → (3) calibration (abs-max, percentile) on real activations → (4) INT8 GEMM path + accuracy delta → (5) NF4/FP4 4-bit weights (QLoRA-style) → (6) size/speed/quality table across schemes. **Done when:** Int8 keeps quality within a small measured delta at ~4× size reduction; NF4 at ~8×, with the accuracy cost quantified. **Videos:** 4 eps — why fewer bits (the memory/bandwidth story); Int8 quantization slowly; calibration; 4-bit & NF4. **AlphaDesk:** shrinks AlphaSLM to serve faster on the 4070; pairs with QLoRA (T39). **Effort:** ~16 h. *Ref: LLM.int8() 2022, GPTQ 2022, QLoRA/NF4 2023.*

#### T41 · AI Gateway (load balancing / failover)
**Build:** an AI gateway in front of multiple backends (your `tickerd`, an API provider, a local fallback): routing policies, load balancing, health checks, failover, retries with backoff, rate limiting, per-request cost/latency accounting, and a unified OpenAI-compatible API. **Ladder:** (1) reverse-proxy with a unified schema over 2+ backends → (2) load-balancing strategies (round-robin, least-latency) → (3) health checks + automatic failover → (4) retries/backoff + circuit breaker → (5) rate limiting + cost/latency dashboard. **Done when:** killing a backend mid-load triggers seamless failover with no dropped requests; the dashboard shows per-backend cost/latency. **Videos:** 3 eps — why you need a gateway; load balancing & failover; observability & cost control. **AlphaDesk:** the **single entry point** routing between the local desk model and cloud models — an architecture story that resonates at Director/VP level. **Effort:** ~12 h.

### Phase 7 · GPU Kernels — the 4070 lane (weeks 24–25)

These three build on the Phase-1 CPU parts. Code is written + correctness-verified in-cloud (Triton interpreter / NumPy reference); **real speedups are measured on your 4070** via `gpu-runner/`. The videos show both the code and your measured numbers.

#### T16 · Matmul kernel — Part B: GPU (Triton/CUDA)
**Build:** a tiled GPU matmul in Triton (and an annotated raw-CUDA version): shared-memory tiling, coalesced loads, and autotuned block sizes, benchmarked vs cuBLAS on the 4070. **Ladder:** (1) naive Triton matmul, correctness in interpreter mode → (2) shared-memory tiling → (3) coalescing + `tl.dot` → (4) autotune block/warp config → (5) 4070 benchmark vs cuBLAS (report % of peak). **Done when:** your kernel reaches a respectable fraction of cuBLAS throughput and you can read the roofline from your own numbers. **Videos:** 3 eps — the GPU execution model (slowly — threads/warps/blocks/SMs); tiling in shared memory; measuring against cuBLAS. **Effort:** ~14 h. *Ref: Triton tutorials, CUDA matmul.*

#### T45 · Softmax kernel — Part B: GPU (fused)
**Build:** a fused GPU softmax (one kernel, one pass) using the online-softmax algorithm from Phase 1, with block-level reductions, benchmarked vs PyTorch. **Ladder:** (1) row-per-block softmax in Triton → (2) block reduction for max/sum → (3) fuse to a single pass (online) → (4) numerical checks at fp16 → (5) 4070 benchmark vs `torch.softmax`. **Done when:** the fused kernel matches PyTorch accuracy and beats a naive multi-pass version measurably. **Videos:** 2 eps — reductions on a GPU; fusing softmax (paying off the Phase-1 online-softmax episode). **Effort:** ~8 h.

#### T7 · Flash Attention kernel (CUDA/Triton)
**Build:** FlashAttention from scratch: tiled Q/K/V blocks, online-softmax accumulation (no materialized N×N attention matrix), causal masking, and the backward pass — the capstone kernel, benchmarked vs PyTorch SDPA on the 4070. **Ladder:** (1) attention as tiled block-matmuls → (2) online-softmax accumulation across K/V blocks (this is where Phase-1 softmax + Phase-2 attention converge) → (3) causal masking in-kernel → (4) forward benchmark vs `F.scaled_dot_product_attention` (speed + memory) → (5) backward pass → (6) plug into your Phase-2 transformer, verify identical outputs. **Done when:** forward matches SDPA outputs, uses O(N) not O(N²) memory (measured), and is faster than a naive attention kernel; the transformer runs end-to-end on it. **Videos:** 5 eps — why attention is memory-bound (slowly); the tiling insight; online-softmax accumulation (the heart, very slow); causal masking; the backward pass. **Effort:** ~22 h. *Ref: Dao 2022 FlashAttention, FlashAttention-2.*

### Phase 8 · Multimodal (weeks 26–28)

#### T32 · Vision Transformer (ViT)
**Build:** ViT from scratch: patch embedding, class token, positional embeddings, the (reused) transformer encoder, trained on a small image task, plus attention-rollout visualization — and a finance use: classifying candlestick-chart patterns. **Ladder:** (1) patchify + linear embed → (2) prepend cls token + pos embeds → (3) reuse Phase-2 encoder blocks → (4) train on CIFAR-10-scale (or a chart-pattern set) → (5) attention-rollout maps. **Done when:** ViT trains to a sensible accuracy and attention maps focus on meaningful regions; the chart-pattern classifier works above baseline. **Videos:** 3 eps — images as sequences; patch embeddings; what ViT looks at. **AlphaDesk:** a chart-pattern classifier tool for the agent. ◇ **Effort:** ~12 h. *Ref: Dosovitskiy 2020 ViT.*

#### T26 · Multimodal projector (CLIP)
**Build:** a CLIP-style dual encoder: image encoder (your ViT) + text encoder (your transformer) trained with contrastive loss to a shared space, enabling text↔image retrieval — trained small on your 4070. **Ladder:** (1) two encoders → shared projection → (2) symmetric InfoNCE (image↔text) → (3) train on a small captioned set → (4) zero-shot classification via text prompts → (5) text→image and image→text retrieval eval. **Done when:** zero-shot classification beats random by a wide margin and retrieval returns sensible matches. **Videos:** 3 eps — one space for pictures and words; contrastive alignment; zero-shot magic explained. **AlphaDesk:** "find charts that look like this pattern" via text or image query. ◇ **Effort:** ~12 h. *Ref: Radford 2021 CLIP.*

#### T33 · Whisper-style ASR model
**Build:** an encoder-decoder speech-to-text model: log-mel frontend, a conv+transformer audio encoder, a text decoder with cross-attention, trained on a small speech set — so you can *speak* to AlphaDesk. **Ladder:** (1) audio → log-mel spectrogram frontend → (2) audio encoder → (3) text decoder + cross-attention → (4) train on a small ASR set (e.g., a subset of LibriSpeech/Common Voice) → (5) greedy decode + WER eval. **Done when:** the model transcribes held-out clips at a WER that clearly beats chance and handles finance vocabulary tolerably. **Videos:** 4 eps — sound as data (spectrograms, slowly); the audio encoder; cross-attention decoding; measuring ASR (WER). **AlphaDesk:** voice input — "Hey desk, what's my exposure to financials?" (ties to your EchoGraph audio-native interest). ◇ **Effort:** ~14 h. *Ref: Radford 2022 Whisper.*

#### T35 · Audio Spectrogram Transformer ◇
**Build:** an AST for audio *classification* (not transcription): patchify the spectrogram (ViT-on-audio), train to classify audio events — a compact companion to ASR that reuses ViT. **Ladder:** (1) spectrogram patches → (2) reuse the ViT encoder → (3) train on an audio-classification set (e.g., ESC-50-scale) → (4) evaluate; compare to a CNN baseline. **Done when:** AST beats the CNN baseline (or ties with fewer params) on the classification set. **Videos:** 2 eps — ViT for ears; when attention beats convolution on audio. **AlphaDesk:** minor — audio-alert classification (e.g., detecting a specific tone in a squawk feed). ◇ **Effort:** ~8 h. *Ref: Gong 2021 AST.*

#### T34 · Text-to-Speech pipeline
**Build:** a TTS pipeline: a text→mel acoustic model (small FastSpeech2-style) + a vocoder (Griffin-Lim first, then a small neural vocoder), giving AlphaDesk a voice — and closing the loop with the very Kokoro-family stack that narrates your lesson videos (the videos explain what Kokoro does internally). **Ladder:** (1) text → phonemes (espeak-ng) → (2) duration + mel prediction (small transformer) → (3) Griffin-Lim vocoder (fast, teaches the phase-reconstruction problem) → (4) a small neural vocoder for quality → (5) synthesize; compare to Kokoro output and explain the gap. **Done when:** intelligible speech is synthesized end-to-end from your own model, and you can explain precisely how it differs from production Kokoro. **Videos:** 4 eps — the TTS pipeline (slowly); mel spectrograms & duration; vocoders & the phase problem; your model vs Kokoro. **AlphaDesk:** spoken responses; a satisfying full-circle finale with the video narration stack. ◇ **Effort:** ~14 h. *Ref: FastSpeech2, HiFi-GAN, Kokoro/StyleTTS2 lineage.*

#### T10 · Diffusion model (UNet + scheduler)
**Build:** a DDPM/DDIM diffusion model from scratch: the forward noising process, a UNet with time embeddings + attention, the training objective, and DDPM/DDIM samplers — trained on a small image set on the 4070. **Ladder:** (1) forward diffusion (noise schedule) + the reparameterization → (2) UNet with sinusoidal time embeddings → (3) the simple ε-prediction training objective → (4) DDPM sampling → (5) DDIM fast sampling + a classifier-free-guidance toy → (6) train on MNIST/Fashion-scale; generate. **Done when:** the model generates recognizable samples and DDIM produces comparable quality in far fewer steps (measured). **Videos:** 5 eps — diffusion intuition (destroy then rebuild, very slowly); the noise schedule & math; the UNet; sampling (DDPM→DDIM); guidance. **AlphaDesk:** the outlier/most-independent topic — framed as a "generate synthetic scenario charts / market-regime images" tool, but primarily here for completeness and because it's foundational modern ML. ◇ **Effort:** ~18 h. *Ref: Ho 2020 DDPM, Song 2021 DDIM.*

### Phase 9 · Capstone Assembly (weeks 29–30)

No new topics — this phase hardens AlphaDesk into a coherent, demoable product and produces the portfolio artifacts. See Section 6.5 for the assembly checklist and the final portfolio video.

---
## 5. The video production system (Remotion + Kokoro)

Every topic ships a mini-course of episodes. This section is the reusable spec Opus follows so all ~140 episodes look and sound like one series. **This is Opus's work, not yours** — you watch and learn.

### 5.1 Series design (slow learning, by construction)

The user asked for detailed explanation and slow pacing. That's enforced by rules, not vibes:

- **One idea per scene.** A scene introduces exactly one concept, holds it on screen, narrates it fully, then transitions. No scene shorter than 8 seconds.
- **Narration cadence.** Kokoro is driven at a calm rate (~0.85× the default speed setting) with explicit pauses (300–600 ms) inserted at sentence boundaries and after every new term. Target ~120–135 words/minute (unhurried lecture pace), not the 160+ of typical explainer videos.
- **Show the math three times.** Every formula appears as (1) plain-English sentence, (2) symbolic equation (typeset), (3) the code line that implements it — narrated in that order, so a symbol is never shown before its meaning.
- **Code walkthroughs are line-highlighted.** Code scenes highlight the active line, dim the rest, and the narration reads intent before syntax. Long files are revealed progressively, never dumped.
- **Recap bookends.** Each episode opens with "here's where we are in the build" (a map highlighting the current topic in the AlphaDesk architecture) and closes with a 3-bullet recap + "what breaks if we skip this."
- **Episode length.** 6–12 minutes each; a topic's mini-course is 2–5 episodes (≈ 20–50 minutes). This yields the requested *detailed* treatment without any single video becoming unwatchable.

### 5.2 Technical pipeline (runs in the CPU cloud sandbox)

```
script.md  ──►  narration segmentation  ──►  Kokoro-82M (ONNX, CPU)  ──►  audio/*.wav + timings.json
   │                                                                              │
   ▼                                                                              ▼
Remotion scene components (React/TSX)  ◄── timings drive scene durations ──►  render (Chromium, 720p)
   │
   ▼
lesson.mp4  ──►  SendUserFile (delivered in chat)   [mp4 not committed; script+audio manifest committed]
```

- **Narration:** Kokoro-82M via ONNX Runtime on CPU (matches your kids-pipeline stack). Phonemization via `misaki` (the model's companion) or `espeak-ng` fallback. A default narrator voice is fixed for the whole series for consistency (an American-English voice such as `af_*`/`am_*`; final pick chosen in the setup session and recorded in `video/VOICE.md`). Weights are Apache-2.0. Output is 24 kHz wav per narration segment, plus a `timings.json` mapping each segment to its duration.
- **Timing-driven durations:** Remotion reads `timings.json` so each visual scene lasts exactly as long as its narration — no manual sync. This is the single most important trick for a maintainable pipeline.
- **Rendering:** `remotion render` using the preinstalled Chromium, 1280×720 @ 30 fps, H.264. ~1–2× realtime on container CPUs; a full topic's episodes render inside one session. If a session is tight on time, episodes render one at a time and deliver incrementally.
- **Captions:** burned-in captions generated from the narration text + timings (accessibility + silent viewing), toggleable via a Remotion prop.

### 5.3 Reusable component library (`video/src/components/`)

Built once in the setup session, reused by every topic (this is why episodes stay consistent and cheap):

`TitleCard`, `ConceptScene` (big idea + supporting visual), `MathReveal` (the three-way formula reveal), `CodeWalkthrough` (line-highlighted code with a scrollytelling reveal), `DiagramScene` (animated architecture/graph diagrams — e.g., HNSW layer descent, attention heatmaps, KV paging blocks), `ChartScene` (plots: loss curves, benchmark bars, recall/latency), `ArchitectureMap` (the AlphaDesk diagram with the current topic lit up), `RecapScene`, `Callout` (gotchas/warnings), `Transition`. A shared theme file (`video/src/theme.ts`) fixes palette, type, spacing, and the dark "terminal-lecture" aesthetic across the series.

### 5.4 Per-topic video assets committed to the repo

For each topic: `video/topics/<id>/script.md` (narration + scene directions), `scenes.tsx` (the composition), `audio/` manifest + regeneration script, and `render.sh`. The mp4s themselves are delivered in chat and regenerable — keeping the repo light. (Optional: your existing YouTube Data API v3 pipeline can publish the rendered episodes; that's a bolt-on, not part of the core plan.)

### 5.5 Storyboard example — T4 Transformer, Episode 1 ("Attention, slowly")

To make the pacing concrete: E1 opens on the ArchitectureMap with "Attention" lit (0:00–0:20, recap of where we are). ConceptScene: "a token asking every other token a question" metaphor, held 20 s. MathReveal for Q/K/V: English ("each token emits a query, a key, and a value") → symbols (Q, K, V projections) → code (three `nn.Linear`s). A DiagramScene animates one query attending over 5 keys with actual softmax weights appearing as a heatmap row, narrated number by number. MathReveal for scaled dot-product with the √d_k scaling explained as "keeping the variance sane." CodeWalkthrough of the 12-line attention function, one line at a time. RecapScene: 3 bullets + "skip this and multi-head in E2 is meaningless." Total ≈ 9 minutes at ~125 wpm. This template (metaphor → three-way math → animated mechanism → code → recap) is what every flagship episode follows.

---
## 6. AlphaDesk — the capital-markets capstone

**AlphaDesk** is a fictional, paper-trading-only AI trading desk. It is not a product and never touches real orders, real money, or real brokerage systems — every screen carries an "educational simulation" disclaimer. Its job is to be the one place every one of the 51 topics plugs into, so your learning compounds into a single demoable system that mirrors your LPL/Fidelity OMS/EMS background.

### 6.1 What it does (three surfaces)

1. **Research Copilot** — ask questions over filings, transcripts, news, and market data; get cited, reasoned answers. (RAG, GraphRAG, KG, CoT, agent, text-to-SQL.)
2. **Order Workflow (paper)** — an OMS/EMS-style flow: build an order ticket in natural language, validate it against a grammar + risk rules, preview it, and "place" it into a simulated book with fills. (Function calling, CFG order-DSL, guardrails, feature store.)
3. **Compliance & Ops** — every interaction passes an input/output guardrail perimeter; a surveillance view flags anomalies; an eval harness scores quality continuously; a gateway routes model traffic. (Guardrails, adversarial hardening, eval harness, AI gateway, interpretability panel.)

### 6.2 Architecture (how the pieces connect)

```
              ┌──────────────── Guardrails perimeter (T28) + audit log ───────────────┐
   user ─►  Semantic Router (T36) ─►  ReAct Agent (T2) ─► Function Router (T24)         │
              │                          │  reasons with CoT (T1)      │                 │
              │                          ▼                             ▼                 │
              │                 Research Copilot            Tools: SQL (T40) · Sandbox    │
              │                 RAG (T6)/GraphRAG (T20)     (T17s) · Order-DSL (T25) ·     │
              │                 over Vector DB (T5) +       Recsys (T42) · KG (T37)        │
              │                 Vector Driver (T50) +                                     │
              │                 Embeddings (T43) + KG (T37)                               │
              └───────────────────────────────────────────────────────────────────────┘
   models:  AlphaSLM (T15) = Transformer (T4)+FinTok (T30), LoRA-tuned (T17), DPO/PPO-aligned (T19/T14),
            distilled (T49), quantized (T8);  served by tickerd (Rust, T3) with KV-paging (T12),
            speculative decoding, prompt caching, logit processors, CFG;  fronted by AI Gateway (T41).
   data:    Data curation (T38) → corpus; Synthetic data (T23) → instruct/preference sets;
            Feature store (T48) → leak-free market features.
   observe: Eval harness (T27) scores everything; SAE (T22) explainability panel; Adversarial (T44) red-teams.
   multimodal add-ons: voice in (ASR T33), voice out (TTS T34), chart-pattern vision (ViT T32 / CLIP T26).
```

### 6.3 AlphaDesk milestone per phase (the desk grows every phase)

| After phase | AlphaDesk can… | Topics wired in |
|---|---|---|
| 1 | tokenize financial text with FinTok; run a toy signal model on your own autograd | T30, T31 |
| 2 | generate market commentary from AlphaSLM; embed & compare filings | T4, T15, T43, (T13, T9) |
| 3 | answer cited research questions over filings + run NL analytics queries | T5, T50, T6, T36, T37, T20, T40, T48, T42 |
| 4 | act as a guarded agent: reason, call tools, run code safely, be evaluated | T1, T2, T24, T17s, T28, T27, T22, (T44) |
| 5 | run a tuned, aligned, compliant desk model you trained yourself | T38, T23, T17, T19, T14, (T39,T47,T21,T11,T46) |
| 6 | serve that model fast & cheap: Rust server, paging, spec-decode, caching, quant, gateway | T3, T12, T12s, T29, T8, T25(logit/CFG), T41 |
| 7 | prove the attention/matmul/softmax speedups on real GPU | T16, T45, T7 |
| 8 | take voice input, speak answers, read charts | T33, T34, T32, T26, (T35, T10) |
| 9 | full end-to-end demo + portfolio video | — |

### 6.4 Complete topic → coverage map (all 51 user-listed topics)

Reasoner/CoT ✓T1 · Agent/ReAct ✓T2 · Inference server (Rust) ✓T3 · Transformer ✓T4 · Vector DB/HNSW ✓T5 · RAG ✓T6 · Flash Attention ✓T7 · Quantization ✓T8 · MoE ✓T9 · Distributed training ✓T11 · KV paging ✓T12 · Speculative decoding ✓T12s · Mamba/SSM ✓T13 · RLHF/PPO ✓T14 · SLM ✓T15 · Matmul kernel ✓T16(A CPU/B GPU) · LoRA ✓T17 · Code interpreter sandbox ✓T17s · DPO ✓T19 · Graph RAG ✓T20 · Model merger ✓T21 · Interpretability/SAE ✓T22 · Synthetic data ✓T23 · Function-calling router ✓T24 · Structured output/CFG ✓T25 · Logit processor ✓T25-logit · Multimodal projector/CLIP ✓T26 · LLM eval harness ✓T27 · Guardrails ✓T28 · Prompt caching ✓T29 · Tokenizer/BPE ✓T30 · Autograd ✓T31 · ViT ✓T32 · Whisper ASR ✓T33 · TTS ✓T34 · Audio Spectrogram Transformer ✓T35 · Semantic router ✓T36 · Knowledge Graph builder ✓T37 · Data curation/MinHash ✓T38 · AI Gateway ✓T41 · PEFT ✓T39 · Text-to-SQL ✓T40 · Recsys/two-tower ✓T42 · Embedding model ✓T43 · Adversarial ✓T44 · Softmax kernel ✓T45(A/B) · Neural Architecture Search ✓T46 · Model distillation ✓T47 · Feature store ✓T48 · Diffusion ✓T10 · Vector DB driver ✓T50. **Count: 51/51 mapped.** (Softmax & matmul each = one topic with a CPU pass in Phase 1 and a GPU pass in Phase 7.)

### 6.5 Phase 9 assembly checklist

A single `make demo` spins up: gateway → tickerd serving quantized AlphaSLM → agent with all tools → guardrails → a small web UI (or CLI + a Remotion "product tour"). Deliverables: a 10-minute recorded end-to-end demo, an architecture one-pager, an `ADR/` folder of decision records, and the capstone portfolio video. Explicit non-goals restated on screen: no real orders, no real money, no market-data redistribution.

---
## 7. Repository structure & conventions

One private repo, `AISystemsLearningProgram`, generation happening on a dedicated branch. Layout (built on Day 1, filled over the run):

```
AISystemsLearningProgram/
├── index.html                  ← the navigation hub (regenerated from the ledger each run)
├── README.md                   ← quickstart + how to navigate
├── MASTER_PLAN.md              ← this document (source of truth)
├── EXECUTION/
│   ├── DAY_PLAN.md             ← the 15-day work-package breakdown
│   ├── LEDGER.md               ← machine + human readable progress (updated every run)
│   ├── DAILY_PROMPT.md         ← the exact self-contained instructions each scheduled run follows
│   └── runs/                   ← one dated log per run (what got done, what spilled)
├── common/                     ← shared Python utils, the market-data loaders, FinTok, AlphaSLM
│   ├── data/                   ← NSE/EDGAR/yfinance loaders + cached samples
│   └── alphadesk/              ← the growing capstone package
├── phases/
│   ├── p0-python-ramp/
│   ├── p1-foundations/
│   │   ├── t31-autograd/        ← src/ tests/ README.md NOTES.md steps/ (step-by-step commits)
│   │   ├── t16a-matmul-cpu/
│   │   ├── t45a-softmax-cpu/
│   │   └── t30-tokenizer/
│   ├── p2-transformers/ … p9-capstone/
│   └── (every topic = its own folder with the same shape)
├── video/
│   ├── src/components/          ← reusable Remotion component library + theme
│   ├── topics/<id>/             ← script.md, scenes.tsx, audio manifest, render.sh
│   └── VOICE.md                 ← fixed narrator voice + Kokoro settings
├── gpu-runner/                  ← scripts you run on the RTX 4070; write results.json back
├── ADR/                         ← architecture decision records
└── .gitignore                  ← excludes *.mp4, model weights, __pycache__, node_modules, data caches
```

**Per-topic folder contract** (every topic looks identical, so navigation is muscle memory): `README.md` (what/why, the paper, how to run), `NOTES.md` (the intuition + gotchas, mirrors the video script), `src/` (the implementation), `tests/` (pytest/cargo, must pass), `steps/` (the step ladder as ordered, individually-runnable checkpoints), `bench/` (verification results), and an AlphaDesk wiring note. **Conventions:** Python via `uv`, `ruff` + `pytest`; Rust as a Cargo workspace; every topic's README links back to `index.html` and forward to its video episodes and its AlphaDesk hook. `.mp4` files are gitignored and delivered in chat.

## 8. The 15-day generation schedule (what the scheduled task produces)

The 15 days are **content generation**, not your study schedule — they fill the repo so you can learn afterward at your own ~6-month pace. The cadence is a pacing/credit-spreading mechanism: a fresh Opus Cowork session fires ~3:00 AM IST each day, does the next work package, commits, and hands you the day's videos for download. If a day's package overflows a session's budget, the remainder carries into the next run automatically (the ledger tracks per-topic status), so the calendar is a target, not a guarantee — it may take 15–20 runs to fully complete. The task keeps firing daily until the ledger shows everything done, then deletes itself.

| Day | Work package (code + videos + docs) |
|---|---|
| 1 | **Foundation & infra:** repo scaffold, `.gitignore`, common/ utils, market-data loaders, Remotion component library + theme, Kokoro pipeline + VOICE.md, the `index.html` generator, Phase 0 Python-ramp content (P0.1–P0.3) |
| 2 | **P1 Foundations:** T31 autograd · T16a matmul-CPU · T45a softmax-CPU · T30 tokenizer/FinTok |
| 3 | **P2a:** T4 transformer (flagship, 5 eps) · begin T15 SLM |
| 4 | **P2b:** finish T15 SLM · T43 embeddings · T13 Mamba · T9 MoE |
| 5 | **P3a Retrieval:** T5 HNSW · T50 vector driver · T6 RAG |
| 6 | **P3b:** T36 semantic router · T37 KG builder · T20 GraphRAG |
| 7 | **P3c:** T40 text-to-SQL · T48 feature store · T42 two-tower recsys |
| 8 | **P4a Agents:** T1 CoT · T2 ReAct · T24 function router · T17s sandbox |
| 9 | **P4b Safety:** T28 guardrails · T27 eval harness · T44 adversarial · T22 SAE |
| 10 | **P5a Training:** T38 data curation · T23 synthetic data · T17 LoRA · T39 PEFT |
| 11 | **P5b Alignment:** T19 DPO · T14 RLHF/PPO · T47 distillation · T21 merger |
| 12 | **P5c + P6a:** T11 FSDP/TP · T46 NAS · T3 tickerd (Rust server) |
| 13 | **P6b Inference:** T12 KV paging · T12s speculative · T29 prompt caching · T25 CFG/logit · T8 quantization · T41 gateway |
| 14 | **P7 Kernels + P8a:** T16b matmul-GPU · T45b softmax-GPU · T7 Flash Attention · T32 ViT · T26 CLIP |
| 15 | **P8b + P9 Capstone:** T33 ASR · T34 TTS · T35 AST · T10 diffusion · AlphaDesk end-to-end assembly · portfolio video · final index.html |

*(Heavier days — 3, 11, 13, 15 — are the most likely to spill; that's expected and handled.)*

## 9. The daily-run contract (what each scheduled session does)

Every fresh Opus session follows `EXECUTION/DAILY_PROMPT.md`, which encodes this loop:

1. **Clone** the branch (token-auth), read `MASTER_PLAN.md`, `DAY_PLAN.md`, and `LEDGER.md`.
2. **Pick the work:** the earliest *incomplete* work package (so any spillover from yesterday is finished first), then today's package.
3. **Build code first (must-have):** implement each topic's step ladder with passing tests; a topic's code is "done" only when its verification test passes. Commit per step.
4. **Then videos (best-effort):** write `script.md`, build/parametrize scenes from the component library, synthesize narration with Kokoro, render at 720p. If budget runs short, code stays done and remaining renders are marked `video: pending` for the next run.
5. **GPU topics:** commit CUDA/Triton source + CPU-verified reference + `gpu-runner/` script; mark `bench: awaiting-4070` (you run it and commit `results.json`).
6. **Regenerate `index.html`** from the ledger so links/status reflect reality.
7. **Deliver:** commit locally, then `git bundle create forge-day-N.bundle main` and **SendUserFile** the bundle (you push it with one `git fetch … && git push` command the run prints) plus the day's rendered `.mp4`s. Nothing pushes from the cloud — the bundle is how code/docs reach the repo.
8. **Update `LEDGER.md`** (per-topic: code/tests/video/bench status + notes), write a `runs/<date>.md` log, push.
9. **Budget guard:** if low on time/tokens, checkpoint (commit + ledger) and stop cleanly — never leave the repo half-broken.
10. **Termination:** when the ledger shows all topics `code:done` and the capstone assembled, the run deletes the scheduled task and posts a "course complete" summary.

**Definition of done (per topic):** README + NOTES written · code implemented · tests pass · verification benchmark met · AlphaDesk hook wired · at least the E1 video scripted (render may lag) · ledger updated.

## 10. Timeline, risks, and your part

**Generation timeline:** ~15–20 daily runs (≈ 2.5–3 weeks) to fill the repo. **Your learning timeline:** ~26 weeks at 20 h/week (Section 3), or the ~16-week accelerated core path. These are independent — the course is ready long before you finish studying it.

**Risk table:**

| Risk | Likelihood | Mitigation |
|---|---|---|
| A day's package overflows one session | High | Ledger-driven catch-up; calendar is a target; code-first so nothing critical is lost |
| Video render too slow / files too big for git | Medium | 720p, one-at-a-time renders, mp4s delivered in chat not committed, regenerable from scripts |
| Cloud proxy blocks git push | Certain (by design) | Public repo for reads; each run delivers a git bundle you push with one command — no tokens involved |
| A day's videos too big for one delivery | Low | 720p, delivered one at a time; regenerable from committed scripts if a delivery is missed |
| GPU numbers can't be produced in cloud | Certain (by design) | 4070 lane: source + CPU reference in cloud, real benchmarks via gpu-runner you run |
| Public repo exposes work-in-progress | Low/accepted | It's an educational portfolio; no secrets are ever committed (.gitignore covers tokens/weights/data) |
| Scope creep / a topic balloons | Medium | Fixed per-topic folder contract + step ladder + "small scale, real behavior" rule |
| A daily run fails entirely | Low | Next run's catch-up logic resumes from the ledger; runs are independent |

**Your part (small):** (1) one-time: make the repo public and push the scaffold bundle to `main`; (2) each morning, push the delivered day-bundle with the one-line command the run prints, and skim the delivered videos; (3) for GPU topics, run the one-line `gpu-runner` command on your 4070 and commit `results.json`; (4) then, when the course is complete, start learning.

---

## 11. One-time setup (to make the daily runs work)

1. **Make the repo public** — `shail-eesh/AISystemsLearningProgram` → Settings → General → Change visibility → Public. (The cloud proxy can read a public repo but not a private one.)
2. **Push the scaffold** to `main` from the delivered bundle: `git clone ai-systems-forge.bundle repo && cd repo && git remote set-url origin https://github.com/shail-eesh/AISystemsLearningProgram.git && git push -u origin main`.
3. That's it — the task fires ~3:00 AM IST daily on `claude-opus-5`. Each run clones the public repo, builds the next work package, and hands you a day-bundle to push plus the day's videos.

*Everything in AlphaDesk is a fictional educational simulation — no real orders, money, brokerage systems, or market-data redistribution.*
