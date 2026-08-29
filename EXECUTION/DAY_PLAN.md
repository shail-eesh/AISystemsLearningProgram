# 15-Day Generation Schedule

The 15 days are **content generation** (filling this repo), not your study schedule. A fresh Claude
Opus Cowork session fires ~3:00 AM IST daily, does the next work package, commits code/docs, and
delivers the day's rendered videos in chat. Overflow carries to the next run automatically via the
[ledger](LEDGER.md); the calendar is a target (expect ~15–20 runs). The task self-deletes when the
ledger shows everything complete.

Total topics: **52** · Total planned episodes: **168**.

### Day 1 — Foundation & infra
repo scaffold · common utils + data loaders · Remotion components + theme · Kokoro pipeline · index.html generator · P0.1 Python for the .NET veteran · P0.2 NumPy as your new LINQ · P0.3 PyTorch tensors & the training loop

_Planned video episodes: 3_

### Day 2 — P1 Foundations
**T31** Autograd engine (micrograd-style) · **T16A** Matrix multiplication kernel — CPU · **T45A** Softmax & online softmax — CPU · **T30** Tokenizer (BPE) → FinTok

_Planned video episodes: 10_

### Day 3 — P2a Transformer + SLM start
**T4** Transformer from scratch (Attention Is All You Need) · **T15** Small Language Model (AlphaSLM-40M)

_Planned video episodes: 9_

### Day 4 — P2b SLM/embeddings/Mamba/MoE
**T43** Embedding model (contrastive) · **T13** State Space Model (Mamba) · **T9** Mixture of Experts routing layer

_Planned video episodes: 9_

### Day 5 — P3a HNSW/driver/RAG
**T5** Vector database (HNSW index) · **T50** Database driver for vectors · **T6** RAG pipeline

_Planned video episodes: 10_

### Day 6 — P3b router/KG/GraphRAG
**T36** Semantic router · **T37** Knowledge Graph builder · **T20** Graph RAG system

_Planned video episodes: 8_

### Day 7 — P3c text-to-SQL/feature-store/recsys
**T40** Text-to-SQL engine · **T48** Feature store (point-in-time) · **T42** Recommendation system (two-tower)

_Planned video episodes: 8_

### Day 8 — P4a CoT/ReAct/router/sandbox
**T1** Reasoner (Chain-of-Thought) · **T2** Agent loop (ReAct) · **T24** Function-calling router · **T17S** Code interpreter sandbox

_Planned video episodes: 12_

### Day 9 — P4b guardrails/eval/adversarial/SAE
**T28** Guardrails (input/output filtering) · **T27** LLM eval harness · **T44** Adversarial attack generator · **T22** Interpretability tool (Sparse Autoencoders)

_Planned video episodes: 12_

### Day 10 — P5a curation/synthetic/LoRA/PEFT
**T38** Data curation (MinHash/dedup) · **T23** Synthetic data generator · **T17** LoRA trainer · **T39** PEFT library (QLoRA/prefix/IA3)

_Planned video episodes: 12_

### Day 11 — P5b DPO/PPO/distill/merge
**T19** DPO loss function · **T14** RLHF pipeline (PPO) · **T47** Model distillation pipeline · **T21** Model merger (Soups/SLERP/TIES)

_Planned video episodes: 13_

### Day 12 — P5c FSDP/NAS + P6a Rust server
**T11** Distributed training (FSDP/Tensor Parallel) · **T46** Neural Architecture Search · **T3** Inference server (Rust, tickerd)

_Planned video episodes: 12_

### Day 13 — P6b inference systems (heavy)
**T12** KV-cache paging (like vLLM) · **T12S** Speculative decoding · **T29** Prompt caching (RadixAttention) · **T25** Structured output (CFG) + logit processors · **T8** Quantization library (Int8/FP4) · **T41** AI Gateway (load balancing/failover)

_Planned video episodes: 19_

### Day 14 — P7 GPU kernels + P8a ViT/CLIP
**T16B** Matmul kernel — GPU (Triton/CUDA) · **T45B** Softmax kernel — GPU (fused) · **T7** Flash Attention kernel (CUDA/Triton) · **T32** Vision Transformer (ViT) · **T26** Multimodal projector (CLIP)

_Planned video episodes: 16_

### Day 15 — P8b audio/diffusion + P9 capstone
**T33** Whisper-style ASR model · **T34** Text-to-Speech pipeline · **T35** Audio Spectrogram Transformer · **T10** Diffusion model (UNet + scheduler)

_Planned video episodes: 15_

Heaviest days (most likely to spill): **13, 14, 15, 11**. That is expected and handled — code is
always committed first; remaining video renders are marked `Video: pending` for the next run.
