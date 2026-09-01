# Progress Ledger  ·  9/55 code-complete (16%)

Regenerated from `EXECUTION/status.json` by `scripts/gen_index.py`. Legend: ✅ done · 🟡 in progress/pending · 🖥️ awaiting-4070 · ⏭️ deferred · ⬜ scheduled.
A run picks the earliest row whose **Code** isn't ✅, finishes it, continues down. Video may lag (best-effort).

| Day | Phase | ID | Topic | Code | Tests | Bench | Video | Wired | Notes |
|----:|:------|:---|:------|:----:|:-----:|:-----:|:-----:|:-----:|:------|
| 1 | P0 | [P0.1](phases/p0/p0-1-python-for-dotnet/) | Python for the .NET veteran | ✅ | ✅ | ✅ | ✅ | ✅ | OMS domain model · 63 tests · tape replay reconciles to 3.1e-07 · E1 15m47s |
| 1 | P0 | [P0.2](phases/p0/p0-2-numpy-as-linq/) | NumPy as your new LINQ | ✅ | ✅ | ✅ | ✅ | ✅ | Vectorised indicators · 116 tests · 70/70 parity, worst 4.7e-14, 72x over the loop · E1 14m27s |
| 1 | P0 | [P0.3](phases/p0/p0-3-pytorch-training-loop/) | PyTorch tensors & the training loop | ✅ | ✅ | ✅ | ✅ | ✅ | Training loop · 27 tests · memorises 64 rows, honest split at chance (z=+0.30) · E1 13m58s |
| 2 | P1 | [T31](phases/p1/t31-autograd/) | Autograd engine (micrograd-style) | ✅ | ✅ | ✅ | ✅ | ✅ | Scalar Value + NumPy Tensor autodiff · 61 tests · gradcheck 3.6e-10 · loss curve matches a hand-derived reference to 6e-17 · E1 11m05s · E2 8m10s · E3 10m42s |
| 2 | P1 | [T16A](phases/p1/t16a-matmul-cpu/) | Matrix multiplication kernel — CPU | ✅ | ✅ | ✅ | ✅ | ✅ | 5 CPU matmul kernels via ctypes · 36 tests · blocking alone 22.6x over naive C, best 35.6x, 14% of OpenBLAS · E1 9m05s · E2 9m42s |
| 2 | P1 | [T45A](phases/p1/t45a-softmax-cpu/) | Softmax & online softmax — CPU | ✅ | ✅ | ✅ | ✅ | ✅ | naive/stable/two-pass/online softmax + fused CE · 49 tests · online bit-identical to two-pass in f64/f32/f16 · E1 7m28s · E2 11m14s |
| 2 | P1 | [T30](phases/p1/t30-tokenizer-bpe/) | Tokenizer (BPE) → FinTok | ✅ | ✅ | ✅ | ✅ | ✅ | Byte-level BPE + FinTok-3.5k · 54 tests · 100k-string round-trip fuzz clean · 2.73x over a matched general vocab · E1 8m28s · E2 6m52s · E3 10m39s |
| 3 | P2 | [T4](phases/p2/t4-transformer/) | Transformer from scratch (Attention Is All You Need) | ✅ | ✅ | ✅ | ✅ | ✅ | Decoder-only GPT (RoPE/learned/sinusoidal positions, RMS/LayerNorm, tied head, KV cache) · 121 tests · logit parity 3.8e-07 vs an independent reference · loss curve 0.17x the seed-noise band · 100% well-formed generation · induction heads 9.1x chance with a depth control (94% vs 65%) · cache 3.19x and token-identical · E1 10m41s · E2 10m27s · E3 12m04s · E4 9m26s · E5 10m23s |
| 3 | P2 | [T15](phases/p2/t15-slm/) | Small Language Model (AlphaSLM-40M) | ✅ | ✅ | 🖥️ | ✅ | ✅ | AlphaSLM: tagged 12MB corpus, FinTok-packed uint16 shards, restartable harness · 68 tests · CPU bench 6/6 (resume bit-exact 0.0; 3-rung scaling ordering holds, power law checked on a held-out rung at +0.33%; 1.014x held-out filing perplexity, explained by a 19.5%/3.56-nat digit floor) · the 15M and 40M rungs need the 4070: gpu-runner/t15_alphaslm_40m.py · E1 8m25s · E2 8m00s · E3 8m43s · E4 9m06s |
| 4 | P2 | [T43](phases/p2/t43-embedding-model/) | Embedding model (contrastive) | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |  |
| 4 | P2 | [T13](phases/p2/t13-mamba-ssm/) | State Space Model (Mamba) | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |  |
| 4 | P2 | [T9](phases/p2/t9-moe/) | Mixture of Experts routing layer | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |  |
| 5 | P3 | [T5](phases/p3/t5-hnsw-vector-db/) | Vector database (HNSW index) | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |  |
| 5 | P3 | [T50](phases/p3/t50-vector-driver/) | Database driver for vectors | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |  |
| 5 | P3 | [T6](phases/p3/t6-rag-pipeline/) | RAG pipeline | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |  |
| 6 | P3 | [T36](phases/p3/t36-semantic-router/) | Semantic router | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |  |
| 6 | P3 | [T37](phases/p3/t37-knowledge-graph/) | Knowledge Graph builder | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |  |
| 6 | P3 | [T20](phases/p3/t20-graph-rag/) | Graph RAG system | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |  |
| 7 | P3 | [T40](phases/p3/t40-text-to-sql/) | Text-to-SQL engine | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |  |
| 7 | P3 | [T48](phases/p3/t48-feature-store/) | Feature store (point-in-time) | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |  |
| 7 | P3 | [T42](phases/p3/t42-two-tower-recsys/) | Recommendation system (two-tower) | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |  |
| 8 | P4 | [T1](phases/p4/t1-cot-reasoner/) | Reasoner (Chain-of-Thought) | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |  |
| 8 | P4 | [T2](phases/p4/t2-react-agent/) | Agent loop (ReAct) | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |  |
| 8 | P4 | [T24](phases/p4/t24-function-router/) | Function-calling router | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |  |
| 8 | P4 | [T17S](phases/p4/t17s-code-sandbox/) | Code interpreter sandbox | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |  |
| 9 | P4 | [T28](phases/p4/t28-guardrails/) | Guardrails (input/output filtering) | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |  |
| 9 | P4 | [T27](phases/p4/t27-eval-harness/) | LLM eval harness | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |  |
| 9 | P4 | [T44](phases/p4/t44-adversarial/) | Adversarial attack generator | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |  |
| 9 | P4 | [T22](phases/p4/t22-sae-interpretability/) | Interpretability tool (Sparse Autoencoders) | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |  |
| 10 | P5 | [T38](phases/p5/t38-data-curation/) | Data curation (MinHash/dedup) | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |  |
| 10 | P5 | [T23](phases/p5/t23-synthetic-data/) | Synthetic data generator | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |  |
| 10 | P5 | [T17](phases/p5/t17-lora-trainer/) | LoRA trainer | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |  |
| 10 | P5 | [T39](phases/p5/t39-peft-library/) | PEFT library (QLoRA/prefix/IA3) | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |  |
| 11 | P5 | [T19](phases/p5/t19-dpo/) | DPO loss function | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |  |
| 11 | P5 | [T14](phases/p5/t14-rlhf-ppo/) | RLHF pipeline (PPO) | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |  |
| 11 | P5 | [T47](phases/p5/t47-distillation/) | Model distillation pipeline | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |  |
| 11 | P5 | [T21](phases/p5/t21-model-merger/) | Model merger (Soups/SLERP/TIES) | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |  |
| 12 | P5 | [T11](phases/p5/t11-distributed-training/) | Distributed training (FSDP/Tensor Parallel) | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |  |
| 12 | P5 | [T46](phases/p5/t46-nas/) | Neural Architecture Search | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |  |
| 12 | P6 | [T3](phases/p6/t3-rust-inference-server/) | Inference server (Rust, tickerd) | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |  |
| 13 | P6 | [T12](phases/p6/t12-kv-paging/) | KV-cache paging (like vLLM) | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |  |
| 13 | P6 | [T12S](phases/p6/t12s-speculative-decoding/) | Speculative decoding | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |  |
| 13 | P6 | [T29](phases/p6/t29-prompt-caching/) | Prompt caching (RadixAttention) | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |  |
| 13 | P6 | [T25](phases/p6/t25-cfg-structured-output/) | Structured output (CFG) + logit processors | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |  |
| 13 | P6 | [T8](phases/p6/t8-quantization/) | Quantization library (Int8/FP4) | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |  |
| 13 | P6 | [T41](phases/p6/t41-ai-gateway/) | AI Gateway (load balancing/failover) | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |  |
| 14 | P7 | [T16B](phases/p7/t16b-matmul-gpu/) | Matmul kernel — GPU (Triton/CUDA) | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |  |
| 14 | P7 | [T45B](phases/p7/t45b-softmax-gpu/) | Softmax kernel — GPU (fused) | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |  |
| 14 | P7 | [T7](phases/p7/t7-flash-attention/) | Flash Attention kernel (CUDA/Triton) | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |  |
| 14 | P8 | [T32](phases/p8/t32-vit/) | Vision Transformer (ViT) | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |  |
| 14 | P8 | [T26](phases/p8/t26-clip-projector/) | Multimodal projector (CLIP) | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |  |
| 15 | P8 | [T33](phases/p8/t33-whisper-asr/) | Whisper-style ASR model | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |  |
| 15 | P8 | [T34](phases/p8/t34-tts-pipeline/) | Text-to-Speech pipeline | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |  |
| 15 | P8 | [T35](phases/p8/t35-audio-spectrogram-transformer/) | Audio Spectrogram Transformer | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |  |
| 15 | P8 | [T10](phases/p8/t10-diffusion/) | Diffusion model (UNet + scheduler) | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |  |

_Last updated: 2026-09-01._
