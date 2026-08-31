"""The AlphaSLM size ladder.

Five configurations on one axis, so the scaling study compares models that
differ in *size* and nothing else: same tokenizer, same corpus, same context,
same optimiser, same schedule shape. Width and depth grow together, roughly as
``d ~ 64 * sqrt(L)``, which is the shape real model families use — growing depth
alone gives you a slow model, growing width alone gives you a shallow one.

The three CPU-scale rungs run in this sandbox in minutes. ``alphaslm-40m`` is the
4070 rung: it is defined here, its parameter count is asserted by a test, and it
is trained by ``gpu-runner/t15_alphaslm_40m.py``. Nothing about the code changes
between them — that is the point of writing the ladder as data.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
T4_SRC = REPO / "phases" / "p2" / "t4-transformer" / "src"


def _gpt_config_cls():
    """T4's ``GPTConfig``. AlphaSLM is T4's architecture — not a copy of it."""
    if str(T4_SRC) not in sys.path:
        sys.path.insert(0, str(T4_SRC))
    from t4_transformer import GPTConfig

    return GPTConfig


@dataclass(frozen=True)
class Rung:
    name: str
    n_layer: int
    n_head: int
    n_embd: int
    block_size: int
    device: str          # "cpu" (runs here) or "cuda" (the 4070 lane)

    def gpt_config(self, vocab_size: int):
        cls = _gpt_config_cls()
        return cls(vocab_size=vocab_size, block_size=self.block_size,
                   n_layer=self.n_layer, n_head=self.n_head, n_embd=self.n_embd,
                   position="rope", norm="rmsnorm", dropout=0.0, bias=False,
                   tie_weights=True)

    def parameters(self, vocab_size: int) -> int:
        """Closed-form count, so the ladder can be checked without building it.

        Embedding (tied, so counted once) + per layer: qkv (3d^2), attention
        projection (d^2), MLP (8d^2), two RMSNorm gains (2d); plus the final
        norm. No biases — ``bias=False`` throughout, LLaMA-style.
        """
        d, layer = self.n_embd, self.n_layer
        embedding = vocab_size * d
        per_layer = 4 * d * d + 8 * d * d + 2 * d
        return embedding + layer * per_layer + d

    def chinchilla_tokens(self, vocab_size: int = 3495) -> int:
        """The Chinchilla rule of thumb: ~20 training tokens per parameter.

        Stated here so the scaling study can say out loud how far under it we
        are. Every CPU rung is trained on a fraction of this — the corpus is
        750k tokens — which means the study measures models in the
        *data-limited* regime, not the compute-optimal one. That is a real
        caveat, not a footnote, and it is why the study reports tokens-per-
        parameter alongside every loss.
        """
        return 20 * self.parameters(vocab_size)


LADDER: dict[str, Rung] = {
    "alphaslm-0.6m": Rung("alphaslm-0.6m", n_layer=2, n_head=4, n_embd=96,
                          block_size=128, device="cpu"),
    "alphaslm-1.8m": Rung("alphaslm-1.8m", n_layer=4, n_head=4, n_embd=160,
                          block_size=128, device="cpu"),
    "alphaslm-5m": Rung("alphaslm-5m", n_layer=6, n_head=6, n_embd=264,
                        block_size=128, device="cpu"),
    "alphaslm-15m": Rung("alphaslm-15m", n_layer=8, n_head=8, n_embd=384,
                         block_size=256, device="cuda"),
    "alphaslm-40m": Rung("alphaslm-40m", n_layer=12, n_head=8, n_embd=512,
                         block_size=512, device="cuda"),
}

CPU_RUNGS = [r for r in LADDER.values() if r.device == "cpu"]
GPU_RUNGS = [r for r in LADDER.values() if r.device == "cuda"]


def describe_ladder(vocab_size: int = 3495) -> str:
    lines = [f"{'rung':<16} {'L':>3} {'H':>3} {'d':>5} {'ctx':>5} {'params':>12} "
             f"{'chinchilla tokens':>18} {'lane':>6}"]
    for r in LADDER.values():
        lines.append(f"{r.name:<16} {r.n_layer:>3} {r.n_head:>3} {r.n_embd:>5} "
                     f"{r.block_size:>5} {r.parameters(vocab_size):>12,} "
                     f"{r.chinchilla_tokens(vocab_size):>18,} {r.device:>6}")
    return "\n".join(lines)
