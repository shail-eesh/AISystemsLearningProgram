"""Turning a logit vector into the next token — four ways, in order of nerve.

Greedy is deterministic and boring: it produces loops ("the the the") because
the argmax of a language model is not a fluent sentence, it is the safest token
at every step, forever.

Temperature, top-k and top-p all do the same structural thing — reshape the
distribution before sampling — and differ in *what* they promise:

* **temperature** ``p ∝ exp(logits / T)``: T < 1 sharpens (timid), T > 1
  flattens (unhinged). It never removes an option, only reweights.
* **top-k** keeps the k most likely tokens. A fixed budget, regardless of how
  confident the model is — which is the flaw: k=40 is far too generous when
  the model is certain, and far too mean when it is genuinely unsure.
* **top-p / nucleus** keeps the smallest set whose probability mass reaches p.
  The budget adapts to the model's confidence, which is why it won.

All four are *pure functions of logits*, so they are tested by construction
rather than by staring at generated text.
"""

from __future__ import annotations

import torch
from torch import Tensor

NEG_INF = float("-inf")


def apply_temperature(logits: Tensor, temperature: float) -> Tensor:
    if temperature < 0:
        raise ValueError("temperature must be >= 0")
    if temperature == 0:
        # The limit T -> 0 is the argmax; implement it exactly rather than
        # dividing by something near zero and hoping.
        out = torch.full_like(logits, NEG_INF)
        out.scatter_(-1, logits.argmax(-1, keepdim=True), 0.0)
        return out
    return logits / temperature


def top_k_filter(logits: Tensor, k: int) -> Tensor:
    """Mask everything outside the k largest logits (per row)."""
    if k <= 0:
        raise ValueError("k must be >= 1")
    k = min(k, logits.shape[-1])
    kth = logits.topk(k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < kth, NEG_INF)


def top_p_filter(logits: Tensor, p: float) -> Tensor:
    """Nucleus filtering: keep the smallest prefix of the sorted distribution
    whose cumulative probability is >= p.

    The off-by-one that everyone writes: the token that *crosses* the threshold
    must be kept, not dropped. Shifting the mask by one is what does that, and
    it is why ``top_p(0.0)`` still leaves exactly one token alive instead of
    zero.
    """
    if not 0.0 < p <= 1.0:
        raise ValueError("p must be in (0, 1]")
    sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
    probs = sorted_logits.softmax(-1).cumsum(-1)
    drop = probs - sorted_logits.softmax(-1) >= p    # mass *before* this token
    drop[..., 0] = False                             # always keep the argmax
    mask = torch.zeros_like(drop).scatter(-1, sorted_idx, drop)
    return logits.masked_fill(mask, NEG_INF)


def sample_next(logits: Tensor, *, temperature: float = 1.0, top_k: int | None = None,
                top_p: float | None = None, generator: torch.Generator | None = None) -> Tensor:
    """logits ``(B, vocab)`` -> token ids ``(B, 1)``.

    Order matters: temperature first (it changes the *shape* the filters see),
    then top-k, then top-p. Applying top-p before temperature would select the
    nucleus of a distribution you are about to throw away.
    """
    logits = apply_temperature(logits, temperature)
    if top_k is not None:
        logits = top_k_filter(logits, top_k)
    if top_p is not None:
        logits = top_p_filter(logits, top_p)
    probs = logits.softmax(-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)


def greedy_next(logits: Tensor) -> Tensor:
    return logits.argmax(-1, keepdim=True)


def distribution_stats(logits: Tensor) -> dict[str, float]:
    """Entropy and effective vocabulary — the two numbers that tell you whether
    a sampler is doing anything. ``perplexity`` here is per-step: exp(entropy),
    read as "how many tokens is the model effectively choosing between"."""
    p = logits.softmax(-1)
    nz = p > 0
    entropy = float(-(p[nz] * p[nz].log()).sum() / max(p.shape[0], 1))
    return {
        "entropy_nats": entropy,
        "effective_choices": float(torch.exp(torch.tensor(entropy))),
        "alive_tokens": float((p > 0).sum()) / max(p.shape[0], 1),
        "max_prob": float(p.max()),
    }
