"""Looking inside: attention statistics, and the induction-head detector.

An attention map is a ``(T, T)`` row-stochastic matrix per head. Untrained, the
rows are near-uniform over the past — every head is a running average. Trained,
distinct *patterns* appear, and three of them are so common they have names:

* **previous-token head** — attends to ``i-1``. Copies the last token's
  identity into the current position's residual stream.
* **positional / diagonal head** — attends to a fixed short offset.
* **induction head** — at position ``i``, finds an earlier occurrence of the
  current token and attends to *the token after it*. This implements "the
  pattern AB happened before, and I have just seen A, so B is next."

Induction heads are the reason a transformer can do in-context learning at all,
and they emerge on their own from ordinary next-token training — you can watch
it happen as a sudden bend in the loss curve on a repeated-sequence task
(Olsson et al. 2022, "In-context Learning and Induction Heads").

``induction_scores`` measures the third pattern directly: on a sequence built as
``prefix + prefix``, the induction target for position ``i`` in the second half
is exactly position ``i - half + 1``, so the score is the attention weight
placed there, averaged over positions and batch. Chance level is roughly
``1 / (number of visible positions)``.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .data import repeated_sequence_batch
from .model import GPT


def attention_summary(attn: Tensor) -> dict[str, float]:
    """Per-head diagnostics for one ``(B, H, T, T)`` map, averaged over batch.

    ``entropy`` is in nats and compared against the entropy of the uniform
    distribution over the *visible* positions — a head at that value is doing
    nothing but averaging.
    """
    b, h, t, _ = attn.shape
    idx = torch.arange(t)
    visible = (idx + 1).float()                       # causal: row i sees i+1 slots
    p = attn.clamp_min(1e-12)
    ent = -(attn * p.log()).sum(-1)                   # (B,H,T)
    uniform_ent = visible.log().to(ent.dtype)
    prev = attn[..., idx[1:], idx[:-1]].mean().item() if t > 1 else float("nan")
    self_w = attn[..., idx, idx].mean().item()
    return {
        "mean_entropy_nats": float(ent.mean()),
        "uniform_entropy_nats": float(uniform_ent.mean()),
        "entropy_ratio": float(ent.mean() / uniform_ent.mean().clamp_min(1e-9)),
        "prev_token_weight": float(prev),
        "self_weight": float(self_w),
        "max_weight": float(attn.max()),
    }


def induction_scores(model: GPT, *, half: int = 24, batch: int = 32,
                     seed: int = 0) -> dict[str, object]:
    """Attention mass on the induction target, per layer and head.

    Returns per-head scores plus the chance baseline, plus the best head, plus
    the model's own *behavioural* score: how often it predicts the repeated
    token correctly in the second half.
    """
    gen = torch.Generator().manual_seed(seed)
    vocab = model.config.vocab_size
    seq = repeated_sequence_batch(batch, half, vocab, generator=gen)
    seq = seq[:, : model.config.block_size]
    t = seq.shape[1]
    maps = model.attention_maps(seq)

    positions = [i for i in range(half, t) if 0 <= i - half + 1 <= i]
    targets = [i - half + 1 for i in positions]
    chance = float(sum(1.0 / (i + 1) for i in positions) / max(len(positions), 1))

    per_head: list[dict] = []
    for layer, attn in enumerate(maps):
        for head in range(attn.shape[1]):
            weights = attn[:, head]                    # (B,T,T)
            score = float(weights[:, positions, targets].mean())
            per_head.append({
                "layer": layer, "head": head, "score": score,
                "over_chance": score / chance if chance else float("nan"),
            })
    best = max(per_head, key=lambda d: d["score"])

    with torch.no_grad():
        logits, _ = model(seq[:, :-1])
    pred = logits.argmax(-1)
    tgt = seq[:, 1:]
    second = slice(half - 1, None)
    accuracy = float((pred[:, second] == tgt[:, second]).float().mean())

    return {
        "chance": chance,
        "per_head": per_head,
        "best": best,
        "second_half_accuracy": accuracy,
        "sequence_length": t,
        "half": half,
    }


def train_induction_model(*, n_layer: int = 2, n_head: int = 4, n_embd: int = 64,
                          vocab: int = 32, half: int = 24, steps: int = 600,
                          batch: int = 32, lr: float = 3e-3, seed: int = 0,
                          record_every: int = 25) -> tuple[GPT, list[dict]]:
    """Train a tiny model on ``prefix+prefix`` and record when induction appears.

    This is the smallest experiment that shows a *phase change*: loss sits flat
    near chance for a while (the model is learning unigram statistics, which do
    not help), then falls off a cliff as the induction circuit forms. The
    recorded trace is what Episode 4's chart animates.
    """
    from .config import GPTConfig
    from .train import TrainConfig  # noqa: F401  (kept for symmetry with T15)

    torch.manual_seed(seed)
    cfg = GPTConfig(vocab_size=vocab, block_size=2 * half, n_layer=n_layer,
                    n_head=n_head, n_embd=n_embd, position="learned", dropout=0.0)
    model = GPT(cfg)
    opt = model.configure_optimizers(lr=lr, weight_decay=0.0)
    gen = torch.Generator().manual_seed(seed + 1)
    trace: list[dict] = []
    model.train()
    for step in range(steps):
        seq = repeated_sequence_batch(batch, half, vocab, generator=gen)
        _, loss = model(seq[:, :-1], seq[:, 1:])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % record_every == 0 or step == steps - 1:
            probe = induction_scores(model, half=half, batch=16, seed=99)
            trace.append({
                "step": step,
                "loss": float(loss.detach()),
                "best_head_score": probe["best"]["score"],
                "over_chance": probe["best"]["over_chance"],
                "second_half_accuracy": probe["second_half_accuracy"],
            })
            model.train()
    return model, trace
