"""AlphaDesk wiring for T15.

AlphaSLM is *the desk's local model* — the thing every later topic modifies
rather than replaces: LoRA-tuned (T17), DPO-aligned (T19), distilled (T47),
quantized (T8), served by tickerd (T3), and evaluated by the harness (T27).
Registering it here is what makes that chain visible in ``Registry.describe()``.

Three components:

* ``models.alphaslm`` — a factory taking a rung name; builds the architecture
  and loads a checkpoint if one is present, otherwise returns the untrained
  model rather than raising. A desk that boots without weights is a desk you can
  still demo.
* ``data.pretrain_shards`` — the packed corpus, so T38 (curation) and T23
  (synthetic data) have something concrete to improve.
* ``models.alphaslm_eval`` — held-out perplexity by register.

AlphaDesk is a fictional educational simulation. Generated commentary is a
language-model artefact about fictional issuers; it is never a market view,
never advice, and never routed to an order.
"""

from __future__ import annotations

import pathlib
from typing import Any

from common.alphadesk import Surface, register

CHECKPOINTS = pathlib.Path(__file__).resolve().parent / "artifacts" / "checkpoints"


@register(
    topic="T15",
    name="alphaslm",
    surface=Surface.MODELS,
    summary="AlphaSLM — the desk's own pretrained LM (T4 architecture + FinTok, size ladder)",
    requires=("T4", "T30"),
)
def build_alphaslm(rung: str = "alphaslm-1.8m", *, checkpoint: str | None = None,
                   vocab_size: int = 3495) -> Any:
    """The desk's local model. Loads weights when they exist, never demands them."""
    import torch

    from .config import LADDER
    from .harness import Trainer  # noqa: F401  (imported for the module's side effects)

    if rung not in LADDER:
        raise KeyError(f"unknown rung {rung!r}; have {sorted(LADDER)}")
    import sys
    from pathlib import Path

    t4 = Path(__file__).resolve().parents[5] / "phases/p2/t4-transformer/src"
    if str(t4) not in sys.path:
        sys.path.insert(0, str(t4))
    from t4_transformer import GPT

    model = GPT(LADDER[rung].gpt_config(vocab_size))
    path = pathlib.Path(checkpoint) if checkpoint else CHECKPOINTS / f"{rung}.pt"
    if path.exists():
        state = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"] if "model" in state else state)
    return model


@register(
    topic="T15",
    name="pretrain_shards",
    surface=Surface.DATA,
    summary="Packed uint16 FinTok shards of the AlphaSLM corpus (train/val, split by document)",
)
def build_pretrain_shards(block_size: int = 128) -> Any:
    from .shards import ensure_shards

    train, val, meta = ensure_shards(block_size=block_size)
    return {"train": train, "val": val, "meta": meta}


@register(
    topic="T15",
    name="alphaslm_eval",
    surface=Surface.MODELS,
    summary="Held-out perplexity by register (filing / commentary / announcement / tape)",
)
def build_alphaslm_eval() -> dict[str, Any]:
    from .evaluate import compare_models, perplexity_by_tag, perplexity_on_documents

    return {
        "perplexity_on_documents": perplexity_on_documents,
        "perplexity_by_tag": perplexity_by_tag,
        "compare_models": compare_models,
    }
