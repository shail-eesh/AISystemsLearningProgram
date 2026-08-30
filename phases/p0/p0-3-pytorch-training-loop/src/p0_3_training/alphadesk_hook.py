"""AlphaDesk wiring for P0.3.

The desk's Foundation surface gets the training-loop skeleton itself: a
factory for the model and one for the trainer. Everything from T15 (AlphaSLM)
to T17 (LoRA) reuses this shape — build a module, hand it a config, get back a
fitted model plus its history.

The registered signal model is deliberately labelled as a *demonstration of
overfitting*, not as a trading signal, and the desk never routes an order from
it.
"""

from __future__ import annotations

from typing import Any

from common.alphadesk import Surface, register

from .data import FEATURE_COLUMNS


@register(
    topic="P0.3",
    name="toy_signal_model",
    surface=Surface.FOUNDATION,
    summary=(
        "Tiny MLP over 8 causal price features — a teaching artefact that scores at "
        "chance out-of-sample; never wired to an order"
    ),
)
def build_toy_signal_model(hidden: int = 32, dropout: float = 0.1) -> Any:
    from .model import TinyMLP

    return TinyMLP(len(FEATURE_COLUMNS), hidden=hidden, dropout=dropout)


@register(
    topic="P0.3",
    name="training_loop",
    surface=Surface.FOUNDATION,
    summary="Reusable train/evaluate skeleton (seeding, batching, eval mode, checkpointing)",
)
def build_training_loop() -> dict[str, Any]:
    from .loop import TrainConfig, evaluate, seed_everything, train

    return {
        "train": train,
        "evaluate": evaluate,
        "seed_everything": seed_everything,
        "TrainConfig": TrainConfig,
    }
