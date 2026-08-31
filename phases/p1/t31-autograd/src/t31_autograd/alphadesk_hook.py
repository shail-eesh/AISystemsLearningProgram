"""AlphaDesk wiring for T31.

The desk's Foundation surface gains two things: the differentiation engine
itself (so later topics can point at it and say "this is what `.backward()`
does"), and the *first model AlphaDesk trains without PyTorch* — a tiny MLP
over the same causal price features P0.3 used, fitted entirely by this engine.

The honest caveat from P0.3 carries forward unchanged: this model memorises its
training window and scores at chance out of sample. It is registered as a
teaching artefact. AlphaDesk is a fictional educational simulation; nothing
here is wired to an order, and no real money or brokerage exists anywhere in
this repository.
"""

from __future__ import annotations

from typing import Any

from common.alphadesk import Surface, register


@register(
    topic="T31",
    name="autograd",
    surface=Surface.FOUNDATION,
    summary="Reverse-mode autodiff (scalar Value + NumPy Tensor) with gradcheck",
)
def build_autograd() -> dict[str, Any]:
    from . import Tensor, Value, gradcheck_tensors, gradcheck_values

    return {
        "Value": Value,
        "Tensor": Tensor,
        "gradcheck_tensors": gradcheck_tensors,
        "gradcheck_values": gradcheck_values,
    }


@register(
    topic="T31",
    name="handrolled_signal_model",
    surface=Surface.FOUNDATION,
    summary=(
        "Toy up/down MLP trained by the hand-rolled engine -- a demonstration that "
        "the engine learns, not a trading signal; never routed to an order"
    ),
    requires=("P0.3",),
)
def build_handrolled_signal_model(hidden: int = 16, seed: int = 0) -> Any:
    from .train import SignalTrainer

    return SignalTrainer(hidden=hidden, seed=seed)
