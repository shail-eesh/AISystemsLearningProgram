"""AlphaDesk wiring for T45A.

No product surface — the capsule says so plainly, and it is right: nothing on
the desk calls a softmax directly. What the desk gets is the numerics library
every later model depends on: the attention block in T4, the cross-entropy loss
in T15, the logit processors in T25, and the fused GPU kernel in T45B all use
these exact routines or a direct descendant of them.

Registering it makes that dependency visible in `Registry.describe()` instead
of implicit in five separate files.

AlphaDesk is a fictional educational simulation — no orders, money or venues.
"""

from __future__ import annotations

from typing import Any

from common.alphadesk import Surface, register


@register(
    topic="T45A",
    name="softmax_numerics",
    surface=Surface.FOUNDATION,
    summary=(
        "Numerically safe softmax family: stable, two-pass fused, one-pass online "
        "(the Flash-Attention primitive), log-softmax and fused cross-entropy"
    ),
)
def build_softmax_numerics() -> dict[str, Any]:
    from . import (
        cross_entropy,
        cross_entropy_grad,
        log_softmax,
        logsumexp,
        online_normalizer,
        online_softmax,
        stable_softmax,
        two_pass_softmax,
    )

    return {
        "softmax": stable_softmax,
        "two_pass_softmax": two_pass_softmax,
        "online_softmax": online_softmax,
        "online_normalizer": online_normalizer,
        "logsumexp": logsumexp,
        "log_softmax": log_softmax,
        "cross_entropy": cross_entropy,
        "cross_entropy_grad": cross_entropy_grad,
    }
