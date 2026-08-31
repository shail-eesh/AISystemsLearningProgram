"""AlphaDesk wiring for T16A.

The capsule is honest that this topic has no product hook: nothing on the desk
calls a hand-written matmul, and nothing should — AlphaDesk uses NumPy/BLAS
like any sane system. What the desk *does* get is the literacy artefact: the
kernels and their measured throughput, registered on the foundation surface so
the Phase 9 capstone can print "this is the hardware floor everything else runs
on", and so Phase 7's GPU pass has a CPU baseline to beat.

AlphaDesk is a fictional educational simulation. Nothing here touches money,
orders, or a real venue.
"""

from __future__ import annotations

from typing import Any

from common.alphadesk import Surface, register


@register(
    topic="T16A",
    name="cpu_matmul_kernels",
    surface=Surface.FOUNDATION,
    summary=(
        "Five CPU matmul kernels (naive -> loop-order -> blocked -> threaded -> "
        "register-tiled) with FLOP accounting; the baseline the Phase-7 GPU pass beats"
    ),
)
def build_cpu_matmul() -> dict[str, Any]:
    from . import arithmetic_intensity, environment, flops, matmul_numpy
    from .native import ALL_KERNELS, AVAILABLE

    def call(name: str, A, B, **kw):
        if not AVAILABLE:
            raise RuntimeError("no C compiler available in this environment")
        from .native import kernels

        return kernels().call(name, A, B, **kw)

    return {
        "kernel_names": ALL_KERNELS,
        "available": AVAILABLE,
        "call": call,
        "reference": matmul_numpy,
        "flops": flops,
        "arithmetic_intensity": arithmetic_intensity,
        "environment": environment,
    }
