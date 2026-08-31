"""T16A · matrix multiplication on a CPU, from the triple loop to the roofline.

    from t16a_matmul import kernels, matmul_numpy, arithmetic_intensity

`reference.py` holds the Python implementations and the FLOP accounting;
`kernels.c` holds the five C kernels; `native.py` compiles and binds them.
"""

from .native import (
    ALL_KERNELS,
    AVAILABLE,
    BLOCKED_KERNELS,
    PLAIN_KERNELS,
    Kernels,
    environment,
    kernels,
)
from .reference import (
    Timing,
    arithmetic_intensity,
    bytes_touched,
    flops,
    matmul_numpy,
    matmul_python_naive,
    random_pair,
    time_call,
)

__all__ = [
    "ALL_KERNELS", "AVAILABLE", "BLOCKED_KERNELS", "Kernels", "PLAIN_KERNELS",
    "Timing", "arithmetic_intensity", "bytes_touched", "environment", "flops",
    "kernels", "matmul_numpy", "matmul_python_naive", "random_pair", "time_call",
]
