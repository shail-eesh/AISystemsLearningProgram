"""T45A · softmax numerics: overflow, the shift identity, and online softmax.

    from t45a_softmax import online_softmax, log_softmax, cross_entropy

The GPU half of this topic is T45B (Phase 7), which fuses the same algorithm
into a single kernel. Everything here is CPU/NumPy so the *numerics* can be
studied without a launch configuration in the way.
"""

from .softmax import (
    SoftmaxState,
    cross_entropy,
    cross_entropy_grad,
    log_softmax,
    logsumexp,
    naive_softmax,
    online_normalizer,
    online_softmax,
    stable_softmax,
    two_pass_softmax,
)

__all__ = [
    "SoftmaxState", "cross_entropy", "cross_entropy_grad", "log_softmax",
    "logsumexp", "naive_softmax", "online_normalizer", "online_softmax",
    "stable_softmax", "two_pass_softmax",
]
