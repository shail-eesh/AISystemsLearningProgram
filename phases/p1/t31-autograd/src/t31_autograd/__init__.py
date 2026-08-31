"""T31 · a reverse-mode autodiff engine, scalars first, then tensors.

    from t31_autograd import Value, Tensor, MLP, Adam, gradcheck_tensors

Read in this order: `engine.py` (the idea, on scalars), `tensor.py` (the same
idea on ndarrays, plus broadcasting), `nn.py` and `optim.py` (the library that
falls out for free), `gradcheck.py` (why you believe any of it).
"""

from .engine import Value, draw_ascii
from .gradcheck import GradCheckResult, gradcheck_tensors, gradcheck_values, rel_error
from .nn import (
    MLP,
    Linear,
    Module,
    ReLU,
    Sequential,
    Tanh,
    bce_with_logits,
    mse_loss,
    softmax_cross_entropy,
)
from .optim import SGD, Adam, Optimizer
from .tensor import Tensor, randn, tensor, unbroadcast, zeros

__all__ = [
    "MLP", "SGD", "Adam", "GradCheckResult", "Linear", "Module", "Optimizer",
    "ReLU", "Sequential", "Tanh", "Tensor", "Value", "bce_with_logits",
    "draw_ascii", "gradcheck_tensors", "gradcheck_values", "mse_loss", "randn",
    "rel_error", "softmax_cross_entropy", "tensor", "unbroadcast", "zeros",
]
