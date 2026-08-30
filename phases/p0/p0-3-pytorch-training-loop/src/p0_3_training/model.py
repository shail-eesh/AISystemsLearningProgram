"""The model: a deliberately tiny MLP, plus the manual equivalent.

`TinyMLP` is ~1.5k parameters. That is not a limitation, it is the design: the
whole lesson of this topic is that a model with more capacity than the signal
warrants will memorise, and you want that failure to be small, fast and legible.

`ManualLinear` reimplements a logistic regression with hand-written forward and
backward passes. Running the two side by side is what makes `nn.Module` and
`autograd` stop being magic — and it is the on-ramp to T31, where you build the
autograd engine itself.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class TinyMLP(nn.Module):
    """input -> hidden -> hidden -> 1 logit.

    Returns **logits**, never probabilities: `BCEWithLogitsLoss` fuses the
    sigmoid into the loss for numerical stability. Applying `sigmoid` yourself
    and then using `BCELoss` is the classic first-week bug — mathematically
    identical, numerically worse, and it silently produces NaNs once a logit
    reaches about ±40.
    """

    def __init__(self, n_features: int, hidden: int = 32, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


class ManualLinear:
    """Logistic regression with the calculus written out by hand.

    forward:  z = xW + b,  p = sigmoid(z),  loss = BCE(p, y)
    backward: dL/dz = (p - y) / n,  dL/dW = xᵀ (dL/dz),  dL/db = sum(dL/dz)

    The `(p - y)` is not a coincidence: the sigmoid's derivative cancels
    against the log in the cross-entropy, which is *why* this pairing is the
    standard one. T31 rebuilds this cancellation as a graph.
    """

    def __init__(self, n_features: int, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 0.1, size=(n_features, 1))
        self.b = np.zeros((1,))

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        # The stable form: exp() of a large positive number overflows.
        out = np.empty_like(z)
        pos = z >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        ez = np.exp(z[~pos])
        out[~pos] = ez / (1.0 + ez)
        return out

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self._sigmoid(x @ self.W + self.b)

    def loss(self, p: np.ndarray, y: np.ndarray) -> float:
        eps = 1e-12
        return float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))

    def step(self, x: np.ndarray, y: np.ndarray, lr: float) -> float:
        p = self.forward(x)
        n = x.shape[0]
        dz = (p - y) / n
        self.W -= lr * (x.T @ dz)
        self.b -= lr * dz.sum(axis=0)
        return self.loss(p, y)
