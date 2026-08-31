"""Step 4b: the two optimisers you actually need.

SGD with momentum is a ball rolling downhill with friction; Adam is that ball
with a per-parameter learning rate derived from the recent gradient magnitude.
Adam's bias correction matters more than it looks: the running averages start
at zero, so without dividing by `1 - beta^t` the first ~1/(1-beta) steps are
badly under-scaled, and a model that "needs a warmup" often just needs this.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .tensor import Tensor


class Optimizer:
    def __init__(self, params: Iterable[Tensor], lr: float):
        self.params = list(params)
        self.lr = lr
        self.t = 0

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()

    def step(self) -> None:  # pragma: no cover - abstract
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, params, lr: float = 0.1, momentum: float = 0.0, weight_decay: float = 0.0):
        super().__init__(params, lr)
        self.momentum, self.weight_decay = momentum, weight_decay
        self.velocity = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        self.t += 1
        for i, p in enumerate(self.params):
            g = p.grad + self.weight_decay * p.data
            self.velocity[i] = self.momentum * self.velocity[i] + g
            p.data -= self.lr * self.velocity[i]


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        super().__init__(params, lr)
        self.b1, self.b2 = betas
        self.eps, self.weight_decay = eps, weight_decay
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        self.t += 1
        for i, p in enumerate(self.params):
            g = p.grad + self.weight_decay * p.data
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            m_hat = self.m[i] / (1 - self.b1**self.t)
            v_hat = self.v[i] / (1 - self.b2**self.t)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
