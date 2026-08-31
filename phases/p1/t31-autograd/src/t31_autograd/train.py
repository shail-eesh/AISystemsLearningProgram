"""Step 5: train the Phase-0 task on *your* engine and compare the loss curve.

The claim this file has to earn is narrow and testable: *a network trained by
this engine follows the same optimisation trajectory as one trained by a
reference implementation.* Not "it gets good accuracy" — the Phase-0 lesson was
that good accuracy here is an artefact. The comparison is between two loss
curves on identical data, identical initialisation and identical hyper-
parameters, one differentiated by hand-written closures and one by an
independently-derived analytic gradient.

If those two curves separate, one of the two is wrong, and the finite-
difference gradcheck in `bench/` tells you which.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .nn import MLP, bce_with_logits
from .optim import Adam
from .tensor import Tensor


def make_features(n: int = 512, d: int = 8, *, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A separable-but-noisy toy task, standing in for causal price features.

    Deliberately synthetic and deliberately easy: the point of step 5 is to
    compare two optimisers, and a task with real market noise would hide the
    comparison under its own variance.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    w = rng.standard_normal((d, 1))
    logits = X @ w + 0.5 * rng.standard_normal((n, 1))
    y = (logits > 0).astype(np.float64)
    return X, y


class ReferenceMLP:
    """The same architecture with hand-derived gradients — no engine involved.

    One hidden tanh layer, one linear head, BCE-with-logits. Every derivative
    below was written out on paper. This is the independent implementation the
    engine is checked against; sharing code between them would make the test
    vacuous.
    """

    def __init__(self, d_in: int, hidden: int, rng: np.random.Generator):
        self.W1 = rng.standard_normal((d_in, hidden)) / np.sqrt(d_in)
        self.b1 = np.zeros((1, hidden))
        self.W2 = rng.standard_normal((hidden, 1)) / np.sqrt(hidden)
        self.b2 = np.zeros((1, 1))

    @property
    def params(self) -> list[np.ndarray]:
        return [self.W1, self.b1, self.W2, self.b2]

    def forward(self, X: np.ndarray):
        z1 = X @ self.W1 + self.b1
        a1 = np.tanh(z1)
        z2 = a1 @ self.W2 + self.b2
        return z1, a1, z2

    def loss_and_grads(self, X: np.ndarray, y: np.ndarray):
        n = X.shape[0]
        _z1, a1, z2 = self.forward(X)
        loss = float(np.mean(np.maximum(z2, 0) - z2 * y + np.log1p(np.exp(-np.abs(z2)))))
        # d(BCE)/dz2 = sigmoid(z2) - y, averaged over the batch.
        dz2 = (1.0 / (1.0 + np.exp(-z2)) - y) / n
        gW2 = a1.T @ dz2
        gb2 = dz2.sum(axis=0, keepdims=True)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (1 - a1 * a1)          # tanh'
        gW1 = X.T @ dz1
        gb1 = dz1.sum(axis=0, keepdims=True)
        return loss, [gW1, gb1, gW2, gb2]


@dataclass
class TrainResult:
    losses: list[float] = field(default_factory=list)
    accuracy: float = 0.0
    steps: int = 0


def train_with_engine(
    X: np.ndarray,
    y: np.ndarray,
    *,
    hidden: int = 16,
    steps: int = 200,
    lr: float = 0.05,
    seed: int = 0,
) -> tuple[TrainResult, MLP]:
    model = MLP([X.shape[1], hidden, 1], activation="tanh", rng=np.random.default_rng(seed))
    opt = Adam(model.parameters(), lr=lr)
    Xt = Tensor(X, requires_grad=False)
    yt = Tensor(y, requires_grad=False)
    result = TrainResult(steps=steps)
    for _ in range(steps):
        loss = bce_with_logits(model(Xt), yt)
        model.zero_grad()
        loss.backward()
        opt.step()
        result.losses.append(float(loss.data))
    logits = model(Xt).data
    result.accuracy = float((((logits > 0).astype(float)) == y).mean())
    return result, model


def train_reference(
    X: np.ndarray,
    y: np.ndarray,
    *,
    hidden: int = 16,
    steps: int = 200,
    lr: float = 0.05,
    seed: int = 0,
) -> TrainResult:
    """Same init, same Adam, gradients from `ReferenceMLP`."""
    rng = np.random.default_rng(seed)
    ref = ReferenceMLP(X.shape[1], hidden, rng)
    m = [np.zeros_like(p) for p in ref.params]
    v = [np.zeros_like(p) for p in ref.params]
    b1, b2, eps = 0.9, 0.999, 1e-8
    result = TrainResult(steps=steps)
    for t in range(1, steps + 1):
        loss, grads = ref.loss_and_grads(X, y)
        result.losses.append(loss)
        for i, (p, g) in enumerate(zip(ref.params, grads, strict=True)):
            m[i] = b1 * m[i] + (1 - b1) * g
            v[i] = b2 * v[i] + (1 - b2) * g * g
            p -= lr * (m[i] / (1 - b1**t)) / (np.sqrt(v[i] / (1 - b2**t)) + eps)
    _z1, _a1, z2 = ref.forward(X)
    result.accuracy = float((((z2 > 0).astype(float)) == y).mean())
    return result


class SignalTrainer:
    """The object AlphaDesk's registry hands out — see `alphadesk_hook.py`."""

    disclaimer = (
        "Teaching artefact. AlphaDesk is a fictional simulation; this model is "
        "not a trading signal and is never routed to an order."
    )

    def __init__(self, hidden: int = 16, seed: int = 0):
        self.hidden, self.seed = hidden, seed
        self.model: MLP | None = None

    def fit(self, X: np.ndarray | None = None, y: np.ndarray | None = None, steps: int = 200):
        if X is None or y is None:
            X, y = make_features(seed=self.seed)
        result, self.model = train_with_engine(
            X, y, hidden=self.hidden, steps=steps, seed=self.seed
        )
        return result

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("fit() first")
        logits = self.model(Tensor(X, requires_grad=False)).data
        return 1.0 / (1.0 + np.exp(-logits))
