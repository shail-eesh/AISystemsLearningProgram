"""Step 4: a mini `nn` on top of the tensor engine.

Everything here is ~120 lines because the engine already did the work. That is
the point of the phase: once `Tensor` composes correctly, a layer library is
just bookkeeping — which arrays are parameters, and what shape they are.

Initialisation is not bookkeeping, though. Weights drawn from N(0,1) into a
5-layer tanh net produce saturated activations and a dead loss curve; that is
the failure the video shows before fixing it. Kaiming/He scaling
(`sqrt(2/fan_in)` for ReLU) and Xavier/Glorot (`sqrt(1/fan_in)` for tanh) keep
the *variance* of activations roughly constant layer to layer.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from .tensor import Tensor


class Module:
    """Base class: parameter discovery by attribute walk, like `nn.Module`."""

    def parameters(self) -> list[Tensor]:
        found: list[Tensor] = []
        seen: set[int] = set()

        def visit(obj) -> None:
            if isinstance(obj, Tensor):
                if obj.requires_grad and id(obj) not in seen:
                    seen.add(id(obj))
                    found.append(obj)
            elif isinstance(obj, Module):
                for value in vars(obj).values():
                    visit(value)
            elif isinstance(obj, list | tuple):
                for value in obj:
                    visit(value)

        visit(self)
        return found

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()

    def num_parameters(self) -> int:
        return sum(p.data.size for p in self.parameters())

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def forward(self, x: Tensor) -> Tensor:  # pragma: no cover - abstract
        raise NotImplementedError


class Linear(Module):
    """y = x @ W + b, with `x` of shape (batch, in_features)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        nonlinearity: str = "tanh",
        rng: np.random.Generator | None = None,
    ):
        rng = rng or np.random.default_rng(0)
        gain = {"relu": np.sqrt(2.0), "tanh": 1.0, "linear": 1.0}[nonlinearity]
        scale = gain / np.sqrt(in_features)
        self.weight = Tensor(rng.standard_normal((in_features, out_features)) * scale, label="W")
        self.bias = Tensor(np.zeros((1, out_features)), label="b") if bias else None
        self.in_features, self.out_features = in_features, out_features

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight
        # (batch, out) + (1, out) -> the broadcast whose backward pass is the
        # whole reason `unbroadcast` exists.
        return out + self.bias if self.bias is not None else out


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()


class Sequential(Module):
    def __init__(self, *layers: Module):
        self.layers = list(layers)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def __iter__(self) -> Iterator[Module]:
        return iter(self.layers)


class MLP(Sequential):
    """The Phase-0 architecture, rebuilt on your own engine."""

    def __init__(
        self,
        sizes: list[int],
        *,
        activation: str = "tanh",
        rng: np.random.Generator | None = None,
    ):
        rng = rng or np.random.default_rng(0)
        act = {"tanh": Tanh, "relu": ReLU}[activation]
        layers: list[Module] = []
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:], strict=True)):
            last = i == len(sizes) - 2
            layers.append(
                Linear(a, b, nonlinearity="linear" if last else activation, rng=rng)
            )
            if not last:
                layers.append(act())
        super().__init__(*layers)


# -- losses ----------------------------------------------------------------
def mse_loss(pred: Tensor, target) -> Tensor:
    target = Tensor._coerce(target)
    diff = pred - target
    return (diff * diff).mean()


def bce_with_logits(logits: Tensor, target) -> Tensor:
    """Numerically safe binary cross-entropy, straight from the identity

        log(1 + e^z) = max(z, 0) + log(1 + e^-|z|)

    which is the same max-subtraction trick T45A derives for softmax. Writing
    `sigmoid(z).log()` instead overflows the moment |z| passes ~700.
    """
    target = Tensor._coerce(target)
    z = logits
    # |z| must stay differentiable: d|z|/dz = sign(z), so build it as z*sign
    # with the *sign* detached. Writing `Tensor(np.abs(z.data))` instead is the
    # classic bug -- it forward-computes correctly and silently zeroes the
    # gradient of that term, which gradcheck catches and a loss curve does not.
    sign = Tensor(np.sign(z.data), requires_grad=False)
    softplus = z.relu() + (Tensor(1.0, requires_grad=False) + (-(z * sign)).exp()).log()
    return (softplus - z * target).mean()


def softmax_cross_entropy(logits: Tensor, target_index: np.ndarray) -> Tensor:
    """Mean cross-entropy over a batch, with the log-sum-exp stabilised."""
    idx = np.asarray(target_index, dtype=int)
    shifted = logits - Tensor(logits.data.max(axis=-1, keepdims=True), requires_grad=False)
    logsumexp = shifted.exp().sum(axis=-1, keepdims=True).log()
    log_probs = shifted - logsumexp
    picked = log_probs[np.arange(len(idx)), idx]
    return -picked.mean()
