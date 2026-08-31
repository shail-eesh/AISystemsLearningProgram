"""Step 3: the same engine on NumPy arrays — where broadcasting bites.

The scalar engine is honest but useless: an MLP with 8 inputs and 32 hidden
units is ~300 `Value` objects *per sample*, each one a Python object with a
closure. The tensor version keeps the identical structure — node, local
closure, topological backward — and swaps the payload for an `np.ndarray`, so
one node now carries a whole matrix and the inner loop happens in C.

**The hard part, stated plainly.** NumPy silently broadcasts `(32, 8) + (8,)`
into `(32, 8)`. Forward that is free. Backward it is not: the bias saw its
value used 32 times, so its gradient is the *sum* of 32 upstream gradients. The
rule is mechanical and worth memorising:

    forward broadcast  ==  backward sum-and-reshape

Every axis NumPy stretched must be summed over on the way back; every axis it
invented (because the operand had fewer dimensions) must be summed away
entirely. :func:`unbroadcast` is those two sentences in six lines, and it is
the single most common source of silently-wrong hand-rolled autograd.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

ArrayLike = "Tensor | np.ndarray | float | int"


def unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Reduce `grad` back to `shape`, undoing NumPy's forward broadcast.

    Two reductions, in this order:
      1. Leading axes that did not exist in `shape` are summed away.
      2. Axes that were length 1 in `shape` and stretched are summed with
         `keepdims=True`, so the result keeps its rank.
    """
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(shape)


class Tensor:
    """An ndarray with a gradient and a place in a graph."""

    __slots__ = ("data", "grad", "requires_grad", "_backward", "_prev", "_op", "label")

    def __init__(
        self,
        data,
        _children: Iterable[Tensor] = (),
        _op: str = "",
        requires_grad: bool = True,
        label: str = "",
    ):
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self._backward = lambda: None
        self._prev = tuple(_children)
        self._op = _op
        self.label = label

    # -- basics -----------------------------------------------------------
    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, op={self._op or 'leaf'}{', ' + self.label if self.label else ''})"

    @staticmethod
    def _coerce(other) -> Tensor:
        return other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)

    def _accumulate(self, g: np.ndarray) -> None:
        if self.requires_grad:
            self.grad = self.grad + unbroadcast(g, self.data.shape)

    def detach(self) -> Tensor:
        return Tensor(self.data.copy(), requires_grad=False)

    def zero_grad(self) -> None:
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)

    # -- elementwise ops ---------------------------------------------------
    def __add__(self, other) -> Tensor:
        other = self._coerce(other)
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            self._accumulate(out.grad)
            other._accumulate(out.grad)

        out._backward = _backward
        return out

    def __mul__(self, other) -> Tensor:
        other = self._coerce(other)
        out = Tensor(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            self._accumulate(other.data * out.grad)
            other._accumulate(self.data * out.grad)

        out._backward = _backward
        return out

    def __pow__(self, exponent: float) -> Tensor:
        out = Tensor(self.data**exponent, (self,), f"**{exponent}")

        def _backward() -> None:
            self._accumulate(exponent * self.data ** (exponent - 1) * out.grad)

        out._backward = _backward
        return out

    def __matmul__(self, other) -> Tensor:
        other = self._coerce(other)
        out = Tensor(self.data @ other.data, (self, other), "@")

        def _backward() -> None:
            # dL/dA = dL/dC @ B^T, dL/dB = A^T @ dL/dC. Derive it once by
            # index notation and you never look it up again: C_ij = sum_k
            # A_ik B_kj, so dC_ij/dA_ik = B_kj.
            self._accumulate(out.grad @ np.swapaxes(other.data, -1, -2))
            other._accumulate(np.swapaxes(self.data, -1, -2) @ out.grad)

        out._backward = _backward
        return out

    def exp(self) -> Tensor:
        e = np.exp(self.data)
        out = Tensor(e, (self,), "exp")
        out._backward = lambda: self._accumulate(e * out.grad)
        return out

    def log(self) -> Tensor:
        out = Tensor(np.log(self.data), (self,), "log")
        out._backward = lambda: self._accumulate(out.grad / self.data)
        return out

    def tanh(self) -> Tensor:
        t = np.tanh(self.data)
        out = Tensor(t, (self,), "tanh")
        out._backward = lambda: self._accumulate((1 - t * t) * out.grad)
        return out

    def relu(self) -> Tensor:
        mask = self.data > 0
        out = Tensor(np.where(mask, self.data, 0.0), (self,), "relu")
        out._backward = lambda: self._accumulate(mask * out.grad)
        return out

    def sigmoid(self) -> Tensor:
        s = 1.0 / (1.0 + np.exp(-self.data))
        out = Tensor(s, (self,), "sigmoid")
        out._backward = lambda: self._accumulate(s * (1 - s) * out.grad)
        return out

    # -- reductions & shape -------------------------------------------------
    def sum(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")

        def _backward() -> None:
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            # A sum in the forward pass is a *broadcast* on the way back: every
            # element that was added into the total gets the same gradient.
            self._accumulate(np.broadcast_to(g, self.data.shape).copy())

        out._backward = _backward
        return out

    def mean(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        n = self.data.size if axis is None else np.prod(np.array(self.data.shape)[np.array(axis)])
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / float(n))

    def max(self, axis: int | None = None, keepdims: bool = False) -> Tensor:
        m = self.data.max(axis=axis, keepdims=True)
        out = Tensor(m if keepdims else m.squeeze(axis), (self,), "max")

        def _backward() -> None:
            g = out.grad if keepdims or axis is None else np.expand_dims(out.grad, axis)
            # Ties: NumPy's argmax picks the first, but splitting the gradient
            # evenly across ties is what keeps gradcheck honest under
            # perturbation. Both are defensible; we document the choice.
            mask = (self.data == m).astype(np.float64)
            mask /= mask.sum(axis=axis, keepdims=True)
            self._accumulate(mask * g)

        out._backward = _backward
        return out

    def reshape(self, *shape: int) -> Tensor:
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        out = Tensor(self.data.reshape(shape), (self,), "reshape")
        out._backward = lambda: self._accumulate(out.grad.reshape(self.data.shape))
        return out

    def transpose(self) -> Tensor:
        out = Tensor(np.swapaxes(self.data, -1, -2), (self,), "T")
        out._backward = lambda: self._accumulate(np.swapaxes(out.grad, -1, -2))
        return out

    @property
    def T(self) -> Tensor:  # noqa: N802 - matching NumPy's spelling on purpose
        return self.transpose()

    def __getitem__(self, idx) -> Tensor:
        out = Tensor(self.data[idx], (self,), "getitem")

        def _backward() -> None:
            g = np.zeros_like(self.data)
            np.add.at(g, idx, out.grad)  # `+=` would drop repeated indices
            self._accumulate(g)

        out._backward = _backward
        return out

    # -- sugar --------------------------------------------------------------
    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-self._coerce(other))

    def __rsub__(self, other):
        return self._coerce(other) + (-self)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * (self._coerce(other) ** -1.0)

    def __rtruediv__(self, other):
        return self._coerce(other) * (self**-1.0)

    # -- backward -----------------------------------------------------------
    def topo_order(self) -> list[Tensor]:
        order: list[Tensor] = []
        seen: set[int] = set()
        stack: list[tuple[Tensor, bool]] = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
                continue
            if id(node) in seen:
                continue
            seen.add(id(node))
            stack.append((node, True))
            for child in node._prev:
                if id(child) not in seen:
                    stack.append((child, False))
        return order

    def backward(self) -> None:
        if self.data.size != 1:
            raise RuntimeError(
                f"backward() starts from a scalar; this tensor is {self.shape}. "
                "Reduce it first (.sum() or .mean())."
            )
        order = self.topo_order()
        for node in order:
            node.zero_grad()
        self.grad = np.ones_like(self.data)
        for node in reversed(order):
            node._backward()


# -- convenience constructors ---------------------------------------------
def tensor(data, **kw) -> Tensor:
    return Tensor(data, **kw)


def zeros(*shape: int, **kw) -> Tensor:
    return Tensor(np.zeros(shape), **kw)


def randn(*shape: int, rng: np.random.Generator | None = None, scale: float = 1.0, **kw) -> Tensor:
    rng = rng or np.random.default_rng(0)
    return Tensor(rng.standard_normal(shape) * scale, **kw)
