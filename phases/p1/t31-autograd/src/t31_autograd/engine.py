"""Step 1–2: reverse-mode autodiff on scalars.

The whole idea in one paragraph. A numeric expression is a DAG: leaves are the
inputs, interior nodes are operations. Evaluating it forwards is ordinary
arithmetic. The derivative of the *final* node with respect to every other node
can be computed in one sweep backwards, because each node knows only one local
fact — how its output changes when each of its inputs changes. The chain rule
glues those local facts together.

.NET analogy: think of an expression tree (`System.Linq.Expressions`) where
each node also carries a closure that *pushes* the accumulated derivative into
its children. `backward()` walks the tree in reverse topological order and
calls those closures once each.

Two rules that make this correct, and which are where hand-rolled engines
usually go wrong:

1. **Accumulate, never assign.** If a node is used twice (`y = x*x + x`), it
   receives gradient from two parents; the total is the sum. `+=`, always.
2. **Reverse topological order.** A node's `_backward` may only run once *all*
   of its consumers have pushed into it. A topological sort of the DAG,
   reversed, guarantees that. Sorting by insertion order or by depth does not.
"""

from __future__ import annotations

import math
from collections.abc import Iterable


class Value:
    """A scalar and its gradient slot.

    `data` is the forward value; `grad` is d(final)/d(self), filled in by
    :meth:`backward`. `_backward` is the local chain-rule closure installed by
    whichever op produced this node.
    """

    __slots__ = ("data", "grad", "_backward", "_prev", "_op", "label")

    def __init__(self, data: float, _children: Iterable[Value] = (), _op: str = "", label: str = ""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = tuple(_children)
        self._op = _op
        self.label = label

    # -- construction helpers --------------------------------------------
    @staticmethod
    def _coerce(other: Value | float | int) -> Value:
        return other if isinstance(other, Value) else Value(other)

    # -- arithmetic -------------------------------------------------------
    def __add__(self, other: Value | float) -> Value:
        other = self._coerce(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            # d(a+b)/da = 1, d(a+b)/db = 1 -> the upstream gradient passes
            # through unchanged. Addition is a gradient *router*.
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other: Value | float) -> Value:
        other = self._coerce(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            # d(ab)/da = b -> multiplication is a gradient *switcher*: each
            # input gets the upstream gradient scaled by the *other* input.
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, exponent: float) -> Value:
        if not isinstance(exponent, int | float):
            raise TypeError("only int/float powers are supported (x**y with y a Value needs exp/log)")
        out = Value(self.data**exponent, (self,), f"**{exponent}")

        def _backward() -> None:
            self.grad += exponent * (self.data ** (exponent - 1)) * out.grad

        out._backward = _backward
        return out

    def exp(self) -> Value:
        e = math.exp(self.data)
        out = Value(e, (self,), "exp")

        def _backward() -> None:
            # The one function that is its own derivative; the forward value is
            # reused rather than recomputed.
            self.grad += e * out.grad

        out._backward = _backward
        return out

    def log(self) -> Value:
        if self.data <= 0:
            raise ValueError(f"log of non-positive value {self.data}")
        out = Value(math.log(self.data), (self,), "log")

        def _backward() -> None:
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> Value:
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward() -> None:
            self.grad += (1 - t * t) * out.grad

        out._backward = _backward
        return out

    def relu(self) -> Value:
        out = Value(self.data if self.data > 0 else 0.0, (self,), "relu")

        def _backward() -> None:
            # At exactly 0 the derivative does not exist; every framework picks
            # a subgradient. We pick 0, matching PyTorch.
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self) -> Value:
        s = 1.0 / (1.0 + math.exp(-self.data))
        out = Value(s, (self,), "sigmoid")

        def _backward() -> None:
            self.grad += s * (1 - s) * out.grad

        out._backward = _backward
        return out

    # -- sugar ------------------------------------------------------------
    def __neg__(self) -> Value:
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-self._coerce(other))

    def __rsub__(self, other):
        return self._coerce(other) + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * (self._coerce(other) ** -1)

    def __rtruediv__(self, other):
        return self._coerce(other) * (self**-1)

    def __repr__(self) -> str:
        tag = f" {self.label}" if self.label else ""
        return f"Value({self.data:.6g}, grad={self.grad:.6g}{tag})"

    # -- the backward pass -------------------------------------------------
    def topo_order(self) -> list[Value]:
        """Nodes of this node's subgraph, parents strictly after children.

        Iterative (not recursive) on purpose: a 40-layer MLP unrolled to
        scalars is tens of thousands of nodes deep and Python's recursion limit
        is 1000. The two-phase visit — push a node, then push its children,
        emit the node only when its children are done — is the standard
        explicit-stack post-order.
        """
        order: list[Value] = []
        seen: set[int] = set()
        stack: list[tuple[Value, bool]] = [(self, False)]
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
        """Fill `.grad` on every node this one depends on."""
        order = self.topo_order()
        for node in order:
            node.grad = 0.0
        self.grad = 1.0  # d(self)/d(self)
        for node in reversed(order):
            node._backward()


def draw_ascii(root: Value, max_nodes: int = 40) -> str:
    """A tiny text rendering of the graph — the thing the video animates."""
    lines = []
    for i, node in enumerate(root.topo_order()):
        if i >= max_nodes:
            lines.append("  ...")
            break
        op = node._op or "leaf"
        name = node.label or f"n{i}"
        lines.append(f"  {name:<10} {op:<8} data={node.data:>10.4f} grad={node.grad:>10.4f}")
    return "\n".join(lines)
