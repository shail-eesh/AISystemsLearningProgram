#!/usr/bin/env python3
"""Step 1 — POCOs become dataclasses; ctor guards become __post_init__.

Run:  python3 steps/step1_dataclasses_vs_pocos.py

What you are translating:

    // C#
    public sealed record Money(decimal Amount, string Currency = "INR");

    public sealed class Order {
        public Order(string id, int qty) {
            if (qty <= 0) throw new ArgumentException(nameof(qty));
            ...
        }
    }

Three things differ and all three bite .NET engineers:

1. `@dataclass(frozen=True)` is *shallow*. A frozen dataclass holding a list
   still lets you mutate the list. C# `readonly` has the same hole, but
   `record struct` value semantics hide it more often.
2. There is no constructor overloading. One `__init__`; alternatives are
   `@classmethod` factories (`Money.zero()`), which is what C# static factory
   methods do anyway.
3. Mutable defaults are a language-level trap with no C# equivalent:
   `def f(x=[])` shares ONE list across every call. Hence `field(default_factory=list)`.
"""

from dataclasses import FrozenInstanceError, dataclass, field, fields

import _bootstrap  # noqa: F401  (path setup; the import IS the side effect)
from p0_1_oms import Money, Order, Quantity, Side, ValidationError


def demo_value_semantics() -> None:
    a, b = Money("101.50"), Money("101.50")
    assert a == b, "dataclasses give you structural equality for free"
    assert a is not b, "...but they are still two distinct objects"
    assert hash(a) == hash(b), "frozen => hashable => usable as a dict key"
    assert {a: "px"}[b] == "px"
    print(f"value semantics: {a} == {b} -> {a == b}")


def demo_immutability_is_shallow() -> None:
    try:
        Money("1").amount = 2  # type: ignore[misc]
    except FrozenInstanceError as exc:
        print(f"frozen blocks rebinding: {type(exc).__name__}")

    order = Order("O1", "ALPHAINFRA", Side.BUY, Quantity(10), limit_price=Money("100"))
    order.fills.append("not really a fill")  # type: ignore[arg-type]
    print(f"...but a mutable field is still mutable: fills={order.fills}")
    order.fills.clear()


def demo_mutable_default_trap() -> None:
    @dataclass
    class Bad:
        # `tags: list = []` would not even compile — dataclasses catch this one.
        tags: list[str] = field(default_factory=list)

    x, y = Bad(), Bad()
    x.tags.append("a")
    assert y.tags == [], "default_factory gives each instance its own list"

    def classic_trap(item, bucket=[]):  # noqa: B006 - demonstrating the bug on purpose
        bucket.append(item)
        return bucket

    first, second = classic_trap(1), classic_trap(2)
    assert first is second and first == [1, 2], "the default list is shared across calls"
    print(f"mutable-default trap: two calls returned the same list {second}")


def demo_guard_clauses() -> None:
    for bad, why in [
        (dict(order_id="", symbol="X", side=Side.BUY, quantity=Quantity(1)), "empty id"),
        (dict(order_id="O", symbol="lower", side=Side.BUY, quantity=Quantity(1)), "lowercase symbol"),
        (dict(order_id="O", symbol="X", side=Side.BUY, quantity=Quantity(0)), "zero quantity"),
    ]:
        try:
            Order(**bad, limit_price=Money("1"))
        except ValidationError as exc:
            print(f"rejected ({why}): {exc}")


def demo_introspection() -> None:
    names = [f.name for f in fields(Order)]
    print(f"fields(Order) -> {names}")
    print("  (C# needs reflection + attributes for this; dataclasses hand it to you)")


if __name__ == "__main__":
    demo_value_semantics()
    demo_immutability_is_shallow()
    demo_mutable_default_trap()
    demo_guard_clauses()
    demo_introspection()
    print("\nstep 1 OK")
