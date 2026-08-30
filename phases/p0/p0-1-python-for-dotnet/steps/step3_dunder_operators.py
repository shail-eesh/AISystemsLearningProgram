#!/usr/bin/env python3
"""Step 3 — operator overloading, equality, and the arithmetic of money.

Run:  python3 steps/step3_dunder_operators.py

C#: `public static Money operator +(Money a, Money b)`, `Equals`/`GetHashCode`,
`IComparable<T>`, `ToString()`.
Python: `__add__`, `__eq__`/`__hash__`, `__lt__` (or `order=True`), `__str__`
vs `__repr__`.

Two rules that have no C# analogue and cause real bugs:

* Return `NotImplemented` (not `NotImplementedError`) from a binary dunder when
  you do not know the other operand. Python then tries the *reflected* method
  on the right-hand side before raising TypeError. Returning `False` or raising
  here breaks `other + self` forever.
* `__bool__` decides truthiness. Money(0) is falsy here — deliberate — which is
  exactly the trap the AlphaDesk registry hit on Day 1: `registry or DEFAULT`
  silently swapped an *empty* registry for the global one.
"""

from decimal import Decimal

import _bootstrap  # noqa: F401
from p0_1_oms import Money, Quantity, ValidationError


def demo_arithmetic() -> None:
    px, qty = Money("101.25"), Quantity(400)
    print(f"notional: {qty} x {px} = {qty.notional(px)}")
    assert qty.notional(px) == Money("40500")
    assert Money("10") + Money("5") == Money("15")
    assert Money("10") - Money("15") == Money("-5")
    assert 3 * Money("10") == Money("30"), "__rmul__ makes int * Money work"


def demo_float_is_rejected() -> None:
    for bad in (101.25, True):
        try:
            Money(bad)
        except ValidationError as exc:
            print(f"rejected {bad!r}: {exc}")
    lhs = Decimal("0.1") + Decimal("0.2")
    print(f"Decimal('0.1')+Decimal('0.2') == Decimal('0.3') -> {lhs == Decimal('0.3')}")
    print(f"       0.1 +        0.2 ==        0.3 -> {0.1 + 0.2 == 0.3}   <-- why money is Decimal")


def demo_not_implemented_protocol() -> None:
    class Weird:
        def __radd__(self, other):
            return "handled by the right-hand side"

    # Money.__add__ returns NotImplemented, so Python falls back to Weird.__radd__.
    assert Money("1") + Weird() == "handled by the right-hand side"
    print("NotImplemented let the right operand take over — never raise here")

    try:
        Money("1") + 1
    except TypeError as exc:
        print(f"when neither side knows: TypeError({exc})")


def demo_currency_guard() -> None:
    try:
        Money("1", "INR") + Money("1", "USD")
    except ValidationError as exc:
        print(f"currency mismatch: {exc}")


def demo_ordering_and_truthiness() -> None:
    prices = [Money("103.10"), Money("99.95"), Money("101.25")]
    print(f"sorted: {[str(p) for p in sorted(prices)]}")
    assert max(prices) == Money("103.10"), "order=True gives <,<=,>,>= for free"
    assert not Money("0"), "zero money is falsy"
    assert not Quantity(0), "zero quantity is falsy"
    assert Money("0.0001"), "...but a hundredth of a paisa is not"
    print("truthiness: Money(0) is falsy — check `is None`, not `if value:`")


def demo_repr_vs_str() -> None:
    m = Money("1234.5")
    print(f"__str__ (for humans):     {m}")
    print(f"__repr__ (for debugging): {m!r}")


if __name__ == "__main__":
    demo_arithmetic()
    demo_float_is_rejected()
    demo_not_implemented_protocol()
    demo_currency_guard()
    demo_ordering_and_truthiness()
    demo_repr_vs_str()
    print("\nstep 3 OK")
