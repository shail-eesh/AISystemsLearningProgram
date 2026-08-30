#!/usr/bin/env python3
"""Step 2 — enums that carry behaviour, and interfaces you never declare.

Run:  python3 steps/step2_enums_protocols.py

C# enums are named integers; you bolt behaviour on with extension methods.
Python enums are real objects with methods and properties, so `Side.BUY.sign`
lives on the enum itself. And `IRiskCheck` disappears entirely: a Protocol is
checked structurally, so a class satisfies it by *shape*, never by declaration.
"""

from dataclasses import dataclass

import _bootstrap  # noqa: F401
from p0_1_oms import Money, Order, Quantity, RiskLimitBreached, Side
from p0_1_oms.oms import AllowedSymbols, MaxNotional, OrderStatus, RiskCheck


def demo_enum_behaviour() -> None:
    assert Side.BUY.sign == 1 and Side.SELL.sign == -1
    assert Side.BUY.opposite() is Side.SELL
    assert Side("BUY") is Side.BUY, "str-backed enums parse from the wire for free"
    assert Side.BUY == "BUY", "...and compare equal to their string, unlike a C# enum"
    assert OrderStatus.FILLED.is_terminal and not OrderStatus.PARTIAL.is_terminal
    print(f"enum with behaviour: {Side.SELL} sign={Side.SELL.sign} opposite={Side.SELL.opposite()}")


def demo_structural_typing() -> None:
    @dataclass(frozen=True)
    class NoShortSelling:
        """Never declares RiskCheck. Still satisfies it."""

        def check(self, order: Order) -> None:
            if order.side is Side.SELL:
                raise RiskLimitBreached("this book is long-only")

    check = NoShortSelling()
    assert isinstance(check, RiskCheck), "@runtime_checkable verifies the method exists"
    print(f"structural typing: {type(check).__name__} satisfies RiskCheck without inheriting it")

    sell = Order("O2", "ALPHAINFRA", Side.SELL, Quantity(10), limit_price=Money("100"))
    try:
        sell.validate([check])
    except RiskLimitBreached as exc:
        print(f"  rejected: {exc}; order is now {sell.status.value}")
        assert sell.status is OrderStatus.REJECTED


def demo_composed_checks() -> None:
    checks = [MaxNotional(Money("50000")), AllowedSymbols(frozenset({"ALPHAINFRA", "COASTBANK"}))]
    ok = Order("O3", "ALPHAINFRA", Side.BUY, Quantity(100), limit_price=Money("101"))
    ok.validate(checks)
    print(f"passes both checks: {ok}")

    too_big = Order("O4", "ALPHAINFRA", Side.BUY, Quantity(5000), limit_price=Money("101"))
    try:
        too_big.validate(checks)
    except RiskLimitBreached as exc:
        print(f"  breach: {exc}")

    off_universe = Order("O5", "ZZTOP", Side.BUY, Quantity(1), limit_price=Money("1"))
    try:
        off_universe.validate(checks)
    except RiskLimitBreached as exc:
        print(f"  breach: {exc}")


if __name__ == "__main__":
    demo_enum_behaviour()
    demo_structural_typing()
    demo_composed_checks()
    print("\nstep 2 OK")
