"""AlphaDesk wiring for P0.1.

The Order Workflow surface needs a book to place paper orders into. That is
this topic's contribution: the domain model itself, exposed as a factory the
desk can call. Every later topic follows exactly this pattern.
"""

from __future__ import annotations

from common.alphadesk import Surface, register

from .money import Money
from .oms import AllowedSymbols, MaxNotional, Portfolio

#: The sample universe AlphaDesk is allowed to "trade" (fictional issuers).
UNIVERSE = frozenset({"ALPHAINFRA", "BHARATCHEM", "COASTBANK", "DECCANMOT", "EASTPOWER"})


@register(
    topic="P0.1",
    name="paper_book",
    surface=Surface.ORDERS,
    summary="Order/Fill/Position domain model backing the paper order workflow",
)
def build_paper_book(currency: str = "INR") -> Portfolio:
    """A fresh simulated book. No venue, no broker, no money."""
    return Portfolio(currency=currency)


@register(
    topic="P0.1",
    name="default_risk_checks",
    surface=Surface.COMPLIANCE,
    summary="Pre-trade checks (max notional, tradable universe) for the paper book",
)
def build_default_risk_checks(max_notional: str = "5000000") -> list[object]:
    return [MaxNotional(Money(max_notional)), AllowedSymbols(UNIVERSE)]
