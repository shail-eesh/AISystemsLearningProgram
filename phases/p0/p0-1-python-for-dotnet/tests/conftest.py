"""Shared fixtures — pytest's answer to xUnit class fixtures / IClassFixture."""

from datetime import UTC, datetime

import pytest
from p0_1_oms import Fill, Money, Order, Portfolio, Quantity, Side


@pytest.fixture
def now() -> datetime:
    return datetime(2024, 6, 3, 9, 30, tzinfo=UTC)


@pytest.fixture
def book() -> Portfolio:
    return Portfolio()


@pytest.fixture
def buy_order() -> Order:
    return Order("O1", "ALPHAINFRA", Side.BUY, Quantity(300), limit_price=Money("102.00"))


@pytest.fixture
def make_fill(now):
    def _make(order_id: str, qty: int, px: str, seq: int = 1) -> Fill:
        return Fill(f"{order_id}-F{seq}", order_id, Quantity(qty), Money(px), now)

    return _make
