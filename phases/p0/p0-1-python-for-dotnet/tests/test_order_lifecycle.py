"""Order state machine: the part an OMS bug report is always about."""

import pytest
from p0_1_oms import (
    Fill,
    Money,
    Order,
    OrderStatus,
    OrderType,
    Quantity,
    RiskLimitBreached,
    Side,
    ValidationError,
)
from p0_1_oms.oms import AllowedSymbols, MaxNotional


def test_new_order_starts_empty(buy_order):
    assert buy_order.status is OrderStatus.NEW
    assert buy_order.filled_quantity == Quantity(0)
    assert buy_order.leaves_quantity == Quantity(300)
    assert buy_order.average_fill_price is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"order_id": ""},
        {"symbol": "alphainfra"},
        {"quantity": Quantity(0)},
        {"order_type": OrderType.MARKET},                       # limit price present
    ],
)
def test_invariants_are_enforced_at_construction(kwargs):
    base = dict(order_id="O1", symbol="ALPHAINFRA", side=Side.BUY,
                quantity=Quantity(10), limit_price=Money("100"))
    with pytest.raises(ValidationError):
        Order(**{**base, **kwargs})


def test_market_order_needs_no_price_but_has_no_notional():
    o = Order("M1", "ALPHAINFRA", Side.BUY, Quantity(10), order_type=OrderType.MARKET)
    assert o.limit_price is None
    with pytest.raises(ValidationError):
        _ = o.notional


def test_limit_order_requires_a_price():
    with pytest.raises(ValidationError):
        Order("L1", "ALPHAINFRA", Side.BUY, Quantity(10))


def test_partial_then_complete_fill(buy_order, make_fill):
    buy_order.apply(make_fill("O1", 120, "101.00", 1))
    assert buy_order.status is OrderStatus.PARTIAL
    assert buy_order.leaves_quantity == Quantity(180)
    buy_order.apply(make_fill("O1", 180, "101.50", 2))
    assert buy_order.status is OrderStatus.FILLED
    # VWAP: (120*101.00 + 180*101.50) / 300
    assert buy_order.average_fill_price == Money("101.3000")


def test_overfill_is_refused(buy_order, make_fill):
    buy_order.apply(make_fill("O1", 280, "101.00", 1))
    with pytest.raises(ValidationError, match="overfill"):
        buy_order.apply(make_fill("O1", 30, "101.00", 2))
    assert buy_order.filled_quantity == Quantity(280)


def test_fill_from_a_different_order_is_refused(buy_order, make_fill):
    with pytest.raises(ValidationError, match="belongs to"):
        buy_order.apply(make_fill("OTHER", 10, "101.00"))


def test_fill_through_the_limit_is_refused(buy_order, make_fill):
    with pytest.raises(ValidationError, match="through"):
        buy_order.apply(make_fill("O1", 10, "102.50"))
    sell = Order("S1", "ALPHAINFRA", Side.SELL, Quantity(10), limit_price=Money("102.00"))
    with pytest.raises(ValidationError, match="through"):
        sell.apply(make_fill("S1", 10, "101.50"))
    sell.apply(make_fill("S1", 10, "102.50"))       # price improvement is fine
    assert sell.status is OrderStatus.FILLED


def test_terminal_states_reject_further_activity(buy_order, make_fill):
    buy_order.cancel("done for the day")
    assert buy_order.status is OrderStatus.CANCELLED
    with pytest.raises(ValidationError):
        buy_order.apply(make_fill("O1", 10, "101.00"))
    with pytest.raises(ValidationError):
        buy_order.cancel()


def test_partially_filled_order_cannot_be_rejected(buy_order, make_fill):
    buy_order.apply(make_fill("O1", 10, "101.00"))
    with pytest.raises(ValidationError):
        buy_order.reject("too late")


@pytest.mark.parametrize("bad", ["zero_qty", "naive_ts", "bad_venue", "zero_price"])
def test_fill_invariants(bad, now):
    kwargs = dict(fill_id="F", order_id="O1", quantity=Quantity(10),
                  price=Money("101"), at=now)
    if bad == "zero_qty":
        kwargs["quantity"] = Quantity(0)
    elif bad == "naive_ts":
        kwargs["at"] = now.replace(tzinfo=None)
    elif bad == "bad_venue":
        kwargs["venue"] = "NSE"
    else:
        kwargs["price"] = Money("0")
    with pytest.raises(ValidationError):
        Fill(**kwargs)


def test_risk_checks_reject_and_record(buy_order):
    checks = [MaxNotional(Money("1000")), AllowedSymbols(frozenset({"ALPHAINFRA"}))]
    with pytest.raises(RiskLimitBreached):
        buy_order.validate(checks)
    assert buy_order.status is OrderStatus.REJECTED
    assert "exceeds limit" in (buy_order.reject_reason or "")


def test_risk_checks_pass_quietly(buy_order):
    buy_order.validate([MaxNotional(Money("10000000")),
                        AllowedSymbols(frozenset({"ALPHAINFRA"}))])
    assert buy_order.status is OrderStatus.NEW
