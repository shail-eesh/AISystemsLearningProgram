"""Average-cost accounting. Every branch of `Position.apply` has a test."""

import pytest
from p0_1_oms import Money, Order, Portfolio, Quantity, Side, ValidationError
from p0_1_oms.money import Money as M


@pytest.fixture
def desk(book, make_fill):
    """A helper that places an order and fills it in one line."""

    seq = {"n": 0}

    def trade(symbol: str, side: Side, qty: int, px: str, limit: str | None = None) -> Portfolio:
        seq["n"] += 1
        oid = f"T{seq['n']}"
        lim = limit or (str(float(px) + 5) if side is Side.BUY else str(float(px) - 5))
        book.place(Order(oid, symbol, side, Quantity(qty), limit_price=Money(lim)))
        book.execute(make_fill(oid, qty, px))
        return book

    return trade


def test_open_long(desk, book):
    desk("ALPHAINFRA", Side.BUY, 100, "100.00")
    pos = book["ALPHAINFRA"]
    assert pos.net_quantity == 100
    assert pos.average_cost == M("100")
    assert pos.realised_pnl == M("0")
    assert pos.is_long and not pos.is_flat


def test_adding_reweights_average_cost(desk, book):
    desk("ALPHAINFRA", Side.BUY, 100, "100.00")
    desk("ALPHAINFRA", Side.BUY, 300, "104.00")
    pos = book["ALPHAINFRA"]
    assert pos.net_quantity == 400
    assert pos.average_cost == M("103.00")      # (100*100 + 300*104)/400
    assert pos.realised_pnl == M("0")


def test_partial_reduction_realises_and_keeps_basis(desk, book):
    desk("ALPHAINFRA", Side.BUY, 400, "103.00")
    desk("ALPHAINFRA", Side.SELL, 100, "106.00")
    pos = book["ALPHAINFRA"]
    assert pos.net_quantity == 300
    assert pos.average_cost == M("103.00"), "reducing must not move the cost basis"
    assert pos.realised_pnl == M("300.00")      # 100 * (106 - 103)


def test_flat_close_zeroes_the_basis(desk, book):
    desk("COASTBANK", Side.BUY, 50, "200.00")
    desk("COASTBANK", Side.SELL, 50, "190.00")
    pos = book["COASTBANK"]
    assert pos.is_flat
    assert pos.average_cost == M("0")
    assert pos.realised_pnl == M("-500.00")


def test_flip_through_zero_reopens_at_the_fill_price(desk, book):
    desk("EASTPOWER", Side.BUY, 100, "50.00")
    desk("EASTPOWER", Side.SELL, 250, "55.00")
    pos = book["EASTPOWER"]
    assert pos.net_quantity == -150
    assert pos.realised_pnl == M("500.00")       # only the 100 closed
    assert pos.average_cost == M("55.00"), "the short leg opens at the flip price"


def test_short_then_cover(desk, book):
    desk("DECCANMOT", Side.SELL, 200, "80.00")
    assert book["DECCANMOT"].net_quantity == -200
    assert book["DECCANMOT"].average_cost == M("80")
    desk("DECCANMOT", Side.BUY, 200, "72.00")
    assert book["DECCANMOT"].realised_pnl == M("1600.00")   # 200 * (80 - 72)


def test_unrealised_pnl_signs(desk, book):
    desk("ALPHAINFRA", Side.BUY, 100, "100.00")
    assert book["ALPHAINFRA"].unrealised_pnl(M("110")) == M("1000")
    assert book["ALPHAINFRA"].unrealised_pnl(M("90")) == M("-1000")
    desk("COASTBANK", Side.SELL, 100, "100.00")
    assert book["COASTBANK"].unrealised_pnl(M("90")) == M("1000"), "shorts gain when price falls"


def test_flat_position_has_no_unrealised(book):
    assert book["NOTHELD"].unrealised_pnl(M("100")) == M("0")


def test_portfolio_behaves_like_a_collection(desk, book):
    desk("ALPHAINFRA", Side.BUY, 10, "100.00")
    desk("COASTBANK", Side.BUY, 10, "200.00")
    assert len(book) == 2
    assert "ALPHAINFRA" in book and "ZZTOP" not in book
    assert {p.symbol for p in book} == {"ALPHAINFRA", "COASTBANK"}
    assert book["ZZTOP"].is_flat, "unknown symbols read as a flat position, not KeyError"


def test_portfolio_aggregates(desk, book):
    desk("ALPHAINFRA", Side.BUY, 100, "100.00")
    desk("ALPHAINFRA", Side.SELL, 40, "105.00")
    desk("COASTBANK", Side.BUY, 10, "200.00")
    assert book.realised_pnl == M("200.00")
    marks = {"ALPHAINFRA": M("110"), "COASTBANK": M("190")}
    assert book.unrealised_pnl(marks) == M("500.00")   # 60*10 + 10*(-10)
    assert book.gross_exposure(marks) == M("8500.00")  # 60*110 + 10*190


def test_duplicate_order_ids_and_unknown_fills(book, make_fill):
    o = Order("D1", "ALPHAINFRA", Side.BUY, Quantity(10), limit_price=M("100"))
    book.place(o)
    with pytest.raises(ValidationError, match="duplicate"):
        book.place(Order("D1", "ALPHAINFRA", Side.BUY, Quantity(10), limit_price=M("100")))
    with pytest.raises(ValidationError, match="unknown order"):
        book.execute(make_fill("NOPE", 10, "100"))


def test_position_is_immutable(desk, book):
    desk("ALPHAINFRA", Side.BUY, 10, "100.00")
    before = book["ALPHAINFRA"]
    desk("ALPHAINFRA", Side.BUY, 10, "120.00")
    assert before.net_quantity == 10, "the earlier Position object is untouched"
    assert book["ALPHAINFRA"].net_quantity == 20
