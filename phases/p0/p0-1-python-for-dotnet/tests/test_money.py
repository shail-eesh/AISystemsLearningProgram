"""Value-object arithmetic. xUnit's [Theory]/[InlineData] is pytest.parametrize."""

from decimal import Decimal

import pytest
from p0_1_oms import Money, Quantity, ValidationError


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(1, "1.000000"), ("2.5", "2.500000"), (Decimal("3.14159"), "3.141590"),
     ("-4", "-4.000000"), ("1.23456789", "1.234568")],
)
def test_construction_quantizes_to_six_dp(raw, expected):
    assert str(Money(raw).amount) == expected


def test_display_is_four_dp_even_though_storage_is_six():
    assert str(Money("1.23456789")) == "INR 1.2346"


@pytest.mark.parametrize("bad", [1.5, True, None, [], "abc"])
def test_construction_rejects_non_exact_inputs(bad):
    with pytest.raises(ValidationError):
        Money(bad)


@pytest.mark.parametrize("ccy", ["", "IN", "INRS", "1NR", None])
def test_currency_must_be_three_letters(ccy):
    with pytest.raises((ValidationError, TypeError)):
        Money(1, ccy)


def test_half_up_rounding_not_bankers():
    # Python's float round() does banker's rounding; money must not.
    assert Money("1.0000005").amount == Decimal("1.000001")
    assert Money("1.0000015").amount == Decimal("1.000002")


def test_addition_subtraction_and_scaling():
    assert Money("10.25") + Money("0.75") == Money("11.00")
    assert Money("10") - Money("12.5") == Money("-2.5")
    assert Money("2.5") * 4 == Money("10")
    assert 4 * Money("2.5") == Money("10")
    assert -Money("3") == Money("-3")
    assert abs(Money("-3")) == Money("3")


def test_cross_currency_arithmetic_is_refused():
    with pytest.raises(ValidationError):
        Money(1, "INR") + Money(1, "USD")
    with pytest.raises(ValidationError):
        Money(1, "INR") - Money(1, "USD")


def test_float_scaling_is_refused():
    with pytest.raises(ValidationError):
        Money("1") * 1.5


def test_unknown_operand_defers_to_reflected_method():
    class Weird:
        def __radd__(self, other):
            return "right-hand"

    assert Money("1") + Weird() == "right-hand"
    with pytest.raises(TypeError):
        Money("1") + 1


def test_value_semantics_and_hashing():
    assert Money("1.5") == Money("1.5")
    assert hash(Money("1.5")) == hash(Money("1.5"))
    assert {Money("1.5"): "x"}[Money("1.5")] == "x"
    assert Money("1.5") != Money("1.5", "USD")


def test_ordering_and_truthiness():
    assert Money("1") < Money("2") <= Money("2")
    assert max([Money("1"), Money("9"), Money("3")]) == Money("9")
    assert not Money("0")
    assert Money("0.0001")


@pytest.mark.parametrize("bad", [-1, 1.0, True, "5", None])
def test_quantity_rejects_bad_inputs(bad):
    with pytest.raises(ValidationError):
        Quantity(bad)


def test_quantity_arithmetic_and_notional():
    assert Quantity(10) + Quantity(5) == Quantity(15)
    assert Quantity(10) - Quantity(4) == Quantity(6)
    with pytest.raises(ValidationError):
        Quantity(4) - Quantity(10)
    assert Quantity(400).notional(Money("101.25")) == Money("40500")
    assert int(Quantity(7)) == 7
    assert not Quantity(0)
