"""Value objects: Money and Quantity.

The first real .NET→Python lesson. In C# you would reach for `decimal` for
money and never for `double`; Python's equivalent is `decimal.Decimal`, and
`float` is exactly as wrong here as `double` is there. 0.1 + 0.2 != 0.3 in both
languages for the same IEEE-754 reason.

`@dataclass(frozen=True)` gives you what a C# `readonly record struct` gives
you: value equality, a usable `ToString`, immutability, and a hash — but the
immutability is a convention enforced by `__setattr__`, not by the runtime
laying the value out on the stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Union

from .errors import ValidationError

Numeric = Union[int, str, Decimal, "Money"]
# Storage precision. Six decimal places, not two: a *price* needs four, but a
# weighted-average cost basis is a quotient, and rounding it at every fill
# accumulates a visible drift over a few hundred orders (the P0.1 benchmark
# measures exactly this). Display stays at four dp — store precise, show tidy.
_QUANTUM = Decimal("0.000001")


def _to_decimal(value: object, field: str) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, bool):  # bool is an int subclass — a classic Python trap
        raise ValidationError(f"{field} must be numeric, got bool")
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, float):
        raise ValidationError(
            f"{field} was a float ({value!r}). Money never comes from binary floating "
            "point — pass a str or Decimal, exactly as you would avoid double in C#."
        )
    if isinstance(value, str):
        try:
            return Decimal(value)
        except InvalidOperation as exc:
            raise ValidationError(f"{field} is not a number: {value!r}") from exc
    raise ValidationError(f"{field} must be int, str or Decimal, got {type(value).__name__}")


@dataclass(frozen=True, order=True)
class Money:
    """A currency amount. Immutable, exact, and closed under +, -, and scaling."""

    amount: Decimal
    currency: str = "INR"

    def __init__(self, amount: Numeric, currency: str = "INR") -> None:
        raw = amount.amount if isinstance(amount, Money) else _to_decimal(amount, "amount")
        if not currency or len(currency) != 3 or not currency.isalpha():
            raise ValidationError(f"currency must be a 3-letter code, got {currency!r}")
        object.__setattr__(self, "amount", raw.quantize(_QUANTUM, rounding=ROUND_HALF_UP))
        object.__setattr__(self, "currency", currency.upper())

    # -- operators: C# operator overloading, spelled with dunders -----------
    def _same_currency(self, other: Money) -> None:
        if self.currency != other.currency:
            raise ValidationError(f"cannot mix {self.currency} and {other.currency}")

    def __add__(self, other: Money) -> Money:
        if not isinstance(other, Money):
            return NotImplemented          # lets Python try other.__radd__
        self._same_currency(other)
        return Money(self.amount + other.amount, self.currency)

    def __sub__(self, other: Money) -> Money:
        if not isinstance(other, Money):
            return NotImplemented
        self._same_currency(other)
        return Money(self.amount - other.amount, self.currency)

    def __mul__(self, factor: int | Decimal) -> Money:
        if isinstance(factor, float):
            raise ValidationError("scale Money by int or Decimal, never float")
        if not isinstance(factor, int | Decimal):
            return NotImplemented
        return Money(self.amount * Decimal(factor), self.currency)

    __rmul__ = __mul__

    def __neg__(self) -> Money:
        return Money(-self.amount, self.currency)

    def __abs__(self) -> Money:
        return Money(abs(self.amount), self.currency)

    def __bool__(self) -> bool:
        return self.amount != 0

    def __str__(self) -> str:
        return f"{self.currency} {self.amount:,.4f}"

    @classmethod
    def zero(cls, currency: str = "INR") -> Money:
        return cls(0, currency)


@dataclass(frozen=True, order=True)
class Quantity:
    """A share count. Integral, non-negative, and lot-aware."""

    shares: int

    def __init__(self, shares: int) -> None:
        if isinstance(shares, bool) or not isinstance(shares, int):
            raise ValidationError(f"quantity must be an int, got {type(shares).__name__}")
        if shares < 0:
            raise ValidationError(f"quantity must be non-negative, got {shares}")
        object.__setattr__(self, "shares", shares)

    def __add__(self, other: Quantity) -> Quantity:
        return Quantity(self.shares + other.shares) if isinstance(other, Quantity) else NotImplemented

    def __sub__(self, other: Quantity) -> Quantity:
        if not isinstance(other, Quantity):
            return NotImplemented
        if other.shares > self.shares:
            raise ValidationError(f"cannot subtract {other.shares} from {self.shares}")
        return Quantity(self.shares - other.shares)

    def __bool__(self) -> bool:
        return self.shares != 0

    def __int__(self) -> int:
        return self.shares

    def __str__(self) -> str:
        return f"{self.shares:,}"

    def notional(self, price: Money) -> Money:
        """shares x price — the one place the two value objects meet."""
        return price * self.shares
