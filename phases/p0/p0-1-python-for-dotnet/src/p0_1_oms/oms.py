"""Order, Fill, Position, Portfolio — the AlphaDesk paper OMS domain model.

AlphaDesk is a fictional educational simulation: nothing here reaches a venue,
a broker, or real money. `SIMBOOK` is an in-memory book that only exists in
this repo.

The C# original this is ported from would have been roughly:

    public sealed class Order {
        public string OrderId { get; }
        public Side Side { get; }
        ...
        public void Apply(Fill fill) { /* guard clauses, mutate, raise event */ }
    }

The Python port keeps the same domain and drops the ceremony: dataclasses
instead of POCOs with backing fields, an Enum instead of a C# enum, a Protocol
instead of an interface, and exceptions instead of `TryX(out ...)` pairs.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime
from decimal import Decimal
from enum import Enum
from typing import Protocol, runtime_checkable

from .errors import RiskLimitBreached, ValidationError
from .money import Money, Quantity

VENUE = "SIMBOOK"  # the simulated book; never a real MIC code


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"

    @property
    def sign(self) -> int:
        """+1 for BUY, -1 for SELL — the only place direction is encoded."""
        return 1 if self is Side.BUY else -1

    def opposite(self) -> Side:
        return Side.SELL if self is Side.BUY else Side.BUY


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderStatus(str, Enum):
    NEW = "NEW"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"

    @property
    def is_terminal(self) -> bool:
        return self in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED)


@runtime_checkable
class RiskCheck(Protocol):
    """Structural typing: anything with a `check(order)` method qualifies.

    C# would need `class MaxNotional : IRiskCheck`. Python needs nothing — a
    class that happens to have `check(order)` already satisfies this, which is
    why Protocols are called *static duck typing*.
    """

    def check(self, order: Order) -> None:
        """Raise RiskLimitBreached to reject; return None to accept."""
        ...


@dataclass(frozen=True)
class Fill:
    """An execution against an order. Immutable — fills are facts, not state."""

    fill_id: str
    order_id: str
    quantity: Quantity
    price: Money
    at: datetime
    venue: str = VENUE

    def __post_init__(self) -> None:
        if not self.quantity:
            raise ValidationError("a fill must move a non-zero quantity")
        if self.price.amount <= 0:
            raise ValidationError(f"fill price must be positive, got {self.price}")
        if self.venue != VENUE:
            raise ValidationError("AlphaDesk fills only ever come from the simulated book")
        if self.at.tzinfo is None:
            raise ValidationError("fill timestamps must be timezone-aware (UTC)")

    @property
    def notional(self) -> Money:
        return self.quantity.notional(self.price)


@dataclass
class Order:
    """A parent order and its execution state.

    Mutable by design: an order *is* a lifecycle. Its fills are an append-only
    list so the state can always be rebuilt from them — the same reason an OMS
    keeps an execution log.
    """

    order_id: str
    symbol: str
    side: Side
    quantity: Quantity
    order_type: OrderType = OrderType.LIMIT
    limit_price: Money | None = None
    trade_date: date = field(default_factory=lambda: datetime.now(UTC).date())
    fills: list[Fill] = field(default_factory=list)
    status: OrderStatus = OrderStatus.NEW
    reject_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.order_id:
            raise ValidationError("order_id is required")
        if not self.symbol or not self.symbol.isupper():
            raise ValidationError(f"symbol must be an uppercase ticker, got {self.symbol!r}")
        if not self.quantity:
            raise ValidationError("order quantity must be positive")
        if self.order_type is OrderType.LIMIT and self.limit_price is None:
            raise ValidationError("a LIMIT order needs a limit_price")
        if self.order_type is OrderType.MARKET and self.limit_price is not None:
            raise ValidationError("a MARKET order must not carry a limit_price")

    # -- derived state (C# would make these expression-bodied properties) ----
    @property
    def filled_quantity(self) -> Quantity:
        return Quantity(sum((f.quantity.shares for f in self.fills), 0))

    @property
    def leaves_quantity(self) -> Quantity:
        return self.quantity - self.filled_quantity

    @property
    def average_fill_price(self) -> Money | None:
        filled = self.filled_quantity.shares
        if filled == 0:
            return None
        gross = sum((f.notional.amount for f in self.fills), Decimal(0))
        return Money(gross / Decimal(filled), self.fills[0].price.currency)

    @property
    def notional(self) -> Money:
        """Worst-case exposure used by pre-trade risk."""
        if self.limit_price is None:
            raise ValidationError("market orders have no ex-ante notional")
        return self.quantity.notional(self.limit_price)

    # -- behaviour ----------------------------------------------------------
    def apply(self, fill: Fill) -> Order:
        """Apply an execution. Returns self so calls chain, like a fluent API."""
        if self.status.is_terminal:
            raise ValidationError(f"{self.order_id} is {self.status.value}; it cannot fill")
        if fill.order_id != self.order_id:
            raise ValidationError(f"fill {fill.fill_id} belongs to {fill.order_id}")
        if fill.quantity.shares > self.leaves_quantity.shares:
            raise ValidationError(
                f"overfill: {fill.quantity} against {self.leaves_quantity} leaves"
            )
        if self.order_type is OrderType.LIMIT and self.limit_price is not None:
            through = (
                fill.price > self.limit_price
                if self.side is Side.BUY
                else fill.price < self.limit_price
            )
            if through:
                raise ValidationError(
                    f"fill at {fill.price} is through the {self.side.value} "
                    f"limit {self.limit_price}"
                )
        self.fills.append(fill)
        self.status = OrderStatus.FILLED if not self.leaves_quantity else OrderStatus.PARTIAL
        return self

    def cancel(self, reason: str = "user") -> Order:
        if self.status.is_terminal:
            raise ValidationError(f"{self.order_id} is already {self.status.value}")
        self.status = OrderStatus.CANCELLED
        self.reject_reason = reason
        return self

    def reject(self, reason: str) -> Order:
        if self.fills:
            raise ValidationError("a partially filled order cannot be rejected")
        self.status = OrderStatus.REJECTED
        self.reject_reason = reason
        return self

    def validate(self, checks: Iterable[RiskCheck]) -> Order:
        """Run pre-trade checks; reject on the first breach."""
        for chk in checks:
            try:
                chk.check(self)
            except RiskLimitBreached as exc:
                self.reject(str(exc))
                raise
        return self

    def __str__(self) -> str:
        px = f"@{self.limit_price}" if self.limit_price else "@MKT"
        return (
            f"{self.order_id} {self.side.value} {self.quantity} {self.symbol} {px} "
            f"[{self.status.value} {self.filled_quantity}/{self.quantity}]"
        )


@dataclass(frozen=True)
class Position:
    """A net position with a weighted-average cost basis.

    Immutable: every fill produces a *new* Position. That makes the P&L
    arithmetic trivially testable and rules out the classic OMS bug where a
    position is mutated on a background thread mid-read.
    """

    symbol: str
    net_quantity: int = 0                       # signed: negative is short
    average_cost: Money = field(default_factory=Money.zero)
    realised_pnl: Money = field(default_factory=Money.zero)

    @property
    def is_flat(self) -> bool:
        return self.net_quantity == 0

    @property
    def is_long(self) -> bool:
        return self.net_quantity > 0

    def apply(self, side: Side, fill: Fill) -> Position:
        """Fold one fill into the position (average-cost accounting)."""
        signed = side.sign * fill.quantity.shares
        old, px = self.net_quantity, fill.price.amount
        new = old + signed

        if old == 0 or (old > 0) == (signed > 0):
            # opening or adding: re-weight the average cost, realise nothing
            gross = self.average_cost.amount * abs(old) + px * abs(signed)
            avg = gross / Decimal(abs(new)) if new else Decimal(0)
            return replace(self, net_quantity=new,
                           average_cost=Money(avg, fill.price.currency))

        # reducing, closing, or flipping: realise against the average cost
        closed = min(abs(old), abs(signed))
        direction = Decimal(1 if old > 0 else -1)
        realised = (px - self.average_cost.amount) * Decimal(closed) * direction
        pnl = self.realised_pnl + Money(realised, fill.price.currency)

        if new == 0:
            return replace(self, net_quantity=0,
                           average_cost=Money.zero(fill.price.currency), realised_pnl=pnl)
        if (new > 0) == (old > 0):          # partial reduction: basis unchanged
            return replace(self, net_quantity=new, realised_pnl=pnl)
        # flipped through zero: the remainder opens at the fill price
        return replace(self, net_quantity=new, average_cost=Money(px, fill.price.currency),
                       realised_pnl=pnl)

    def unrealised_pnl(self, mark: Money) -> Money:
        if self.is_flat:
            return Money.zero(mark.currency)
        diff = (mark.amount - self.average_cost.amount) * Decimal(self.net_quantity)
        return Money(diff, mark.currency)

    def __str__(self) -> str:
        return (
            f"{self.symbol} {self.net_quantity:+,} @ {self.average_cost} "
            f"(realised {self.realised_pnl})"
        )


class Portfolio:
    """A book of positions, driven by orders and fills.

    Implements `__len__`, `__iter__`, `__getitem__` and `__contains__` so it
    behaves like a first-class collection — the Python answer to implementing
    `IReadOnlyDictionary<string, Position>`.
    """

    def __init__(self, currency: str = "INR") -> None:
        self.currency = currency
        self._positions: dict[str, Position] = {}
        self._orders: dict[str, Order] = {}

    # -- collection protocol ------------------------------------------------
    def __len__(self) -> int:
        return len(self._positions)

    def __iter__(self) -> Iterator[Position]:
        return iter(self._positions.values())

    def __contains__(self, symbol: object) -> bool:
        return symbol in self._positions

    def __getitem__(self, symbol: str) -> Position:
        return self._positions.get(
            symbol,
            Position(symbol, average_cost=Money.zero(self.currency),
                     realised_pnl=Money.zero(self.currency)),
        )

    # -- behaviour ----------------------------------------------------------
    def place(self, order: Order, checks: Iterable[RiskCheck] = ()) -> Order:
        if order.order_id in self._orders:
            raise ValidationError(f"duplicate order id {order.order_id}")
        order.validate(checks)
        self._orders[order.order_id] = order
        return order

    def execute(self, fill: Fill) -> Position:
        try:
            order = self._orders[fill.order_id]
        except KeyError as exc:
            raise ValidationError(f"unknown order {fill.order_id}") from exc
        order.apply(fill)
        pos = self[order.symbol].apply(order.side, fill)
        self._positions[order.symbol] = pos
        return pos

    @property
    def orders(self) -> tuple[Order, ...]:
        return tuple(self._orders.values())

    @property
    def realised_pnl(self) -> Money:
        total = Money.zero(self.currency)
        for p in self._positions.values():
            total = total + p.realised_pnl
        return total

    def unrealised_pnl(self, marks: dict[str, Money]) -> Money:
        total = Money.zero(self.currency)
        for p in self._positions.values():
            if p.symbol in marks:
                total = total + p.unrealised_pnl(marks[p.symbol])
        return total

    def gross_exposure(self, marks: dict[str, Money]) -> Money:
        total = Money.zero(self.currency)
        for p in self._positions.values():
            if p.symbol in marks:
                total = total + abs(marks[p.symbol] * abs(p.net_quantity))
        return total


# --------------------------------------------------------------------------- #
# Concrete risk checks (they satisfy RiskCheck structurally, not by inheritance)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MaxNotional:
    limit: Money

    def check(self, order: Order) -> None:
        if order.order_type is OrderType.MARKET:
            return
        if order.notional > self.limit:
            raise RiskLimitBreached(
                f"{order.order_id} notional {order.notional} exceeds limit {self.limit}"
            )


@dataclass(frozen=True)
class AllowedSymbols:
    universe: frozenset[str]

    def check(self, order: Order) -> None:
        if order.symbol not in self.universe:
            raise RiskLimitBreached(f"{order.symbol} is not in the tradable universe")
