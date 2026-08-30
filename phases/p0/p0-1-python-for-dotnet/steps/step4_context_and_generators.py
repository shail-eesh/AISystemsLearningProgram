#!/usr/bin/env python3
"""Step 4 — `using` becomes `with`; `IEnumerable`/`yield return` becomes a generator.

Run:  python3 steps/step4_context_and_generators.py

`using (var tx = ...)` -> `with transaction(...)`. `IDisposable` is the pair
`__enter__`/`__exit__`, and `@contextmanager` writes both from one function
whose `yield` is the body. Unlike `IDisposable`, `__exit__` sees the exception
and can swallow it by returning True — powerful, and easy to abuse.

`IEnumerable<T>` + `yield return` maps almost exactly onto a generator
function: lazy, single-pass, and O(1) in memory. `itertools` is the LINQ
standard library.
"""

import itertools
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime

import _bootstrap  # noqa: F401
from p0_1_oms import Fill, Money, Order, Portfolio, Quantity, Side, ValidationError


@contextmanager
def order_batch(book: Portfolio, label: str) -> Iterator[list[Order]]:
    """All-or-nothing placement. The `yield` is where the `with` body runs."""
    staged: list[Order] = []
    print(f"  [{label}] batch open")
    try:
        yield staged
    except Exception as exc:
        print(f"  [{label}] rolled back {len(staged)} order(s): {exc}")
        for o in staged:
            if o.order_id in {x.order_id for x in book.orders} and not o.fills:
                o.cancel("batch rollback")
        raise
    else:
        print(f"  [{label}] committed {len(staged)} order(s)")
    finally:
        print(f"  [{label}] batch closed (this runs either way, like a finally block)")


def demo_context_manager() -> None:
    book = Portfolio()
    with order_batch(book, "good") as staged:
        for i, sym in enumerate(["ALPHAINFRA", "COASTBANK"], start=1):
            staged.append(book.place(Order(f"B{i}", sym, Side.BUY, Quantity(50),
                                           limit_price=Money("100"))))
    try:
        with order_batch(book, "bad") as staged:
            staged.append(book.place(Order("B3", "EASTPOWER", Side.BUY, Quantity(50),
                                           limit_price=Money("100"))))
            raise ValidationError("risk system offline")
    except ValidationError:
        pass
    print(f"  statuses: {[o.status.value for o in book.orders]}")


def fills_for(order: Order, slices: list[int], prices: list[str]) -> Iterator[Fill]:
    """A generator: the Python `IEnumerable<Fill>` with `yield return`."""
    for n, (qty, px) in enumerate(zip(slices, prices, strict=True), start=1):
        yield Fill(f"{order.order_id}-F{n}", order.order_id, Quantity(qty),
                   Money(px), datetime.now(UTC))


def demo_generators_are_lazy() -> None:
    book = Portfolio()
    order = book.place(Order("G1", "ALPHAINFRA", Side.BUY, Quantity(300),
                             limit_price=Money("102")))
    stream = fills_for(order, [100, 100, 100], ["101.00", "101.50", "101.90"])
    print(f"  generator built, nothing executed yet: {stream}")
    first = next(stream)
    print(f"  pulled one: {first.fill_id} {first.quantity} @ {first.price}")
    for f in stream:                       # resumes where it left off
        book.execute(f)
    book.execute(first)
    print(f"  {order}  avg={order.average_fill_price}")
    assert next(stream, None) is None, "a generator is single-pass; it is now exhausted"


def demo_itertools_is_linq() -> None:
    book = Portfolio()
    for i, (sym, side) in enumerate(
        [("ALPHAINFRA", Side.BUY), ("COASTBANK", Side.BUY), ("ALPHAINFRA", Side.SELL)]
    ):
        book.place(Order(f"L{i}", sym, side, Quantity(10), limit_price=Money("100")))

    # C#: orders.GroupBy(o => o.Symbol).OrderBy(g => g.Key)
    by_symbol = itertools.groupby(sorted(book.orders, key=lambda o: o.symbol),
                                  key=lambda o: o.symbol)
    for symbol, group in by_symbol:
        print(f"  {symbol}: {[o.order_id for o in group]}")
    # C#: orders.Take(2) — lazily, without materialising the list
    print(f"  first two ids: {[o.order_id for o in itertools.islice(book.orders, 2)]}")


if __name__ == "__main__":
    demo_context_manager()
    demo_generators_are_lazy()
    demo_itertools_is_linq()
    print("\nstep 4 OK")
