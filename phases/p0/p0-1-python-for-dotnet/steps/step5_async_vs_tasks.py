#!/usr/bin/env python3
"""Step 5 — async/await: same keywords, different machine.

Run:  python3 steps/step5_async_vs_tasks.py

The keywords look identical to C#'s and the semantics are not. What actually
differs, in the order it bites you:

1. **A coroutine does nothing until awaited.** `Task` in .NET is usually *hot*
   — `DoAsync()` has already started before you await it. Calling an `async def`
   returns a cold coroutine object; forget the `await` and the work never runs
   (Python at least warns you: "coroutine was never awaited").
2. **One thread, one event loop.** There is no thread pool underneath.
   `async` buys you concurrency, never parallelism; a CPU-bound `await` blocks
   every other task. `Task.Run(...)` for CPU work maps to
   `run_in_executor`/`ProcessPoolExecutor`, not to `async def`.
3. **No `ConfigureAwait(false)`, no sync context, so no classic deadlock.**
   The flip side: you cannot block on a coroutine from sync code at all —
   `asyncio.run` is the only entry point, and it fails if a loop is running.
4. **`Task.WhenAll` is `asyncio.gather`**; `WhenAny` is `asyncio.wait(...,
   return_when=FIRST_COMPLETED)`; `CancellationToken` is task cancellation
   delivered as a `CancelledError` *exception* at the next await point.
5. **`async with` / `async for`** exist as first-class forms — the async
   `IAsyncDisposable` / `IAsyncEnumerable` equivalents.
"""

import asyncio
import time
from datetime import UTC, datetime

import _bootstrap  # noqa: F401
from p0_1_oms import Fill, Money, Order, Portfolio, Quantity, Side


async def quote(symbol: str, latency: float) -> tuple[str, Money]:
    """Pretend to ask the simulated book for a price."""
    await asyncio.sleep(latency)
    return symbol, Money("100") + Money(str(len(symbol)))


async def demo_cold_coroutines() -> None:
    coro = quote("ALPHAINFRA", 0.01)
    print(f"  calling an async def returned {type(coro).__name__} — nothing ran yet")
    print(f"  awaited: {(await coro)[1]}")


async def demo_gather_is_when_all() -> None:
    symbols = {"ALPHAINFRA": 0.10, "COASTBANK": 0.10, "EASTPOWER": 0.10}
    t0 = time.perf_counter()
    results = await asyncio.gather(*(quote(s, d) for s, d in symbols.items()))
    concurrent = time.perf_counter() - t0

    t0 = time.perf_counter()
    for s, d in symbols.items():
        await quote(s, d)
    sequential = time.perf_counter() - t0

    print(f"  gather (WhenAll): {concurrent:.2f}s for {len(symbols)} quotes")
    print(f"  sequential:       {sequential:.2f}s  -> {sequential / concurrent:.1f}x")
    assert concurrent < sequential / 2
    print(f"  quotes: {[(s, str(m)) for s, m in results]}")


async def demo_cancellation() -> None:
    async def never() -> None:
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            print("  CancelledError arrived at the await point (this is the token)")
            raise

    task = asyncio.create_task(never())
    await asyncio.sleep(0.01)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        print("  ...and propagated to the awaiter")


async def demo_timeout_and_async_context() -> None:
    try:
        async with asyncio.timeout(0.05):
            await quote("SLOWFEED", 5.0)
    except TimeoutError:
        print("  `async with asyncio.timeout(...)` is the CancellationTokenSource timer")


async def demo_async_iteration_over_fills() -> None:
    book = Portfolio()
    order = book.place(Order("A1", "ALPHAINFRA", Side.BUY, Quantity(200),
                             limit_price=Money("102")))

    async def stream():
        """`IAsyncEnumerable<Fill>`: an async generator."""
        for i, (qty, px) in enumerate([(120, "101.20"), (80, "101.60")], start=1):
            await asyncio.sleep(0.01)
            yield Fill(f"A1-F{i}", "A1", Quantity(qty), Money(px), datetime.now(UTC))

    async for fill in stream():                # `await foreach`
        book.execute(fill)
    print(f"  {order} avg={order.average_fill_price}")


def demo_cpu_bound_warning() -> None:
    print("  reminder: `async def` never parallelises CPU work — the GIL and the")
    print("  single event loop mean a tight numeric loop starves every other task.")
    print("  For that, Phase 0.2 switches to NumPy (vectorise) not to threads.")


async def main() -> None:
    print("cold coroutines:")
    await demo_cold_coroutines()
    print("gather vs sequential:")
    await demo_gather_is_when_all()
    print("cancellation:")
    await demo_cancellation()
    print("timeout:")
    await demo_timeout_and_async_context()
    print("async iteration:")
    await demo_async_iteration_over_fills()
    print("cpu-bound:")
    demo_cpu_bound_warning()


if __name__ == "__main__":
    asyncio.run(main())          # the ONLY way into the loop from sync code
    print("\nstep 5 OK")
