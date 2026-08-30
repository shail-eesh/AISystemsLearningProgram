#!/usr/bin/env python3
"""Verification benchmark for P0.1 — replay the synthetic OMS tape.

    python3 phases/p0/p0-1-python-for-dotnet/bench/replay_orders.py

Every row of `common/data/samples/orders_sample.csv` is placed through the
domain model and filled at its recorded average price. Two independent
reconciliations then have to agree:

1. **Net quantity** against a one-line pandas groupby of signed filled shares.
2. **Realised P&L** against a *separately written* float-based average-cost
   reducer in this file — deliberately not the `Position` code under test.

Agreement to 1e-6 (relative) is the "done when" for this topic. The float
reference is also where you see the cost of the shortcut: the residual is not
zero, it is float noise.
"""

from __future__ import annotations

import json
import pathlib
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime

TOPIC = pathlib.Path(__file__).resolve().parent.parent
REPO = TOPIC.parents[2]
sys.path[:0] = [str(REPO), str(TOPIC / "src")]

import pandas as pd  # noqa: E402
from p0_1_oms import Fill, Money, Order, OrderStatus, Portfolio, Quantity, Side  # noqa: E402

from common.data import load_orders  # noqa: E402


def float_reference(rows: pd.DataFrame) -> dict[str, dict[str, float]]:
    """An independent average-cost reducer, in plain floats. No shared code."""
    net: dict[str, float] = defaultdict(float)
    cost: dict[str, float] = defaultdict(float)
    realised: dict[str, float] = defaultdict(float)
    for r in rows.itertuples():
        if r.filled_quantity == 0:
            continue
        sym = r.symbol
        signed = r.filled_quantity * (1 if r.side == "BUY" else -1)
        px, old = float(r.avg_fill_price), net[sym]
        new = old + signed
        if old == 0 or (old > 0) == (signed > 0):
            cost[sym] = (cost[sym] * abs(old) + px * abs(signed)) / abs(new) if new else 0.0
        else:
            closed = min(abs(old), abs(signed))
            realised[sym] += (px - cost[sym]) * closed * (1 if old > 0 else -1)
            if new == 0:
                cost[sym] = 0.0
            elif (new > 0) != (old > 0):
                cost[sym] = px
        net[sym] = new
    return {
        s: {"net": net[s], "avg_cost": cost[s], "realised": realised[s]} for s in sorted(net)
    }


def replay(rows: pd.DataFrame) -> tuple[Portfolio, dict[str, int]]:
    book = Portfolio()
    counts = {"placed": 0, "filled": 0, "partial": 0, "unfilled": 0}
    for r in rows.itertuples():
        side = Side(r.side)
        order = Order(
            order_id=r.order_id,
            symbol=r.symbol,
            side=side,
            quantity=Quantity(int(r.quantity)),
            limit_price=Money(f"{r.limit_price:.4f}"),
            trade_date=pd.Timestamp(r.date).date(),
        )
        book.place(order)
        counts["placed"] += 1
        if r.filled_quantity == 0:
            counts["unfilled"] += 1
            continue
        book.execute(
            Fill(
                fill_id=f"{r.order_id}-F1",
                order_id=r.order_id,
                quantity=Quantity(int(r.filled_quantity)),
                price=Money(f"{r.avg_fill_price:.4f}"),
                at=datetime.combine(pd.Timestamp(r.date).date(), datetime.min.time(), tzinfo=UTC),
            )
        )
        counts["filled" if order.status is OrderStatus.FILLED else "partial"] += 1
    return book, counts


def main() -> int:
    rows = load_orders().sort_values(["date", "order_id"], kind="stable")
    t0 = time.perf_counter()
    book, counts = replay(rows)
    elapsed = time.perf_counter() - t0

    ref = float_reference(rows)

    # -- reconciliation 1: net quantity, against a pandas groupby -----------
    signed = rows["filled_quantity"] * rows["side"].map({"BUY": 1, "SELL": -1})
    pandas_net = signed.groupby(rows["symbol"]).sum().to_dict()
    qty_breaks = {
        s: (int(book[s].net_quantity), int(pandas_net.get(s, 0)))
        for s in pandas_net
        if int(book[s].net_quantity) != int(pandas_net[s])
    }

    # -- reconciliation 2: realised P&L, against the float reference --------
    pnl_rows, worst = [], 0.0
    for sym, r in ref.items():
        model = float(book[sym].realised_pnl.amount)
        denom = max(1.0, abs(r["realised"]))
        rel = abs(model - r["realised"]) / denom
        worst = max(worst, rel)
        pnl_rows.append(
            {
                "symbol": sym,
                "net_quantity": int(book[sym].net_quantity),
                "avg_cost_model": float(book[sym].average_cost.amount),
                "avg_cost_reference": round(r["avg_cost"], 6),
                "realised_model": model,
                "realised_reference": round(r["realised"], 6),
                "relative_error": rel,
            }
        )

    tolerance = 1e-6
    passed = not qty_breaks and worst <= tolerance
    results = {
        "topic": "P0.1",
        "benchmark": "OMS tape replay + dual reconciliation",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "input": "common/data/samples/orders_sample.csv (synthetic)",
        "orders": counts,
        "symbols": len(pnl_rows),
        "elapsed_seconds": round(elapsed, 4),
        "orders_per_second": round(counts["placed"] / elapsed, 1),
        "tolerance": tolerance,
        "worst_relative_error": worst,
        "quantity_breaks": qty_breaks,
        "per_symbol": pnl_rows,
        "portfolio_realised_pnl_inr": float(book.realised_pnl.amount),
        "passed": passed,
        "note": (
            "Decimal model vs an independently written float reducer. The residual is "
            "float noise in the reference, not error in the model."
        ),
    }
    out = pathlib.Path(__file__).parent / "results.json"
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print(f"replayed {counts['placed']} orders in {elapsed:.3f}s "
          f"({results['orders_per_second']}/s)")
    print(f"  filled={counts['filled']} partial={counts['partial']} unfilled={counts['unfilled']}")
    print(f"  symbols reconciled: {len(pnl_rows)}  quantity breaks: {len(qty_breaks)}")
    print(f"  worst relative P&L error: {worst:.3e} (tolerance {tolerance:.0e})")
    print(f"  book realised P&L: {book.realised_pnl}")
    print(f"  -> {'PASS' if passed else 'FAIL'}; wrote {out.relative_to(REPO)}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
