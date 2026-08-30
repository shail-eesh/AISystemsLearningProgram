#!/usr/bin/env python3
"""Generate the committed sample datasets.

Deterministic from `CONFIG.seed`, so re-running this reproduces byte-identical
files. The data is **synthetic**: a geometric-Brownian-motion price path per
fictional issuer, dressed in the column names real feeds use. That gives the
course realistic *shape* (gaps, weekends skipped, volume skew, corporate
announcements) without redistributing anyone's licensed data.

    python3 common/data/make_samples.py
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from common.config import CONFIG, paths

SAMPLE_SYMBOLS = ["ALPHAINFRA", "BHARATCHEM", "COASTBANK", "DECCANMOT", "EASTPOWER"]
SERIES = "EQ"
START = date(2024, 1, 1)
N_DAYS = 260  # ~1 trading year


def _trading_days(start: date, n: int) -> list[date]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:  # Mon-Fri; the course does not model NSE holidays
            out.append(d)
        d += timedelta(days=1)
    return out


def _price_path(rng: np.random.Generator, n: int, s0: float, mu: float, sigma: float) -> np.ndarray:
    shocks = rng.normal(mu / n, sigma / np.sqrt(n), size=n)
    return s0 * np.exp(np.cumsum(shocks))


def build_ohlcv(rng: np.random.Generator) -> pd.DataFrame:
    days = _trading_days(START, N_DAYS)
    frames = []
    for i, sym in enumerate(SAMPLE_SYMBOLS):
        close = _price_path(rng, N_DAYS, s0=120.0 + 90 * i, mu=0.08 + 0.03 * i, sigma=0.28 + 0.05 * i)
        intraday = np.abs(rng.normal(0.0, 0.011, size=N_DAYS)) + 0.002
        open_ = close * (1 + rng.normal(0, 0.004, size=N_DAYS))
        high = np.maximum(open_, close) * (1 + intraday)
        low = np.minimum(open_, close) * (1 - intraday)
        prev_close = np.concatenate([[close[0] * (1 + rng.normal(0, 0.004))], close[:-1]])
        volume = rng.lognormal(mean=12.4 + 0.2 * i, sigma=0.55, size=N_DAYS).astype(np.int64)
        frames.append(
            pd.DataFrame(
                {
                    "date": [d.isoformat() for d in days],
                    "symbol": sym,
                    "open": np.round(open_, 2),
                    "high": np.round(high, 2),
                    "low": np.round(low, 2),
                    "close": np.round(close, 2),
                    "prev_close": np.round(prev_close, 2),
                    "volume": volume,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def to_bhavcopy(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Re-dress OHLCV in NSE bhavcopy column names (the real feed's schema)."""
    df = ohlcv.copy()
    turnover = ((df["high"] + df["low"] + df["close"]) / 3.0) * df["volume"]
    return pd.DataFrame(
        {
            "SYMBOL": df["symbol"],
            "SERIES": SERIES,
            "OPEN": df["open"],
            "HIGH": df["high"],
            "LOW": df["low"],
            "CLOSE": df["close"],
            "LAST": df["close"],
            "PREVCLOSE": df["prev_close"],
            "TOTTRDQTY": df["volume"],
            "TOTTRDVAL": np.round(turnover, 2),
            "TIMESTAMP": pd.to_datetime(df["date"]).dt.strftime("%d-%b-%Y").str.upper(),
            "TOTALTRADES": (df["volume"] // 180).astype("int64") + 7,
            "ISIN": ["INE" + f"{abs(hash(s)) % 10**6:06d}" + "01011" for s in df["symbol"]],
        }
    )


def build_orders(rng: np.random.Generator, ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Synthetic OMS traffic: parent orders with partial fills. Paper only."""
    rows = []
    px = ohlcv.set_index(["symbol", "date"])["close"].to_dict()
    days = sorted(ohlcv["date"].unique())[-60:]
    oid = 100000
    for day in days:
        for _ in range(int(rng.integers(2, 6))):
            sym = SAMPLE_SYMBOLS[int(rng.integers(0, len(SAMPLE_SYMBOLS)))]
            ref = px[(sym, day)]
            side = "BUY" if rng.random() < 0.55 else "SELL"
            qty = int(rng.integers(1, 25)) * 50
            limit = round(ref * (1 + rng.normal(0, 0.004)), 2)
            filled = int(qty * min(1.0, max(0.0, rng.beta(6, 2))))
            filled -= filled % 5
            avg = round(limit * (1 + rng.normal(0, 0.0015)), 4) if filled else 0.0
            status = "FILLED" if filled == qty else ("PARTIAL" if filled else "NEW")
            oid += 1
            rows.append(
                {
                    "order_id": f"AD-{oid}",
                    "date": day,
                    "symbol": sym,
                    "side": side,
                    "order_type": "LIMIT",
                    "quantity": qty,
                    "limit_price": limit,
                    "filled_quantity": filled,
                    "avg_fill_price": avg,
                    "status": status,
                    "venue": "SIMBOOK",
                }
            )
    return pd.DataFrame(rows)


FILING_TEMPLATE = {
    "risk_factors": (
        "Our results depend on commodity input prices, which have been volatile. "
        "A sustained increase in {input_} costs that we cannot pass through to customers "
        "would compress gross margin. We operate in a concentrated customer base; the "
        "loss of any of our three largest customers would materially reduce revenue."
    ),
    "mdna": (
        "Revenue for the period was {rev} crore, {dir_} {pct} percent year over year, driven "
        "primarily by {driver}. Gross margin was {gm} percent compared with {gm_prior} percent "
        "in the prior-year period. Operating expenses grew {opex} percent, reflecting continued "
        "investment in distribution."
    ),
    "liquidity": (
        "As of the balance sheet date the Company held {cash} crore in cash and cash equivalents "
        "and had {debt} crore of long-term borrowings. Management believes existing liquidity is "
        "sufficient to fund operations for at least the next twelve months."
    ),
}


def build_filings(rng: np.random.Generator) -> list[dict]:
    inputs = ["naphtha", "steel", "polymer", "copper", "coking coal"]
    drivers = [
        "higher realizations in the domestic segment",
        "volume growth in the export book",
        "the full-period contribution of the Pune facility",
        "improved capacity utilisation",
        "a favourable product mix",
    ]
    docs = []
    for i, sym in enumerate(SAMPLE_SYMBOLS):
        for q, form in enumerate(["10-K", "10-Q", "10-Q"]):
            rev = round(float(rng.uniform(400, 3200)), 1)
            pct = round(float(rng.uniform(-14, 31)), 1)
            gm = round(float(rng.uniform(18, 41)), 1)
            fields = {
                "input_": inputs[i % len(inputs)],
                "rev": rev,
                "dir_": "up" if pct >= 0 else "down",
                "pct": abs(pct),
                "driver": drivers[(i + q) % len(drivers)],
                "gm": gm,
                "gm_prior": round(gm - float(rng.uniform(-4, 4)), 1),
                "opex": round(float(rng.uniform(1, 22)), 1),
                "cash": round(float(rng.uniform(50, 900)), 1),
                "debt": round(float(rng.uniform(0, 1400)), 1),
            }
            for section, template in FILING_TEMPLATE.items():
                docs.append(
                    {
                        "doc_id": f"{sym}-{form}-{2024 + q // 3}-{q}-{section}",
                        "issuer": sym,
                        "form": form,
                        "fiscal_period": f"FY{2024 + q // 3}Q{(q % 4) + 1}",
                        "section": section,
                        "text": template.format(**fields),
                        "synthetic": True,
                    }
                )
    return docs


def main(out_dir: Path | None = None) -> dict[str, Path]:
    out = out_dir or paths.samples
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(CONFIG.seed)

    ohlcv = build_ohlcv(rng)
    bhav = to_bhavcopy(ohlcv)
    orders = build_orders(rng, ohlcv)
    filings = build_filings(rng)

    written = {}
    ohlcv.to_csv(out / "ohlcv_sample.csv", index=False)
    written["ohlcv"] = out / "ohlcv_sample.csv"
    bhav.to_csv(out / "nse_bhavcopy_sample.csv", index=False)
    written["bhavcopy"] = out / "nse_bhavcopy_sample.csv"
    orders.to_csv(out / "orders_sample.csv", index=False)
    written["orders"] = out / "orders_sample.csv"
    with (out / "filings_sample.jsonl").open("w", encoding="utf-8") as fh:
        for d in filings:
            fh.write(json.dumps(d, sort_keys=True) + "\n")
    written["filings"] = out / "filings_sample.jsonl"

    (out / "README.md").write_text(
        "# Sample datasets (synthetic)\n\n"
        "Generated by `common/data/make_samples.py` from seed "
        f"`{CONFIG.seed}`. Issuers are fictional and prices are simulated — nothing here is\n"
        "real or licensed market data. The column names mirror the real feeds (NSE bhavcopy,\n"
        "yfinance OHLCV, EDGAR filing sections) so the loaders you write against these samples\n"
        "work unchanged on live data.\n\n"
        "| file | rows | shape mirrors |\n|---|---:|---|\n"
        f"| `ohlcv_sample.csv` | {len(ohlcv)} | yfinance daily OHLCV |\n"
        f"| `nse_bhavcopy_sample.csv` | {len(bhav)} | NSE cash-market bhavcopy |\n"
        f"| `orders_sample.csv` | {len(orders)} | OMS parent orders + fills (paper) |\n"
        f"| `filings_sample.jsonl` | {len(filings)} | EDGAR 10-K/10-Q sections |\n",
        encoding="utf-8",
    )
    written["readme"] = out / "README.md"
    return written


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    for k, v in main().items():
        print(f"{k:10s} -> {v.relative_to(paths.repo)}")
