"""Loaders for the course's four data surfaces.

Design note (the part worth internalising): each loader has an **offline path**
that reads a committed sample and an **online path** that hits a real source.
Only the offline path is exercised by tests. The online path exists so that the
same function signature works when you point it at the real world — that is the
whole trick behind a data layer you can test.

.NET analogy: think of the offline sample as an in-memory repository fake and
the online fetch as the SQL-backed implementation, except Python lets one
function carry both without an interface ceremony.
"""

from __future__ import annotations

import io
import json
from collections.abc import Iterable, Iterator
from pathlib import Path

import pandas as pd

from common.config import CONFIG, paths
from common.logging_utils import get_logger

log = get_logger(__name__)

SAMPLE_SYMBOLS = ["ALPHAINFRA", "BHARATCHEM", "COASTBANK", "DECCANMOT", "EASTPOWER"]

OHLCV_COLUMNS = ["date", "symbol", "open", "high", "low", "close", "prev_close", "volume"]
BHAVCOPY_COLUMNS = [
    "SYMBOL", "SERIES", "OPEN", "HIGH", "LOW", "CLOSE", "LAST", "PREVCLOSE",
    "TOTTRDQTY", "TOTTRDVAL", "TIMESTAMP", "TOTALTRADES", "ISIN",
]


class NetworkDisabled(RuntimeError):
    """Raised when an online fetch is attempted without FORGE_ALLOW_NETWORK=1."""


def _require_network(what: str) -> None:
    if not CONFIG.allow_network:
        raise NetworkDisabled(
            f"{what} needs the network. Re-run with FORGE_ALLOW_NETWORK=1 "
            "(tests must never need this)."
        )


def _sample(name: str) -> Path:
    p = paths.samples / name
    if not p.exists():  # pragma: no cover - only if the repo is incomplete
        raise FileNotFoundError(
            f"missing sample {p}; regenerate with `python3 common/data/make_samples.py`"
        )
    return p


# --------------------------------------------------------------------------- #
# Offline loaders                                                             #
# --------------------------------------------------------------------------- #
def load_ohlcv(symbols: Iterable[str] | None = None, *, as_index: bool = False) -> pd.DataFrame:
    """Daily OHLCV for the sample universe, sorted by (symbol, date).

    Args:
        symbols: restrict to these tickers; ``None`` means all of them.
        as_index: return a (symbol, date) MultiIndex instead of plain columns.
    """
    df = pd.read_csv(_sample("ohlcv_sample.csv"), parse_dates=["date"])
    if symbols is not None:
        wanted = {s.upper() for s in symbols}
        unknown = wanted - set(df["symbol"].unique())
        if unknown:
            raise KeyError(f"unknown symbols: {sorted(unknown)}")
        df = df[df["symbol"].isin(wanted)]
    df = df.sort_values(["symbol", "date"], kind="stable").reset_index(drop=True)
    return df.set_index(["symbol", "date"]) if as_index else df


def load_bhavcopy(*, normalise: bool = False) -> pd.DataFrame:
    """The NSE cash-market bhavcopy sample.

    With ``normalise=True`` the exchange's SHOUTY schema is mapped onto the
    tidy OHLCV schema, which is what every downstream topic actually wants.
    """
    df = pd.read_csv(_sample("nse_bhavcopy_sample.csv"))
    if not normalise:
        return df
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df["TIMESTAMP"], format="%d-%b-%Y"),
            "symbol": df["SYMBOL"],
            "open": df["OPEN"],
            "high": df["HIGH"],
            "low": df["LOW"],
            "close": df["CLOSE"],
            "prev_close": df["PREVCLOSE"],
            "volume": df["TOTTRDQTY"],
        }
    )
    return out.sort_values(["symbol", "date"], kind="stable").reset_index(drop=True)


def load_orders() -> pd.DataFrame:
    """Synthetic OMS order book (paper only — no venue ever sees these)."""
    return pd.read_csv(_sample("orders_sample.csv"), parse_dates=["date"])


def load_filings(
    issuer: str | None = None, section: str | None = None
) -> list[dict]:
    """EDGAR-shaped filing excerpts, filtered by issuer and/or section."""
    docs: list[dict] = []
    with _sample("filings_sample.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            if issuer and d["issuer"].upper() != issuer.upper():
                continue
            if section and d["section"] != section:
                continue
            docs.append(d)
    return docs


def iter_filing_texts() -> Iterator[str]:
    """Convenience stream of raw filing text — the corpus for FinTok (T30)."""
    for d in load_filings():
        yield d["text"]


# --------------------------------------------------------------------------- #
# Online fetchers (opt-in; cached; never used by tests)                        #
# --------------------------------------------------------------------------- #
def _cache_path(name: str) -> Path:
    paths.ensure()
    return paths.cache / name


def fetch_ohlcv_live(symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """Daily bars from yfinance, cached under `common/data/cache/` (gitignored)."""
    _require_network("fetch_ohlcv_live")
    cache = _cache_path(f"yf_{symbol}_{period}_{interval}.csv")
    if cache.exists():
        log.info("cache hit %s", cache.name)
        return pd.read_csv(cache, parse_dates=["date"])
    import yfinance as yf  # imported lazily: an optional dependency

    raw = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=False)
    df = (
        raw.reset_index()
        .rename(columns={c: c.lower().replace(" ", "_") for c in raw.reset_index().columns})
        .assign(symbol=symbol.upper())
    )
    df["prev_close"] = df["close"].shift(1)
    df = df[[c for c in OHLCV_COLUMNS if c in df.columns]]
    df.to_csv(cache, index=False)
    return df


def fetch_bhavcopy_live(day: str) -> pd.DataFrame:
    """NSE cash-market bhavcopy for one trading day (``YYYY-MM-DD``).

    Personal-use terms only; the course never redistributes what this returns.
    """
    _require_network("fetch_bhavcopy_live")
    import requests

    d = pd.Timestamp(day)
    url = (
        "https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_"
        f"{d.strftime('%d%m%Y')}.csv"
    )
    cache = _cache_path(f"bhav_{d.date()}.csv")
    if cache.exists():
        return pd.read_csv(cache)
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "text/csv,*/*"}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    df.columns = [c.strip().upper() for c in df.columns]
    df.to_csv(cache, index=False)
    return df


def fetch_edgar_filing_live(cik: str, form: str = "10-K", limit: int = 1) -> list[dict]:
    """Recent EDGAR filing metadata for a CIK (fair-use excerpts only)."""
    _require_network("fetch_edgar_filing_live")
    import requests

    cik10 = str(cik).lstrip("0").zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{cik10}.json"
    resp = requests.get(url, headers={"User-Agent": CONFIG.edgar_user_agent}, timeout=30)
    resp.raise_for_status()
    recent = resp.json()["filings"]["recent"]
    out = []
    for i, f in enumerate(recent["form"]):
        if f != form:
            continue
        out.append(
            {
                "cik": cik10,
                "form": f,
                "accession": recent["accessionNumber"][i],
                "filing_date": recent["filingDate"][i],
                "primary_document": recent["primaryDocument"][i],
            }
        )
        if len(out) >= limit:
            break
    return out
