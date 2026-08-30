"""The data layer's contract: offline, deterministic, and shaped like the real feeds."""

import pytest

from common.config import Config
from common.data import loaders
from common.data.loaders import (
    BHAVCOPY_COLUMNS,
    OHLCV_COLUMNS,
    NetworkDisabled,
    iter_filing_texts,
    load_bhavcopy,
    load_filings,
    load_ohlcv,
    load_orders,
)


def test_ohlcv_schema_and_ordering():
    df = load_ohlcv()
    assert list(df.columns) == OHLCV_COLUMNS
    assert len(df) == 1300
    for _, g in df.groupby("symbol"):
        assert g["date"].is_monotonic_increasing


def test_ohlcv_is_internally_consistent():
    df = load_ohlcv()
    assert (df["high"] >= df["low"]).all()
    assert (df["high"] >= df[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (df["low"] <= df[["open", "close"]].min(axis=1) + 1e-9).all()
    assert (df["volume"] > 0).all()


def test_ohlcv_symbol_filter_and_unknown_symbol():
    df = load_ohlcv(["ALPHAINFRA"])
    assert set(df["symbol"]) == {"ALPHAINFRA"}
    with pytest.raises(KeyError):
        load_ohlcv(["NOTLISTED"])


def test_bhavcopy_raw_schema():
    raw = load_bhavcopy()
    assert list(raw.columns) == BHAVCOPY_COLUMNS
    assert set(raw["SERIES"]) == {"EQ"}


def test_bhavcopy_normalises_to_ohlcv():
    norm = load_bhavcopy(normalise=True)
    ohlcv = load_ohlcv()
    assert set(norm.columns) == set(OHLCV_COLUMNS)
    assert len(norm) == len(ohlcv)
    assert norm["close"].tolist() == ohlcv["close"].tolist()
    assert norm["date"].tolist() == ohlcv["date"].tolist()


def test_orders_are_paper_only_and_well_formed():
    o = load_orders()
    assert set(o["venue"]) == {"SIMBOOK"}, "no real venue may ever appear here"
    assert (o["filled_quantity"] <= o["quantity"]).all()
    assert set(o["status"]) <= {"NEW", "PARTIAL", "FILLED"}
    filled = o[o["status"] == "FILLED"]
    assert (filled["filled_quantity"] == filled["quantity"]).all()
    assert (o.loc[o["filled_quantity"] == 0, "avg_fill_price"] == 0).all()


def test_filings_filtering():
    all_docs = load_filings()
    assert len(all_docs) == 45
    assert all(d["synthetic"] is True for d in all_docs), "no licensed text may be committed"
    mdna = load_filings(section="mdna")
    assert len(mdna) == 15
    one = load_filings(issuer="COASTBANK")
    assert {d["issuer"] for d in one} == {"COASTBANK"}
    assert sum(1 for _ in iter_filing_texts()) == len(all_docs)


def test_online_paths_refuse_without_opt_in(monkeypatch: pytest.MonkeyPatch):
    # Config is frozen, so swap the whole object rather than poking a field.
    monkeypatch.setattr(loaders, "CONFIG", Config(allow_network=False))
    for fn, args in [
        (loaders.fetch_ohlcv_live, ("AAPL",)),
        (loaders.fetch_bhavcopy_live, ("2024-01-02",)),
        (loaders.fetch_edgar_filing_live, ("320193",)),
    ]:
        with pytest.raises(NetworkDisabled):
            fn(*args)
