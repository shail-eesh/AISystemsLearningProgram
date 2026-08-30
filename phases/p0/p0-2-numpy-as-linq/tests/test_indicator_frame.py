"""The multi-symbol feature table — where the classic groupby bug lives."""

import numpy as np
import pandas as pd
import pytest
from p0_2_indicators import INDICATOR_COLUMNS, indicator_frame, sma

from common.data import load_ohlcv


def test_frame_has_every_column_and_row():
    prices = load_ohlcv()
    out = indicator_frame(prices)
    assert len(out) == len(prices)
    assert set(INDICATOR_COLUMNS) <= set(out.columns)
    assert out["symbol"].nunique() == 5


def test_missing_columns_are_reported():
    with pytest.raises(KeyError, match="missing columns"):
        indicator_frame(pd.DataFrame({"symbol": ["A"], "close": [1.0]}))


def test_indicators_do_not_bleed_across_symbols():
    """Compute per symbol; compare with the same symbol computed alone."""
    prices = load_ohlcv()
    out = indicator_frame(prices)
    for sym, g in out.groupby("symbol"):
        alone = load_ohlcv([sym])["close"].to_numpy(dtype=np.float64)
        expected = sma(alone, 20)
        got = g.sort_values("date")["sma_20"].to_numpy(dtype=np.float64)
        assert np.array_equal(got, expected, equal_nan=True), f"{sym} contaminated"


def test_naive_whole_frame_computation_would_be_wrong():
    """Proof the groupby is load-bearing, not decoration."""
    prices = load_ohlcv().sort_values(["symbol", "date"])
    naive = sma(prices["close"].to_numpy(dtype=np.float64), 20)
    correct = indicator_frame(prices)["sma_20"].to_numpy(dtype=np.float64)
    boundary = 260          # first row of the second symbol
    assert not np.array_equal(naive[boundary:boundary + 19],
                              correct[boundary:boundary + 19], equal_nan=True), (
        "computing across the concatenated frame must differ at symbol boundaries"
    )
    assert np.isnan(correct[boundary:boundary + 19]).all(), "each symbol re-warms up"


def test_warmup_is_per_symbol():
    out = indicator_frame(load_ohlcv())
    for _, g in out.groupby("symbol"):
        g = g.sort_values("date")
        assert g["sma_20"].iloc[:19].isna().all()
        assert not g["sma_20"].iloc[19:].isna().any()
