"""Fixtures: real (synthetic) price series plus a deterministic random walk."""

import numpy as np
import pytest

from common.data import load_ohlcv

SYMBOLS = ["ALPHAINFRA", "BHARATCHEM", "COASTBANK", "DECCANMOT", "EASTPOWER"]


@pytest.fixture(scope="session")
def prices():
    return load_ohlcv()


@pytest.fixture(params=SYMBOLS)
def series(request, prices):
    """One symbol's OHLCV as float64 arrays, parametrized over the universe."""
    g = prices[prices["symbol"] == request.param]
    return {
        "symbol": request.param,
        "close": g["close"].to_numpy(dtype=np.float64),
        "high": g["high"].to_numpy(dtype=np.float64),
        "low": g["low"].to_numpy(dtype=np.float64),
        "prev_close": g["prev_close"].to_numpy(dtype=np.float64),
        "volume": g["volume"].to_numpy(dtype=np.float64),
    }


@pytest.fixture
def walk():
    rng = np.random.default_rng(1729)
    return 100.0 + np.cumsum(rng.normal(0, 0.8, 500))
