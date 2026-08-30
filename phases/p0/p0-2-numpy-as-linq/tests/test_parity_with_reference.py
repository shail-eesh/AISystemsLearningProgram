"""Parity against the pandas reference — the "done when" of P0.2.

TA-Lib will not build in this sandbox, so pandas (`rolling`, `ewm`) is the
independent implementation. Tolerance is 1e-6 relative, per the master plan;
in practice everything lands at machine epsilon.
"""

import numpy as np
import pytest
from p0_2_indicators import atr, bollinger, ema, macd, rsi, sma, vwap, wilder_rma
from p0_2_indicators import references as ref

TOL = 1e-6


def close_enough(got: np.ndarray, want: np.ndarray, tol: float = TOL) -> float:
    """Assert NaN masks match, return the worst relative error on the rest."""
    assert got.shape == want.shape
    assert np.array_equal(np.isnan(got), np.isnan(want)), "warm-up NaN masks differ"
    m = ~np.isnan(got)
    if not m.any():
        return 0.0
    scale = max(1.0, float(np.max(np.abs(want[m]))))
    err = float(np.max(np.abs(got[m] - want[m])) / scale)
    assert err <= tol, f"relative error {err:.3e} exceeds {tol:.0e}"
    return err


@pytest.mark.parametrize("window", [2, 5, 20, 50])
def test_sma_matches(series, window):
    close_enough(sma(series["close"], window), ref.ref_sma(series["close"], window))


@pytest.mark.parametrize("span", [5, 12, 26])
def test_ema_matches(series, span):
    close_enough(ema(series["close"], span), ref.ref_ema(series["close"], span))


@pytest.mark.parametrize("period", [7, 14, 21])
def test_wilder_matches(series, period):
    close_enough(wilder_rma(series["close"], period), ref.ref_wilder_rma(series["close"], period))


@pytest.mark.parametrize("period", [7, 14])
def test_rsi_matches(series, period):
    close_enough(rsi(series["close"], period), ref.ref_rsi(series["close"], period))


def test_bollinger_matches(series):
    for got, want in zip(bollinger(series["close"]), ref.ref_bollinger(series["close"]),
                         strict=True):
        close_enough(got, want)


def test_vwap_matches(series):
    args = (series["high"], series["low"], series["close"], series["volume"])
    close_enough(vwap(*args), ref.ref_vwap(*args))
    close_enough(vwap(*args, window=20), ref.ref_vwap(*args, window=20))


def test_macd_matches(series):
    for got, want in zip(macd(series["close"]), ref.ref_macd(series["close"]), strict=True):
        close_enough(got, want)


def test_atr_matches(series):
    args = (series["high"], series["low"], series["prev_close"])
    close_enough(atr(*args, 14), ref.ref_atr(*args, 14))


def test_naive_python_loop_agrees_too(walk):
    close_enough(sma(walk, 20), ref.ref_sma_python_loop(walk, 20))
