"""Properties that must hold whatever the input — including the finance ones."""

import numpy as np
import pytest
from p0_2_indicators import (
    atr,
    bollinger,
    ema,
    macd,
    rolling_max,
    rolling_min,
    rsi,
    sliding_windows,
    sma,
    true_range,
    vwap,
    wilder_rma,
)


def test_constant_series_is_its_own_average():
    x = np.full(50, 42.0)
    assert np.allclose(sma(x, 10)[9:], 42.0)
    assert np.allclose(ema(x, 10), 42.0)
    assert np.allclose(wilder_rma(x, 10)[9:], 42.0)


def test_warmup_nan_counts_are_exact(walk):
    assert np.isnan(sma(walk, 20)[:19]).all() and not np.isnan(sma(walk, 20)[19:]).any()
    assert not np.isnan(ema(walk, 20)).any(), "seed='first' is defined from bar 0"
    assert np.isnan(ema(walk, 20, seed="sma")[:19]).all()
    assert np.isnan(rsi(walk, 14)[:14]).all() and not np.isnan(rsi(walk, 14)[14:]).any()


def test_rsi_is_bounded_and_saturates():
    rising = np.arange(1.0, 60.0)
    falling = rising[::-1].copy()
    assert np.nanmax(rsi(rising, 14)) == pytest.approx(100.0)
    assert np.nanmin(rsi(falling, 14)) == pytest.approx(0.0)
    rng = np.random.default_rng(5)
    noisy = 100 + np.cumsum(rng.normal(0, 1, 400))
    r = rsi(noisy, 14)
    assert np.nanmin(r) >= 0.0 and np.nanmax(r) <= 100.0


def test_rsi_flat_series_has_no_division_error():
    flat = np.full(40, 10.0)
    r = rsi(flat, 14)
    assert np.nanmax(r) == 100.0, "zero average loss pins RSI at 100 by convention"
    assert np.isfinite(r[14:]).all()


def test_bollinger_bands_are_ordered_and_symmetric(walk):
    lower, mid, upper = bollinger(walk, 20, 2.0)
    m = ~np.isnan(mid)
    assert (lower[m] <= mid[m]).all() and (mid[m] <= upper[m]).all()
    assert np.allclose(upper[m] - mid[m], mid[m] - lower[m])
    wide = bollinger(walk, 20, 3.0)
    assert (wide[2][m] >= upper[m]).all(), "more sigma means wider bands"


def test_vwap_stays_inside_the_typical_price_range(series):
    h, low, c, v = series["high"], series["low"], series["close"], series["volume"]
    typical = (h + low + c) / 3.0
    got = vwap(h, low, c, v)
    for i in range(1, len(got)):
        assert typical[: i + 1].min() - 1e-9 <= got[i] <= typical[: i + 1].max() + 1e-9


def test_rolling_min_max_bracket_the_series(walk):
    lo, hi = rolling_min(walk, 10), rolling_max(walk, 10)
    m = ~np.isnan(lo)
    assert (lo[m] <= hi[m]).all()
    assert (lo[m] <= walk[m]).all() and (walk[m] <= hi[m]).all()


def test_true_range_is_non_negative_and_at_least_the_bar_range(series):
    tr = true_range(series["high"], series["low"], series["prev_close"])
    assert (tr >= 0).all()
    assert (tr >= series["high"] - series["low"] - 1e-9).all()
    a = atr(series["high"], series["low"], series["prev_close"], 14)
    assert (a[~np.isnan(a)] > 0).all()


def test_macd_histogram_is_the_difference(walk):
    line, signal, hist = macd(walk)
    assert np.allclose(hist, line - signal, equal_nan=True)


def test_no_lookahead(walk):
    """The finance-critical property: today's value cannot depend on tomorrow.

    Truncate the series and every earlier value must be bit-identical.
    """
    cut = 300
    for name, fn in [
        ("sma", lambda x: sma(x, 20)),
        ("ema", lambda x: ema(x, 12)),
        ("wilder", lambda x: wilder_rma(x, 14)),
        ("rsi", lambda x: rsi(x, 14)),
        ("bb_upper", lambda x: bollinger(x)[2]),
        ("macd", lambda x: macd(x)[0]),
    ]:
        full = fn(walk)[:cut]
        partial = fn(walk[:cut])
        assert np.array_equal(full, partial, equal_nan=True), f"{name} peeks at the future"


def test_sliding_window_view_is_a_view_not_a_copy(walk):
    w = sliding_windows(walk, 20)
    assert w.base is not None
    assert w.shape == (walk.size - 19, 20)


@pytest.mark.parametrize(
    ("fn", "args"),
    [(sma, (0,)), (sma, (-3,)), (sma, (10_000,)), (sma, (2.5,)), (sma, (True,))],
)
def test_window_validation(fn, args, walk):
    with pytest.raises((ValueError, TypeError)):
        fn(walk, *args)


def test_two_dimensional_input_is_refused():
    with pytest.raises(ValueError):
        sma(np.ones((3, 4)), 2)


def test_ema_seed_must_be_known(walk):
    with pytest.raises(ValueError):
        ema(walk, 12, seed="magic")


def test_macd_spans_must_be_ordered(walk):
    with pytest.raises(ValueError):
        macd(walk, fast=26, slow=12)


def test_vwap_length_mismatch_is_refused(walk):
    with pytest.raises(ValueError):
        vwap(walk, walk, walk, walk[:-1])
