"""The dataset's guarantees. These are the tests that keep the topic honest."""

import numpy as np
import pytest
from p0_3_training import FEATURE_COLUMNS, build_dataset, chronological_split, standardise
from p0_3_training.data import _features_for_symbol


def test_shapes_and_dtypes(data):
    assert data.X.shape[1] == len(FEATURE_COLUMNS)
    assert data.X.dtype == np.float32 and data.y.dtype == np.float32
    assert data.X.shape[0] == data.y.shape[0] == data.dates.shape[0]
    assert set(np.unique(data.y)) <= {0.0, 1.0}


def test_no_nans_survive(data):
    assert np.isfinite(data.X).all()
    assert np.isfinite(data.y).all()


def test_base_rate_is_near_a_coin_flip(data):
    assert 0.4 < data.base_rate < 0.6, "a random walk should be near 50/50"


def test_features_are_causal():
    """Truncating the price history must not change any earlier feature row."""
    from common.data import load_ohlcv

    g = load_ohlcv(["ALPHAINFRA"]).sort_values("date")
    full = _features_for_symbol(g)
    cut = 200
    partial = _features_for_symbol(g.iloc[:cut])
    for col in FEATURE_COLUMNS:
        a = full[col].to_numpy()[: cut - 1]
        b = partial[col].to_numpy()[: cut - 1]
        assert np.array_equal(a, b, equal_nan=True), f"{col} depends on the future"


def test_label_is_tomorrow_not_today():
    from common.data import load_ohlcv

    g = load_ohlcv(["ALPHAINFRA"]).sort_values("date")
    f = _features_for_symbol(g)
    close = g["close"].to_numpy()
    expected = (close[1:] > close[:-1]).astype(float)
    assert np.array_equal(f["label"].to_numpy()[:-1], expected)
    assert np.isnan(f["label"].to_numpy()[-1]), "the last day has no tomorrow"


def test_split_is_chronological_and_disjoint(data):
    tr, va, te = chronological_split(data)
    assert len(tr) + len(va) + len(te) == len(data)
    assert tr.dates.max() < va.dates.min() <= va.dates.max() < te.dates.min()
    assert not set(tr.dates) & set(va.dates) & set(te.dates)


def test_split_does_not_straddle_a_trading_day(data):
    """Five symbols share each date; a row-count split would cut through one."""
    tr, va, te = chronological_split(data)
    for a, b in ((tr, va), (va, te)):
        assert not (set(a.dates) & set(b.dates))


@pytest.mark.parametrize("bad", [(0.0, 0.2), (1.0, 0.2), (0.9, 0.2), (0.5, 0.0)])
def test_split_fractions_validated(data, bad):
    with pytest.raises(ValueError):
        chronological_split(data, *bad)


def test_standardise_uses_training_statistics_only(data):
    tr, va, te = chronological_split(data)
    (tr_s, va_s, te_s), mu, sd = standardise(tr, va, te)
    assert np.allclose(tr_s.X.mean(axis=0), 0.0, atol=1e-4)
    assert np.allclose(tr_s.X.std(axis=0), 1.0, atol=1e-3)
    # The test split is NOT centred — that is the proof no future stats leaked in.
    assert not np.allclose(te_s.X.mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(mu, tr.X.mean(axis=0), atol=1e-5)


def test_symbol_subset_builds(data):
    small = build_dataset(["ALPHAINFRA"])
    assert set(np.unique(small.symbols)) == {"ALPHAINFRA"}
    assert len(small) < len(data)
