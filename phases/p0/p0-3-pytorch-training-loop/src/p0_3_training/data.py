"""The toy dataset: predict tomorrow's direction from today's features.

This is deliberately a *bad* prediction problem, and the point of the topic is
to feel exactly how bad while the loss curve looks encouraging. The features
are honest (all computed from information available at time t) and the label is
`close[t+1] > close[t]`.

The two habits that make it honest, and that almost every first attempt gets
wrong, are both here:

* **`chronological_split`** — never `train_test_split(shuffle=True)` on a time
  series. Shuffling lets the model interpolate between days it has already
  seen, and the score becomes fiction.
* **`standardise(fit_on=train)`** — the mean and std come from the training
  window only. Fitting the scaler on the whole series leaks the future into
  every training row through the normalisation constants alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from common.data import load_ohlcv

FEATURE_COLUMNS = [
    "ret_1", "ret_5", "ret_20",
    "range_pct", "close_loc",
    "vol_20", "volume_z", "gap_pct",
]


def _features_for_symbol(g: pd.DataFrame) -> pd.DataFrame:
    """Eight simple, causal features. Nothing here peeks past row t."""
    c = g["close"].to_numpy(dtype=np.float64)
    h = g["high"].to_numpy(dtype=np.float64)
    low = g["low"].to_numpy(dtype=np.float64)
    o = g["open"].to_numpy(dtype=np.float64)
    pc = g["prev_close"].to_numpy(dtype=np.float64)
    v = g["volume"].to_numpy(dtype=np.float64)

    def lag_return(k: int) -> np.ndarray:
        out = np.full(c.size, np.nan)
        out[k:] = c[k:] / c[:-k] - 1.0
        return out

    ret_1 = lag_return(1)
    with np.errstate(invalid="ignore"):
        vol_20 = pd.Series(ret_1).rolling(20).std(ddof=0).to_numpy()
        vol_mean = pd.Series(v).rolling(20).mean().to_numpy()
        vol_std = pd.Series(v).rolling(20).std(ddof=0).to_numpy()

    out = pd.DataFrame(
        {
            "symbol": g["symbol"].to_numpy(),
            "date": g["date"].to_numpy(),
            "close": c,
            "ret_1": ret_1,
            "ret_5": lag_return(5),
            "ret_20": lag_return(20),
            "range_pct": (h - low) / c,
            "close_loc": np.where(h > low, (c - low) / (h - low), 0.5),
            "vol_20": vol_20,
            "volume_z": (v - vol_mean) / np.where(vol_std > 0, vol_std, np.nan),
            "gap_pct": o / pc - 1.0,
        }
    )
    # Label: does the NEXT close exceed today's? The shift is the only place the
    # future is allowed to appear, and it appears in y, never in X.
    out["label"] = (np.concatenate([c[1:], [np.nan]]) > c).astype(np.float64)
    out.loc[out.index[-1], "label"] = np.nan          # last day has no tomorrow
    return out


@dataclass(frozen=True)
class Dataset:
    """Plain arrays plus the metadata needed to split them honestly."""

    X: np.ndarray            # (n, n_features) float32
    y: np.ndarray            # (n,) float32, 0/1
    dates: np.ndarray        # (n,) datetime64
    symbols: np.ndarray      # (n,) str
    feature_names: list[str]

    def __len__(self) -> int:
        return self.X.shape[0]

    @property
    def base_rate(self) -> float:
        """Fraction of up-days — the accuracy of always predicting 'up'."""
        return float(self.y.mean())


def build_dataset(symbols: list[str] | None = None) -> Dataset:
    prices = load_ohlcv(symbols)
    frames = [
        _features_for_symbol(g.sort_values("date", kind="stable"))
        for _, g in prices.groupby("symbol", sort=True)
    ]
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=[*FEATURE_COLUMNS, "label"]).sort_values(
        ["date", "symbol"], kind="stable"
    )
    return Dataset(
        X=df[FEATURE_COLUMNS].to_numpy(dtype=np.float32),
        y=df["label"].to_numpy(dtype=np.float32),
        dates=df["date"].to_numpy(),
        symbols=df["symbol"].to_numpy(),
        feature_names=list(FEATURE_COLUMNS),
    )


def chronological_split(
    data: Dataset, train_frac: float = 0.7, val_frac: float = 0.15
) -> tuple[Dataset, Dataset, Dataset]:
    """Split by DATE, not by row order or randomness.

    Rows are cut at date boundaries so the same trading day never straddles two
    splits — otherwise five symbols from one day land on both sides and the
    model sees a neighbour of every validation row.
    """
    if not 0 < train_frac < 1 or not 0 < val_frac < 1 or train_frac + val_frac >= 1:
        raise ValueError("train_frac and val_frac must be positive and sum to < 1")
    unique_dates = np.unique(data.dates)
    n = unique_dates.size
    train_end = unique_dates[int(n * train_frac)]
    val_end = unique_dates[int(n * (train_frac + val_frac))]

    def take(mask: np.ndarray) -> Dataset:
        return Dataset(
            X=data.X[mask], y=data.y[mask], dates=data.dates[mask],
            symbols=data.symbols[mask], feature_names=data.feature_names,
        )

    return (
        take(data.dates < train_end),
        take((data.dates >= train_end) & (data.dates < val_end)),
        take(data.dates >= val_end),
    )


def standardise(
    train: Dataset, *others: Dataset
) -> tuple[list[Dataset], np.ndarray, np.ndarray]:
    """Z-score every split using the TRAINING mean and std only."""
    mu = train.X.mean(axis=0)
    sd = train.X.std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0).astype(np.float32)
    mu = mu.astype(np.float32)

    def apply(d: Dataset) -> Dataset:
        return Dataset(X=(d.X - mu) / sd, y=d.y, dates=d.dates, symbols=d.symbols,
                       feature_names=d.feature_names)

    return [apply(train), *(apply(o) for o in others)], mu, sd


class ReturnsDataset:
    """A `torch.utils.data.Dataset` without importing torch at module level.

    Torch is an optional dependency for Phase 0 (`pip install -e .[torch]`), so
    the import is deferred into the constructor. Everything above this line
    works with NumPy alone.
    """

    def __init__(self, data: Dataset) -> None:
        import torch

        self.X = torch.from_numpy(np.ascontiguousarray(data.X))
        self.y = torch.from_numpy(np.ascontiguousarray(data.y)).unsqueeze(1)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]
