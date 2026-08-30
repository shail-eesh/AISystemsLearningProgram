#!/usr/bin/env python3
"""Verification benchmark for P0.2 — indicator parity + the speed-up.

    python3 phases/p0/p0-2-numpy-as-linq/bench/indicator_parity.py

Two claims are measured:

1. **Parity.** Every indicator, on all five symbols, against the independent
   pandas reference. Pass condition: NaN warm-up masks identical, worst
   relative error <= 1e-6 (the master plan's tolerance).
2. **Speed.** The 20-day SMA computed by a nested Python loop vs the
   vectorised implementation, on 200k synthetic bars.

TA-Lib is the conventional reference for this exercise; it is a C library that
does not build in this sandbox, so pandas stands in. It is still an
independent implementation by other authors, which is what a reference is for.
"""

from __future__ import annotations

import json
import pathlib
import platform
import sys
import time
from datetime import UTC, datetime

TOPIC = pathlib.Path(__file__).resolve().parent.parent
REPO = TOPIC.parents[2]
sys.path[:0] = [str(REPO), str(TOPIC / "src")]

import numpy as np  # noqa: E402
from p0_2_indicators import atr, bollinger, ema, macd, rsi, sma, vwap, wilder_rma  # noqa: E402
from p0_2_indicators import references as ref  # noqa: E402

from common.data import SAMPLE_SYMBOLS, load_ohlcv  # noqa: E402

TOLERANCE = 1e-6


def compare(name: str, got: np.ndarray, want: np.ndarray) -> dict:
    nan_match = bool(np.array_equal(np.isnan(got), np.isnan(want)))
    m = ~np.isnan(got) & ~np.isnan(want)
    scale = max(1.0, float(np.max(np.abs(want[m])))) if m.any() else 1.0
    abs_err = float(np.max(np.abs(got[m] - want[m]))) if m.any() else 0.0
    return {
        "indicator": name,
        "n_compared": int(m.sum()),
        "nan_mask_matches": nan_match,
        "max_abs_error": abs_err,
        "max_rel_error": abs_err / scale,
        "passed": nan_match and abs_err / scale <= TOLERANCE,
    }


def parity_rows() -> list[dict]:
    rows = []
    for sym in SAMPLE_SYMBOLS:
        g = load_ohlcv([sym])
        c = g["close"].to_numpy(dtype=np.float64)
        h = g["high"].to_numpy(dtype=np.float64)
        low = g["low"].to_numpy(dtype=np.float64)
        v = g["volume"].to_numpy(dtype=np.float64)
        pc = g["prev_close"].to_numpy(dtype=np.float64)
        checks = [
            ("sma_20", sma(c, 20), ref.ref_sma(c, 20)),
            ("sma_50", sma(c, 50), ref.ref_sma(c, 50)),
            ("ema_12", ema(c, 12), ref.ref_ema(c, 12)),
            ("ema_26", ema(c, 26), ref.ref_ema(c, 26)),
            ("wilder_14", wilder_rma(c, 14), ref.ref_wilder_rma(c, 14)),
            ("rsi_14", rsi(c, 14), ref.ref_rsi(c, 14)),
            ("bb_lower", bollinger(c)[0], ref.ref_bollinger(c)[0]),
            ("bb_mid", bollinger(c)[1], ref.ref_bollinger(c)[1]),
            ("bb_upper", bollinger(c)[2], ref.ref_bollinger(c)[2]),
            ("vwap_session", vwap(h, low, c, v), ref.ref_vwap(h, low, c, v)),
            ("vwap_20", vwap(h, low, c, v, 20), ref.ref_vwap(h, low, c, v, 20)),
            ("macd", macd(c)[0], ref.ref_macd(c)[0]),
            ("macd_signal", macd(c)[1], ref.ref_macd(c)[1]),
            ("atr_14", atr(h, low, pc, 14), ref.ref_atr(h, low, pc, 14)),
        ]
        for name, got, want in checks:
            rows.append({"symbol": sym, **compare(name, got, want)})
    return rows


def speed_row() -> dict:
    n, window, sub = 200_000, 20, 20_000
    rng = np.random.default_rng(1729)
    x = 100.0 + np.cumsum(rng.normal(0, 0.5, n))

    t0 = time.perf_counter()
    ref.ref_sma_python_loop(x[:sub], window)
    loop_sub = time.perf_counter() - t0
    loop_scaled = loop_sub * (n / sub)

    t0 = time.perf_counter()
    sma(x, window)
    vec = time.perf_counter() - t0

    t0 = time.perf_counter()
    ref.ref_sma(x, window)
    pandas_t = time.perf_counter() - t0

    return {
        "bars": n,
        "window": window,
        "python_loop_seconds_scaled": round(loop_scaled, 4),
        "vectorised_seconds": round(vec, 5),
        "pandas_rolling_seconds": round(pandas_t, 5),
        "speedup_vs_loop": round(loop_scaled / vec, 1),
    }


def main() -> int:
    rows = parity_rows()
    worst = max(rows, key=lambda r: r["max_rel_error"])
    failures = [r for r in rows if not r["passed"]]
    speed = speed_row()
    passed = not failures

    results = {
        "topic": "P0.2",
        "benchmark": "indicator parity vs pandas reference + vectorisation speed-up",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "reference": "pandas rolling/ewm (TA-Lib is a C library and does not build here)",
        "tolerance": TOLERANCE,
        "checks": len(rows),
        "failures": failures,
        "worst": {k: worst[k] for k in ("symbol", "indicator", "max_abs_error", "max_rel_error")},
        "speed": speed,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "machine": platform.machine(),
        },
        "per_check": rows,
        "passed": passed,
    }
    out = pathlib.Path(__file__).parent / "results.json"
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print(f"parity: {len(rows)} checks across {len(SAMPLE_SYMBOLS)} symbols")
    print(f"  worst: {worst['indicator']} on {worst['symbol']} "
          f"rel={worst['max_rel_error']:.3e} (tolerance {TOLERANCE:.0e})")
    print(f"  failures: {len(failures)}")
    print(f"speed on {speed['bars']:,} bars, window {speed['window']}:")
    print(f"  python loop (scaled): {speed['python_loop_seconds_scaled'] * 1e3:8.1f} ms")
    print(f"  vectorised:           {speed['vectorised_seconds'] * 1e3:8.1f} ms  "
          f"({speed['speedup_vs_loop']}x)")
    print(f"  pandas rolling:       {speed['pandas_rolling_seconds'] * 1e3:8.1f} ms")
    print(f"  -> {'PASS' if passed else 'FAIL'}; wrote {out.relative_to(REPO)}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
