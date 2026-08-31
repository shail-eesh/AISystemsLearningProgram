#!/usr/bin/env python3
"""Verification benchmark for T45A.

    python3 phases/p1/t45a-softmax-cpu/bench/numerics.py

The capsule's claim: *the online version matches the reference at fp32/fp16 on
adversarial inputs (+/-1e4 logits).* Four measurements:

1. **Failure of the naive formula** — how many of the adversarial rows produce
   inf or nan, per dtype. (The point of keeping the broken version around.)
2. **Online vs two-pass, within each dtype.** The real claim: streaming the
   normaliser costs no accuracy. Compared inside the dtype, because the dtype's
   own precision is a separate question.
3. **Every implementation vs a float64 reference**, on the adversarial grid.
4. **Chunk-size independence** — the same answer for chunks from 1 to the whole
   row, which is what makes the algorithm safe to tile.

Plus a throughput table, mostly to make the honest point that on a CPU with
NumPy the online version is *slower* — it exists for a memory-hierarchy reason
that only pays off in a fused kernel (T45B, T7).
"""

from __future__ import annotations

import json
import pathlib
import platform
import sys
from datetime import UTC, datetime

TOPIC = pathlib.Path(__file__).resolve().parent.parent
REPO = TOPIC.parents[2]
sys.path[:0] = [str(REPO), str(TOPIC / "src")]

import numpy as np  # noqa: E402
from t45a_softmax import (  # noqa: E402
    cross_entropy,
    log_softmax,
    naive_softmax,
    online_softmax,
    stable_softmax,
    two_pass_softmax,
)

DTYPES = {"float64": np.float64, "float32": np.float32, "float16": np.float16}
TOLERANCE = {"float64": 1e-15, "float32": 1e-7, "float16": 1e-3}


def adversarial_grid(rows: int = 64, cols: int = 128, seed: int = 45) -> np.ndarray:
    """Rows that break naive softmax: giant positives, giant negatives, ties.

    The giant values are placed at *representable* magnitudes so that low
    precision is tested on the algorithm rather than on its own mantissa —
    fp16 cannot tell 10000 from 9999 and that is a dtype fact, not a softmax
    fact (see `tests/test_softmax_numerics.py`).
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((rows, cols)) * 3.0
    giant = rng.random((rows, cols))
    x = np.where(giant > 0.93, 1e4, x)
    x = np.where(giant < 0.07, -1e4, x)
    x[0] = 800.0                    # every entry identical and huge
    x[1] = -1e4                     # every entry identical and hugely negative
    return x


def naive_failure_rate(x: np.ndarray) -> dict:
    out = {}
    for name, dtype in DTYPES.items():
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            p = naive_softmax(x.astype(dtype))
        bad = int((~np.isfinite(p)).any(axis=-1).sum())
        out[name] = {"rows": int(x.shape[0]), "rows_with_inf_or_nan": bad}
    return out


def accuracy(x: np.ndarray) -> dict:
    ref = stable_softmax(x.astype(np.float64))
    impls = {"stable": stable_softmax, "two_pass": two_pass_softmax, "online": online_softmax}
    per_dtype = {}
    failures = []
    for dname, dtype in DTYPES.items():
        xd = x.astype(dtype)
        within = float(np.abs(online_softmax(xd).astype(np.float64)
                              - stable_softmax(xd).astype(np.float64)).max())
        vs_ref = {}
        for iname, fn in impls.items():
            err = float(np.abs(fn(xd).astype(np.float64) - ref).max())
            vs_ref[iname] = err
            if err > TOLERANCE[dname]:
                failures.append({"dtype": dname, "impl": iname, "max_abs_error": err})
        if within > TOLERANCE[dname]:
            failures.append({"dtype": dname, "impl": "online-vs-two-pass", "max_abs_error": within})
        per_dtype[dname] = {
            "tolerance": TOLERANCE[dname],
            "online_vs_two_pass": within,
            "vs_float64_reference": vs_ref,
        }
    return {"per_dtype": per_dtype, "failures": failures}


def chunk_independence(x: np.ndarray) -> dict:
    ref = online_softmax(x, chunk=10**9)
    rows = []
    for chunk in (1, 3, 16, 64, 512):
        err = float(np.abs(online_softmax(x, chunk=chunk) - ref).max())
        rows.append({"chunk": chunk, "max_abs_error": err})
    return {"rows": rows, "max": max(r["max_abs_error"] for r in rows)}


def throughput() -> dict:
    import time

    rng = np.random.default_rng(0)
    x = rng.standard_normal((256, 4096)).astype(np.float32) * 5
    out = {}
    for name, fn in (("stable", stable_softmax), ("two_pass", two_pass_softmax),
                     ("online(chunk=512)", lambda a: online_softmax(a, chunk=512))):
        fn(x)
        best = float("inf")
        for _ in range(3):
            t0 = time.perf_counter()
            fn(x)
            best = min(best, time.perf_counter() - t0)
        out[name] = {"ms": round(best * 1e3, 2), "gb_per_s": round(x.nbytes / best / 1e9, 2)}
    out["_note"] = (
        "On a CPU with NumPy the online version is slower: it pays Python loop "
        "overhead per chunk to save a pass it does not need here. Its value is "
        "in a fused kernel where the row never fits in fast memory (T45B, T7)."
    )
    return out


def loss_sanity() -> dict:
    logits = np.array([[0.0, -900.0], [100.0, 0.0, 0.0][:2]])
    fused = float(cross_entropy(logits, np.array([1, 0])))
    with np.errstate(divide="ignore"):
        unfused = float(-np.log(stable_softmax(logits))[np.arange(2), [1, 0]].mean())
    return {
        "fused_cross_entropy": fused,
        "unfused_minus_log_softmax": unfused if np.isfinite(unfused) else "inf",
        "log_softmax_of_minus_800": float(log_softmax(np.array([[0.0, -800.0]]))[0, 1]),
    }


def main() -> int:
    x = adversarial_grid()
    naive = naive_failure_rate(x)
    acc = accuracy(x)
    chunks = chunk_independence(x)
    tput = throughput()
    losses = loss_sanity()

    failures = list(acc["failures"])
    if chunks["max"] > 0.0:
        failures.append({"case": "chunk independence", "value": chunks["max"]})
    if naive["float64"]["rows_with_inf_or_nan"] == 0:
        failures.append({"case": "naive softmax was expected to fail and did not"})

    report = {
        "topic": "T45A",
        "benchmark": "softmax numerics: overflow, online-vs-two-pass, dtype accuracy, chunk independence",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "passed": not failures,
        "failures": failures,
        "adversarial_grid": {"shape": list(x.shape), "range": [float(x.min()), float(x.max())]},
        "naive_softmax_failure": naive,
        "accuracy": acc,
        "chunk_independence": chunks,
        "throughput": tput,
        "loss_sanity": losses,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "machine": platform.machine()},
    }
    (TOPIC / "bench" / "results.json").write_text(json.dumps(report, indent=2) + "\n")

    print(f"adversarial grid : {x.shape[0]}x{x.shape[1]}, logits in "
          f"[{x.min():.0f}, {x.max():.0f}]")
    for d in DTYPES:
        n = naive[d]["rows_with_inf_or_nan"]
        a = acc["per_dtype"][d]
        print(f"  {d:<8} naive breaks on {n:>3}/{x.shape[0]} rows | "
              f"online vs two-pass {a['online_vs_two_pass']:.2e} | "
              f"online vs f64 {a['vs_float64_reference']['online']:.2e} "
              f"(tol {a['tolerance']:.0e})")
    print(f"chunk independence: max error across chunks 1..512 = {chunks['max']:.2e}")
    print("throughput (256x4096 f32): " + "  ".join(
        f"{k}={v['ms']}ms" for k, v in tput.items() if not k.startswith("_")))
    print(f"\n-> {'PASS' if not failures else 'FAIL: ' + str(failures)}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
