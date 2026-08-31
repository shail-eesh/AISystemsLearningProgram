#!/usr/bin/env python3
"""Verification benchmark for T16A.

    python3 phases/p1/t16a-matmul-cpu/bench/roofline.py

The capsule's claim: *the blocked kernel beats naive C by at least 10x at
1024x1024, and the memory-hierarchy chart comes from measurements rather than
folklore.* Four measurements:

1. Correctness of all five kernels against NumPy.
2. Throughput table: every kernel x {256, 512, 1024}, plus OpenBLAS.
3. Block-size sweep, so the chosen tile is an observation and not a guess.
4. Read+write bandwidth vs working-set size — the cliffs are the cache levels.

Everything is compiled with identical flags. Handicapping the naive kernel
would make a better-looking number and a worse lesson.
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
from t16a_matmul import (  # noqa: E402
    ALL_KERNELS,
    AVAILABLE,
    arithmetic_intensity,
    environment,
    random_pair,
    time_call,
)

SIZES = (256, 512, 1024)
REQUIRED_SPEEDUP = 10.0


def correctness() -> dict:
    from t16a_matmul.native import kernels

    lib = kernels()
    worst = {}
    for name in ALL_KERNELS:
        err = 0.0
        for m, k, n in [(65, 33, 129), (128, 128, 128), (37, 53, 41)]:
            rng = np.random.default_rng(m * k * n)
            A, B = rng.standard_normal((m, k)), rng.standard_normal((k, n))
            err = max(err, float(np.abs(lib.call(name, A, B, mc=16, kc=16, nc=32) - A @ B).max()))
        worst[name] = err
    return {"max_abs_error": worst, "passed": all(v < 1e-9 for v in worst.values())}


def throughput_table() -> dict:
    from t16a_matmul.native import kernels

    lib = kernels()
    rows = []
    for n in SIZES:
        A, B = random_pair(n)
        row = {"n": n}
        for name in ALL_KERNELS:
            t = time_call(name, lambda name=name, A=A, B=B: lib.call(name, A, B), n, n, n, repeats=2)
            row[name] = {"ms": round(t.seconds * 1e3, 2), "gflops": round(t.gflops, 2)}
        blas = time_call("numpy", lambda A=A, B=B: A @ B, n, n, n, repeats=3)
        row["numpy_openblas"] = {"ms": round(blas.seconds * 1e3, 3), "gflops": round(blas.gflops, 2)}
        row["arithmetic_intensity"] = round(arithmetic_intensity(n, n, n), 1)
        rows.append(row)
    return {"rows": rows}


def headline(rows: list[dict]) -> dict:
    big = rows[-1]
    naive = big["matmul_naive_ijk"]["ms"]
    blocked_only = big["matmul_blocked"]["ms"]
    best_name = min(
        (k for k in ALL_KERNELS if k != "matmul_naive_ijk"),
        key=lambda k: big[k]["ms"],
    )
    best = big[best_name]["ms"]
    blas = big["numpy_openblas"]["gflops"]
    return {
        "n": big["n"],
        "naive_c_ms": naive,
        "blocked_single_thread_ms": blocked_only,
        "blocked_single_thread_speedup": round(naive / blocked_only, 1),
        "best_kernel": best_name,
        "best_kernel_ms": best,
        "best_kernel_speedup_over_naive": round(naive / best, 1),
        "best_kernel_gflops": big[best_name]["gflops"],
        "openblas_gflops": blas,
        "fraction_of_openblas": round(big[best_name]["gflops"] / blas, 3),
        "required_speedup": REQUIRED_SPEEDUP,
        "passed": (naive / blocked_only) >= REQUIRED_SPEEDUP,
    }


def block_sweep(n: int = 1024) -> dict:
    from t16a_matmul.native import kernels

    lib = kernels()
    A, B = random_pair(n)
    out = []
    for mc, kc, nc in [(64, 64, 128), (128, 128, 256), (256, 128, 256),
                       (128, 256, 512), (64, 256, 1024), (256, 256, 256)]:
        t = time_call("blocked", lambda mc=mc, kc=kc, nc=nc:
                      lib.call("matmul_blocked", A, B, mc=mc, kc=kc, nc=nc), n, n, n, repeats=2)
        out.append({"mc": mc, "kc": kc, "nc": nc,
                    "b_tile_kb": kc * nc * 8 // 1024,
                    "ms": round(t.seconds * 1e3, 2), "gflops": round(t.gflops, 2)})
    best = min(out, key=lambda r: r["ms"])
    spread = max(r["ms"] for r in out) / best["ms"]
    return {"n": n, "sweep": out, "best": best, "worst_over_best": round(spread, 2)}


def bandwidth_curve() -> dict:
    rows = []
    for kb in (16, 64, 256, 1024, 4096, 16384, 65536):
        a = np.ones(kb * 1024 // 8, dtype=np.float64)
        passes = max(3, int(2e8 // a.nbytes))
        a *= 1.0000001
        t0 = time.perf_counter()
        for _ in range(passes):
            a *= 1.0000001
        elapsed = time.perf_counter() - t0
        rows.append({"working_set_kb": kb, "passes": passes,
                     "gb_per_s": round(2 * a.nbytes * passes / elapsed / 1e9, 1)})
    return {"note": "in-place scale; 2 bytes moved per array byte (one read, one write)",
            "rows": rows}


def main() -> int:
    env = environment()
    if not AVAILABLE:
        report = {"topic": "T16A", "passed": False, "skipped": True, "environment": env}
        (TOPIC / "bench" / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        print("no C compiler available; nothing measured")
        return 1

    correct = correctness()
    table = throughput_table()
    head = headline(table["rows"])
    sweep = block_sweep()
    bw = bandwidth_curve()

    report = {
        "topic": "T16A",
        "benchmark": "CPU matmul ladder: correctness, throughput vs OpenBLAS, block sweep, bandwidth",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "passed": correct["passed"] and head["passed"],
        "correctness": correct,
        "headline": head,
        "throughput": table,
        "block_sweep": sweep,
        "bandwidth": bw,
        "environment": env | {"python": platform.python_version(), "numpy": np.__version__},
    }
    (TOPIC / "bench" / "results.json").write_text(json.dumps(report, indent=2) + "\n")

    print(f"build flags      : {env.get('flags')}  ({env.get('threads')} OpenMP threads)")
    print(f"correctness      : worst abs error {max(correct['max_abs_error'].values()):.1e}")
    print(f"n={head['n']} naive C   : {head['naive_c_ms']:.0f} ms")
    print(f"       blocked 1T: {head['blocked_single_thread_ms']:.0f} ms "
          f"-> {head['blocked_single_thread_speedup']}x  (need >= {REQUIRED_SPEEDUP}x)")
    print(f"       best ({head['best_kernel'].replace('matmul_', '')}): "
          f"{head['best_kernel_ms']:.0f} ms -> {head['best_kernel_speedup_over_naive']}x, "
          f"{head['best_kernel_gflops']} GFLOP/s")
    print(f"       OpenBLAS  : {head['openblas_gflops']} GFLOP/s "
          f"(we reach {head['fraction_of_openblas']:.0%} of it)")
    print(f"block sweep      : best {sweep['best']['mc']}/{sweep['best']['kc']}/{sweep['best']['nc']}, "
          f"spread worst/best {sweep['worst_over_best']}x")
    print("bandwidth cliffs : " + "  ".join(
        f"{r['working_set_kb']}KB={r['gb_per_s']}GB/s" for r in bw["rows"]))
    print(f"\n-> {'PASS' if report['passed'] else 'FAIL'}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
