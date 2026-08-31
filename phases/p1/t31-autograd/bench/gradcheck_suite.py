#!/usr/bin/env python3
"""Verification benchmark for T31.

    python3 phases/p1/t31-autograd/bench/gradcheck_suite.py

Three claims from the capsule, measured:

1. **Gradcheck passes on 30 random graphs.** Scalar engine, six expression
   templates x five random draws, central finite differences at 1e-6.
2. **The tensor engine survives broadcasting.** Ten shape-mixing expressions,
   every entry (or a 200-entry random subset) probed.
3. **The MLP matches a reference loss curve within noise.** Identical init,
   identical Adam, gradients from two independent derivations — one by this
   engine's closures, one hand-derived on paper in `train.ReferenceMLP`.

"Within noise" turns out to be a wild understatement, and that is the useful
result: the two curves agree to ~1e-16, because they are computing the *same*
arithmetic in a different order. An autodiff engine is not an approximation of
the derivative, it is the derivative.
"""

from __future__ import annotations

import json
import pathlib
import platform
import random
import sys
import time
from datetime import UTC, datetime

TOPIC = pathlib.Path(__file__).resolve().parent.parent
REPO = TOPIC.parents[2]
sys.path[:0] = [str(REPO), str(TOPIC / "src")]

import numpy as np  # noqa: E402
from t31_autograd import (  # noqa: E402
    Tensor,
    Value,
    gradcheck_tensors,
    gradcheck_values,
    mse_loss,
)
from t31_autograd.train import make_features, train_reference, train_with_engine  # noqa: E402

TOLERANCE = 1e-6


def scalar_suite() -> dict:
    random.seed(31)
    templates = {
        "a*b + tanh(a)": lambda a, b: a * b + a.tanh(),
        "(a+b)^3": lambda a, b: (a + b) ** 3,
        "a/b - exp(b)": lambda a, b: a / b - b.exp(),
        "sigmoid(a*b)": lambda a, b: (a * b).sigmoid(),
        "relu(a)*b + b/a": lambda a, b: a.relu() * b + b / a,
        "log(a^2+b)*(a-b)": lambda a, b: ((a * a + b).log()) * (a - b),
    }
    graphs, worst, failures = 0, 0.0, []
    per = []
    for name, fn in templates.items():
        tmpl_worst = 0.0
        for _ in range(5):
            a, b = Value(random.uniform(0.4, 2.0)), Value(random.uniform(0.4, 2.0))
            res = gradcheck_values(fn, [a, b], tolerance=TOLERANCE)
            graphs += 1
            tmpl_worst = max(tmpl_worst, res.max_rel_error)
            if not res.ok:
                failures.append({"template": name, "rel_error": res.max_rel_error})
        per.append({"template": name, "draws": 5, "max_rel_error": tmpl_worst})
        worst = max(worst, tmpl_worst)
    return {"graphs": graphs, "max_rel_error": worst, "failures": failures, "per_template": per}


def tensor_suite() -> dict:
    rng = np.random.default_rng(11)
    cases = {
        "matmul + bias broadcast": (
            lambda A, B, c: ((A @ B + c).tanh()).sum(),
            lambda: [Tensor(rng.standard_normal((5, 3))), Tensor(rng.standard_normal((3, 4))),
                     Tensor(rng.standard_normal((1, 4)))],
        ),
        "column broadcast (n,1)": (
            lambda A, c: ((A * c) ** 2).sum(),
            lambda: [Tensor(rng.standard_normal((4, 3))), Tensor(rng.standard_normal((4, 1)))],
        ),
        "rank promotion (3,)": (
            lambda A, c: ((A + c).tanh()).sum(),
            lambda: [Tensor(rng.standard_normal((4, 3))), Tensor(rng.standard_normal((3,)))],
        ),
        "sum along axis, reused": (
            lambda A: ((A - A.sum(axis=1, keepdims=True)) ** 2).sum(),
            lambda: [Tensor(rng.standard_normal((4, 3)))],
        ),
        "mean x sum": (
            lambda A: A.mean() * A.sum(),
            lambda: [Tensor(rng.standard_normal((4, 3)))],
        ),
        "max along axis": (
            lambda A: (A.max(axis=1) ** 3).sum(),
            lambda: [Tensor(rng.standard_normal((4, 3)))],
        ),
        "reshape + transpose + matmul": (
            lambda A: ((A.reshape(3, 4).T @ A.reshape(3, 4)) ** 2).sum(),
            lambda: [Tensor(rng.standard_normal((4, 3)))],
        ),
        "divide": (
            lambda A, B: (A / B).sum(),
            lambda: [Tensor(rng.standard_normal((3, 2))), Tensor(rng.random((3, 2)) + 0.5)],
        ),
        "exp/log/sigmoid chain": (
            lambda A: ((A.exp() + 1.0).log() * A.sigmoid()).sum(),
            lambda: [Tensor(rng.standard_normal((3, 3)))],
        ),
        "relu": (
            lambda A: (A.relu() ** 2).sum(),
            lambda: [Tensor(rng.standard_normal((4, 4)) + 0.3)],
        ),
    }
    per, worst, failures, entries = [], 0.0, [], 0
    for name, (fn, make) in cases.items():
        res = gradcheck_tensors(fn, make(), tolerance=TOLERANCE)
        per.append({"case": name, "entries": res.n_checked, "max_rel_error": res.max_rel_error})
        entries += res.n_checked
        worst = max(worst, res.max_rel_error)
        if not res.ok:
            failures.append({"case": name, "rel_error": res.max_rel_error})
    return {"cases": len(cases), "entries": entries, "max_rel_error": worst,
            "failures": failures, "per_case": per}


def whole_network_gradcheck() -> dict:
    from t31_autograd import MLP

    rng = np.random.default_rng(9)
    model = MLP([4, 6, 2], rng=np.random.default_rng(0))
    x = Tensor(rng.standard_normal((8, 4)), requires_grad=False)
    target = Tensor(rng.standard_normal((8, 2)), requires_grad=False)
    res = gradcheck_tensors(lambda *ps: mse_loss(model(x), target), model.parameters())
    return {"parameters": model.num_parameters(), "entries": res.n_checked,
            "max_rel_error": res.max_rel_error, "passed": res.ok}


def loss_curve_parity() -> dict:
    X, y = make_features(n=512, d=8, seed=0)
    t0 = time.perf_counter()
    engine, model = train_with_engine(X, y, hidden=16, steps=200, seed=0)
    engine_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    reference = train_reference(X, y, hidden=16, steps=200, seed=0)
    ref_seconds = time.perf_counter() - t0

    a, b = np.array(engine.losses), np.array(reference.losses)
    return {
        "steps": 200,
        "samples": int(X.shape[0]),
        "parameters": model.num_parameters(),
        "engine_first_loss": float(a[0]),
        "engine_final_loss": float(a[-1]),
        "reference_final_loss": float(b[-1]),
        "max_abs_curve_divergence": float(np.abs(a - b).max()),
        "engine_seconds": round(engine_seconds, 3),
        "reference_seconds": round(ref_seconds, 3),
        "overhead_vs_handwritten": round(engine_seconds / ref_seconds, 2),
        "train_accuracy": engine.accuracy,
    }


def scalar_vs_tensor_cost() -> dict:
    """Why the scalar engine is a teaching tool and not a library."""
    rng = np.random.default_rng(0)
    n, d, h = 64, 8, 16
    X = rng.standard_normal((n, d))
    W1 = [[Value(v) for v in row] for row in rng.standard_normal((d, h))]

    t0 = time.perf_counter()
    total = Value(0.0)
    for i in range(n):
        for j in range(h):
            acc = Value(0.0)
            for k in range(d):
                acc = acc + W1[k][j] * X[i, k]
            total = total + acc.tanh()
    total.backward()
    scalar_seconds = time.perf_counter() - t0
    nodes = len(total.topo_order())

    Wt = Tensor(rng.standard_normal((d, h)))
    Xt = Tensor(X, requires_grad=False)
    t0 = time.perf_counter()
    (Xt @ Wt).tanh().sum().backward()
    tensor_seconds = time.perf_counter() - t0

    return {
        "shape": f"({n},{d}) @ ({d},{h})",
        "scalar_graph_nodes": nodes,
        "scalar_seconds": round(scalar_seconds, 4),
        "tensor_seconds": round(tensor_seconds, 6),
        "speedup": round(scalar_seconds / tensor_seconds, 1),
    }


def main() -> int:
    scalar = scalar_suite()
    tensor = tensor_suite()
    network = whole_network_gradcheck()
    parity = loss_curve_parity()
    cost = scalar_vs_tensor_cost()

    failures = scalar["failures"] + tensor["failures"]
    if not network["passed"]:
        failures.append({"case": "whole-network gradcheck"})
    if parity["max_abs_curve_divergence"] > 1e-9:
        failures.append({"case": "loss-curve parity", "value": parity["max_abs_curve_divergence"]})

    report = {
        "topic": "T31",
        "benchmark": "finite-difference gradcheck + loss-curve parity vs a hand-derived reference",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "tolerance": TOLERANCE,
        "passed": not failures,
        "failures": failures,
        "scalar_gradcheck": scalar,
        "tensor_gradcheck": tensor,
        "whole_network_gradcheck": network,
        "loss_curve_parity": parity,
        "scalar_vs_tensor_cost": cost,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "machine": platform.machine(),
        },
    }
    out = TOPIC / "bench" / "results.json"
    out.write_text(json.dumps(report, indent=2) + "\n")

    print(f"scalar gradcheck : {scalar['graphs']} graphs, worst {scalar['max_rel_error']:.2e}")
    print(f"tensor gradcheck : {tensor['cases']} cases / {tensor['entries']} entries, "
          f"worst {tensor['max_rel_error']:.2e}")
    print(f"whole-network    : {network['parameters']} params, worst {network['max_rel_error']:.2e}")
    print(f"loss-curve parity: max divergence {parity['max_abs_curve_divergence']:.2e} over 200 steps")
    print(f"                   engine {parity['engine_seconds']}s vs hand-written "
          f"{parity['reference_seconds']}s ({parity['overhead_vs_handwritten']}x)")
    print(f"scalar vs tensor : {cost['scalar_graph_nodes']} nodes, {cost['speedup']}x slower")
    print(f"\n-> {'PASS' if not failures else 'FAIL: ' + str(failures)}  (written to {out.relative_to(REPO)})")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
