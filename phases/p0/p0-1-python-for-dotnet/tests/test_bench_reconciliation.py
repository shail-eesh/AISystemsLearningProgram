"""The verification benchmark, run as a test.

"Done when" for P0.1 is: replaying the synthetic OMS tape reconciles against
two independent references to 1e-6. Keeping that in pytest means the claim
cannot rot silently.
"""

import importlib.util
import json
import pathlib

BENCH = pathlib.Path(__file__).resolve().parent.parent / "bench" / "replay_orders.py"


def _load():
    spec = importlib.util.spec_from_file_location("p0_1_bench", BENCH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_replay_reconciles_within_tolerance():
    bench = _load()
    assert bench.main() == 0
    results = json.loads((BENCH.parent / "results.json").read_text())
    assert results["passed"] is True
    assert results["quantity_breaks"] == {}
    assert results["worst_relative_error"] <= results["tolerance"]
    assert results["orders"]["placed"] == 210


def test_float_reference_is_independent_of_the_model():
    """Guard against the reference quietly starting to call the code under test."""
    source = BENCH.read_text()
    ref = source.split("def float_reference")[1].split("def replay")[0]
    for banned in ("Money", "Position", "Portfolio", "Quantity"):
        assert banned not in ref, f"the reference reducer must not use {banned}"


def test_results_json_is_committed_and_current():
    results = json.loads((BENCH.parent / "results.json").read_text())
    assert results["topic"] == "P0.1"
    assert results["symbols"] == 5
