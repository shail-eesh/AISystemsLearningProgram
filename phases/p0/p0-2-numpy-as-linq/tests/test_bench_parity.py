"""Run the P0.2 benchmark as a test so the parity claim cannot rot."""

import importlib.util
import json
import pathlib

BENCH = pathlib.Path(__file__).resolve().parent.parent / "bench" / "indicator_parity.py"


def _load():
    spec = importlib.util.spec_from_file_location("p0_2_bench", BENCH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_all_70_parity_checks_pass():
    bench = _load()
    assert bench.main() == 0
    results = json.loads((BENCH.parent / "results.json").read_text())
    assert results["passed"] is True
    assert results["checks"] == 70
    assert results["failures"] == []
    assert results["worst"]["max_rel_error"] <= results["tolerance"]


def test_vectorisation_is_actually_faster():
    results = json.loads((BENCH.parent / "results.json").read_text())
    assert results["speed"]["speedup_vs_loop"] > 20
