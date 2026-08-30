"""The benchmark as a test, plus the AlphaDesk registration."""

import importlib.util
import json
import pathlib

from common.alphadesk import Registry

BENCH = pathlib.Path(__file__).resolve().parent.parent / "bench" / "train_and_report.py"


def _load():
    spec = importlib.util.spec_from_file_location("p0_3_bench", BENCH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_benchmark_passes_both_conditions():
    bench = _load()
    assert bench.main() == 0
    r = json.loads((BENCH.parent / "results.json").read_text())
    assert r["passed"] is True
    assert r["a_overfit_tiny_batch"]["final_train_accuracy"] >= 0.99
    assert abs(r["b_honest_evaluation"]["z_vs_coinflip"]) < 2.0
    assert r["b_honest_evaluation"]["beats_majority_baseline"] is False


def test_the_centred_window_leak_is_large():
    r = json.loads((BENCH.parent / "results.json").read_text())
    centred = next(v for v in r["c_leakage_audit"]["variants"] if "centred" in v["name"])
    assert centred["uplift"] > 0.10, "the leakage demo must actually demonstrate a leak"


def test_topic_registers_two_foundation_components():
    reg = Registry()
    errors = reg.load_modules(["p0_3_training.alphadesk_hook"])
    assert errors == {}
    assert {c.name for c in reg.by_topic("P0.3")} == {"toy_signal_model", "training_loop"}
    model = reg.get("foundation.toy_signal_model").build()
    assert sum(p.numel() for p in model.parameters()) > 0
    loop = reg.get("foundation.training_loop").build()
    assert set(loop) == {"train", "evaluate", "seed_everything", "TrainConfig"}
