"""The benchmark's own machinery, the recorded results, and the AlphaDesk hook.

The full benchmark takes seven minutes, so it is not run here. What *is* run is
everything that could silently rot: the reference model, the weight transfer,
the induction probe's arithmetic, and the committed results.json.
"""

import importlib.util
import json
import pathlib
import sys

import pytest
import torch
from t4_transformer import GPT, GPTConfig, induction_scores, variable_period_batch

from common.alphadesk import Registry

BENCH = pathlib.Path(__file__).resolve().parent.parent / "bench"
sys.path.insert(0, str(BENCH))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"t4_bench_{name}", BENCH / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


# -- the reference model ---------------------------------------------------


def test_reference_produces_identical_logits():
    """The core parity claim, at test scale: two independently written models,
    one set of weights, the same logits."""
    ref_mod = _load("reference")
    cfg = GPTConfig(vocab_size=24, block_size=16, n_layer=2, n_head=4, n_embd=32,
                    position="learned", norm="layernorm", dropout=0.0)
    torch.manual_seed(0)
    ours = GPT(cfg).eval()
    ref = ref_mod.RefGPT(24, 16, 2, 4, 32).eval()
    ref_mod.copy_weights_from(ours, ref)
    idx = torch.randint(0, 24, (3, 12))
    with torch.no_grad():
        a, _ = ours(idx)
        b, _ = ref(idx)
    assert float((a - b).abs().max()) / float(a.abs().max()) < 1e-5


def test_reference_uses_a_genuinely_different_code_path():
    """If the reference imported our attention the parity check would be a
    tautology. This pins that it does not."""
    src = (BENCH / "reference.py").read_text()
    assert "scaled_dot_product_attention" in src
    assert "from t4_transformer" not in src


def test_gpt2_init_makes_the_reference_match_our_scale():
    ref_mod = _load("reference")
    torch.manual_seed(0)
    plain = ref_mod.RefGPT(24, 16, 4, 4, 32)
    torch.manual_seed(0)
    inited = ref_mod.init_like_gpt2(ref_mod.RefGPT(24, 16, 4, 4, 32), 4)
    assert float(plain.wte.weight.detach().std()) > 0.5
    assert float(inited.wte.weight.detach().std()) == pytest.approx(0.02, rel=0.2)
    assert inited.head.weight is inited.wte.weight


# -- the induction probe ---------------------------------------------------


def test_variable_period_rows_repeat_with_their_own_period():
    g = torch.Generator().manual_seed(0)
    seq, periods = variable_period_batch(16, 48, 64, min_period=10, max_period=14,
                                         generator=g)
    assert seq.shape == (16, 48)
    assert int(periods.min()) >= 10 and int(periods.max()) <= 14
    assert len(set(periods.tolist())) > 1        # genuinely variable
    for b in range(16):
        p = int(periods[b])
        assert torch.equal(seq[b, :48 - p], seq[b, p:])


def test_variable_period_rejects_impossible_ranges():
    with pytest.raises(ValueError, match="min_period"):
        variable_period_batch(2, 10, 8, min_period=8, max_period=4)


def test_induction_score_of_an_untrained_model_is_chance():
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=64, block_size=48, n_layer=2, n_head=4, n_embd=64))
    probe = induction_scores(model, batch=16, seed=0, min_period=10, max_period=14)
    assert probe["best"]["over_chance"] < 2.0
    assert probe["repeat_accuracy"] < 0.15


def test_induction_score_of_a_planted_head_is_exactly_one():
    """Calibrate the metric itself.

    A model whose attention is *replaced* by a head that looks precisely at the
    induction target must score 1.0. Without this, ``over_chance`` is a number
    nobody has checked the top of. The planted map is built from the same
    ``variable_period_batch`` seed the probe uses, so the periods line up.
    """
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=64, block_size=48, n_layer=1, n_head=1, n_embd=32))
    g = torch.Generator().manual_seed(0)
    _, periods = variable_period_batch(8, 48, 64, min_period=10, max_period=14,
                                       generator=g)
    planted = torch.zeros(8, 1, 48, 48)
    for b in range(8):
        p = int(periods[b])
        for i in range(48):
            planted[b, 0, i, max(i - p + 1, 0)] = 1.0
    model.attention_maps = lambda idx: [planted]        # type: ignore[method-assign]
    probe = induction_scores(model, batch=8, seed=0, min_period=10, max_period=14)
    assert probe["best"]["score"] == pytest.approx(1.0)
    assert probe["best"]["over_chance"] > 20


def test_induction_probe_ignores_a_fixed_offset_head():
    """The control that makes the whole probe worth running: a head that always
    looks a constant distance back scores at chance, because the period varies
    per row. This is exactly the shortcut the earlier fixed-period task allowed."""
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=64, block_size=48, n_layer=1, n_head=1, n_embd=32))
    fixed = torch.zeros(8, 1, 48, 48)
    for i in range(48):
        fixed[:, 0, i, max(i - 11, 0)] = 1.0        # always 11 back
    model.attention_maps = lambda idx: [fixed]      # type: ignore[method-assign]
    probe = induction_scores(model, batch=8, seed=0, min_period=10, max_period=14)
    assert probe["best"]["score"] < 0.35


# -- the committed results -------------------------------------------------


def test_results_json_records_a_full_passing_run():
    data = json.loads((BENCH / "results.json").read_text())
    assert data["topic"] == "T4"
    assert data["quick"] is False, "results.json must come from a full run, not --quick"
    assert data["all_passed"] is True
    assert set(data["checks"]) == {"logit_parity", "loss_curve", "generation",
                                   "induction", "kv_cache", "positions"}
    assert all(data["checks"].values())


def test_recorded_numbers_meet_the_capsule_thresholds():
    d = json.loads((BENCH / "results.json").read_text())
    assert d["logit_parity"]["relative_logit_diff"] < 1e-5
    assert d["loss_curve"]["gap_over_noise"] < 1.0
    assert d["generation"]["well_formed_fraction"] >= 0.9
    assert d["induction"]["best_head"]["over_chance"] > 5.0
    assert d["induction"]["depth_accuracy_gap"] > 0.15
    assert d["kv_cache"]["identical"] is True
    assert d["kv_cache"]["speedup"] > 1.5
    assert d["positions"]["best"] == "rope"


# -- AlphaDesk -------------------------------------------------------------


def test_topic_registers_three_components():
    reg = Registry()
    errors = reg.load_modules(["t4_transformer.alphadesk_hook"])
    assert errors == {}
    assert {c.name for c in reg.by_topic("T4")} == {
        "gpt_architecture", "kv_cache", "attention_inspector"}


def test_registered_factory_builds_a_working_model():
    reg = Registry()
    reg.load_modules(["t4_transformer.alphadesk_hook"])
    model = reg.get("models.gpt_architecture").build(vocab_size=32, block_size=16,
                                                     n_layer=1, n_head=2, n_embd=16)
    logits, _ = model(torch.randint(0, 32, (1, 8)))
    assert logits.shape == (1, 8, 32)


def test_registered_inspector_exposes_the_probe():
    reg = Registry()
    reg.load_modules(["t4_transformer.alphadesk_hook"])
    tools = reg.get("models.attention_inspector").build()
    assert callable(tools["induction_scores"])
    assert callable(tools["attention_summary"])


def test_topic_is_in_the_desk_manifest():
    from common.alphadesk import TOPIC_MODULES

    assert TOPIC_MODULES["t4_transformer.alphadesk_hook"] == "T4"


def test_declared_prerequisites_are_real_topics():
    reg = Registry()
    reg.load_modules(["t4_transformer.alphadesk_hook"])
    assert reg.get("models.gpt_architecture").requires == ("T31", "T45A")
