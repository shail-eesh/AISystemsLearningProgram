"""Evaluation semantics, the committed benchmark results, and the AlphaDesk hook."""

import json
import math
import pathlib

import pytest
import torch
from t4_transformer import GPT
from t15_alphaslm import (
    LADDER,
    Trainer,
    TrainSpec,
    compare_by_token_class,
    compare_models,
    loss_by_token_class,
    perplexity_by_tag,
    perplexity_on_documents,
    sample_commentary,
)
from t15_alphaslm.scaling import analyse, extrapolate

from common.alphadesk import Registry

BENCH = pathlib.Path(__file__).resolve().parent.parent / "bench"
VOCAB = 3495


# -- perplexity ------------------------------------------------------------


def test_perplexity_is_exp_of_the_loss(tiny_model, tokenizer, small_corpus):
    result = perplexity_on_documents(tiny_model, tokenizer, small_corpus[:5])
    assert result["perplexity"] == pytest.approx(math.exp(result["loss"]))
    assert result["tokens"] > 0


def test_an_untrained_model_scores_near_the_uniform_baseline(tiny_model, tokenizer,
                                                             small_corpus):
    result = perplexity_on_documents(tiny_model, tokenizer, small_corpus[:5])
    assert result["loss"] == pytest.approx(math.log(VOCAB), abs=0.6)


def test_degenerate_documents_are_skipped_not_scored(tiny_model, tokenizer):
    result = perplexity_on_documents(tiny_model, tokenizer, ["", "a", "hello there"])
    assert result["documents"] <= 2


def test_bits_per_char_is_reported_for_cross_tokenizer_comparison(tiny_model, tokenizer,
                                                                  small_corpus):
    result = perplexity_on_documents(tiny_model, tokenizer, small_corpus[:5])
    assert 0 < result["bits_per_char"] < 8
    # loss is per token; bits/char is per character, and the corpus is ~4.8
    # characters per token, so the second must be the smaller number
    assert result["bits_per_char"] < result["loss"] / math.log(2)


def test_perplexity_splits_by_register(tiny_model, tokenizer, small_corpus):
    by_tag = perplexity_by_tag(tiny_model, tokenizer, small_corpus, max_docs_per_tag=3)
    assert {"filing", "commentary", "announcement", "order", "tape"} <= set(by_tag)
    for stats in by_tag.values():
        assert stats["tokens"] >= 0


def test_token_class_split_covers_every_token(tiny_model, tokenizer, small_corpus):
    out = loss_by_token_class(tiny_model, tokenizer, small_corpus[:6], max_docs=6)
    assert out["numeric"]["share"] + out["prose"]["share"] == pytest.approx(1.0)
    assert out["numeric"]["tokens"] > 0 and out["prose"]["tokens"] > 0
    blended = (out["numeric"]["loss"] * out["numeric"]["share"]
               + out["prose"]["loss"] * out["prose"]["share"])
    assert blended == pytest.approx(out["overall_loss"], rel=1e-6)


def test_compare_models_reports_pairwise_margins(tiny_model, tokenizer, small_corpus):
    torch.manual_seed(1)
    other = GPT(tiny_model.config)
    result = compare_models({"a": tiny_model, "b": other}, tokenizer,
                            small_corpus[:4], max_docs=4)
    assert set(result["scores"]) == {"a", "b"}
    margin = result["margins"]["a vs b"]
    assert margin["loss_delta"] == pytest.approx(
        result["scores"]["a"]["loss"] - result["scores"]["b"]["loss"])
    assert margin["perplexity_ratio"] == pytest.approx(math.exp(margin["loss_delta"]))


def test_compare_by_token_class_reports_both_halves(tiny_model, tokenizer, small_corpus):
    torch.manual_seed(1)
    out = compare_by_token_class({"a": tiny_model, "b": GPT(tiny_model.config)},
                                 tokenizer, small_corpus[:4], max_docs=4)
    assert set(out["per_model"]) == {"a", "b"}
    assert "prose_loss_improvement" in out and "numeric_loss_improvement" in out


def test_generation_starts_from_the_prompt(small_shards, tokenizer):
    train, val, _, _ = small_shards
    torch.manual_seed(15)
    model = GPT(LADDER["alphaslm-0.6m"].gpt_config(VOCAB).scaled(
        block_size=64, n_layer=1, n_embd=64))
    Trainer(model, train, val, TrainSpec(steps=30, batch_size=4, warmup=5,
                                         eval_every=10_000, checkpoint_every=0)).train()
    prompt = "<|commentary|> Market commentary for 2024-06-12."
    text = sample_commentary(model, tokenizer, prompt=prompt, max_new_tokens=20)
    assert text.startswith(prompt)
    assert len(text) > len(prompt)


# -- the scaling analysis --------------------------------------------------


def test_analyse_detects_a_broken_ordering():
    rows = [
        {"name": "s", "params": 1_000, "val_loss": 1.0, "val_perplexity": math.e,
         "final_train_loss": 0.9},
        {"name": "m", "params": 10_000, "val_loss": 1.2, "val_perplexity": 3.3,
         "final_train_loss": 1.1},
        {"name": "l", "params": 100_000, "val_loss": 0.8, "val_perplexity": 2.2,
         "final_train_loss": 0.6},
    ]
    out = analyse(rows)
    assert out["ordering_holds"] is False
    assert out["smallest"] == "s" and out["largest"] == "l"


def test_power_law_fit_passes_through_its_two_anchor_points():
    rows = [
        {"name": "s", "params": 1_000, "val_loss": 2.0, "val_perplexity": 7.4,
         "final_train_loss": 1.9},
        {"name": "m", "params": 10_000, "val_loss": 1.5, "val_perplexity": 4.5,
         "final_train_loss": 1.4},
        {"name": "l", "params": 100_000, "val_loss": 1.0, "val_perplexity": 2.7,
         "final_train_loss": 0.9},
    ]
    out = analyse(rows)
    fit = {r["name"]: r for r in out["power_law_residuals"]}
    assert fit["s"]["relative_error"] == pytest.approx(0.0, abs=1e-9)
    assert fit["l"]["relative_error"] == pytest.approx(0.0, abs=1e-9)
    # the middle point is the only real evidence, and it is *not* forced to fit
    assert abs(fit["m"]["relative_error"]) > 0
    assert extrapolate({"power_law": out["power_law"]}, 1_000) == pytest.approx(2.0)


# -- committed results -----------------------------------------------------


def test_results_json_records_a_full_passing_run():
    data = json.loads((BENCH / "results.json").read_text())
    assert data["topic"] == "T15"
    assert data["quick"] is False, "results.json must come from a full run"
    assert data["all_passed"] is True
    assert set(data["checks"]) == {"shards", "resume", "accumulation", "scaling",
                                   "perplexity", "register"}


def test_recorded_numbers_meet_the_capsule_thresholds():
    d = json.loads((BENCH / "results.json").read_text())
    assert d["shards"]["documents_in_both_splits"] == 0
    assert all(d["shards"]["checksums_match_committed"].values())
    assert d["resume"]["max_parameter_difference"] == 0.0
    assert d["accumulation"]["max_parameter_difference"] < 1e-4
    assert d["scaling"]["ordering_holds"] is True
    assert d["perplexity"]["perplexity_ratio"] > 1.005
    assert d["register"]["hit_rate"] >= 0.8


def test_the_entropy_floor_finding_is_recorded_not_hidden():
    """The scaling margin is small because most of the corpus is irreducible.

    That decomposition is the topic's actual finding, so it has to be in the
    committed results — a reader who sees a 1.4% margin and no explanation has
    been given a number without its meaning.
    """
    d = json.loads((BENCH / "results.json").read_text())
    classes = d["perplexity"]["by_token_class"]["per_model"]
    largest = d["perplexity"]["largest"]
    numeric = classes[largest]["numeric"]
    prose = classes[largest]["prose"]
    assert numeric["loss"] > 3.0, "digits should be close to unpredictable"
    assert prose["loss"] < 0.2, "templated prose should be close to deterministic"
    assert 0.1 < numeric["share"] < 0.3
    assert "entropy_floor_finding" in d["perplexity"]
    # and the smallest rung is already at the prose floor, which is *why* the
    # margin is small — so the prose improvement across the ladder is ~zero
    smallest = d["perplexity"]["smallest"]
    assert abs(classes[smallest]["prose"]["loss"] - prose["loss"]) < 0.05


def test_the_gpu_lane_is_declared_not_silently_skipped():
    d = json.loads((BENCH / "results.json").read_text())
    assert d["gpu_lane"]["status"] == "awaiting-4070"
    assert d["gpu_lane"]["rungs"] == ["alphaslm-15m", "alphaslm-40m"]
    assert pathlib.Path(BENCH.parents[3] / d["gpu_lane"]["runner"]).exists()


def test_scaling_study_json_matches_the_ladder():
    study = json.loads((BENCH / "scaling_study.json").read_text())
    names = [r["name"] for r in study["rungs"]]
    assert names == ["alphaslm-0.6m", "alphaslm-1.8m", "alphaslm-5m"]
    for row in study["rungs"]:
        assert row["params"] == LADDER[row["name"]].parameters(VOCAB)


# -- AlphaDesk -------------------------------------------------------------


def test_topic_registers_three_components():
    reg = Registry()
    assert reg.load_modules(["t15_alphaslm.alphadesk_hook"]) == {}
    assert {c.name for c in reg.by_topic("T15")} == {
        "alphaslm", "pretrain_shards", "alphaslm_eval"}


def test_alphaslm_builds_without_weights():
    """A desk with no checkpoint still boots — that is the registry's contract."""
    reg = Registry()
    reg.load_modules(["t15_alphaslm.alphadesk_hook"])
    model = reg.get("models.alphaslm").build("alphaslm-0.6m")
    logits, _ = model(torch.randint(0, VOCAB, (1, 16)))
    assert logits.shape == (1, 16, VOCAB)


def test_alphaslm_rejects_an_unknown_rung():
    reg = Registry()
    reg.load_modules(["t15_alphaslm.alphadesk_hook"])
    with pytest.raises(KeyError, match="unknown rung"):
        reg.get("models.alphaslm").build("alphaslm-70b")


def test_alphaslm_declares_its_prerequisites():
    reg = Registry()
    reg.load_modules(["t15_alphaslm.alphadesk_hook"])
    assert reg.get("models.alphaslm").requires == ("T4", "T30")


def test_eval_component_exposes_the_measurement_functions():
    reg = Registry()
    reg.load_modules(["t15_alphaslm.alphadesk_hook"])
    tools = reg.get("models.alphaslm_eval").build()
    assert set(tools) == {"perplexity_on_documents", "perplexity_by_tag",
                          "compare_models"}


def test_topic_is_in_the_desk_manifest():
    from common.alphadesk import TOPIC_MODULES

    assert TOPIC_MODULES["t15_alphaslm.alphadesk_hook"] == "T15"
