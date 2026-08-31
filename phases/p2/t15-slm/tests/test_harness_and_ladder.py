"""The size ladder, the schedule, and the harness's restart guarantees."""

import json
import math

import pytest
import torch

from t15_alphaslm import (
    CPU_RUNGS,
    GPU_RUNGS,
    LADDER,
    Trainer,
    TrainSpec,
    describe_ladder,
    lr_at,
)
from t4_transformer import GPT

VOCAB = 3495


# -- the ladder ------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(LADDER))
def test_closed_form_parameter_count_matches_a_built_model(name):
    """The ladder is data, so its arithmetic has to be checked against reality —
    otherwise the 40M rung's 'plan' is a guess printed in a nice table."""
    rung = LADDER[name]
    model = GPT(rung.gpt_config(VOCAB))
    assert rung.parameters(VOCAB) == model.num_params()


def test_ladder_is_monotone_in_size():
    sizes = [r.parameters(VOCAB) for r in LADDER.values()]
    assert sizes == sorted(sizes)


def test_rung_names_are_honest_about_their_size():
    for name, rung in LADDER.items():
        claimed = float(name.split("-")[1].rstrip("m"))
        actual = rung.parameters(VOCAB) / 1e6
        assert abs(actual - claimed) / claimed < 0.25, f"{name} is really {actual:.1f}M"


def test_every_rung_uses_the_phase_2_architecture():
    for rung in LADDER.values():
        cfg = rung.gpt_config(VOCAB)
        assert cfg.position == "rope"
        assert cfg.norm == "rmsnorm"
        assert cfg.tie_weights is True
        assert cfg.bias is False
        assert cfg.n_embd % cfg.n_head == 0


def test_lanes_are_declared():
    assert [r.name for r in CPU_RUNGS] == ["alphaslm-0.6m", "alphaslm-1.8m", "alphaslm-5m"]
    assert [r.name for r in GPU_RUNGS] == ["alphaslm-15m", "alphaslm-40m"]


def test_chinchilla_target_is_twenty_tokens_per_parameter():
    for rung in LADDER.values():
        assert rung.chinchilla_tokens(VOCAB) == 20 * rung.parameters(VOCAB)


def test_describe_ladder_lists_every_rung():
    text = describe_ladder(VOCAB)
    for name in LADDER:
        assert name in text


# -- the schedule ----------------------------------------------------------


def test_warmup_rises_linearly_to_the_peak():
    spec = TrainSpec(steps=1000, warmup=100, lr=1e-3)
    warm = [lr_at(s, spec) for s in range(100)]
    assert warm == sorted(warm)
    assert warm[0] == pytest.approx(1e-5)
    assert warm[-1] == pytest.approx(1e-3)


def test_cosine_decays_to_the_floor():
    spec = TrainSpec(steps=1000, warmup=100, lr=1e-3, min_lr_ratio=0.1)
    assert lr_at(999, spec) == pytest.approx(1e-4, rel=0.02)
    # the floor applies to the decay phase; warmup deliberately starts below it
    assert min(lr_at(s, spec) for s in range(100, 3000)) >= 1e-4 - 1e-12
    decay = [lr_at(s, spec) for s in range(100, 1000)]
    assert decay == sorted(decay, reverse=True)


def test_zero_warmup_still_works():
    spec = TrainSpec(steps=10, warmup=0, lr=1e-3)
    assert lr_at(0, spec) == pytest.approx(1e-3, rel=0.01)


# -- the harness -----------------------------------------------------------


def _spec(**kw):
    base = dict(steps=20, batch_size=4, warmup=4, eval_every=10_000,
                checkpoint_every=0, seed=15)
    return TrainSpec(**(base | kw))


def test_training_reduces_the_loss(small_shards, tiny_model):
    train, val, _, _ = small_shards
    t = Trainer(tiny_model, train, val, _spec(steps=60))
    state = t.train()
    first = sum(h["loss"] for h in state.history[:5]) / 5
    last = sum(h["loss"] for h in state.history[-5:]) / 5
    assert last < first
    assert state.step == 60
    assert state.tokens == 60 * 4 * 64


def test_evaluate_returns_perplexity_consistent_with_the_loss(small_shards, tiny_model):
    train, val, _, _ = small_shards
    t = Trainer(tiny_model, train, val, _spec())
    result = t.evaluate(batches=3)
    assert result["perplexity"] == pytest.approx(math.exp(result["loss"]))
    assert result["batches"] <= 3


def test_evaluate_is_deterministic(small_shards, tiny_model):
    """Sequential windows, so the same model gives the same number every time."""
    train, val, _, _ = small_shards
    t = Trainer(tiny_model, train, val, _spec())
    assert t.evaluate(batches=3)["loss"] == t.evaluate(batches=3)["loss"]


def test_evaluate_restores_training_mode(small_shards, tiny_model):
    train, val, _, _ = small_shards
    t = Trainer(tiny_model, train, val, _spec())
    tiny_model.train()
    t.evaluate(batches=1)
    assert tiny_model.training


def test_resume_is_bit_identical(small_shards, tmp_path):
    """The claim the whole checkpoint format exists for."""
    train, val, _, _ = small_shards
    rung = LADDER["alphaslm-0.6m"]

    def build():
        torch.manual_seed(15)
        return GPT(rung.gpt_config(VOCAB).scaled(block_size=64, n_layer=1, n_embd=64))

    spec = dict(steps=40, batch_size=4, warmup=8, eval_every=10_000,
                checkpoint_every=20, seed=15)
    straight = build()
    Trainer(straight, train, val, TrainSpec(**spec), run_dir=tmp_path / "a").train()

    part = build()
    Trainer(part, train, val, TrainSpec(**spec), run_dir=tmp_path / "b").train(until=20)
    torch.manual_seed(4321)                 # poison the global RNG between halves
    resumed = build()
    t = Trainer(resumed, train, val, TrainSpec(**spec), run_dir=tmp_path / "b")
    assert t.load().step == 20
    t.train()

    for a, b in zip(straight.state_dict().values(), resumed.state_dict().values(),
                    strict=True):
        assert torch.equal(a, b)


def test_checkpoint_carries_the_optimizer_and_both_rng_states(small_shards, tmp_path,
                                                              tiny_model):
    train, val, _, _ = small_shards
    t = Trainer(tiny_model, train, val, _spec(), run_dir=tmp_path)
    t.train()
    ckpt = torch.load(t.save(), map_location="cpu", weights_only=False)
    assert set(ckpt) == {"model", "optimizer", "state", "spec", "config",
                         "numpy_rng", "torch_rng"}
    assert ckpt["optimizer"]["state"], "Adam moments must be in the checkpoint"
    assert ckpt["state"]["step"] == 20


def test_accumulation_matches_the_full_batch(small_shards):
    """Same windows, same update — to float32 reassociation, not to the bit."""
    train, val, _, _ = small_shards
    rung = LADDER["alphaslm-0.6m"]
    outs = []
    for batch, micro in ((8, 1), (2, 4)):
        torch.manual_seed(15)
        model = GPT(rung.gpt_config(VOCAB).scaled(block_size=64, n_layer=1, n_embd=64))
        Trainer(model, train, val,
                _spec(steps=15, batch_size=batch, micro_batches=micro)).train()
        outs.append(model)
    worst = max(float((a - b).abs().max())
                for a, b in zip(outs[0].state_dict().values(),
                                outs[1].state_dict().values(), strict=True))
    assert worst < 1e-4
    assert worst > 0.0, "exactly zero would mean the accumulation path is not running"


def test_gradient_clipping_is_applied(small_shards, tiny_model):
    train, val, _, _ = small_shards
    t = Trainer(tiny_model, train, val, _spec(steps=5, grad_clip=0.01))
    state = t.train()
    # clip_grad_norm_ returns the norm *before* clipping; the update used 0.01
    assert all(h["grad_norm"] > 0 for h in state.history)


def test_metrics_log_is_one_json_object_per_line(small_shards, tmp_path, tiny_model):
    train, val, _, _ = small_shards
    t = Trainer(tiny_model, train, val,
                _spec(steps=20, log_every=5, eval_every=10, eval_batches=2),
                run_dir=tmp_path)
    t.train()
    lines = (tmp_path / "metrics.jsonl").read_text().splitlines()
    assert len(lines) >= 4
    for line in lines:
        record = json.loads(line)
        assert "step" in record or "eval" in record


def test_wall_clock_budget_stops_cleanly(small_shards, tmp_path, tiny_model):
    train, val, _, _ = small_shards
    t = Trainer(tiny_model, train, val,
                _spec(steps=100_000, max_seconds=0.5, checkpoint_every=0),
                run_dir=tmp_path)
    state = t.train()
    assert 0 < state.step < 100_000
    assert t.checkpoint_path().exists()


def test_trainer_without_a_run_dir_refuses_to_checkpoint(small_shards, tiny_model):
    train, val, _, _ = small_shards
    t = Trainer(tiny_model, train, val, _spec())
    with pytest.raises(ValueError, match="no run_dir"):
        t.save()
