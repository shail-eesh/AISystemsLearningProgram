"""The corpus, the batching contract, and the training harness."""

import math

import pytest
import torch
from t4_transformer import (
    GPT,
    CharVocab,
    GPTConfig,
    TrainConfig,
    build_corpus,
    estimate_loss,
    get_batch,
    lr_at,
    repeated_sequence_batch,
    smoothed,
    train,
)


def test_corpus_is_deterministic_and_grammatical():
    a, b = build_corpus(), build_corpus()
    assert a == b
    lines = a.splitlines()
    assert len(lines) > 1000
    tape = [ln for ln in lines if ln.startswith("2024-")]
    assert len(tape) == 1300
    for ln in tape[:20]:
        parts = ln.split()
        assert parts[2] == "O" and parts[4] == "H" and parts[6] == "L"
        assert parts[8] == "C" and parts[10] == "V"


def test_vocab_round_trips(tape):
    _, _, vocab = tape
    text = build_corpus()[:5000]
    assert vocab.decode(vocab.encode(text)) == text


def test_vocab_rejects_unseen_characters():
    v = CharVocab.from_text("abc")
    assert v.size == 3
    with pytest.raises(KeyError, match="absent"):
        v.encode("abcd")


def test_split_is_chronological_not_shuffled(tape):
    train_ids, val_ids, vocab = tape
    text = build_corpus()
    n = len(train_ids)
    assert vocab.decode(train_ids) == text[:n]
    assert vocab.decode(val_ids) == text[n:]


def test_targets_are_inputs_shifted_by_one(tape):
    data, _, _ = tape
    g = torch.Generator().manual_seed(0)
    x, y = get_batch(data, 4, 16, generator=g)
    assert x.shape == y.shape == (4, 16)
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_get_batch_needs_more_data_than_the_block():
    with pytest.raises(ValueError, match="need more than"):
        get_batch(torch.arange(8), 2, 8)


def test_repeated_sequence_is_two_identical_halves():
    g = torch.Generator().manual_seed(0)
    seq = repeated_sequence_batch(4, 10, 30, generator=g)
    assert seq.shape == (4, 20)
    assert torch.equal(seq[:, :10], seq[:, 10:])


def test_lr_schedule_warms_up_then_decays():
    cfg = TrainConfig(steps=100, warmup=10, lr=1e-3, min_lr_ratio=0.1)
    warm = [lr_at(s, cfg) for s in range(10)]
    assert warm == sorted(warm)
    assert warm[-1] == pytest.approx(1e-3)
    decay = [lr_at(s, cfg) for s in range(10, 100)]
    assert decay == sorted(decay, reverse=True)
    assert lr_at(99, cfg) == pytest.approx(1e-4, rel=0.02)


def test_lr_schedule_never_goes_below_the_floor():
    cfg = TrainConfig(steps=50, warmup=5, lr=1e-3, min_lr_ratio=0.1)
    assert min(lr_at(s, cfg) for s in range(200)) >= 1e-4 - 1e-12


def test_smoothed_is_a_trailing_mean():
    assert smoothed([1.0, 3.0, 5.0], window=2) == [1.0, 2.0, 4.0]


def test_training_reduces_loss_on_the_tape(tape):
    train_ids, val_ids, vocab = tape
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=vocab.size, block_size=32, n_layer=2,
                          n_head=4, n_embd=64))
    hist = train(model, train_ids, val_ids,
                 TrainConfig(steps=120, batch_size=8, lr=3e-3, eval_every=60,
                             eval_batches=4))
    assert hist.val_loss[0] > hist.val_loss[-1]
    assert hist.val_loss[0] == pytest.approx(math.log(vocab.size), abs=0.4)
    assert hist.val_loss[-1] < 2.5
    assert len(hist.train_loss) == 120


def test_training_is_reproducible(tape):
    train_ids, val_ids, vocab = tape
    finals = []
    for _ in range(2):
        torch.manual_seed(0)
        model = GPT(GPTConfig(vocab_size=vocab.size, block_size=32, n_layer=2,
                              n_head=2, n_embd=32))
        finals.append(train(model, train_ids, val_ids,
                            TrainConfig(steps=30, batch_size=8, eval_every=30,
                                        eval_batches=2)).final_val)
    assert finals[0] == finals[1]


def test_gradient_clipping_bounds_the_update(tape):
    """Clipping is the seatbelt; this checks it is actually fastened."""
    train_ids, _, vocab = tape
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=vocab.size, block_size=32, n_layer=2,
                          n_head=2, n_embd=32))
    g = torch.Generator().manual_seed(0)
    x, y = get_batch(train_ids, 8, 32, generator=g)
    _, loss = model(x, y)
    (loss * 1000).backward()          # a deliberately enormous gradient
    before = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
    after = torch.norm(torch.stack([p.grad.norm() for p in model.parameters()
                                    if p.grad is not None]))
    assert before > 1.0
    assert float(after) == pytest.approx(1.0, abs=1e-4)


def test_estimate_loss_leaves_the_model_in_training_mode(tape):
    train_ids, val_ids, vocab = tape
    model = GPT(GPTConfig(vocab_size=vocab.size, block_size=32, n_layer=1,
                          n_head=2, n_embd=32))
    model.train()
    estimate_loss(model, val_ids, TrainConfig(batch_size=4, eval_batches=2))
    assert model.training
