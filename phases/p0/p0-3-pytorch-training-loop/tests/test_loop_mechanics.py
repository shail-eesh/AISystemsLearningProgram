"""What the five lines of the training loop actually guarantee."""

import numpy as np
import pytest
import torch
from p0_3_training import TinyMLP, TrainConfig, count_parameters, evaluate, seed_everything, train
from p0_3_training.model import ManualLinear
from torch import nn


def _tensors(d):
    return (
        torch.from_numpy(np.ascontiguousarray(d.X)),
        torch.from_numpy(np.ascontiguousarray(d.y)).unsqueeze(1),
    )


def test_model_shapes_and_size():
    m = TinyMLP(8, hidden=32)
    assert m(torch.randn(5, 8)).shape == (5, 1)
    assert 700 < count_parameters(m) < 1200, "tiny by design"


def test_model_returns_logits_not_probabilities():
    m = TinyMLP(8, hidden=32)
    with torch.no_grad():
        out = m(torch.randn(256, 8) * 50)
    assert (out < 0).any() and (out > 0).any(), "logits are unbounded; probabilities are not"


def test_training_reduces_loss(splits):
    tr, va, _ = splits
    m = TinyMLP(tr.X.shape[1], hidden=32)
    before = evaluate(m, *_tensors(tr))[0]
    m, hist = train(m, tr, va, TrainConfig(epochs=15, seed=1729))
    assert hist.train_loss[-1] < before
    assert len(hist.val_loss) == 15


def test_zero_grad_is_load_bearing(splits):
    """Omit zero_grad and the gradients accumulate — provably different weights."""
    tr, _, _ = splits
    X, y = _tensors(tr)
    X, y = X[:128], y[:128]

    def run(zero: bool):
        seed_everything(0)
        m = TinyMLP(tr.X.shape[1], hidden=16)
        opt = torch.optim.SGD(m.parameters(), lr=0.05)
        crit = nn.BCEWithLogitsLoss()
        for _ in range(5):
            if zero:
                opt.zero_grad()
            crit(m(X), y).backward()
            opt.step()
        return torch.cat([p.detach().flatten() for p in m.parameters()])

    assert not torch.allclose(run(True), run(False)), "zero_grad() changed nothing?"


def test_evaluate_does_not_build_a_graph_or_leave_train_mode(splits):
    tr, _, _ = splits
    m = TinyMLP(tr.X.shape[1], hidden=16, dropout=0.5)
    m.train()
    loss, acc = evaluate(m, *_tensors(tr))
    assert m.training, "evaluate must restore the previous mode"
    assert isinstance(loss, float) and 0.0 <= acc <= 1.0
    for p in m.parameters():
        assert p.grad is None, "evaluation must not populate gradients"


def test_eval_mode_disables_dropout():
    m = TinyMLP(8, hidden=16, dropout=0.5)
    x = torch.randn(64, 8)
    m.train()
    assert not torch.allclose(m(x), m(x))
    m.eval()
    with torch.no_grad():
        assert torch.allclose(m(x), m(x))


def test_training_is_reproducible(splits):
    tr, va, _ = splits

    def run():
        seed_everything(1729)
        m = TinyMLP(tr.X.shape[1], hidden=16)
        m, _ = train(m, tr, va, TrainConfig(epochs=5, seed=1729))
        return torch.cat([p.detach().flatten() for p in m.parameters()])

    assert torch.equal(run(), run()), "same seed must give bit-identical weights"


def test_best_validation_checkpoint_is_restored(splits):
    tr, va, _ = splits
    m = TinyMLP(tr.X.shape[1], hidden=32)
    m, hist = train(m, tr, va, TrainConfig(epochs=25, seed=1729))
    final_val = evaluate(m, *_tensors(va))[0]
    assert final_val <= min(hist.val_loss) + 1e-6, "the best checkpoint was not restored"


def test_manual_gradients_match_autograd():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, 5))
    y = (rng.normal(size=(64, 1)) > 0).astype(np.float64)
    manual = ManualLinear(5, seed=0)

    W = torch.tensor(manual.W, requires_grad=True)
    b = torch.tensor(manual.b, requires_grad=True)
    loss = nn.functional.binary_cross_entropy_with_logits(
        torch.tensor(X) @ W + b, torch.tensor(y)
    )
    loss.backward()

    p = manual.forward(X)
    dz = (p - y) / X.shape[0]
    assert np.allclose(W.grad.numpy(), X.T @ dz, atol=1e-10)
    assert np.allclose(b.grad.numpy(), dz.sum(axis=0), atol=1e-10)
    assert abs(manual.loss(p, y) - loss.item()) < 1e-10


def test_manual_sigmoid_is_numerically_stable():
    m = ManualLinear(1)
    z = np.array([[-800.0], [0.0], [800.0]])
    out = m._sigmoid(z)
    assert np.isfinite(out).all()
    assert out[0, 0] == pytest.approx(0.0) and out[2, 0] == pytest.approx(1.0)
    assert out[1, 0] == pytest.approx(0.5)


def test_manual_linear_learns_a_separable_problem():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(400, 2))
    y = (X[:, :1] + X[:, 1:] > 0).astype(np.float64)
    m = ManualLinear(2, seed=0)
    first = m.step(X, y, lr=0.5)
    for _ in range(500):
        last = m.step(X, y, lr=0.5)
    assert last < first
    assert ((m.forward(X) > 0.5) == y).mean() > 0.95
