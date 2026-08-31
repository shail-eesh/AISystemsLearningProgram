"""The layer library, the optimisers, and the loss-curve parity claim."""

from __future__ import annotations

import numpy as np
import pytest
from t31_autograd import (
    MLP,
    SGD,
    Adam,
    Linear,
    Tensor,
    bce_with_logits,
    gradcheck_tensors,
    mse_loss,
    softmax_cross_entropy,
)
from t31_autograd.train import make_features, train_reference, train_with_engine


def test_linear_shapes_and_parameter_discovery():
    lin = Linear(4, 3)
    out = lin(Tensor(np.ones((7, 4))))
    assert out.shape == (7, 3)
    assert len(lin.parameters()) == 2
    assert lin.num_parameters() == 4 * 3 + 3


def test_mlp_parameter_count_and_no_duplicates():
    m = MLP([8, 16, 1])
    assert m.num_parameters() == 8 * 16 + 16 + 16 * 1 + 1
    ids = [id(p) for p in m.parameters()]
    assert len(ids) == len(set(ids))


def test_initialisation_scales_with_fan_in():
    wide = Linear(1024, 8, rng=np.random.default_rng(0)).weight.data.std()
    narrow = Linear(16, 8, rng=np.random.default_rng(0)).weight.data.std()
    assert wide < narrow / 4


def test_bce_matches_the_closed_form():
    z = np.array([[2.0], [-3.0], [0.0]])
    y = np.array([[1.0], [0.0], [1.0]])
    loss = bce_with_logits(Tensor(z), Tensor(y, requires_grad=False))
    expected = np.mean(np.log1p(np.exp(-(2 * y - 1) * z)))
    assert float(loss.data) == pytest.approx(expected)


def test_bce_gradient_is_sigmoid_minus_target():
    z = Tensor(np.array([[2.0], [-3.0]]))
    y = Tensor(np.array([[1.0], [0.0]]), requires_grad=False)
    bce_with_logits(z, y).backward()
    expected = (1 / (1 + np.exp(-z.data)) - y.data) / 2
    np.testing.assert_allclose(z.grad, expected, rtol=1e-12)


def test_bce_survives_enormous_logits():
    """The whole point of the max-subtraction identity."""
    z = Tensor(np.array([[800.0], [-800.0]]))
    y = Tensor(np.array([[1.0], [0.0]]), requires_grad=False)
    loss = bce_with_logits(z, y)
    assert np.isfinite(float(loss.data))
    assert float(loss.data) == pytest.approx(0.0, abs=1e-12)


def test_softmax_cross_entropy_matches_reference():
    rng = np.random.default_rng(2)
    logits = rng.standard_normal((6, 4))
    idx = rng.integers(0, 4, 6)
    loss = softmax_cross_entropy(Tensor(logits), idx)
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    assert float(loss.data) == pytest.approx(-log_probs[np.arange(6), idx].mean())


@pytest.mark.parametrize("loss_name", ["mse", "bce", "softmax_ce"])
def test_loss_gradients_pass_gradcheck(loss_name):
    rng = np.random.default_rng(5)
    if loss_name == "mse":
        target = Tensor(rng.standard_normal((5, 2)), requires_grad=False)
        fn, inputs = (lambda p: mse_loss(p, target)), [Tensor(rng.standard_normal((5, 2)))]
    elif loss_name == "bce":
        target = Tensor((rng.random((5, 1)) > 0.5).astype(float), requires_grad=False)
        fn, inputs = (lambda z: bce_with_logits(z, target)), [Tensor(rng.standard_normal((5, 1)) * 4)]
    else:
        idx = rng.integers(0, 3, 5)
        fn, inputs = (lambda lg: softmax_cross_entropy(lg, idx)), [Tensor(rng.standard_normal((5, 3)))]
    assert gradcheck_tensors(fn, inputs).ok


def test_end_to_end_gradcheck_through_a_whole_mlp():
    rng = np.random.default_rng(9)
    model = MLP([4, 5, 2], rng=np.random.default_rng(0))
    x = Tensor(rng.standard_normal((6, 4)), requires_grad=False)
    target = Tensor(rng.standard_normal((6, 2)), requires_grad=False)
    res = gradcheck_tensors(lambda *ps: mse_loss(model(x), target), model.parameters())
    assert res.ok, str(res)


def test_adam_bias_correction_makes_the_first_step_full_size():
    p = Tensor(np.array([0.0]))
    p.grad = np.array([1.0])
    Adam([p], lr=0.1).step()
    # Without bias correction the first step would be ~1e-4 * lr, not ~lr.
    assert float(abs(p.data[0])) == pytest.approx(0.1, rel=1e-6)


def test_sgd_momentum_accelerates_a_constant_gradient():
    plain, mom = Tensor(np.array([0.0])), Tensor(np.array([0.0]))
    o1, o2 = SGD([plain], lr=0.1), SGD([mom], lr=0.1, momentum=0.9)
    for _ in range(5):
        plain.grad, mom.grad = np.array([1.0]), np.array([1.0])
        o1.step()
        o2.step()
    assert abs(float(mom.data[0])) > abs(float(plain.data[0]))


def test_weight_decay_pulls_toward_zero():
    p = Tensor(np.array([1.0]))
    opt = SGD([p], lr=0.1, weight_decay=0.5)
    p.grad = np.array([0.0])
    opt.step()
    assert float(p.data[0]) < 1.0


def test_loss_curve_matches_an_independent_implementation():
    """The topic's 'done when': two implementations, one trajectory."""
    X, y = make_features(n=256, d=6, seed=1)
    engine, _model = train_with_engine(X, y, hidden=12, steps=120, seed=1)
    reference = train_reference(X, y, hidden=12, steps=120, seed=1)
    a, b = np.array(engine.losses), np.array(reference.losses)
    assert np.abs(a - b).max() < 1e-9
    assert a[-1] < a[0] / 4


def test_training_actually_reduces_loss():
    X, y = make_features(n=128, d=5, seed=3)
    result, model = train_with_engine(X, y, hidden=8, steps=100, seed=3)
    assert result.losses[-1] < result.losses[0]
    assert result.accuracy > 0.85
    assert model.num_parameters() == 5 * 8 + 8 + 8 + 1
