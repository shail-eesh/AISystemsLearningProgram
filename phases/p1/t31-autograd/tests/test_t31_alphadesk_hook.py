"""T31's contribution to the desk, and the disclaimer that rides with it."""

from __future__ import annotations

import numpy as np

from common.alphadesk import Registry, Surface


def _registry() -> Registry:
    reg = Registry()
    errors = reg.load_modules(["t31_autograd.alphadesk_hook"])
    assert not errors, errors
    return reg


def test_topic_registers_two_components():
    reg = _registry()
    keys = {c.key for c in reg.by_topic("T31")}
    assert keys == {"foundation.autograd", "foundation.handrolled_signal_model"}


def test_autograd_component_exposes_the_engine():
    built = _registry().get("foundation.autograd").build()
    assert set(built) == {"Value", "Tensor", "gradcheck_tensors", "gradcheck_values"}
    v = built["Value"](2.0)
    (v * v).backward()
    assert v.grad == 4.0


def test_signal_model_trains_and_carries_the_disclaimer():
    trainer = _registry().get("foundation.handrolled_signal_model").build(hidden=8)
    assert "not a trading signal" in trainer.disclaimer
    result = trainer.fit(steps=50)
    assert result.losses[-1] < result.losses[0]
    proba = trainer.predict_proba(np.zeros((3, 8)))
    assert proba.shape == (3, 1)
    assert np.all((proba >= 0) & (proba <= 1))


def test_components_land_on_the_foundation_surface():
    reg = _registry()
    assert all(c.surface is Surface.FOUNDATION for c in reg.by_topic("T31"))
