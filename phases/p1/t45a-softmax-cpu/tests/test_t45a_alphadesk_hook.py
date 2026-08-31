"""T45A on the desk: the numerics every later model imports."""

from __future__ import annotations

import numpy as np

from common.alphadesk import Registry, Surface


def _registry() -> Registry:
    reg = Registry()
    errors = reg.load_modules(["t45a_softmax.alphadesk_hook"])
    assert not errors, errors
    return reg


def test_registers_on_the_foundation_surface():
    comp = _registry().get("foundation.softmax_numerics")
    assert comp.topic == "T45A"
    assert comp.surface is Surface.FOUNDATION


def test_exposed_routines_work():
    built = _registry().get("foundation.softmax_numerics").build()
    expected = {"softmax", "two_pass_softmax", "online_softmax", "online_normalizer",
                "logsumexp", "log_softmax", "cross_entropy", "cross_entropy_grad"}
    assert set(built) == expected
    p = built["online_softmax"](np.array([[1e4, -1e4]]))
    np.testing.assert_allclose(p, [[1.0, 0.0]])
    assert float(built["cross_entropy"](np.zeros((1, 4)), np.array([2]))) > 0
