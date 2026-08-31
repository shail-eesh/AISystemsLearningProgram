"""T16A on the desk: literacy, not product."""

from __future__ import annotations

import numpy as np

from common.alphadesk import Registry, Surface


def _registry() -> Registry:
    reg = Registry()
    errors = reg.load_modules(["t16a_matmul.alphadesk_hook"])
    assert not errors, errors
    return reg


def test_registers_on_the_foundation_surface():
    reg = _registry()
    comp = reg.get("foundation.cpu_matmul_kernels")
    assert comp.topic == "T16A"
    assert comp.surface is Surface.FOUNDATION


def test_component_exposes_kernels_and_accounting():
    built = _registry().get("foundation.cpu_matmul_kernels").build()
    assert built["flops"](2, 2, 2) == 16
    assert built["environment"]()["machine"]
    A = np.eye(4)
    np.testing.assert_allclose(built["reference"](A, A), A)
    if built["available"]:
        np.testing.assert_allclose(built["call"]("matmul_ikj", A, A), A, atol=1e-12)
