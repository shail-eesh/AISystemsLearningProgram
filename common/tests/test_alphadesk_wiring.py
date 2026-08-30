"""Smoke test: the desk boots, and every topic built so far is plugged in.

This is the test that catches "topic T-whatever silently stopped registering".
It grows a line per phase; today it covers Phase 0.

One subtlety: some topics depend on an optional extra (P0.3 needs `torch`). The
registry is *designed* to degrade rather than raise in that case, so the tests
below distinguish "did not register because the code is broken" from "did not
register because a dependency is not installed" — and only the first is a
failure.
"""

import importlib.util

import pytest

from common.alphadesk import REGISTRY, Registry, Surface, load_all

EXPECTED_ALWAYS = {"P0.1", "P0.2"}
#: topic -> the optional import it needs
OPTIONAL = {"P0.3": "torch"}


def _available(topic: str) -> bool:
    dep = OPTIONAL.get(topic)
    return dep is None or importlib.util.find_spec(dep) is not None


def _expected() -> set[str]:
    return EXPECTED_ALWAYS | {t for t in OPTIONAL if _available(t)}


@pytest.fixture
def desk() -> Registry:
    reg = Registry()
    errors = load_all(reg)
    real = {
        m: e
        for m, e in errors.items()
        if not any(f"No module named '{dep}'" in e for dep in OPTIONAL.values())
    }
    assert real == {}, f"topic modules failed to import: {real}"
    return reg


def test_phase0_components_are_registered(desk: Registry):
    assert _expected() <= set(desk.topics())
    book = desk.get("orders.paper_book").build()
    assert len(book) == 0, "a fresh paper book starts flat"
    assert len(desk.get("compliance.default_risk_checks").build()) == 2


def test_feature_table_component_is_callable(desk: Registry):
    frame = desk.get("data.technical_features").build(["ALPHAINFRA"])
    assert len(frame) == 260
    assert "sma_20" in frame.columns


def test_no_unmet_requirements_so_far(desk: Registry):
    assert desk.missing_requirements() == {}


def test_describe_mentions_the_surfaces_in_use(desk: Registry):
    text = desk.describe()
    for surface in (Surface.ORDERS, Surface.COMPLIANCE, Surface.DATA):
        assert surface.value in text
    if _available("P0.3"):
        assert Surface.FOUNDATION.value in text


def test_registering_twice_into_the_same_registry_is_refused(desk: Registry):
    """A second load_all into a live registry must not silently duplicate."""
    errors = load_all(desk)
    assert errors, "re-registering the same components should surface a conflict"
    assert all("Duplicate" in e or "already registered" in e for e in errors.values())


def test_global_registry_is_importable(desk: Registry):
    assert isinstance(REGISTRY, Registry)
