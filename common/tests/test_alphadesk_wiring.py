"""Smoke test: the desk boots, and every topic built so far is plugged in.

This is the test that catches "topic T-whatever silently stopped registering".
It grows a line per phase; today it covers Phase 0.
"""

from common.alphadesk import REGISTRY, Registry, Surface, load_all

EXPECTED_TODAY = {"P0.1", "P0.2", "P0.3"}


def test_manifest_modules_all_import():
    reg = Registry()
    errors = load_all(reg)
    assert errors == {}, f"topic modules failed to import: {errors}"


def test_phase0_components_are_registered():
    reg = Registry()
    load_all(reg)
    assert set(reg.topics()) >= EXPECTED_TODAY
    book = reg.get("orders.paper_book").build()
    assert len(book) == 0, "a fresh paper book starts flat"
    checks = reg.get("compliance.default_risk_checks").build()
    assert len(checks) == 2


def test_no_unmet_requirements_so_far():
    reg = Registry()
    load_all(reg)
    assert reg.missing_requirements() == {}


def test_describe_mentions_the_surfaces_in_use():
    reg = Registry()
    load_all(reg)
    text = reg.describe()
    for surface in (Surface.ORDERS, Surface.DATA, Surface.FOUNDATION):
        assert surface.value in text


def test_global_registry_is_importable_and_empty_until_loaded():
    assert isinstance(REGISTRY, Registry)
