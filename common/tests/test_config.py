"""Config is boring on purpose — but a wrong seed silently ruins reproducibility."""


import pytest

from common.config import Config, Paths, seed_everything


def test_paths_resolve_under_repo():
    p = Paths()
    assert p.samples.is_dir(), "committed samples must exist in a fresh clone"
    assert p.samples.is_relative_to(p.repo)
    assert p.execution.joinpath("status.json").exists()


def test_env_override(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FORGE_SEED", "42")
    monkeypatch.setenv("FORGE_ALLOW_NETWORK", "1")
    cfg = Config.from_env()
    assert cfg.seed == 42
    assert cfg.allow_network is True


def test_bad_seed_is_loud(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FORGE_SEED", "not-a-number")
    with pytest.raises(ValueError):
        Config.from_env()


def test_network_defaults_off():
    assert Config().allow_network is False, "tests must never depend on the network"


def test_seed_everything_is_reproducible():
    import numpy as np

    seed_everything(7)
    a = np.random.rand(4)
    seed_everything(7)
    b = np.random.rand(4)
    assert (a == b).all()
