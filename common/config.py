"""Course-wide configuration.

One immutable `Config` object, resolved once, overridable by environment
variables prefixed with ``FORGE_``. Keeping this in one place means a topic
never hard-codes a path and every run is reproducible from the same seed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser().resolve() if raw else default


@dataclass(frozen=True)
class Paths:
    """Every directory the course writes to, resolved from the repo root."""

    repo: Path = REPO_ROOT
    common: Path = field(default_factory=lambda: REPO_ROOT / "common")
    data: Path = field(default_factory=lambda: REPO_ROOT / "common" / "data")
    samples: Path = field(default_factory=lambda: REPO_ROOT / "common" / "data" / "samples")
    cache: Path = field(default_factory=lambda: REPO_ROOT / "common" / "data" / "cache")
    phases: Path = field(default_factory=lambda: REPO_ROOT / "phases")
    video: Path = field(default_factory=lambda: REPO_ROOT / "video")
    execution: Path = field(default_factory=lambda: REPO_ROOT / "EXECUTION")

    def ensure(self) -> Paths:
        """Create the writable directories (cache only; the rest are committed)."""
        self.cache.mkdir(parents=True, exist_ok=True)
        return self


@dataclass(frozen=True)
class Config:
    """Global knobs. Small by design — this course never trains for days."""

    seed: int = 1729
    #: Offline by default: tests and CI must never depend on a network call.
    allow_network: bool = False
    #: Politeness for the SEC EDGAR API, which requires a descriptive UA.
    edgar_user_agent: str = "AI Systems Forge educational course (contact: local)"
    #: Default float precision for numeric comparisons across the course.
    tolerance: float = 1e-6
    paths: Paths = field(default_factory=Paths)

    @classmethod
    def from_env(cls) -> Config:
        return cls(
            seed=_env_int("FORGE_SEED", 1729),
            allow_network=os.environ.get("FORGE_ALLOW_NETWORK", "0") not in ("0", "", "false", "False"),
            edgar_user_agent=os.environ.get("FORGE_EDGAR_UA", cls.edgar_user_agent),
            paths=Paths(repo=_env_path("FORGE_REPO_ROOT", REPO_ROOT)),
        )


CONFIG = Config.from_env()
paths = CONFIG.paths


def seed_everything(seed: int | None = None) -> int:
    """Seed every RNG the course touches. Returns the seed actually used."""
    import random

    import numpy as np

    s = CONFIG.seed if seed is None else seed
    random.seed(s)
    np.random.seed(s % (2**32))
    try:  # torch is optional in Phase 0/1
        import torch

        torch.manual_seed(s)
    except ModuleNotFoundError:
        pass
    return s
