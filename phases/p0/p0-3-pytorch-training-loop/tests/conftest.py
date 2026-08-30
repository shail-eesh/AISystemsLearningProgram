"""Shared fixtures. The dataset is built once per session — it is not free."""

import pytest
from p0_3_training import build_dataset, chronological_split, standardise


@pytest.fixture(scope="session")
def data():
    return build_dataset()


@pytest.fixture(scope="session")
def splits(data):
    tr, va, te = chronological_split(data)
    (tr, va, te), _, _ = standardise(tr, va, te)
    return tr, va, te
