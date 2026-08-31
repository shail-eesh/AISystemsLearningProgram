"""Shared fixtures. The dataset is built once per session — it is not free.

`torch` is an *optional* extra (`pip install -e '.[torch]'`). Without it this
whole directory is skipped rather than erroring at collection: the course must
stay green on a machine that has not installed the extras, exactly as
`common/alphadesk` degrades rather than dies when a topic is missing.
"""

import importlib.util

import pytest

if importlib.util.find_spec("torch") is None:  # pragma: no cover - environment dependent
    collect_ignore_glob = ["*.py"]
else:
    from p0_3_training import build_dataset, chronological_split, standardise

    @pytest.fixture(scope="session")
    def data():
        return build_dataset()

    @pytest.fixture(scope="session")
    def splits(data):
        tr, va, te = chronological_split(data)
        (tr, va, te), _, _ = standardise(tr, va, te)
        return tr, va, te
