"""torch is an optional extra; without it this directory is skipped, not failed.

Same contract as P0.3: the course must stay green on a machine that has not
installed `.[torch]`, exactly as `common/alphadesk` degrades rather than dies
when a topic is missing.
"""

import importlib.util

import pytest

if importlib.util.find_spec("torch") is None:  # pragma: no cover - environment dependent
    collect_ignore_glob = ["*.py"]
else:
    import torch
    from t4_transformer import GPT, GPTConfig, char_dataset

    @pytest.fixture(scope="session")
    def tape():
        return char_dataset()

    @pytest.fixture
    def small_config():
        return GPTConfig(vocab_size=32, block_size=24, n_layer=2, n_head=4,
                         n_embd=32, dropout=0.0)

    @pytest.fixture
    def small_model(small_config):
        torch.manual_seed(0)
        return GPT(small_config).eval()
