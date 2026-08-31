"""torch is an optional extra; without it this directory is skipped, not failed.

The shard fixture builds a *small* corpus into a temporary directory rather than
touching the committed artifacts, so tests never depend on — or clobber — the
packed shards a step or a benchmark may have written.
"""

import importlib.util

import pytest

if importlib.util.find_spec("torch") is None:  # pragma: no cover - environment dependent
    collect_ignore_glob = ["*.py"]
else:
    import torch

    from t15_alphaslm import (
        LADDER,
        build_corpus,
        load_tokenizer,
        open_shards,
        pack_documents,
        split_documents,
    )
    from t4_transformer import GPT

    @pytest.fixture(scope="session")
    def tokenizer():
        return load_tokenizer()

    @pytest.fixture(scope="session")
    def small_corpus():
        return build_corpus(docs=300)

    @pytest.fixture(scope="session")
    def small_shards(tmp_path_factory, small_corpus, tokenizer):
        d = tmp_path_factory.mktemp("shards")
        train_docs, val_docs = split_documents(small_corpus)
        meta = pack_documents(train_docs, val_docs, out_dir=d, tokenizer=tokenizer)
        train, val, _ = open_shards(d, block_size=64)
        return train, val, meta, d

    @pytest.fixture
    def tiny_model():
        torch.manual_seed(15)
        rung = LADDER["alphaslm-0.6m"]
        cfg = rung.gpt_config(3495).scaled(block_size=64, n_layer=1, n_embd=64, n_head=4)
        return GPT(cfg)
