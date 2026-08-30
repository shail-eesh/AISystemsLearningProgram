"""The registry is the capstone's spine — it must degrade, never explode."""

import pytest

from common.alphadesk import DISCLAIMER, Registry, Surface, banner, register
from common.alphadesk.registry import Component, DuplicateComponent


@pytest.fixture
def reg() -> Registry:
    return Registry()


def test_register_decorator_files_component(reg: Registry):
    @register(topic="T30", name="fintok", surface=Surface.MODELS,
              summary="BPE tokenizer", registry=reg)
    class FinTok:
        def __init__(self, vocab: int = 8):
            self.vocab = vocab

    assert len(reg) == 1
    comp = reg.get("models.fintok")
    assert comp.topic == "T30"
    assert comp.build(vocab=16).vocab == 16
    assert FinTok.__alphadesk__ is comp


def test_duplicate_keys_are_rejected_but_replaceable(reg: Registry):
    def make():
        return object()

    reg.add(Component("T1", "x", Surface.RESEARCH, "first", make))
    with pytest.raises(DuplicateComponent):
        reg.add(Component("T2", "x", Surface.RESEARCH, "second", make))
    reg.add(Component("T2", "x", Surface.RESEARCH, "second", make), replace=True)
    assert reg.get("research.x").topic == "T2"


def test_lookup_helpers(reg: Registry):
    reg.add(Component("T5", "hnsw", Surface.RESEARCH, "vector index", dict))
    reg.add(Component("T6", "rag", Surface.RESEARCH, "pipeline", dict, requires=("T5",)))
    reg.add(Component("T28", "guardrails", Surface.COMPLIANCE, "perimeter", dict))
    assert reg.topics() == ["T28", "T5", "T6"]
    assert len(reg.by_surface("research")) == 2
    assert [c.name for c in reg.by_topic("t6")] == ["rag"]
    assert "research.rag" in reg
    with pytest.raises(KeyError):
        reg.get("nope.nope")


def test_missing_requirements_are_reported_not_raised(reg: Registry):
    reg.add(Component("T20", "graphrag", Surface.RESEARCH, "graph rag", dict,
                      requires=("T5", "T37")))
    assert reg.missing_requirements() == {"research.graphrag": ["T5", "T37"]}


def test_load_modules_degrades_gracefully(reg: Registry):
    errors = reg.load_modules(["json", "this_module_does_not_exist_forge"])
    assert list(errors) == ["this_module_does_not_exist_forge"]
    assert "ModuleNotFoundError" in errors["this_module_does_not_exist_forge"]


def test_describe_empty_and_populated(reg: Registry):
    assert "no components" in reg.describe()
    reg.add(Component("T5", "hnsw", Surface.RESEARCH, "vector index", dict))
    text = reg.describe()
    assert "hnsw" in text and "[T5]" in text


def test_disclaimer_is_present_and_boxed():
    assert "fictional educational simulation" in DISCLAIMER
    assert "no real money" in DISCLAIMER.lower()
    lines = banner(width=80).splitlines()
    assert all(len(ln) == 80 for ln in lines)
