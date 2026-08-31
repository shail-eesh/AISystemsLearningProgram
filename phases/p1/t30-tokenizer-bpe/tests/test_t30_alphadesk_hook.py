"""FinTok on the desk — the one Phase-1 topic with a real product hook."""

from __future__ import annotations

from common.alphadesk import Registry, Surface


def _registry() -> Registry:
    reg = Registry()
    errors = reg.load_modules(["t30_fintok.alphadesk_hook"])
    assert not errors, errors
    return reg


def test_registers_the_tokenizer_and_the_trainer():
    reg = _registry()
    assert {c.key for c in reg.by_topic("T30")} == {"models.fintok", "models.bpe_trainer"}
    assert all(c.surface is Surface.MODELS for c in reg.by_topic("T30"))


def test_fintok_component_round_trips():
    tok = _registry().get("models.fintok").build()
    text = "<|order|> BUY 100 ALPHAINFRA @ 251.40"
    assert tok.decode(tok.encode(text)) == text
    assert tok.vocab_size > 1000


def test_trainer_component_can_train_a_small_tokenizer():
    built = _registry().get("models.bpe_trainer").build()
    tok, _stats = built["Tokenizer"].train(["hello hello world world world"], 300)
    assert tok.decode(tok.encode_ordinary("hello world")) == "hello world"
    assert "<|order|>" in built["special_tokens"]
