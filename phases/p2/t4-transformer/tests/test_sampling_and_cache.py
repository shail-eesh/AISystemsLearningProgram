"""Samplers as pure functions, and the KV cache as an exactness claim."""

import pytest
import torch
from t4_transformer import (
    GPT,
    GPTConfig,
    KVCache,
    apply_temperature,
    greedy_next,
    sample_next,
    top_k_filter,
    top_p_filter,
)

LOGITS = torch.tensor([[3.0, 2.0, 1.0, 0.0, -1.0]])
NEG_INF = float("-inf")


def test_temperature_is_a_pure_rescale():
    assert torch.allclose(apply_temperature(LOGITS, 2.0), LOGITS / 2)
    assert torch.allclose(apply_temperature(LOGITS, 1.0), LOGITS)


def test_temperature_zero_is_the_argmax_limit():
    out = apply_temperature(LOGITS, 0.0)
    assert int(out.argmax()) == int(LOGITS.argmax())
    assert float(out.softmax(-1).max()) == 1.0


def test_temperature_monotonically_changes_entropy():
    ent = []
    for t in (0.25, 0.5, 1.0, 2.0, 4.0):
        p = apply_temperature(LOGITS, t).softmax(-1)
        ent.append(float(-(p * p.log()).sum()))
    assert ent == sorted(ent)


def test_negative_temperature_is_rejected():
    with pytest.raises(ValueError, match="temperature"):
        apply_temperature(LOGITS, -1.0)


@pytest.mark.parametrize("k", [1, 2, 3, 5, 99])
def test_top_k_keeps_exactly_k(k):
    out = top_k_filter(LOGITS, k)
    assert int((out > NEG_INF).sum()) == min(k, LOGITS.shape[-1])


def test_top_k_keeps_the_largest():
    out = top_k_filter(LOGITS, 2)
    assert out[0, 0] == 3.0 and out[0, 1] == 2.0
    assert out[0, 2] == NEG_INF


def test_top_k_rejects_zero():
    with pytest.raises(ValueError, match="k must be"):
        top_k_filter(LOGITS, 0)


def test_top_p_keeps_the_crossing_token():
    """probs are .636 .234 .086 .032 .012 — p=0.9 needs three tokens."""
    out = top_p_filter(LOGITS, 0.9)
    assert int((out > NEG_INF).sum()) == 3


def test_top_p_always_keeps_at_least_one():
    assert int((top_p_filter(LOGITS, 0.01) > NEG_INF).sum()) == 1


def test_top_p_one_keeps_everything():
    assert int((top_p_filter(LOGITS, 1.0) > NEG_INF).sum()) == LOGITS.shape[-1]


def test_top_p_adapts_to_confidence_and_top_k_does_not():
    """The argument for nucleus sampling, as a test."""
    confident = torch.tensor([[20.0, 1.0, 0.0, -1.0, -2.0]])
    unsure = torch.tensor([[0.2, 0.1, 0.0, -0.1, -0.2]])
    assert int((top_p_filter(confident, 0.9) > NEG_INF).sum()) == 1
    assert int((top_p_filter(unsure, 0.9) > NEG_INF).sum()) >= 4
    assert int((top_k_filter(confident, 4) > NEG_INF).sum()) == 4
    assert int((top_k_filter(unsure, 4) > NEG_INF).sum()) == 4


def test_top_p_is_batch_independent():
    rows = torch.cat([torch.tensor([[20.0, 1.0, 0.0, -1.0, -2.0]]), LOGITS])
    both = top_p_filter(rows, 0.9)
    assert int((both[0] > NEG_INF).sum()) == 1
    assert int((both[1] > NEG_INF).sum()) == 3


def test_sample_next_never_draws_a_filtered_token():
    g = torch.Generator().manual_seed(0)
    for _ in range(200):
        tok = sample_next(LOGITS, temperature=1.0, top_k=2, generator=g)
        assert int(tok) in (0, 1)


def test_greedy_is_deterministic():
    assert int(greedy_next(LOGITS)) == 0


# -- KV cache -------------------------------------------------------------


def _model(position="rope", block=32):
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=24, block_size=block, n_layer=3, n_head=4,
                         n_embd=32, position=position)).eval()


@pytest.mark.parametrize("position", ["learned", "sinusoidal", "rope", "none"])
def test_cached_generation_is_token_identical(position):
    model = _model(position)
    idx = torch.randint(0, 24, (2, 5))
    a = model.generate(idx, 20, greedy=True, use_cache=False)
    b = model.generate(idx, 20, greedy=True, use_cache=True)
    assert torch.equal(a, b)


def test_cached_logits_match_a_full_forward():
    """Stronger than the token test: the *logits* must match, not just their
    argmax. A subtly wrong cache can survive greedy decoding for a while."""
    model = _model()
    idx = torch.randint(0, 24, (1, 12))
    with torch.no_grad():
        full, _ = model(idx)
        caches = model.new_caches(1)
        model(idx[:, :6], caches=caches)
        tail, _ = model(idx[:, 6:], caches=caches)
    assert torch.allclose(full[:, 6:], tail, atol=1e-5)


def test_cache_grows_by_the_number_of_tokens_fed():
    model = _model()
    caches = model.new_caches(1)
    assert all(c.length == 0 for c in caches)
    model(torch.randint(0, 24, (1, 7)), caches=caches)
    assert all(c.length == 7 for c in caches)
    model(torch.randint(0, 24, (1, 1)), caches=caches)
    assert all(c.length == 8 for c in caches)
    caches[0].reset()
    assert caches[0].length == 0


def test_cache_refuses_to_overflow():
    cache = KVCache.empty(1, 2, 4, 8)
    cache.append(torch.zeros(1, 2, 4, 8), torch.zeros(1, 2, 4, 8))
    with pytest.raises(ValueError, match="cache holds"):
        cache.append(torch.zeros(1, 2, 1, 8), torch.zeros(1, 2, 1, 8))


def test_generate_with_cache_refuses_to_exceed_the_context():
    model = _model(block=16)
    with pytest.raises(ValueError, match="cannot slide"):
        model.generate(torch.randint(0, 24, (1, 8)), 12, greedy=True, use_cache=True)


def test_uncached_generation_slides_the_window_instead():
    model = _model(block=16)
    out = model.generate(torch.randint(0, 24, (1, 8)), 20, greedy=True, use_cache=False)
    assert out.shape == (1, 28)


def test_cache_bytes_are_linear_in_context():
    a = KVCache.empty(1, 4, 128, 32).bytes
    b = KVCache.empty(1, 4, 256, 32).bytes
    assert b == 2 * a
    assert a == 1 * 4 * 128 * 32 * 4 * 2
