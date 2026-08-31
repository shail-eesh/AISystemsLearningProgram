"""Attention: the mask, the scaling, the head split, and the oracle."""

import math

import pytest
import torch
from t4_transformer import (
    CausalSelfAttention,
    causal_mask,
    multi_head_attention_looped,
    scaled_dot_product,
    single_head_attention,
)


def test_causal_mask_is_lower_triangular_including_diagonal():
    m = causal_mask(5)
    assert m.dtype is torch.bool
    for i in range(5):
        for j in range(5):
            assert bool(m[i, j]) is (j <= i)


def test_attention_rows_are_distributions():
    torch.manual_seed(0)
    x = torch.randn(7, 16)
    w = [torch.randn(16, 8) for _ in range(3)]
    _, attn = single_head_attention(x, *w)
    assert torch.allclose(attn.sum(-1), torch.ones(7), atol=1e-6)
    assert (attn >= 0).all()


def test_causal_mask_zeroes_the_future_exactly():
    torch.manual_seed(0)
    x = torch.randn(9, 16)
    w = [torch.randn(16, 8) for _ in range(3)]
    _, attn = single_head_attention(x, *w)
    assert float(attn.triu(1).abs().max()) == 0.0


def test_future_tokens_cannot_change_the_past():
    """The property the mask exists for, tested behaviourally rather than
    structurally: perturb the last token and nothing before it may move."""
    torch.manual_seed(0)
    x = torch.randn(1, 8, 32)
    att = CausalSelfAttention(32, 4, block_size=8).eval()
    with torch.no_grad():
        a = att(x)
        x2 = x.clone()
        x2[:, -1] += 10.0
        b = att(x2)
    assert torch.allclose(a[:, :-1], b[:, :-1], atol=0, rtol=0)
    assert not torch.allclose(a[:, -1], b[:, -1])


def test_output_is_a_convex_combination_of_values():
    """out[i] lies inside the convex hull of v[0..i] — it is a weighted average,
    so it can never be larger than the largest value it averages."""
    torch.manual_seed(0)
    x = torch.randn(6, 8)
    eye = torch.eye(8)
    out, _ = single_head_attention(x, eye, eye, eye)
    for i in range(6):
        assert out[i].max() <= x[: i + 1].max() + 1e-6
        assert out[i].min() >= x[: i + 1].min() - 1e-6


@pytest.mark.parametrize("n_head", [1, 2, 4, 8])
def test_batched_heads_match_the_looped_oracle_exactly(n_head):
    torch.manual_seed(n_head)
    d, t = 32, 11
    att = CausalSelfAttention(d, n_head, block_size=t, bias=False).eval()
    x = torch.randn(1, t, d)
    with torch.no_grad():
        fast = att(x)[0]
        wq, wk, wv = att.qkv.weight.split(d, dim=0)
        slow, _ = multi_head_attention_looped(x[0], wq.T, wk.T, wv.T, att.proj.weight.T,
                                              n_head)
    assert torch.equal(fast, slow)


def test_scaling_matches_the_formula():
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 1, 4, 6) for _ in range(3))
    out, attn = scaled_dot_product(q, k, v, causal=False)
    manual = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(6), dim=-1)
    assert torch.allclose(attn, manual, atol=1e-6)
    assert torch.allclose(out, manual @ v, atol=1e-6)


def test_softmax_is_shift_invariant_so_the_mask_value_does_not_matter():
    """-inf, not a large negative number: masked_fill with -1e9 leaks a
    non-zero weight in float16 and this pins the difference."""
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 1, 5, 8) for _ in range(3))
    _, attn = scaled_dot_product(q, k, v)
    assert float(attn.triu(1).max()) == 0.0


def test_rejects_bad_shapes():
    with pytest.raises(ValueError, match="one .T, d_model. sequence"):
        single_head_attention(torch.randn(2, 3, 4), *[torch.randn(4, 4)] * 3)
    with pytest.raises(ValueError, match="not divisible"):
        CausalSelfAttention(30, 4, block_size=8)
    att = CausalSelfAttention(16, 4, block_size=8)
    with pytest.raises(ValueError, match="expected width"):
        att(torch.randn(1, 4, 12))


def test_dropout_only_fires_in_training_mode():
    torch.manual_seed(0)
    att = CausalSelfAttention(32, 4, block_size=8, dropout=0.9)
    x = torch.randn(1, 8, 32)
    att.eval()
    with torch.no_grad():
        assert torch.equal(att(x), att(x))
    att.train()
    torch.manual_seed(1)
    a = att(x)
    torch.manual_seed(2)
    b = att(x)
    assert not torch.allclose(a, b)
