"""Positions: the sinusoidal table as written in the paper, and RoPE's identity."""

import math

import pytest
import torch
from t4_transformer import (
    GPT,
    GPTConfig,
    LearnedPositional,
    SinusoidalPositional,
    apply_rope,
    relative_phase_property,
    rope_angles,
    sinusoidal_table,
)


def test_sinusoidal_matches_the_paper_formula():
    block, d = 16, 8
    tbl = sinusoidal_table(block, d)
    for pos in range(block):
        for i in range(d // 2):
            denom = 10_000 ** (2 * i / d)
            assert tbl[pos, 2 * i] == pytest.approx(math.sin(pos / denom), abs=1e-6)
            assert tbl[pos, 2 * i + 1] == pytest.approx(math.cos(pos / denom), abs=1e-6)


def test_sinusoidal_position_zero_alternates_zero_and_one():
    tbl = sinusoidal_table(4, 8)
    assert torch.allclose(tbl[0, 0::2], torch.zeros(4), atol=1e-7)
    assert torch.allclose(tbl[0, 1::2], torch.ones(4), atol=1e-7)


def test_sinusoidal_is_scaled_to_the_embedding_init():
    """Scale exists so the table does not drown 0.02-std token embeddings."""
    mod = SinusoidalPositional(32, 16, scale=0.02)
    assert float(mod.table.abs().max()) == pytest.approx(0.02, abs=1e-6)


def test_learned_table_refuses_positions_it_never_saw():
    mod = LearnedPositional(8, 16)
    assert mod(8).shape == (8, 16)
    with pytest.raises(ValueError, match="past the learned table"):
        mod(4, offset=6)


@pytest.mark.parametrize("head_dim", [2, 8, 32, 64])
@pytest.mark.parametrize(("m", "n"), [(7, 3), (12, 12), (1, 9), (31, 0)])
def test_rope_score_depends_only_on_the_gap(head_dim, m, n):
    r = relative_phase_property(head_dim=head_dim, m=m, n=n)
    assert r["max_abs_diff"] < 1e-12   # float64 identity, not an approximation


def test_rope_is_a_rotation_so_it_preserves_length():
    torch.manual_seed(0)
    x = torch.randn(2, 4, 10, 16)
    cos, sin = rope_angles(16, 32)
    y = apply_rope(x, cos, sin)
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-5)


def test_rope_at_position_zero_is_the_identity():
    torch.manual_seed(0)
    x = torch.randn(1, 2, 1, 8)
    cos, sin = rope_angles(8, 4)
    assert torch.allclose(apply_rope(x, cos, sin), x, atol=1e-6)


def test_rope_offset_matches_slicing_a_longer_sequence():
    """The offset path used during cached generation must agree with the
    contiguous path used during a full forward — this is the bug that makes a
    cached model quietly worse than an uncached one."""
    torch.manual_seed(0)
    x = torch.randn(1, 2, 12, 8)
    cos, sin = rope_angles(8, 32)
    full = apply_rope(x, cos, sin)
    for off in (0, 3, 7):
        part = apply_rope(x[:, :, off : off + 4], cos, sin, offset=off)
        assert torch.allclose(part, full[:, :, off : off + 4], atol=1e-6)


def test_rope_requires_even_head_dim():
    with pytest.raises(ValueError, match="even"):
        rope_angles(7, 4)
    with pytest.raises(ValueError, match="head_dim must be even"):
        GPTConfig(n_embd=12, n_head=4, position="rope", vocab_size=8)


def test_rope_angles_are_the_geometric_ladder():
    cos, sin = rope_angles(8, 3, base=10_000.0)
    expected = [10_000.0 ** (-2 * i / 8) for i in range(4)]
    got = torch.atan2(sin[1], cos[1])
    assert torch.allclose(got, torch.tensor(expected, dtype=got.dtype), atol=1e-6)


@pytest.mark.parametrize("position", ["learned", "sinusoidal", "rope", "none"])
def test_every_position_scheme_builds_and_runs(position):
    torch.manual_seed(0)
    m = GPT(GPTConfig(vocab_size=20, block_size=16, n_layer=2, n_head=2, n_embd=16,
                      position=position))
    idx = torch.randint(0, 20, (2, 12))
    logits, loss = m(idx[:, :-1], idx[:, 1:])
    assert logits.shape == (2, 11, 20)
    assert float(loss.detach()) == pytest.approx(math.log(20), abs=0.35)


def test_only_the_learned_scheme_adds_parameters():
    def n(pos):
        return GPT(GPTConfig(vocab_size=20, block_size=16, n_layer=1, n_head=2,
                             n_embd=16, position=pos)).num_params()
    assert n("learned") == n("none") + 16 * 16
    assert n("sinusoidal") == n("none")
    assert n("rope") == n("none")
