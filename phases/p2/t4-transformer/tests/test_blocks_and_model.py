"""Norms, the MLP, the block, and the assembled GPT."""

import math

import pytest
import torch
from t4_transformer import GPT, MLP, Block, GPTConfig, LayerNorm, RMSNorm, build_norm


def test_rmsnorm_normalises_to_unit_rms():
    torch.manual_seed(0)
    x = torch.randn(4, 7, 16) * 5
    y = RMSNorm(16)(x)
    rms = y.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)


def test_rmsnorm_does_not_recentre_and_layernorm_does():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 32) + 10.0        # a large mean, deliberately
    with torch.no_grad():
        assert float(RMSNorm(32)(x).mean().abs()) > 0.1
        assert float(LayerNorm(32)(x).mean().abs()) < 1e-5


def test_rmsnorm_is_scale_equivariant():
    """The defining property: it removes scale and nothing else."""
    torch.manual_seed(0)
    x = torch.randn(1, 5, 24)
    n = RMSNorm(24)
    assert torch.allclose(n(x), n(x * 37.0), atol=1e-4)


def test_layernorm_bias_is_optional():
    assert LayerNorm(8, bias=True).bias is not None
    assert LayerNorm(8, bias=False).bias is None
    assert sum(p.numel() for p in LayerNorm(8, bias=False).parameters()) == 8


def test_layernorm_matches_torch():
    torch.manual_seed(0)
    x = torch.randn(3, 4, 16)
    ours = LayerNorm(16)
    ref = torch.nn.LayerNorm(16)
    assert torch.allclose(ours(x), ref(x), atol=1e-6)


def test_build_norm_rejects_nonsense():
    assert isinstance(build_norm("layernorm", 4), LayerNorm)
    assert isinstance(build_norm("rmsnorm", 4), RMSNorm)
    with pytest.raises(ValueError, match="unknown norm"):
        build_norm("batchnorm", 4)


def test_mlp_widens_by_the_ratio():
    m = MLP(32, ratio=4)
    assert m.fc.weight.shape == (128, 32)
    assert m.proj.weight.shape == (32, 128)
    assert m(torch.randn(2, 5, 32)).shape == (2, 5, 32)


def test_block_is_two_residual_additions():
    torch.manual_seed(0)
    blk = Block(n_embd=32, n_head=4, block_size=8).eval()
    x = torch.randn(1, 8, 32)
    with torch.no_grad():
        out = blk(x)
        a = x + blk.attn(blk.ln1(x))
        b = a + blk.mlp(blk.ln2(a))
    assert torch.allclose(out, b, atol=0, rtol=0)


def test_block_preserves_shape_for_every_norm():
    for norm in ("layernorm", "rmsnorm"):
        blk = Block(n_embd=32, n_head=4, block_size=8, norm=norm)
        assert blk(torch.randn(2, 8, 32)).shape == (2, 8, 32)


# -- the model ------------------------------------------------------------


def test_config_rejects_indivisible_widths():
    with pytest.raises(ValueError, match="must divide evenly"):
        GPTConfig(n_embd=30, n_head=4)
    with pytest.raises(ValueError, match="block_size"):
        GPTConfig(block_size=0)


def test_weight_tying_shares_one_tensor(small_config):
    tied = GPT(small_config)
    assert tied.lm_head.weight is tied.wte.weight
    untied = GPT(small_config.scaled(tie_weights=False))
    assert untied.lm_head.weight is not untied.wte.weight
    saved = small_config.vocab_size * small_config.n_embd
    assert untied.num_params() - tied.num_params() == saved


def test_initial_loss_is_ln_vocab(small_config):
    """A model that has learned nothing must be exactly as surprised as a
    uniform guess. If step 0 is much lower, something leaks the target; much
    higher, the initialisation is broken."""
    torch.manual_seed(0)
    model = GPT(small_config.scaled(vocab_size=100))
    idx = torch.randint(0, 100, (8, 20))
    with torch.no_grad():
        _, loss = model(idx[:, :-1], idx[:, 1:])
    assert float(loss) == pytest.approx(math.log(100), abs=0.25)


def test_residual_projections_are_scaled_by_depth():
    """GPT-2's 0.02/sqrt(2N) init on every residual write, checked."""
    torch.manual_seed(0)
    deep = GPT(GPTConfig(vocab_size=16, block_size=8, n_layer=8, n_head=2, n_embd=16))
    shallow = GPT(GPTConfig(vocab_size=16, block_size=8, n_layer=2, n_head=2, n_embd=16))
    d = float(deep.blocks[0].mlp.proj.weight.detach().std())
    s = float(shallow.blocks[0].mlp.proj.weight.detach().std())
    assert d < s
    assert s / d == pytest.approx(math.sqrt(8 / 2), rel=0.15)


def test_parameter_report_adds_up(small_model):
    rep = small_model.parameter_report()
    assert rep["total"] == small_model.num_params()
    assert rep["head"] == 0                      # tied
    assert rep["mlp"] > rep["attention"]          # the MLP holds the budget


def test_forward_shapes_and_loss(small_model):
    idx = torch.randint(0, 32, (3, 20))
    logits, loss = small_model(idx)
    assert logits.shape == (3, 20, 32)
    assert loss is None
    _, loss = small_model(idx[:, :-1], idx[:, 1:])
    assert loss.ndim == 0


def test_forward_refuses_sequences_past_the_context(small_model):
    with pytest.raises(ValueError, match="exceeds block_size"):
        small_model(torch.randint(0, 32, (1, 25)))


def test_gradients_reach_layer_zero(small_config):
    """The point of the residual stream, tested rather than asserted."""
    torch.manual_seed(0)
    model = GPT(small_config.scaled(n_layer=6))
    idx = torch.randint(0, 32, (4, 16))
    _, loss = model(idx[:, :-1], idx[:, 1:])
    loss.backward()
    assert model.wte.weight.grad is not None
    assert float(model.wte.weight.grad.abs().max()) > 0
    first = float(model.blocks[0].attn.qkv.weight.grad.norm())
    last = float(model.blocks[-1].attn.qkv.weight.grad.norm())
    assert first > 0 and last > 0
    assert 0.02 < first / last < 50      # same order of magnitude, not vanished


def test_optimizer_decays_matrices_but_not_vectors(small_model):
    opt = small_model.configure_optimizers()
    decay, no_decay = opt.param_groups
    assert decay["weight_decay"] > 0
    assert no_decay["weight_decay"] == 0
    assert all(p.dim() >= 2 for p in decay["params"])
    assert all(p.dim() < 2 for p in no_decay["params"])


def test_optimizer_counts_the_tied_tensor_once(small_model):
    opt = small_model.configure_optimizers()
    n = sum(len(g["params"]) for g in opt.param_groups)
    assert n == len({id(p) for p in small_model.parameters()})


def test_attention_maps_are_per_layer_distributions(small_model):
    idx = torch.randint(0, 32, (2, 12))
    maps = small_model.attention_maps(idx)
    assert len(maps) == small_model.config.n_layer
    for m in maps:
        assert m.shape == (2, small_model.config.n_head, 12, 12)
        assert torch.allclose(m.sum(-1), torch.ones_like(m.sum(-1)), atol=1e-5)
        assert float(m.triu(1).abs().max()) == 0.0


def test_dropout_off_means_deterministic_forward(small_model):
    idx = torch.randint(0, 32, (2, 10))
    with torch.no_grad():
        assert torch.equal(small_model(idx)[0], small_model(idx)[0])
