import torch

from husbandry.models.moe import MoEConfig, MoETransformer


def tiny(**kw) -> MoETransformer:
    cfg = dict(
        vocab_size=50,
        d_model=32,
        n_layers=2,
        n_heads=4,
        d_ff=64,
        n_experts=4,
        top_k=2,
        max_seq_len=16,
    )
    torch.manual_seed(0)
    return MoETransformer(MoEConfig(**(cfg | kw)))


def test_forward_shapes_and_stats():
    m = tiny()
    x = torch.randint(0, 50, (3, 16))
    logits, loss, stats = m(x, x)
    assert logits.shape == (3, 16, 50)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert stats["expert_load"].shape == (2, 4)
    torch.testing.assert_close(stats["expert_load"].sum(-1), torch.ones(2))


def test_causal():
    """Changing a future token must not change earlier logits."""
    m = tiny().eval()
    x = torch.randint(0, 50, (1, 16))
    y = x.clone()
    y[0, 10] = (y[0, 10] + 1) % 50
    a, _, _ = m(x)
    b, _, _ = m(y)
    torch.testing.assert_close(a[:, :10], b[:, :10])
    assert not torch.allclose(a[:, 10:], b[:, 10:])


def test_dense_equivalence_when_one_expert():
    m = tiny(n_experts=1, top_k=1)
    counts = m.param_counts()
    assert counts["active"] == counts["total"]


def test_param_counts_active_fraction():
    counts = tiny(n_experts=8, top_k=2).param_counts()
    assert counts["active"] == counts["total"] - counts["expert"] * 3 // 4


def test_overfits_single_batch():
    m = tiny()
    x = torch.randint(0, 50, (4, 16))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    first = None
    for _ in range(100):
        _, loss, stats = m(x[:, :-1], x[:, 1:])
        first = first or stats["ce_loss"].item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert stats["ce_loss"].item() < first * 0.2


def test_router_gets_gradient():
    m = tiny()
    x = torch.randint(0, 50, (2, 16))
    _, loss, _ = m(x, x)
    loss.backward()
    for b in m.blocks:
        assert b.moe.router.weight.grad.abs().sum() > 0
        assert b.moe.w_gate.grad.abs().sum() > 0
