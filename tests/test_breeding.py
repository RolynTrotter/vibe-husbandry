import json

import pytest
import torch

from husbandry.breeding import (
    IncompatibleMates,
    choose_experts,
    conceive,
    evaluate,
    merge,
    prune,
    report,
    total_cost,
)
from husbandry.data import DataConfig
from husbandry.tasks import eval_items, is_heldout, train_docs
from husbandry.train import load_run, train


@pytest.fixture
def family(smoke_cfg, tmp_path):
    """founder -> dam, sire (trunk frozen) -> embryo."""
    runs = tmp_path / "runs"
    train(smoke_cfg(run_name="founder", max_steps=10))
    for name in ("dam", "sire"):
        train(smoke_cfg(run_name=name, init_from=str(runs / "founder"), trainable="moe",
                        stage="specialize", max_steps=10, seed=hash(name) % 100))  # fmt: skip
    conceive(runs / "dam", runs / "sire", runs, "embryo")
    return runs


def test_conceive_concatenates_experts(family):
    dam, _, _ = load_run(family / "dam")
    sire, _, _ = load_run(family / "sire")
    child, cfg, lin = load_run(family / "embryo")
    assert child.cfg.n_experts == 8 and child.cfg.top_k == 2
    for layer in range(child.cfg.n_layers):
        cw = child.blocks[layer].moe.w_gate
        torch.testing.assert_close(cw[:4], dam.blocks[layer].moe.w_gate)
        torch.testing.assert_close(cw[4:], sire.blocks[layer].moe.w_gate)
    # frozen trunks are identical, so the averaged trunk equals each parent's
    torch.testing.assert_close(child.embed.weight, dam.embed.weight)
    assert lin.generation == 1 and lin.parents == ["dam", "sire"]
    assert lin.expert_origin[0][0] == "dam:founder.e0"
    assert lin.expert_origin[0][4] == "sire:founder.e0"
    ops = [h["op"] for h in lin.history]
    assert ops.count("pretrain") == 1  # shared founder billed once
    assert ops == ["pretrain", "specialize", "specialize", "conceive"]
    assert total_cost(lin).tokens == 3 * 10 * 8 * 64


def test_species_gate(smoke_cfg, family, tmp_path):
    train(smoke_cfg(run_name="stranger", max_steps=5, seed=99))
    with pytest.raises(IncompatibleMates, match="species"):
        conceive(family / "dam", tmp_path / "runs" / "stranger", family, "nope")


def test_child_trains_and_prunes(smoke_cfg, family):
    train(smoke_cfg(run_name="child", init_from=str(family / "embryo"), trainable="moe",
                    stage="gestate", max_steps=10))  # fmt: skip
    for method in ("reap", "allele", "random", "frequency"):
        out = prune(family / "child", family, f"adult-{method}", keep=4, method=method,
                    n_batches=2, batch_size=4, device="cpu")  # fmt: skip
        adult, _, lin = load_run(out)
        assert adult.cfg.n_experts == 4
        assert len(lin.expert_origin[0]) == 4
        assert lin.history[-1]["op"] == "prune"
    # allele keeps exactly one of each homologous pair
    _, _, lin = load_run(family / "adult-allele")
    for row in lin.expert_origin:
        assert sorted(o.split(":", 1)[1] for o in row) == [f"founder.e{k}" for k in range(4)]


def test_pruned_weights_are_the_kept_experts(family):
    out = prune(family / "embryo", family, "adult", keep=4, method="random",
                n_batches=1, batch_size=2, device="cpu", seed=3)  # fmt: skip
    child, _, _ = load_run(family / "embryo")
    adult, _, lin = load_run(out)
    kept = lin.history[-1]["kept"]
    for layer, idx in enumerate(kept):
        torch.testing.assert_close(adult.blocks[layer].moe.w_up, child.blocks[layer].moe.w_up[idx])
        torch.testing.assert_close(
            adult.blocks[layer].moe.router.weight, child.blocks[layer].moe.router.weight[idx]
        )


def test_choose_experts_allele():
    score = torch.tensor([[1.0, 5.0, 3.0, 0.0, 2.0, 1.0, 4.0, 9.0]])
    assert choose_experts("allele", 4, score, score) == [[1, 4, 6, 7]]
    assert choose_experts("reap", 2, score, score) == [[1, 7]]


def test_merge_baselines(family):
    avg = merge(family / "dam", family / "sire", family, "avg")
    ties = merge(family / "dam", family / "sire", family, "ties", base_dir=family / "founder")
    dam, _, _ = load_run(family / "dam")
    sire, _, _ = load_run(family / "sire")
    m, _, _ = load_run(avg)
    torch.testing.assert_close(m.blocks[0].moe.w_up, 0.5 * (dam.blocks[0].moe.w_up
                                                            + sire.blocks[0].moe.w_up))  # fmt: skip
    t, _, lin = load_run(ties)
    assert t.cfg.n_experts == 4 and lin.history[-1]["op"] == "merge"


def test_evaluate_and_report(family, tmp_path):
    lm = [
        DataConfig(source=f"file:{tmp_path}/docs/*.txt", cache_dir=str(tmp_path), val_fraction=0.2)
    ]
    r = evaluate(family / "embryo", lm, ["arith"], n_items=8, n_batches=2, batch_size=2,
                 device="cpu")  # fmt: skip
    assert set(r["loss"]) == {"x.txt"} and 0 <= r["acc"]["arith"] <= 1
    assert len(r["correct"]["arith"]) == 8
    saved = json.loads((family / "embryo" / "eval.json").read_text())
    assert saved["n_experts"] == 8
    assert "| embryo | 1 | 8 |" in report([family / "embryo"])


def test_task_holdout_is_disjoint():
    items = eval_items("arith", 50)
    assert len(items) == 50 and all(is_heldout(p) for p, _ in items)
    for doc in train_docs("arith", 20):
        for line in doc.splitlines():
            prompt = line[: line.index("=") + 1]
            assert not is_heldout(prompt)
    for p, a in eval_items("reverse", 5):
        assert a.split() == p.removeprefix("reverse: ").removesuffix(" ->").split()[::-1]


def test_conceive_trunk_task_arithmetic(smoke_cfg, tmp_path):
    runs = tmp_path / "runs"
    train(smoke_cfg(run_name="founder", max_steps=5))
    for name, seed in (("dam", 1), ("sire", 2)):
        train(smoke_cfg(run_name=name, init_from=str(runs / "founder"), trainable="all",
                        max_steps=5, seed=seed))  # fmt: skip
    with pytest.raises(ValueError, match="base_dir"):
        conceive(runs / "dam", runs / "sire", runs, "x", trunk_lambda=1.0)
    conceive(runs / "dam", runs / "sire", runs, "kid", trunk_lambda=1.0, base_dir=runs / "founder")
    f, d, s, k = (load_run(runs / n)[0] for n in ("founder", "dam", "sire", "kid"))
    w = lambda m: m.blocks[0].attn.qkv.weight  # noqa: E731
    torch.testing.assert_close(w(k), w(f) + (w(d) - w(f)) + (w(s) - w(f)))
