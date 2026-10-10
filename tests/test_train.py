import json

import pytest
import torch
from conftest import SMOKE

from husbandry.config import RunConfig
from husbandry.data import (
    BatchSampler,
    ByteTokenizer,
    DataConfig,
    MixtureSampler,
    build_tokenizer,
    load_tokens,
    prepare,
    tokenizer_from_run,
)
from husbandry.memory import GB, estimate_training_memory
from husbandry.train import load_run, lr_at, train


def test_byte_tokenizer_roundtrip():
    tok = ByteTokenizer()
    assert tok.decode(tok.encode("héllo")) == "héllo"


def test_prepare_caches_and_splits(tmp_path, corpus):
    cfg = DataConfig(source=f"file:{corpus}/*.txt", cache_dir=str(tmp_path), val_fraction=0.2)
    out = prepare(cfg, ByteTokenizer())
    meta = json.loads((out / "meta.json").read_text())
    assert meta["n_docs"] == 5
    assert len(load_tokens(out, "train")) == meta["n_tokens"]["train"]
    assert len(load_tokens(out, "val")) == meta["n_tokens"]["val"] > 0
    assert prepare(cfg, ByteTokenizer()) == out  # cached


def test_bpe_tokenizer_trains(tmp_path, corpus):
    cfg = DataConfig(source=f"file:{corpus}/*.txt", cache_dir=str(tmp_path))
    tok = build_tokenizer("bpe:300", cfg)
    assert tok.vocab_size <= 300
    assert tok.decode(tok.encode("the cat sat")) == "the cat sat"
    tok.save(tmp_path)
    again = tokenizer_from_run(tmp_path)
    assert again.fingerprint == tok.fingerprint and again.eos_id == tok.eos_id


def test_synthetic_source(tmp_path):
    cfg = DataConfig(source="synth:arith", max_docs=20, cache_dir=str(tmp_path))
    out = prepare(cfg, ByteTokenizer())
    text = ByteTokenizer().decode(load_tokens(out, "train")[:200].tolist())
    assert "=" in text and "+" in text or "-" in text


def test_mixture_sampler_respects_weights():
    import numpy as np

    a = BatchSampler(np.zeros(1000, dtype=np.uint16), 64, 8)
    b = BatchSampler(np.ones(1000, dtype=np.uint16), 64, 8)
    x, _ = MixtureSampler([a, b], [3, 1], seed=0).sample(torch.device("cpu"))
    frac_b = x[:, 0].float().mean().item()
    assert 0.1 < frac_b < 0.45


def test_config_rejects_unknown_keys(tmp_path):
    p = tmp_path / "bad.toml"
    p.write_text("[train]\nmax_stepz = 3\n")
    with pytest.raises(ValueError, match="max_stepz"):
        RunConfig.load(p)


def test_config_overrides():
    cfg = RunConfig.load(SMOKE, ["train.max_steps=7", 'data.source="file:x"', 'tokenizer="bpe:9"'])
    assert cfg.train.max_steps == 7 and cfg.data[0].source == "file:x"
    assert cfg.tokenizer == "bpe:9"


def test_config_data_array(tmp_path):
    p = tmp_path / "mix.toml"
    p.write_text('[[data]]\nsource = "synth:arith"\nweight = 2\n[[data]]\nsource = "file:x"\n')
    cfg = RunConfig.load(p, ["data.1.weight=0.5"])
    assert [d.weight for d in cfg.data] == [2, 0.5]


def test_lr_schedule():
    cfg = RunConfig().train
    assert lr_at(0, cfg) < cfg.lr
    assert lr_at(cfg.warmup_steps, cfg) == pytest.approx(cfg.lr)
    assert lr_at(cfg.max_steps, cfg) == pytest.approx(cfg.min_lr)


def test_train_end_to_end_and_reload(smoke_cfg):
    report = train(smoke_cfg())
    assert report.steps == 40
    assert report.final_val_loss is not None and report.final_val_loss < 5.0  # ln(257)=5.55
    assert report.cost.tokens == 40 * 8 * 64 and report.cost.flops > 0

    model, cfg, lineage = load_run(report.run_dir)
    assert lineage.history[-1]["op"] == "pretrain"
    assert lineage.history[-1]["cost"]["tokens"] == report.cost.tokens
    assert lineage.expert_origin[0] == ["smoke.e0", "smoke.e1", "smoke.e2", "smoke.e3"]
    out = model.generate(torch.tensor([[1, 2, 3]]), 5)
    assert out.shape == (1, 8)
    lines = (report.run_dir / "metrics.jsonl").read_text().splitlines()
    assert json.loads(lines[-1])["step"] == 40


def test_init_from_inherits_and_freezes(smoke_cfg):
    parent = train(smoke_cfg(max_steps=10))
    p_model, _, p_lin = load_run(parent.run_dir)
    child = train(smoke_cfg(run_name="kid", init_from=str(parent.run_dir), trainable="moe",
                            stage="specialize", max_steps=10))  # fmt: skip
    c_model, _, c_lin = load_run(child.run_dir)
    assert c_lin.species_id == p_lin.species_id
    assert [h["op"] for h in c_lin.history] == ["pretrain", "specialize"]
    ps, cs = p_model.state_dict(), c_model.state_dict()
    torch.testing.assert_close(ps["blocks.0.attn.qkv.weight"], cs["blocks.0.attn.qkv.weight"])
    assert not torch.equal(ps["blocks.0.moe.w_up"], cs["blocks.0.moe.w_up"])


def test_resume_continues(smoke_cfg):
    train(smoke_cfg(max_steps=20, save_every=10))
    report = train(smoke_cfg(max_steps=30), resume=True)
    assert report.steps == 30
    assert report.cost.tokens == 10 * 8 * 64
    _, _, lineage = load_run(report.run_dir)
    assert [h["steps"] for h in lineage.history] == [20, 30]


def test_memory_budget_refuses(smoke_cfg):
    with pytest.raises(RuntimeError, match="budget"):
        train(smoke_cfg(memory_budget_gb=1e-6))


def test_memory_estimate_matches_readme():
    # README: 2.45B-param child needs ~39 GB for a full AdamW fine-tune (bf16 weights)
    est = estimate_training_memory(int(2.45e9), weight_bytes=2)
    assert 30 * GB < est.total < 45 * GB
