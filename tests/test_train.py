import json
from pathlib import Path

import pytest
import torch

from husbandry.config import RunConfig
from husbandry.data import ByteTokenizer, DataConfig, load_tokens, prepare
from husbandry.memory import GB, estimate_training_memory
from husbandry.train import load_run, lr_at, train

SMOKE = Path(__file__).parent.parent / "configs" / "smoke.toml"


@pytest.fixture
def corpus(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    for i in range(5):
        (d / f"{i}.txt").write_text("the cat sat on the mat. " * 40 + f"story {i}")
    return d


def smoke_cfg(tmp_path, corpus, **train) -> RunConfig:
    cfg = RunConfig.load(SMOKE)
    cfg.data.source = f"file:{corpus}/*.txt"
    cfg.data.cache_dir = str(tmp_path / "data")
    cfg.train.out_dir = str(tmp_path / "runs")
    cfg.train.max_steps = 40
    for k, v in train.items():
        setattr(cfg.train, k, v)
    return cfg


def test_byte_tokenizer_roundtrip():
    tok = ByteTokenizer()
    assert tok.decode(tok.encode("héllo")) == "héllo"


def test_prepare_caches_and_splits(tmp_path, corpus):
    cfg = DataConfig(
        source=f"file:{corpus}/*.txt", tokenizer="byte", cache_dir=str(tmp_path), val_fraction=0.2
    )
    out, tok = prepare(cfg)
    meta = json.loads((out / "meta.json").read_text())
    assert meta["n_docs"] == 5
    assert len(load_tokens(out, "train")) == meta["n_tokens"]["train"]
    assert len(load_tokens(out, "val")) == meta["n_tokens"]["val"] > 0
    assert prepare(cfg)[0] == out  # cached


def test_bpe_tokenizer_trains(tmp_path, corpus):
    cfg = DataConfig(source=f"file:{corpus}/*.txt", tokenizer="bpe:300", cache_dir=str(tmp_path))
    _, tok = prepare(cfg)
    assert tok.vocab_size <= 300
    assert tok.decode(tok.encode("the cat sat")) == "the cat sat"


def test_config_rejects_unknown_keys(tmp_path):
    p = tmp_path / "bad.toml"
    p.write_text("[train]\nmax_stepz = 3\n")
    with pytest.raises(ValueError, match="max_stepz"):
        RunConfig.load(p)


def test_config_overrides():
    cfg = RunConfig.load(SMOKE, ["train.max_steps=7", 'data.source="file:x"'])
    assert cfg.train.max_steps == 7 and cfg.data.source == "file:x"


def test_lr_schedule():
    cfg = RunConfig().train
    assert lr_at(0, cfg) < cfg.lr
    assert lr_at(cfg.warmup_steps, cfg) == pytest.approx(cfg.lr)
    assert lr_at(cfg.max_steps, cfg) == pytest.approx(cfg.min_lr)


def test_train_end_to_end_and_reload(tmp_path, corpus):
    report = train(smoke_cfg(tmp_path, corpus))
    assert report.steps == 40
    assert report.final_val_loss is not None and report.final_val_loss < 5.0  # ln(257)=5.55
    assert report.cost.tokens == 40 * 8 * 64 and report.cost.flops > 0

    model, cfg, lineage = load_run(report.run_dir)
    assert lineage.history[-1]["op"] == "gestate"
    assert lineage.history[-1]["cost"]["tokens"] == report.cost.tokens
    out = model.generate(torch.tensor([[1, 2, 3]]), 5)
    assert out.shape == (1, 8)
    lines = (report.run_dir / "metrics.jsonl").read_text().splitlines()
    assert json.loads(lines[-1])["step"] == 40


def test_resume_continues(tmp_path, corpus):
    train(smoke_cfg(tmp_path, corpus, max_steps=20, save_every=10))
    report = train(smoke_cfg(tmp_path, corpus, max_steps=30), resume=True)
    assert report.steps == 30
    assert report.cost.tokens == 10 * 8 * 64
    _, _, lineage = load_run(report.run_dir)
    assert [h["steps"] for h in lineage.history] == [20, 30]


def test_memory_budget_refuses(tmp_path, corpus):
    with pytest.raises(RuntimeError, match="budget"):
        train(smoke_cfg(tmp_path, corpus, memory_budget_gb=1e-6))


def test_memory_estimate_matches_readme():
    # README: 2.45B-param child needs ~39 GB for a full AdamW fine-tune (bf16 weights)
    est = estimate_training_memory(int(2.45e9), weight_bytes=2)
    assert 30 * GB < est.total < 45 * GB
