from pathlib import Path

import pytest

from husbandry.config import RunConfig

SMOKE = Path(__file__).parent.parent / "configs" / "smoke.toml"


@pytest.fixture
def corpus(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    for i in range(5):
        (d / f"{i}.txt").write_text("the cat sat on the mat. " * 40 + f"story {i}")
    return d


@pytest.fixture
def smoke_cfg(tmp_path, corpus):
    def make(**train) -> RunConfig:
        cfg = RunConfig.load(SMOKE)
        cfg.data[0].source = f"file:{corpus}/*.txt"
        cfg.data[0].cache_dir = str(tmp_path / "data")
        cfg.train.out_dir = str(tmp_path / "runs")
        cfg.train.max_steps = 40
        cfg.train.device = "cpu"
        for k, v in train.items():
            setattr(cfg.train, k, v)
        return cfg

    return make
