"""Run configuration: one TOML file with [model], [data] and [train] tables."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from husbandry.data import DataConfig
from husbandry.models.moe import MoEConfig


@dataclass
class TrainConfig:
    run_name: str = "toy"
    out_dir: str = "runs"
    seed: int = 0
    device: str = "auto"  # auto | mps | cuda | cpu
    precision: str = "auto"  # auto | fp32 | bf16 (autocast); auto = bf16 on CUDA, else fp32
    batch_size: int = 16
    grad_accum: int = 1
    max_steps: int = 2000
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 100
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_every: int = 200
    eval_batches: int = 20
    log_every: int = 10
    save_every: int = 1000
    sample_every: int = 0  # 0 = never; else print a sample from sample_prompt
    sample_prompt: str = "Once upon a time"
    memory_budget_gb: float = 24.0  # refuse to start above this estimate (32 GB Mac)
    compile: bool = False


@dataclass
class RunConfig:
    model: MoEConfig = field(default_factory=MoEConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict:
        return {"model": asdict(self.model), "data": asdict(self.data), "train": asdict(self.train)}

    @classmethod
    def from_dict(cls, d: dict) -> RunConfig:
        unknown = set(d) - {"model", "data", "train"}
        if unknown:
            raise ValueError(f"unknown config sections: {sorted(unknown)}")
        return cls(
            model=_build(MoEConfig, d.get("model", {})),
            data=_build(DataConfig, d.get("data", {})),
            train=_build(TrainConfig, d.get("train", {})),
        )

    @classmethod
    def load(cls, path: str | Path, overrides: list[str] | None = None) -> RunConfig:
        d = tomllib.loads(Path(path).read_text())
        for ov in overrides or []:
            key, _, val = ov.partition("=")
            section, _, name = key.partition(".")
            d.setdefault(section, {})[name] = tomllib.loads(f"v = {val}")["v"] if val else val
        return cls.from_dict(d)


def _build(cls, values: dict):
    names = {f.name for f in fields(cls)}
    unknown = set(values) - names
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**values)
