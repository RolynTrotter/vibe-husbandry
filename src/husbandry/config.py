"""Run configuration: one TOML file.

Top-level ``tokenizer``, one ``[model]`` table, one or more data corpora (``[data]`` or
``[[data]]`` for a weighted mixture), and ``[train]``.
"""

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
    stage: str = "pretrain"  # recorded in the lineage: pretrain | specialize | gestate | ...
    # Start from another run's weights. Architecture and tokenizer then come from that run
    # (the [model] table and tokenizer are ignored) and so does the lineage.
    init_from: str = ""
    trainable: str = "all"  # all | moe (experts + routers) | router
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
    tokenizer: str = "bpe:4096"  # byte | bpe:<vocab> (trained on the first corpus) | hf:<repo>
    model: MoEConfig = field(default_factory=MoEConfig)
    data: list[DataConfig] = field(default_factory=lambda: [DataConfig()])
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict:
        return {
            "tokenizer": self.tokenizer,
            "model": asdict(self.model),
            "data": [asdict(d) for d in self.data],
            "train": asdict(self.train),
        }

    @classmethod
    def from_dict(cls, d: dict) -> RunConfig:
        unknown = set(d) - {"tokenizer", "model", "data", "train"}
        if unknown:
            raise ValueError(f"unknown config sections: {sorted(unknown)}")
        data = d.get("data", [{}])
        if isinstance(data, dict):
            data = [data]
        if not data:
            raise ValueError("need at least one [data] corpus")
        return cls(
            tokenizer=d.get("tokenizer", cls.tokenizer),
            model=_build(MoEConfig, d.get("model", {})),
            data=[_build(DataConfig, x) for x in data],
            train=_build(TrainConfig, d.get("train", {})),
        )

    @classmethod
    def load(cls, path: str | Path, overrides: list[str] | None = None) -> RunConfig:
        """Load TOML, then apply ``section.key=value`` overrides (values in TOML syntax).

        ``data.key=value`` applies to every corpus; ``data.N.key=value`` to corpus N.
        """
        d = tomllib.loads(Path(path).read_text())
        if isinstance(d.get("data"), dict):
            d["data"] = [d["data"]]
        for ov in overrides or []:
            key, _, raw = ov.partition("=")
            val = tomllib.loads(f"v = {raw}")["v"]
            parts = key.split(".")
            if parts == ["tokenizer"]:
                d["tokenizer"] = val
            elif parts[0] == "data" and len(parts) == 3:
                d.setdefault("data", [{}])[int(parts[1])][parts[2]] = val
            elif parts[0] == "data" and len(parts) == 2:
                for corpus in d.setdefault("data", [{}]):
                    corpus[parts[1]] = val
            elif len(parts) == 2:
                d.setdefault(parts[0], {})[parts[1]] = val
            else:
                raise ValueError(f"bad override {ov!r}")
        return cls.from_dict(d)


def _build(cls, values: dict):
    names = {f.name for f in fields(cls)}
    unknown = set(values) - names
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**values)
