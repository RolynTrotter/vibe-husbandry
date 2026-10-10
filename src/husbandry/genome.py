"""The abstract genome and the cost ledger every life-cycle operation reports.

See DESIGN.md. Implemented for the toy MoE: ``gestate`` (husbandry.train) and
``conceive`` / ``prune`` / ``merge`` / ``evaluate`` (husbandry.breeding). ``woo`` comes
later. Every operation appends its Cost to the Lineage, so a whole pedigree has a bill.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field


@dataclass
class Cost:
    """What an operation spent. Summable, so a whole lineage has a total bill."""

    flops: float = 0.0
    tokens: int = 0
    wall_seconds: float = 0.0
    peak_memory_bytes: int = 0
    bytes_communicated: int = 0

    def __add__(self, other: Cost) -> Cost:
        return Cost(
            flops=self.flops + other.flops,
            tokens=self.tokens + other.tokens,
            wall_seconds=self.wall_seconds + other.wall_seconds,
            peak_memory_bytes=max(self.peak_memory_bytes, other.peak_memory_bytes),
            bytes_communicated=self.bytes_communicated + other.bytes_communicated,
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Lineage:
    """Ancestry record saved with every checkpoint."""

    name: str
    species_id: str  # hash of the shared ancestor; only same-species models can mate
    generation: int = 0
    parents: list[str] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)  # one entry per operation + its cost
    # Pedigree of every expert slot, per layer: "founder.e2" for an ancestral expert,
    # "mom:founder.e2" for one the child inherited from parent "mom", and so on.
    expert_origin: list[list[str]] = field(default_factory=list)

    def record(self, op: str, cost: Cost, **info) -> None:
        self.history.append({"op": op, "cost": cost.to_dict(), **info})

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Lineage:
        return cls(**d)


def species_id_for(arch: dict, tokenizer: str, seed: int) -> str:
    """Species of a from-scratch ancestor: architecture minus expert count, tokenizer, seed.

    Expert count is excluded because children legitimately have more experts than
    their parents and are still the same species.
    """
    key = {k: v for k, v in arch.items() if k not in ("n_experts", "top_k")}
    blob = json.dumps({"arch": key, "tokenizer": tokenizer, "seed": seed}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]
