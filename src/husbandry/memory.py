"""Back-of-envelope training memory math (see the README hardware table).

Run ``husbandry memcheck`` before training anything above ~1.4B total parameters.
These are estimates: they ignore allocator fragmentation and framework overhead,
which is why the default budget leaves headroom below 32 GB.
"""

from __future__ import annotations

from dataclasses import dataclass

GB = 1024**3


@dataclass
class MemoryEstimate:
    weights: float
    grads: float
    optimizer: float
    activations: float

    @property
    def total(self) -> float:
        return self.weights + self.grads + self.optimizer + self.activations

    def describe(self) -> str:
        parts = ("weights", "grads", "optimizer", "activations")
        body = ", ".join(f"{p} {getattr(self, p) / GB:.2f}" for p in parts)
        return f"~{self.total / GB:.2f} GB ({body})"


def estimate_training_memory(
    total_params: int,
    trainable_params: int | None = None,
    *,
    weight_bytes: int = 4,
    grad_bytes: int = 4,
    optimizer_bytes: int = 8,  # AdamW: two fp32 moments
    batch_size: int = 1,
    seq_len: int = 1,
    d_model: int = 0,
    n_layers: int = 0,
    n_heads: int = 1,
    act_bytes: int = 4,
) -> MemoryEstimate:
    """Weights + grads + AdamW state + activations (Korthikanti et al. 2022 per-layer
    estimate, ``s*b*h*(34 + 5*a*s/h)`` bytes at 2-byte precision, rescaled to act_bytes).
    """
    trainable = total_params if trainable_params is None else trainable_params
    s, b, h, a = seq_len, batch_size, d_model, n_heads
    per_layer = s * b * h * (34 + 5 * a * s / max(h, 1)) * act_bytes / 2 if h else 0
    return MemoryEstimate(
        weights=total_params * weight_bytes,
        grads=trainable * grad_bytes,
        optimizer=trainable * optimizer_bytes,
        activations=per_layer * n_layers,
    )
