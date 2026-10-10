"""A small from-scratch MoE transformer used to debug the life cycle quickly.

Expert weights are stored *stacked* per layer (``w_gate[E, d, f]`` etc.) so that
genome surgery is plain tensor ops: conceiving a child is ``torch.cat`` along the
expert dimension, pruning is index selection. With ``n_experts=1, top_k=1`` the
model is an ordinary dense transformer (the compute-matched dense baseline).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MoEConfig:
    vocab_size: int = 4096
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 512  # hidden size of each expert
    n_experts: int = 4
    top_k: int = 2
    max_seq_len: int = 256
    rope_theta: float = 10_000.0
    norm_eps: float = 1e-6
    tie_embeddings: bool = True
    aux_loss_coef: float = 0.01  # Switch-style load-balancing loss
    z_loss_coef: float = 1e-3  # router z-loss (ST-MoE), keeps router logits small
    init_std: float = 0.02

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if not 1 <= self.top_k <= self.n_experts:
            raise ValueError("need 1 <= top_k <= n_experts")

    def to_dict(self) -> dict:
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight).to(dtype)


def rope_cache(seq_len: int, head_dim: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = torch.outer(torch.arange(seq_len).float(), inv_freq)  # [T, hd/2]
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, H, T, hd]; rotate pairs (x1, x2) = (first half, second half)
    x1, x2 = x.chunk(2, dim=-1)
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class MoELayer(nn.Module):
    """Top-k routed SwiGLU experts. Experts are the model's "chromosomes"."""

    def __init__(self, cfg: MoEConfig):
        super().__init__()
        E, d, f = cfg.n_experts, cfg.d_model, cfg.d_ff
        self.n_experts, self.top_k = E, cfg.top_k
        self.router = nn.Linear(d, E, bias=False)
        self.w_gate = nn.Parameter(torch.empty(E, d, f))
        self.w_up = nn.Parameter(torch.empty(E, d, f))
        self.w_down = nn.Parameter(torch.empty(E, f, d))
        # REAP saliency accumulator, enabled by ``collect_saliency``: rows are
        # (sum of gate * ||expert output||, number of tokens routed) per expert
        self.saliency: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        B, T, d = x.shape
        x = x.reshape(-1, d)
        # router in fp32 even under autocast: small logit errors flip top-k choices
        with torch.autocast(device_type=x.device.type, enabled=False):
            logits = F.linear(x.float(), self.router.weight.float())  # [N, E]
        probs = logits.softmax(-1)
        top_w, top_i = logits.topk(self.top_k, dim=-1)
        top_w = top_w.softmax(-1).to(x.dtype)  # renormalise over the chosen k (Mixtral)

        out = torch.zeros_like(x)
        counts = torch.zeros(self.n_experts, device=x.device)
        for e in range(self.n_experts):
            tok, slot = (top_i == e).nonzero(as_tuple=True)
            counts[e] = tok.numel()
            if tok.numel() == 0:
                continue
            xe = x[tok]
            h = F.silu(xe @ self.w_gate[e]) * (xe @ self.w_up[e])
            y = (h @ self.w_down[e]) * top_w[tok, slot, None]
            out.index_add_(0, tok, y)
            if self.saliency is not None:
                self.saliency[0, e] += y.detach().float().norm(dim=-1).sum()
                self.saliency[1, e] += tok.numel()

        n_tok = x.shape[0]
        frac_routed = counts / (n_tok * self.top_k)  # f_e, sums to 1
        aux = self.n_experts * (frac_routed * probs.mean(0)).sum()
        z = torch.logsumexp(logits, dim=-1).pow(2).mean()
        return out.view(B, T, d), {"aux": aux, "z": z, "load": frac_routed.detach()}


class Block(nn.Module):
    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.moe_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.moe = MoELayer(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        h, stats = self.moe(self.moe_norm(x))
        return x + h, stats


class MoETransformer(nn.Module):
    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        cos, sin = rope_cache(cfg.max_seq_len, cfg.d_model // cfg.n_heads, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = self.cfg.init_std
        # scale residual-output projections down with depth (GPT-2 style)
        out_std = std / math.sqrt(2 * self.cfg.n_layers)
        for name, p in self.named_parameters():
            if name.endswith("norm.weight") or name == "norm.weight":
                nn.init.ones_(p)
            elif name.endswith(("proj.weight", "w_down")):
                nn.init.normal_(p, std=out_std)
            else:
                nn.init.normal_(p, std=std)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict]:
        T = idx.shape[1]
        if T > self.cfg.max_seq_len:
            raise ValueError(f"sequence length {T} > max_seq_len {self.cfg.max_seq_len}")
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        x = self.embed(idx)
        aux = z = x.new_zeros((), dtype=torch.float32)
        loads = []
        for blk in self.blocks:
            x, s = blk(x, cos, sin)
            aux, z = aux + s["aux"], z + s["z"]
            loads.append(s["load"])
        logits = self.lm_head(self.norm(x))
        n = len(self.blocks)
        stats = {"aux_loss": aux / n, "z_loss": z / n, "expert_load": torch.stack(loads)}
        loss = None
        if targets is not None:
            ce = F.cross_entropy(logits.float().flatten(0, 1), targets.reshape(-1))
            stats["ce_loss"] = ce
            loss = ce + self.cfg.aux_loss_coef * stats["aux_loss"]
            loss = loss + self.cfg.z_loss_coef * stats["z_loss"]
        return logits, loss, stats

    def param_counts(self) -> dict[str, int]:
        """Total, active-per-token, and embedding parameter counts."""
        cfg = self.cfg
        total = sum(p.numel() for p in self.parameters())  # tied weights counted once
        expert = sum(b.moe.w_gate.numel() * 3 for b in self.blocks)
        active_expert = expert * cfg.top_k // cfg.n_experts
        embed = self.embed.weight.numel()
        return {
            "total": total,
            "active": total - expert + active_expert,
            "expert": expert,
            "embedding": embed,
            "active_non_embedding": total - expert + active_expert - embed,
        }

    @torch.no_grad()
    def generate(
        self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int = 50
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            logits, _, _ = self(idx[:, -self.cfg.max_seq_len :])
            logits = logits[:, -1, :].float() / max(temperature, 1e-5)
            if top_k:
                kth = logits.topk(min(top_k, logits.size(-1))).values[:, -1, None]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            idx = torch.cat([idx, torch.multinomial(logits.softmax(-1), 1)], dim=1)
        return idx
