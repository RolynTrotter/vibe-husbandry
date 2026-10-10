"""Pre-training loop for the toy MoE: the first, simplest form of ``gestate``.

Writes ``runs/<run_name>/`` with ``config.json``, ``metrics.jsonl``, ``lineage.json``
and ``ckpt/`` (safetensors weights + optimizer state for resuming).
"""

from __future__ import annotations

import contextlib
import json
import math
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from husbandry.config import RunConfig
from husbandry.data import BatchSampler, load_tokens, prepare
from husbandry.genome import Cost, Lineage, species_id_for
from husbandry.memory import GB, estimate_training_memory
from husbandry.models.moe import MoETransformer


@dataclass
class TrainReport:
    run_dir: Path
    steps: int
    final_train_loss: float
    final_val_loss: float | None
    cost: Cost


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_ctx(device: torch.device, precision: str):
    if precision == "auto":
        precision = "bf16" if device.type == "cuda" else "fp32"
    if precision == "fp32":
        return contextlib.nullcontext()
    if precision == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    raise ValueError(f"unknown precision {precision!r}")


def lr_at(step: int, cfg) -> float:
    """Linear warmup then cosine decay to min_lr."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * min(progress, 1.0)))


def peak_memory_bytes(device: torch.device) -> int:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device)
    if device.type == "mps":
        return torch.mps.driver_allocated_memory()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if sys.platform == "darwin" else rss * 1024  # macOS reports bytes, Linux KiB


def train_flops(model: MoETransformer, tokens: int) -> float:
    """6 * active non-embedding params per token, plus the attention-score term."""
    c = model.cfg
    n = model.param_counts()["active_non_embedding"]
    return float(tokens) * (6 * n + 6 * c.n_layers * c.max_seq_len * c.d_model)


def make_optimizer(model: torch.nn.Module, cfg) -> torch.optim.Optimizer:
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))


@torch.no_grad()
def evaluate(model, sampler: BatchSampler, n_batches: int, device, amp) -> float:
    model.eval()
    losses = []
    for _ in range(n_batches):
        x, y = sampler.sample(device)
        with amp:
            _, _, stats = model(x, y)
        losses.append(stats["ce_loss"].item())
    model.train()
    return sum(losses) / len(losses)


def load_run(run_dir: str | Path, device: str | torch.device = "cpu"):
    """Load (model, RunConfig, Lineage) from a run directory."""
    run_dir = Path(run_dir)
    cfg = RunConfig.from_dict(json.loads((run_dir / "config.json").read_text()))
    model = MoETransformer(cfg.model)
    load_weights(model, run_dir / "ckpt" / "model.safetensors")
    lineage = Lineage.from_dict(json.loads((run_dir / "lineage.json").read_text()))
    return model.to(device), cfg, lineage


def load_weights(model: MoETransformer, path: Path) -> None:
    missing, unexpected = model.load_state_dict(load_file(path), strict=False)
    allowed = {"lm_head.weight"} if model.cfg.tie_embeddings else set()
    if unexpected or set(missing) - allowed:
        raise RuntimeError(f"checkpoint mismatch: missing={missing} unexpected={unexpected}")


def _save(run_dir: Path, model, opt, step: int, lineage: Lineage) -> None:
    ckpt = run_dir / "ckpt"
    ckpt.mkdir(parents=True, exist_ok=True)
    # tied lm_head shares storage with the embedding; safetensors refuses shared tensors
    state = {k: v.contiguous() for k, v in model.state_dict().items() if k != "lm_head.weight"}
    save_file(state, ckpt / "model.safetensors", metadata={"step": str(step)})
    torch.save({"optimizer": opt.state_dict(), "step": step}, ckpt / "trainer.pt")
    (run_dir / "lineage.json").write_text(json.dumps(lineage.to_dict(), indent=2))


def train(cfg: RunConfig, resume: bool = False, force: bool = False) -> TrainReport:
    t = cfg.train
    torch.manual_seed(t.seed)
    device = pick_device(t.device)
    amp = autocast_ctx(device, t.precision)

    data_dir, tok = prepare(cfg.data)
    if tok.vocab_size > cfg.model.vocab_size:
        raise ValueError(f"tokenizer vocab {tok.vocab_size} > model vocab {cfg.model.vocab_size}")
    seq = cfg.model.max_seq_len
    train_s = BatchSampler(load_tokens(data_dir, "train"), t.batch_size, seq, seed=t.seed)
    val_s = BatchSampler(load_tokens(data_dir, "val"), t.batch_size, seq, seed=t.seed + 1)

    model = MoETransformer(cfg.model).to(device)
    counts = model.param_counts()
    mem = estimate_training_memory(
        counts["total"],
        batch_size=t.batch_size,
        seq_len=seq,
        d_model=cfg.model.d_model,
        n_layers=cfg.model.n_layers,
        n_heads=cfg.model.n_heads,
    )
    print(
        f"device={device} params total={counts['total'] / 1e6:.1f}M "
        f"active={counts['active'] / 1e6:.1f}M  memory {mem.describe()}"
    )
    if mem.total > t.memory_budget_gb * GB and not force:
        raise RuntimeError(
            f"estimated {mem.total / GB:.1f} GB > budget {t.memory_budget_gb} GB; "
            "shrink the run or pass --force"
        )

    run_dir = Path(t.out_dir) / t.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))

    opt = make_optimizer(model, t)
    step = 0
    if resume and (run_dir / "ckpt" / "trainer.pt").exists():
        load_weights(model, run_dir / "ckpt" / "model.safetensors")
        state = torch.load(run_dir / "ckpt" / "trainer.pt", map_location=device)
        opt.load_state_dict(state["optimizer"])
        step = state["step"]
        train_s.rng = np.random.default_rng((t.seed, step))  # don't replay the same batches
        lineage = Lineage.from_dict(json.loads((run_dir / "lineage.json").read_text()))
        print(f"resumed from step {step}")
    else:
        species = species_id_for(cfg.model.to_dict(), cfg.data.tokenizer, t.seed)
        lineage = Lineage(name=t.run_name, species_id=species)
    if t.compile:
        model = torch.compile(model)

    tokens_per_step = t.batch_size * t.grad_accum * seq
    start_step, t0 = step, time.time()
    log = (run_dir / "metrics.jsonl").open("a" if step else "w")
    train_loss = float("nan")
    val_loss = None
    while step < t.max_steps:
        lr = lr_at(step, t)
        for g in opt.param_groups:
            g["lr"] = lr
        loss_acc = aux_acc = 0.0
        load = None
        for _ in range(t.grad_accum):
            x, y = train_s.sample(device)
            with amp:
                _, loss, stats = model(x, y)
            (loss / t.grad_accum).backward()
            loss_acc += stats["ce_loss"].item() / t.grad_accum
            aux_acc += stats["aux_loss"].item() / t.grad_accum
            load = stats["expert_load"]
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), t.grad_clip).item()
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        train_loss = loss_acc

        rec = {"step": step, "lr": lr, "loss": train_loss, "aux": aux_acc, "grad_norm": grad_norm}
        if step % t.eval_every == 0 or step == t.max_steps:
            val_loss = evaluate(model, val_s, t.eval_batches, device, amp)
            rec["val_loss"] = val_loss
        if step % t.log_every == 0 or "val_loss" in rec:
            elapsed = time.time() - t0
            tps = (step - start_step) * tokens_per_step / max(elapsed, 1e-9)
            # max-load / mean-load per layer: 1.0 = perfectly balanced experts
            imbalance = (load.max(-1).values * load.shape[-1]).tolist()
            rec |= {"tok_per_s": tps, "expert_imbalance": imbalance}
            print(
                f"step {step:>6} loss {train_loss:.4f}"
                + (f" val {val_loss:.4f}" if "val_loss" in rec else "")
                + f" lr {lr:.2e} gnorm {grad_norm:.2f} aux {aux_acc:.3f}"
                f" imb {max(imbalance):.2f} {tps:,.0f} tok/s",
                flush=True,
            )
            log.write(json.dumps(rec) + "\n")
            log.flush()
        if t.sample_every and step % t.sample_every == 0:
            print("  sample:", repr(sample_text(model, tok, t.sample_prompt, device, 64)))
        if step % t.save_every == 0 and step < t.max_steps:
            _save(run_dir, getattr(model, "_orig_mod", model), opt, step, lineage)
    log.close()

    tokens = (step - start_step) * tokens_per_step
    raw = getattr(model, "_orig_mod", model)
    cost = Cost(
        flops=train_flops(raw, tokens),
        tokens=tokens,
        wall_seconds=time.time() - t0,
        peak_memory_bytes=peak_memory_bytes(device),
    )
    lineage.record("gestate", cost, kind="pretrain", steps=step, data=cfg.data.source)
    _save(run_dir, raw, opt, step, lineage)
    print(
        f"done: {step} steps, {tokens / 1e6:.1f}M tokens, {cost.flops:.2e} FLOPs, "
        f"{cost.wall_seconds:.0f}s, peak mem {cost.peak_memory_bytes / GB:.2f} GB -> {run_dir}"
    )
    return TrainReport(run_dir, step, train_loss, val_loss, cost)


def sample_text(model, tok, prompt: str, device, max_new_tokens: int = 128, **kw) -> str:
    ids = torch.tensor([tok.encode(prompt)], device=device)
    out = getattr(model, "_orig_mod", model).generate(ids, max_new_tokens, **kw)
    return tok.decode(out[0].tolist())
