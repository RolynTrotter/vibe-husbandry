"""Genome operations for the toy MoE: conceive, prune, merge (baseline), evaluate.

Every function reads and writes run directories (see ``train.save_genome``) and appends
its Cost to the genome's Lineage. A child's lineage history is the union of both
parents' histories plus its own operations, so ``total_cost(lineage)`` is the whole
bill for producing that genome, with shared ancestors counted once.
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from husbandry.data import BatchSampler, DataConfig, load_tokens, prepare, tokenizer_from_run
from husbandry.genome import Cost, Lineage
from husbandry.models.moe import MoETransformer
from husbandry.tasks import eval_items
from husbandry.train import load_run, peak_memory_bytes, pick_device, save_genome

EXPERT_KEYS = ("moe.w_gate", "moe.w_up", "moe.w_down", "moe.router.weight")


class IncompatibleMates(ValueError):
    """Raised when two genomes cannot breed (different species or architecture)."""


def is_expert_param(name: str) -> bool:
    return any(k in name for k in EXPERT_KEYS)


def _state_bytes(model: torch.nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())


def _merge_histories(*lineages: Lineage) -> list[dict]:
    """Union of histories, each (op, run) once: shared ancestors are not double-billed."""
    seen, out = set(), []
    for lin in lineages:
        for h in lin.history:
            key = (h["op"], h.get("run"))
            if key not in seen:
                seen.add(key)
                out.append(h)
    return out


def total_cost(lineage: Lineage, exclude: tuple[str, ...] = ("evaluate",)) -> Cost:
    """Everything spent producing this genome (evaluations excluded by default)."""
    total = Cost()
    for h in lineage.history:
        if h["op"] not in exclude:
            total = total + Cost(**h["cost"])
    return total


def check_compatible(a: MoETransformer, b: MoETransformer, la: Lineage, lb: Lineage) -> None:
    """The hard part of the shibboleth gate: same species, same body plan."""
    if la.species_id != lb.species_id:
        raise IncompatibleMates(f"species differ: {la.species_id} vs {lb.species_id}")
    ca, cb = a.cfg.to_dict(), b.cfg.to_dict()
    diff = {k for k in ca if k not in ("n_experts",) and ca[k] != cb[k]}
    if diff:
        raise IncompatibleMates(f"architectures differ in {sorted(diff)}")


# ---------------------------------------------------------------- conceive


def conceive(
    mother_dir: str | Path,
    father_dir: str | Path,
    out_dir: str | Path,
    name: str,
    trunk_lambda: float = 0.5,
    base_dir: str | Path | None = None,
) -> Path:
    """Child = all of mother's experts, then all of father's; router rows likewise.

    The shared trunk (embeddings, attention, norms) is merged by task arithmetic against
    ``base_dir`` (their common ancestor): theta0 + lambda * (tauA + tauB). With
    lambda = 0.5 that is a plain average and no base is needed. If parents were trained
    with a frozen trunk the trunks are identical and this is a no-op.
    """
    t0 = time.time()
    mom, mom_cfg, mom_lin = load_run(mother_dir)
    dad, _, dad_lin = load_run(father_dir)
    check_compatible(mom, dad, mom_lin, dad_lin)
    if trunk_lambda != 0.5 and base_dir is None:
        raise ValueError("trunk_lambda != 0.5 needs base_dir (the common ancestor)")
    base = load_run(base_dir)[0].state_dict() if base_dir is not None else None

    cfg = replace(mom.cfg, n_experts=mom.cfg.n_experts + dad.cfg.n_experts)
    child = MoETransformer(cfg)
    sm, sd = mom.state_dict(), dad.state_dict()
    state = {}
    for k in child.state_dict():
        if is_expert_param(k):
            state[k] = torch.cat([sm[k], sd[k]], dim=0)
        elif base is None:
            state[k] = 0.5 * (sm[k] + sd[k])
        else:
            state[k] = base[k] + trunk_lambda * ((sm[k] - base[k]) + (sd[k] - base[k]))
    child.load_state_dict(state)

    lineage = Lineage(
        name=name,
        species_id=mom_lin.species_id,
        generation=max(mom_lin.generation, dad_lin.generation) + 1,
        parents=[mom_lin.name, dad_lin.name],
        history=_merge_histories(mom_lin, dad_lin),
        expert_origin=[
            [f"{mom_lin.name}:{o}" for o in om] + [f"{dad_lin.name}:{o}" for o in od]
            for om, od in zip(mom_lin.expert_origin, dad_lin.expert_origin, strict=True)
        ],
    )
    n_trunk = sum(v.numel() for k, v in state.items() if not is_expert_param(k))
    cost = Cost(
        flops=3.0 * n_trunk,
        wall_seconds=time.time() - t0,
        peak_memory_bytes=peak_memory_bytes(torch.device("cpu")),
        bytes_communicated=_state_bytes(dad),  # the father's genome travels to the mother
    )
    lineage.record(
        "conceive",
        cost,
        run=name,
        mother=mom_lin.name,
        father=dad_lin.name,
        trunk_lambda=trunk_lambda,
    )
    out = Path(out_dir) / name
    cfg_out = copy.deepcopy(mom_cfg)
    cfg_out.train = replace(cfg_out.train, run_name=name, init_from="")
    save_genome(out, child, cfg_out, lineage, tokenizer_from_run(mother_dir))
    return out


# ---------------------------------------------------------------- prune


@torch.no_grad()
def expert_saliency(
    model: MoETransformer, sampler: BatchSampler, n_batches: int, device
) -> tuple[torch.Tensor, torch.Tensor]:
    """REAP saliency [layers, experts]: mean over routed tokens of gate * ||expert output||.

    Also returns routing frequency (fraction of routed tokens) per expert.
    """
    model.eval()
    E = model.cfg.n_experts
    for b in model.blocks:
        b.moe.saliency = torch.zeros(2, E, device=device)
    for _ in range(n_batches):
        x, _ = sampler.sample(device)
        model(x)
    acc = torch.stack([b.moe.saliency for b in model.blocks]).cpu()  # [L, 2, E]
    for b in model.blocks:
        b.moe.saliency = None
    score = acc[:, 0] / acc[:, 1].clamp(min=1)
    freq = acc[:, 1] / acc[:, 1].sum(-1, keepdim=True)
    return score, freq


def choose_experts(
    method: str, keep: int, score: torch.Tensor, freq: torch.Tensor, seed: int = 0
) -> list[list[int]]:
    """Which expert slots survive in each layer (sorted, so slot order is preserved).

    - reap       highest REAP saliency
    - frequency  most often routed to
    - allele     slots k and k + E/2 are homologous (mother's and father's copy of
                 founder expert k); keep the more salient allele at every locus
    - random     seeded uniform choice (control)
    """
    L, E = score.shape
    if not 1 <= keep <= E:
        raise ValueError(f"keep must be in 1..{E}")
    rng = np.random.default_rng(seed)
    chosen = []
    for layer in range(L):
        if method == "reap":
            idx = score[layer].topk(keep).indices.tolist()
        elif method == "frequency":
            idx = freq[layer].topk(keep).indices.tolist()
        elif method == "allele":
            if E % 2 or keep != E // 2:
                raise ValueError("allele pruning keeps exactly one of each homologous pair")
            h = E // 2
            idx = [k if score[layer, k] >= score[layer, k + h] else k + h for k in range(h)]
        elif method == "random":
            idx = rng.choice(E, keep, replace=False).tolist()
        else:
            raise ValueError(f"unknown prune method {method!r}")
        chosen.append(sorted(idx))
    return chosen


def prune(
    run_dir: str | Path,
    out_dir: str | Path,
    name: str,
    keep: int,
    method: str = "reap",
    calib: list[DataConfig] | None = None,
    n_batches: int = 16,
    batch_size: int = 16,
    device: str = "auto",
    seed: int = 0,
) -> Path:
    """Adolescence -> adulthood: keep ``keep`` experts per layer, drop the rest.

    Calibration defaults to the run's own training mixture (the REAP paper found that
    calibrating on generic text alone destroys domain skills).
    """
    t0 = time.time()
    dev = pick_device(device)
    model, cfg, lineage = load_run(run_dir, dev)
    tok = tokenizer_from_run(run_dir)
    corpora = calib or cfg.data
    seq = model.cfg.max_seq_len
    samplers = [
        BatchSampler(load_tokens(prepare(d, tok), "train"), batch_size, seq, seed=seed + i)
        for i, d in enumerate(corpora)
    ]
    E, L = model.cfg.n_experts, model.cfg.n_layers
    score, freq = torch.zeros(L, E), torch.zeros(L, E)
    for s in samplers:
        sc, fr = expert_saliency(model, s, max(1, n_batches // len(samplers)), dev)
        score, freq = score + sc / len(samplers), freq + fr / len(samplers)
    chosen = choose_experts(method, keep, score, freq, seed)

    new_cfg = replace(model.cfg, n_experts=keep, top_k=min(model.cfg.top_k, keep))
    adult = MoETransformer(new_cfg)
    old = {k: v.cpu() for k, v in model.state_dict().items()}
    state = {}
    for k in adult.state_dict():
        if is_expert_param(k):
            layer = int(k.split(".")[1])
            state[k] = old[k][chosen[layer]]
        else:
            state[k] = old[k]
    adult.load_state_dict(state)

    calib_tokens = len(samplers) * max(1, n_batches // len(samplers)) * batch_size * seq
    lineage = copy.deepcopy(lineage)
    lineage.name = name
    lineage.expert_origin = [
        [lineage.expert_origin[layer][i] for i in idx] for layer, idx in enumerate(chosen)
    ]
    cost = Cost(
        flops=2.0 * model.param_counts()["active_non_embedding"] * calib_tokens,
        tokens=calib_tokens,
        wall_seconds=time.time() - t0,
        peak_memory_bytes=peak_memory_bytes(dev),
    )
    lineage.record(
        "prune", cost, run=name, method=method, keep=keep, kept=chosen, saliency=score.tolist()
    )
    cfg = copy.deepcopy(cfg)
    cfg.train = replace(cfg.train, run_name=name, init_from="")
    out = Path(out_dir) / name
    save_genome(out, adult, cfg, lineage, tok)
    return out


# ---------------------------------------------------------------- merge (baseline)


def merge(
    a_dir: str | Path,
    b_dir: str | Path,
    out_dir: str | Path,
    name: str,
    method: str = "average",
    base_dir: str | Path | None = None,
    density: float = 0.2,
    lam: float = 1.0,
) -> Path:
    """Weight-space merge of two same-shape genomes (the non-sexual baseline).

    - average  0.5 * (A + B)
    - ties     TIES-merging against ``base_dir``: trim each task vector to its top
               ``density`` fraction by magnitude, elect a sign per entry, average the
               entries that agree with it, and add ``lam`` times the result to the base.
    """
    t0 = time.time()
    a, a_cfg, la = load_run(a_dir)
    b, _, lb = load_run(b_dir)
    check_compatible(a, b, la, lb)
    if a.cfg.n_experts != b.cfg.n_experts:
        raise IncompatibleMates("merge needs the same number of experts")
    sa, sb = a.state_dict(), b.state_dict()
    if method == "average":
        state = {k: 0.5 * (sa[k] + sb[k]) for k in sa}
    elif method == "ties":
        if base_dir is None:
            raise ValueError("ties needs base_dir (the common ancestor)")
        base = load_run(base_dir)[0].state_dict()
        state = {k: base[k] + lam * _ties([sa[k] - base[k], sb[k] - base[k]], density) for k in sa}
    else:
        raise ValueError(f"unknown merge method {method!r}")
    model = MoETransformer(a.cfg)
    model.load_state_dict(state)

    lineage = Lineage(
        name=name,
        species_id=la.species_id,
        generation=max(la.generation, lb.generation) + 1,
        parents=[la.name, lb.name],
        history=_merge_histories(la, lb),
        expert_origin=[
            [f"{method}({oa}|{ob})" for oa, ob in zip(ra, rb, strict=True)]
            for ra, rb in zip(la.expert_origin, lb.expert_origin, strict=True)
        ],
    )
    cost = Cost(
        flops=3.0 * sum(v.numel() for v in state.values()),
        wall_seconds=time.time() - t0,
        bytes_communicated=_state_bytes(b),
    )
    lineage.record("merge", cost, run=name, method=method, density=density, lam=lam)
    cfg = copy.deepcopy(a_cfg)
    cfg.train = replace(cfg.train, run_name=name, init_from="")
    out = Path(out_dir) / name
    save_genome(out, model, cfg, lineage, tokenizer_from_run(a_dir))
    return out


def _ties(taus: list[torch.Tensor], density: float) -> torch.Tensor:
    trimmed = []
    for tau in taus:
        k = max(1, int(density * tau.numel()))
        thresh = tau.abs().flatten().kthvalue(tau.numel() - k + 1).values
        trimmed.append(torch.where(tau.abs() >= thresh, tau, torch.zeros_like(tau)))
    stack = torch.stack(trimmed)
    sign = torch.sign(stack.sum(0))
    agree = (torch.sign(stack) == sign) & (stack != 0)
    return (stack * agree).sum(0) / agree.sum(0).clamp(min=1)


# ---------------------------------------------------------------- evaluate


@torch.no_grad()
def greedy_complete(
    model: MoETransformer, tok, prompts: list[str], max_new: int, device, batch_size: int = 64
) -> list[str]:
    """Greedy completions, cut at the first newline or EOS. Batches equal-length prompts.

    Prompts are prefixed with EOS, as every training document is (it follows the
    previous document's EOS in the token stream).
    """
    model.eval()
    encoded = [[tok.eos_id, *tok.encode(p)] for p in prompts]
    out: list[str | None] = [None] * len(prompts)
    by_len: dict[int, list[int]] = {}
    for i, ids in enumerate(encoded):
        by_len.setdefault(len(ids), []).append(i)
    for idxs in by_len.values():
        for j in range(0, len(idxs), batch_size):
            chunk = idxs[j : j + batch_size]
            x = torch.tensor([encoded[i] for i in chunk], device=device)
            start = x.shape[1]
            for _ in range(max_new):
                logits, _, _ = model(x[:, -model.cfg.max_seq_len :])
                x = torch.cat([x, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
            for row, i in zip(x[:, start:].tolist(), chunk, strict=True):
                if tok.eos_id in row:
                    row = row[: row.index(tok.eos_id)]
                out[i] = tok.decode(row).split("\n")[0]
    return out  # type: ignore[return-value]


@torch.no_grad()
def evaluate(
    run_dir: str | Path,
    lm_corpora: list[DataConfig],
    tasks: list[str] = ("facts", "reverse"),
    n_items: int = 200,
    n_batches: int = 20,
    batch_size: int = 16,
    device: str = "auto",
) -> dict:
    """Fitness: held-out LM loss per corpus plus exact-match accuracy per task.

    Keeps per-item correctness vectors (for complementarity-based mate choice later).
    Writes ``eval.json`` into the run directory and records the cost in the lineage.
    """
    t0 = time.time()
    dev = pick_device(device)
    model, _, lineage = load_run(run_dir, dev)
    model.eval()
    tok = tokenizer_from_run(run_dir)
    seq = model.cfg.max_seq_len
    loss = {}
    tokens = 0
    for d in lm_corpora:
        s = BatchSampler(load_tokens(prepare(d, tok), "val"), batch_size, seq, seed=1234)
        vals = []
        for _ in range(n_batches):
            x, y = s.sample(dev)
            vals.append(model(x, y)[2]["ce_loss"].item())
            tokens += x.numel()
        loss[d.name] = float(np.mean(vals))
    acc, correct = {}, {}
    for task in tasks:
        items = eval_items(task, n_items)
        max_new = max(len(tok.encode(a)) for _, a in items) + 2
        preds = greedy_complete(model, tok, [p for p, _ in items], max_new, dev)
        hits = [int(p.strip() == a.strip()) for p, (_, a) in zip(preds, items, strict=True)]
        acc[task], correct[task] = float(np.mean(hits)), hits
        tokens += len(items) * max_new * seq // 2  # rough: no KV cache
    cost = Cost(
        flops=2.0 * model.param_counts()["active_non_embedding"] * tokens,
        tokens=tokens,
        wall_seconds=time.time() - t0,
        peak_memory_bytes=peak_memory_bytes(dev),
    )
    counts = model.param_counts()
    bill = total_cost(lineage)
    result = {
        "name": lineage.name,
        "generation": lineage.generation,
        "parents": lineage.parents,
        "n_experts": model.cfg.n_experts,
        "params_total": counts["total"],
        "params_active": counts["active"],
        "train_tokens": bill.tokens,
        "train_flops": bill.flops,
        "loss": loss,
        "acc": acc,
        "correct": correct,
        "eval_cost": cost.to_dict(),
    }
    run_dir = Path(run_dir)
    (run_dir / "eval.json").write_text(json.dumps(result, indent=2))
    lineage.record("evaluate", cost, run=f"{lineage.name}@eval{len(lineage.history)}")
    (run_dir / "lineage.json").write_text(json.dumps(lineage.to_dict(), indent=2))
    return result


def report(run_dirs: list[str | Path]) -> str:
    """Markdown table comparing evaluated runs."""
    rows = [json.loads((Path(r) / "eval.json").read_text()) for r in run_dirs]
    loss_keys = sorted({k for r in rows for k in r["loss"]})
    acc_keys = sorted({k for r in rows for k in r["acc"]})
    head = ["genome", "gen", "experts", "params (act)", "train tok"]
    head += [f"loss:{k}" for k in loss_keys] + [f"acc:{k}" for k in acc_keys]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        cells = [
            r["name"],
            str(r["generation"]),
            str(r["n_experts"]),
            f"{r['params_total'] / 1e6:.1f}M ({r['params_active'] / 1e6:.1f}M)",
            f"{r['train_tokens'] / 1e6:.1f}M",
        ]
        cells += [f"{r['loss'][k]:.3f}" if k in r["loss"] else "" for k in loss_keys]
        cells += [f"{r['acc'][k]:.1%}" if k in r["acc"] else "" for k in acc_keys]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
