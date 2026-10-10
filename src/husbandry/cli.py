"""Command line.

Life cycle: ``train`` (pretrain / specialize / gestate), ``conceive``, ``prune``,
``evaluate``, ``report``. Baseline: ``merge``. Utilities: ``prepare``, ``sample``,
``memcheck``, ``doctor``.
"""

from __future__ import annotations

import argparse

from husbandry.memory import GB, estimate_training_memory


def _config_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("config")
    p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="override a config value (TOML syntax), e.g. --set train.max_steps=100",
    )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="husbandry")
    sub = p.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train", help="train from a TOML config (also used for gestation)")
    _config_args(tr)
    tr.add_argument("--resume", action="store_true")
    tr.add_argument("--force", action="store_true", help="ignore the memory budget check")

    pr = sub.add_parser("prepare", help="tokenize a config's corpora without training")
    _config_args(pr)

    co = sub.add_parser("conceive", help="child = mother's + father's experts")
    co.add_argument("mother")
    co.add_argument("father")
    co.add_argument("--name", required=True)
    co.add_argument("--out", default="runs")
    co.add_argument("--trunk-lambda", type=float, default=0.5)
    co.add_argument("--base", default=None, help="common ancestor run (for trunk-lambda != 0.5)")

    pn = sub.add_parser("prune", help="keep N experts per layer (adolescence -> adult)")
    pn.add_argument("run_dir")
    pn.add_argument("--name", required=True)
    pn.add_argument("--keep", type=int, required=True)
    pn.add_argument("--method", default="reap", choices=["reap", "frequency", "allele", "random"])
    pn.add_argument("--out", default="runs")
    pn.add_argument("--batches", type=int, default=16)
    pn.add_argument("--device", default="auto")
    pn.add_argument("--seed", type=int, default=0)

    me = sub.add_parser("merge", help="weight-space merge of two genomes (baseline)")
    me.add_argument("a")
    me.add_argument("b")
    me.add_argument("--name", required=True)
    me.add_argument("--method", default="average", choices=["average", "ties"])
    me.add_argument("--base", default=None, help="common ancestor run (needed for ties)")
    me.add_argument("--density", type=float, default=0.2)
    me.add_argument("--lam", type=float, default=1.0)
    me.add_argument("--out", default="runs")

    ev = sub.add_parser("evaluate", help="LM loss + task accuracy; writes eval.json")
    ev.add_argument("run_dirs", nargs="+")
    ev.add_argument("--lm-config", required=True, help="config whose corpora give the LM loss")
    ev.add_argument("--set", action="append", default=[], help="overrides for --lm-config")
    ev.add_argument("--tasks", default="arith,reverse")
    ev.add_argument("--items", type=int, default=200)
    ev.add_argument("--batches", type=int, default=20)
    ev.add_argument("--device", default="auto")

    rp = sub.add_parser("report", help="markdown table of evaluated runs")
    rp.add_argument("run_dirs", nargs="+")

    sa = sub.add_parser("sample", help="generate text from a trained run")
    sa.add_argument("run_dir")
    sa.add_argument("--prompt", default="Once upon a time")
    sa.add_argument("--tokens", type=int, default=200)
    sa.add_argument("--temperature", type=float, default=0.8)
    sa.add_argument("--top-k", type=int, default=50)
    sa.add_argument("--device", default="auto")

    mc = sub.add_parser("memcheck", help="estimate training memory for a parameter count")
    mc.add_argument("params", type=float, help="total parameters, e.g. 1.39e9")
    mc.add_argument(
        "--trainable",
        type=float,
        default=None,
        help="trainable parameters (LoRA/router-only); default = all",
    )
    mc.add_argument("--weight-bytes", type=int, default=2, help="2 = bf16 weights")

    dr = sub.add_parser("doctor", help="print hardware, package versions, matmul throughput")
    dr.add_argument("--no-bench", action="store_true")

    args = p.parse_args(argv)
    if args.cmd in ("train", "prepare"):
        from husbandry.config import RunConfig

        cfg = RunConfig.load(args.config, args.set)
        if args.cmd == "prepare":
            from husbandry.data import build_tokenizer, prepare

            tok = build_tokenizer(cfg.tokenizer, cfg.data[0])
            for d in cfg.data:
                print(f"tokenized {d.source} into {prepare(d, tok)} (vocab {tok.vocab_size})")
        else:
            from husbandry.train import train

            train(cfg, resume=args.resume, force=args.force)
    elif args.cmd == "conceive":
        from husbandry.breeding import conceive

        out = conceive(args.mother, args.father, args.out, args.name, args.trunk_lambda, args.base)
        print(f"conceived {out}")
    elif args.cmd == "prune":
        from husbandry.breeding import prune

        out = prune(args.run_dir, args.out, args.name, args.keep, args.method,
                    n_batches=args.batches, device=args.device, seed=args.seed)  # fmt: skip
        print(f"pruned -> {out}")
    elif args.cmd == "merge":
        from husbandry.breeding import merge

        out = merge(args.a, args.b, args.out, args.name, args.method, args.base,
                    args.density, args.lam)  # fmt: skip
        print(f"merged -> {out}")
    elif args.cmd == "evaluate":
        from husbandry.breeding import evaluate
        from husbandry.config import RunConfig

        corpora = RunConfig.load(args.lm_config, args.set).data
        tasks = [t for t in args.tasks.split(",") if t]
        for run in args.run_dirs:
            r = evaluate(run, corpora, tasks, args.items, args.batches, device=args.device)
            loss = " ".join(f"{k} {v:.3f}" for k, v in r["loss"].items())
            acc = " ".join(f"{k} {v:.1%}" for k, v in r["acc"].items())
            print(f"{r['name']}: loss {loss} | acc {acc}")
    elif args.cmd == "report":
        from husbandry.breeding import report

        print(report(args.run_dirs))
    elif args.cmd == "sample":
        from husbandry.data import tokenizer_from_run
        from husbandry.train import load_run, pick_device, sample_text

        device = pick_device(args.device)
        model, cfg, lineage = load_run(args.run_dir, device)
        model.eval()
        tok = tokenizer_from_run(args.run_dir)
        print(f"[{lineage.name} gen {lineage.generation} species {lineage.species_id}]")
        text = sample_text(model, tok, args.prompt, device, args.tokens,
                           temperature=args.temperature, top_k=args.top_k)  # fmt: skip
        print(text)
    elif args.cmd == "memcheck":
        n = int(args.params)
        trainable = int(args.trainable) if args.trainable is not None else None
        est = estimate_training_memory(n, trainable, weight_bytes=args.weight_bytes)
        print(f"{n / 1e9:.2f}B params: {est.describe()}  (activations not included)")
        verdict = "fits" if est.total < 24 * GB else "tight" if est.total < 30 * GB else "won't fit"
        print(f"32 GB Mac: {verdict}")
    elif args.cmd == "doctor":
        from husbandry.doctor import main as doctor

        doctor(bench=not args.no_bench)


if __name__ == "__main__":
    main()
