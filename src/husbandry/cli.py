"""Command line: ``husbandry {train,sample,prepare,memcheck}``."""

from __future__ import annotations

import argparse

from husbandry.memory import GB, estimate_training_memory


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="husbandry")
    sub = p.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train", help="pre-train a toy MoE from a TOML config")
    tr.add_argument("config")
    tr.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="override a config value, e.g. --set train.max_steps=100",
    )
    tr.add_argument("--resume", action="store_true")
    tr.add_argument("--force", action="store_true", help="ignore the memory budget check")

    pr = sub.add_parser("prepare", help="tokenize a config's corpus without training")
    pr.add_argument("config")
    pr.add_argument("--set", action="append", default=[])

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

    args = p.parse_args(argv)
    if args.cmd in ("train", "prepare"):
        from husbandry.config import RunConfig

        cfg = RunConfig.load(args.config, args.set)
        if args.cmd == "prepare":
            from husbandry.data import prepare

            out, tok = prepare(cfg.data)
            print(f"tokenized into {out} (vocab {tok.vocab_size})")
        else:
            from husbandry.train import train

            train(cfg, resume=args.resume, force=args.force)
    elif args.cmd == "sample":
        from husbandry.data import load_tokenizer
        from husbandry.train import load_run, pick_device, sample_text

        device = pick_device(args.device)
        model, cfg, lineage = load_run(args.run_dir, device)
        model.eval()
        tok = load_tokenizer(cfg.data)
        print(f"[{lineage.name} gen {lineage.generation} species {lineage.species_id}]")
        print(
            sample_text(
                model,
                tok,
                args.prompt,
                device,
                args.tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )
        )
    elif args.cmd == "memcheck":
        n = int(args.params)
        trainable = int(args.trainable) if args.trainable is not None else None
        est = estimate_training_memory(n, trainable, weight_bytes=args.weight_bytes)
        print(f"{n / 1e9:.2f}B params: {est.describe()}  (activations not included)")
        verdict = "fits" if est.total < 24 * GB else "tight" if est.total < 30 * GB else "won't fit"
        print(f"32 GB Mac: {verdict}")


if __name__ == "__main__":
    main()
