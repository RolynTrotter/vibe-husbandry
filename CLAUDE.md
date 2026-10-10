# vibe-husbandry — notes for Claude Code

Research project: "sexual reproduction" of language models. MoE experts are chromosomes;
two same-species parents conceive a child with both parents' experts, the child gestates
and grows up under RL, then is pruned back to adult size. Courtship (wooing) gates which
pairs mate. Read README.md (life cycle, hardware budget, v0 experiment) and DESIGN.md
(genome interface, wooing protocol, scaling plan) before writing code.

Constraints:
- Development machine: Mac Studio, 32 GB unified memory. Prefer MLX (mlx, mlx-lm,
  mlx-lm-lora) for training; PyTorch MPS as fallback. Check memory math before training
  anything larger than ~1.4B total parameters (see README table).
- Species rule: every lineage descends from one ancestor (Qwen3-0.6B for v0).
- Every operation (woo/conceive/gestate/prune/evaluate) goes through the abstract genome
  interface and reports its cost, so the same code can later run at large scale.
- Always include compute-matched baselines (train-longer, direct MoE training, TIES merge).

Layout:
- Poetry project; package in src/husbandry (see README "Using this repo"). Run `poetry run pytest`
  and `poetry run ruff check . && poetry run ruff format --check .` before committing.
- configs/*.toml — run configs ([model], [data], [train]); configs/smoke.toml is the fast e2e check.
- notes/ — paper skims (notes/papers/), data plan, synthesis + open questions.
- papers.tsv — verified arXiv IDs with group and priority; scripts/fetch_papers.sh downloads them to papers/
- scripts/clone_reference_repos.sh — reference implementations into refs/ (study only, not dependencies)
- papers/, refs/, checkpoints/, data/, runs/ are gitignored.
