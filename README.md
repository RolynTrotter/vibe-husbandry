# vibe-husbandry

Animal husbandry, but AIs. Sexual reproduction for language models: two parent MoE models contribute experts
("chromosomes") to a child that gestates in a larger space, grows up under RL, and is
pruned back down to adult size before it can itself reproduce.

See [DESIGN.md](DESIGN.md) for the genome/operations interface, the courtship protocol
(wooing), and the plan for scaling the loop up to large generalist models.

## The life cycle

| Biology | This project | Closest prior work |
|---|---|---|
| Chromosomes | MoE expert FFNs (plus a router) | BTX, MergeME, mergekit-moe |
| Meiosis / homologous pairing | Matching experts and neurons across parents | Git Re-Basin, ZipIt!, Safe Crossover, Transport-and-Merge |
| Gestation in the mother | Child = union of both parents' experts (2× experts) trained inside the mother's compute budget | Sparse upcycling, Drop-Upcycling |
| Growing up | RL (GRPO) on real tasks; router learns which inherited experts to use | DeepSeekMath / R1, mlx-lm-lora |
| Adolescence | Intermediate size, functional but too big to mate | MatFormer (nested sizes), progressive pruning |
| Synaptic pruning / critical period | Prune experts back to adult count | REAP, REAM, NAEE, Minitron |
| Courtship (wooing) | Suitor passes the chooser's filter: displays, gifts, new modalities | PIMP mating preferences, LLM-as-judge, M2N2 "attraction" |
| Kin recognition | Shibboleths: inherited fingerprints + compatibility check | Instructional Fingerprinting, model provenance testing |
| Nuptial gifts | Suitor contributes compute, data, RL environments to gestation | — |
| Species | Only models with a recent common ancestor can interbreed | NEAT historical markings; Git Re-Basin |
| Population & selection | Many lineages, fitness = eval score | GENOME, Evolutionary Model Merge, PBT |

**Species rule.** Start every lineage from one shared ancestor so tokenizers, hidden size
and residual-stream bases match. Cross-family breeding (Qwen × Granite) is a later,
separate experiment that needs the cross-architecture papers in group 05.

## Hardware budget: Mac Studio, 32 GB unified memory

Rough parameter counts (computed from the published configs):

| Setup | Total params | bf16 weights | Full fine-tune w/ AdamW (~16 B/param) |
|---|---|---|---|
| Qwen3-0.6B (dense ancestor) | 0.60B | 1.2 GB | ~10 GB ✅ |
| Qwen3-0.6B upcycled to 4 experts (adult) | 1.39B | 2.8 GB | ~22 GB ⚠️ tight |
| Child: 8 experts (4 + 4 from parents) | 2.45B | 4.9 GB | ~39 GB ❌ |
| Granite-3.1-1B-A400M (native MoE, 32 experts, 8 active) | 1.33B | 2.7 GB | ~21 GB ⚠️ |
| Granite child with 64 experts | 2.54B | 5.1 GB | ~41 GB ❌ |

So the child can't be fully fine-tuned in 32 GB. Options, in order of preference:
1. Freeze attention + embeddings, train router + LoRA on each expert (PEFT or mlx-lm-lora).
2. Keep only the top half of layers as MoE in the child.
3. Use a toy from-scratch MoE (nanoGPT-scale, ~50–150M params) to debug the whole loop fast,
   then move to Qwen3-0.6B.

Model notes:
- **Qwen3-0.6B** is a plain transformer, easiest to hack. Qwen3.5 small models (0.8B/2B)
  are newer but use a Gated DeltaNet hybrid, which complicates surgery; avoid for v1.
- **Granite-3.1-1B-A400M** is the smallest real, open MoE (Apache-2.0), already modular,
  but a different species from Qwen.
- Qwen's own MoEs (Qwen3/3.5/3.6 35B-A3B and up) don't fit for training here.

## First experiment (v0)

1. Ancestor: Qwen3-0.6B → sparse-upcycle to a 4-expert MoE (Drop-Upcycling style).
2. Branch two parents (BTX style): fine-tune parent A on math, parent B on code.
3. Conceive: child gets all 8 experts; attention = average of parents (or task-arithmetic);
   router = concatenated parents' routers + noise. Optionally align experts first (Hungarian
   matching on activations) and compare aligned vs unaligned.
4. Grow up: GRPO on mixed math+code rewards, training router + expert LoRAs only.
5. Adolescence → adult: prune 8 → 4 experts with REAP-style router-weighted saliency,
   short distillation from the adolescent to the adult.
6. Compare child vs each parent vs naive weight-average vs TIES merge on GSM8K + HumanEval/MBPP.
7. Iterate: breed the child with a third lineage; track whether gains compound over generations.

## Using this repo

```bash
poetry install                       # core: torch, datasets, tokenizers, transformers
poetry install --with mlx            # + MLX (macOS only)
poetry install --with research       # + merging/RL/eval libs (peft, trl, POT, lm-eval, ...)

poetry run pytest                    # ~10 s, no downloads
poetry run husbandry train configs/smoke.toml           # seconds; trains on the repo's own *.md
poetry run husbandry train configs/toy-tinystories.toml # toy 8-expert MoE ancestor on TinyStories
poetry run husbandry train configs/toy-dense-baseline.toml  # compute-matched dense baseline
poetry run husbandry sample runs/toy-tinystories --prompt "Once upon a time"
poetry run husbandry memcheck 2.45e9 --trainable 5e7    # memory math before a big run

scripts/fetch_papers.sh 1      # must-reads only (33 PDFs); omit the arg for all 92
scripts/clone_reference_repos.sh
```

Config values can be overridden on the command line: `--set train.max_steps=100`.
Each run writes `runs/<name>/{config.json, metrics.jsonl, lineage.json, ckpt/}`;
`lineage.json` records the species id and the cost (FLOPs, tokens, wall time, peak memory)
of every operation applied to the genome.

Code layout (`src/husbandry/`): `models/moe.py` (toy MoE, stacked expert weights so
conception/pruning are tensor ops), `data.py` (tokenizers + memmapped token files),
`train.py` (pre-training loop = first form of `gestate`), `genome.py` (Cost, Lineage),
`memory.py` (training-memory estimates), `cli.py`.

Research notes: `notes/papers/*.md` (per-group paper skims), `notes/data-plan.md`,
`notes/synthesis.md` (what the papers imply for the plan, and open questions).

## Reading list (priority 1 highlighted)

**01 MoE foundations.** Mixtral (`2401.04088`) and OLMoE (`2409.02060`) are the must-reads;
both analyze what experts actually specialize in (mostly tokens/syntax, not tidy domains).
Background: Shazeer 2017, GShard, Switch, DeepSeekMoE, Qwen2/Qwen3 reports.

**02 Building an MoE from parts.** Sparse Upcycling (`2212.05055`), Branch-Train-MiX
(`2403.07816`), Drop-Upcycling (`2502.19261`), MergeME (`2502.00997`, projector layers for
heterogeneous experts). Also BTM, NVIDIA upcycling, Self-MoE, MergeKit.

**03 Weight-space merging.** Task Arithmetic (`2212.04089`), TIES (`2306.01708`), and the
merging survey (`2408.07666`). Also Model Soups, DARE, Fisher, RegMean, Breadcrumbs.

**04 Alignment ("meiosis").** Git Re-Basin (`2209.04836`), ZipIt! (`2305.03053`),
Safe Crossover (`2003.10306`, names the competing-conventions problem for crossover),
Transport and Merge (`2602.05495`, OT on activations across architectures). Also linear mode
connectivity, OT Fusion, model stitching, Platonic Representation Hypothesis.

**05 Cross-architecture fusion** (for the later Qwen × other-family experiment). FuseLLM,
FuseChat, Unconstrained Model Merging, MergeNet.

**06 Evolution and populations.** Evolutionary Model Merge (`2403.13187`), M2N2
(`2508.16204`, competition for resources + mate attraction), GENOME (`2503.01155`,
crossover/mutation/selection/succession over LLM populations). Also PBT, CycleQD,
Mergenetic, and NEAT (2002, not on arXiv; fetched separately).

**07 Growth ("gestation").** HyperCloning (`2409.12903`, function-preserving width growth)
and MatFormer (`2310.07707`, nested models usable at several sizes — the adolescent).
Also Net2Net, bert2BERT, LiGO, Stacking Your Transformers.

**08 Pruning ("adolescence → adulthood").** Lottery Ticket (`1803.03635`), Minitron
(`2407.14679`, prune + distill), REAP (`2510.13999`, expert pruning beats expert merging for
generative tasks). Also SparseGPT, Wanda, LLM-Pruner, Sheared LLaMA, NAEE, MC-SMoE, REAM, Sub-MoE.

**09 Distillation.** Hinton et al., MiniLLM.

**10 Critical periods.** Achille et al. (`1711.08856`): deep nets have critical learning
periods like animals do; plus the deep-linear-network follow-up. This is the scientific
hook for the adolescence phase.

**11 RL and fine-tuning.** DeepSeekMath (`2402.03300`, introduces GRPO). Also InstructGPT,
DeepSeek-R1, LoRA, QLoRA.

**12 Wooing and mate choice.** "All You Need Is Sex for Diversity" (`2303.17441`) and its
follow-up (`2504.06110`) on evolving mating preferences; LLM-as-a-Judge (`2306.05685`) for
judging displays; Model Provenance Testing (`2502.00706`) for kin recognition. Also the
handicap principle, costly signaling, Instructional Fingerprinting.

**13 New modalities as heritable experts.** MoE-LLaVA (`2401.15947`), CuMo (`2405.05949`).
Also LLaVA, Uni-MoE, MoMa, Chameleon, Whisper, SWE-agent (file systems and shells).

**14 Scaling up.** What Matters for Model Merging at Scale (`2410.03617`), merging during
pre-training (`2505.12082`), DiLoCo (`2311.08105`), Evolution Strategies at Scale
(`2509.24372`). Also OpenDiLoCo, INTELLECT-1, MoE and upcycling scaling laws, Model Swarms.

## Libraries

- **Training:** PyTorch (MPS) + transformers/PEFT/TRL, or the Apple-native MLX stack
  (mlx, mlx-lm, mlx-lm-lora, which supports GRPO). MLX is usually the faster path on a Mac.
- **Merging & evolution:** mergekit (LGPL-3.0, has `mergekit-moe`), mergenetic, cma, pymoo.
- **Alignment:** POT (optimal transport), scipy `linear_sum_assignment`.
- **Eval:** lm-evaluation-harness.
- **Reference code** (cloned to `refs/`): see `scripts/clone_reference_repos.sh`.
  All repo URLs were checked to exist.
