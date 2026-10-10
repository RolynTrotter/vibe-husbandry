#!/usr/bin/env bash
# Shallow-clone reference implementations into refs/ (read-only study material, not dependencies).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p refs
REPOS=(
  # merging toolkits
  arcee-ai/mergekit            # TIES/DARE/SLERP + mergekit-moe (LGPL-3.0)
  tommasomncttn/mergenetic     # evolutionary merging on top of mergekit
  yule-BUAA/MergeLM            # DARE reference code
  prateeky2806/ties-merging    # TIES reference code
  # evolution
  SakanaAI/evolutionary-model-merge
  SakanaAI/natural_niches      # M2N2 (competition + attraction mate selection)
  SakanaAI/CycleQD
  # alignment / "meiosis"
  samuela/git-re-basin         # permutation matching
  gstoica27/ZipIt              # merging models trained on different tasks
  PythonOT/POT                 # optimal transport library
  # cross-architecture fusion
  fanqiwan/FuseAI              # FuseLLM / FuseChat
  # MoE construction
  Taishi-N324/Drop-Upcycling
  allenai/OLMoE                # fully open small MoE, training code + analysis
  # growth (gestation)
  apple/ml-hypercloning        # function-preserving width growth
  # pruning (adolescence)
  CerebrasResearch/reap        # MoE expert pruning
  locuslab/wanda
  IST-DASLab/sparsegpt
  horseee/LLM-Pruner
  # training on a Mac
  ml-explore/mlx-lm
  Goekdeniz-Guelmez/mlx-lm-lora   # SFT/DPO/GRPO on MLX
  karpathy/nanoGPT                # template for a toy from-scratch MoE
  # eval
  EleutherAI/lm-evaluation-harness
  lm-sys/FastChat                 # MT-Bench / LLM-as-judge (courtship judging)
  # new modalities as heritable experts
  PKU-YuanGroup/MoE-LLaVA
  SHI-Labs/CuMo                   # co-upcycled vision experts
  SWE-agent/SWE-agent             # agent-computer interface (file systems, shells)
  # scaling up
  PrimeIntellect-ai/OpenDiLoCo    # low-communication training across "islands"
  VsonicV/es-fine-tuning-paper    # evolution strategies for full-parameter LLM fine-tuning
)
for r in "${REPOS[@]}"; do
  name="${r//\//__}"
  if [ -d "refs/$name" ]; then echo "have  $r"; continue; fi
  echo "clone $r"
  git clone --depth 1 -q "https://github.com/$r" "refs/$name" || echo "  FAILED $r"
done
