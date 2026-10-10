# Toy breeding cycle: shake-down log

The first end-to-end runs of `scripts/toy_cycle.sh` were done on a 4-core cloud CPU, not the
Mac, with step counts scaled down (`F=300 P=100 W=20 G=100`, batch 16). They were a
plumbing test, so the scores below say nothing about whether breeding works.

## Run 1 (2026-10-10): original settings

Settings: parents trained with the trunk frozen (`trainable = "moe"`), and arithmetic with
3-digit operands, answers written most-significant digit first.

| genome | experts | train tok | loss:TinyStories | acc:arith | acc:reverse |
|---|---|---|---|---|---|
| founder | 4 | 1.2M | 3.318 | 0.0% | 0.0% |
| dam | 4 | 1.6M | 3.364 | 0.0% | 0.0% |
| sire | 4 | 1.6M | 3.380 | 0.0% | 0.0% |
| embryo | 8 | 2.0M | 3.400 | 0.0% | 0.0% |
| child | 8 | 2.5M | 3.329 | 0.0% | 0.0% |
| adult-reap | 4 | 2.6M | 3.660 | 0.0% | 0.0% |
| adult-allele | 4 | 2.6M | 3.386 | 0.0% | 0.0% |
| adult-random | 4 | 2.6M | 3.859 | 0.0% | 0.0% |
| longer | 4 | 2.5M | 3.218 | 0.0% | 48.0% |
| direct | 4 | 2.5M | 3.443 | 0.0% | 0.0% |
| avg | 4 | 2.0M | 3.352 | 0.0% | 0.0% |
| ties | 4 | 2.0M | 3.364 | 0.0% | 0.0% |

What it showed:
- **A frozen trunk cannot learn reversal.** Reversal is an attention skill: each output
  word is copied from an earlier position. Expert FFNs alone can't do that.
  - The only model that learned anything was `longer`, which trains everything.
  - With the whole network trainable, the sire reached **93%** on held-out reversal after
    300 steps at batch 16, and its TinyStories loss was unchanged.
  - *Change:* parents and gestation now train the whole network. The trunks therefore
    diverge, and conception merges them as founder + λ·(τ_dam + τ_sire).
- **3-digit addition written most-significant digit first was not learned at all.**
  - Switched to 2-digit operands with the answer reversed (least-significant digit first),
    following Lee et al. 2023 (arXiv 2307.03381).
  - Arithmetic validation loss went from 1.72 to 1.39 at the same budget, and accuracy
    went from 0% to 4% after 300 steps at batch 16.
  - FFN-heavy arithmetic paired with attention-heavy reversal is a good contrast for an
    MoE whose experts are FFNs.
- **Pruning 8 experts back to 4:**
  - Keeping the better "allele" at each homologous slot cost the least TinyStories loss
    (3.329 → 3.386).
  - REAP saliency cost more (→ 3.660) and random selection the most (→ 3.859).
  - This is one run on undertrained models, so it is a hint, not a result.
- **Bugs fixed:**
  - Eval prompts now start with EOS, as training documents do.
  - The cache key for synthetic corpora now hashes a sample document, so a change to a
    task's format can't silently reuse old tokens.
- Compute matching works: `child`, `longer` and `direct` all report 2.5M training tokens.
  The bill in each lineage counts the shared founder only once.
- Running pytest alongside training on 4 cores slowed both down about 25× (thread
  oversubscription). Run them one at a time, or set `OMP_NUM_THREADS`.

## Run 2 (2026-10-10): whole-network parents, facts × reversal

Arithmetic stayed hard even after two changes: answers written least-significant digit
first, and fixed-width operands.
- 6.5% after the full Mac parent budget (1200 steps at batch 16).
- The model learned the format but not the units-digit carry.
- It is kept as `synth:arith` for longer budgets.

The dam now memorises **facts** instead: a fixed table of 1000 made-up names
(`fact: blorvat -> cake`). That is an FFN key-value memory skill, which contrasts well
with the sire's attention-based reversal.

Setup: the run 1 founder; parents at 300 steps, batch 16, whole network trainable; then
conception, gestation of 40 router-only steps plus 150 whole-child steps, and a control.

| genome | steps after founder | facts | reverse | TinyStories |
|---|---|---|---|---|
| founder | 0 | 0.0% | 0.0% | 3.318 |
| dam | 300 | **94.0%** | 0.0% | 3.307 |
| sire | 300 | 0.0% | **85.5%** | 3.322 |
| embryo (λ = 0.5, no gestation) | 600 | 0.5% | 0.0% | 3.343 |
| embryo (λ = 1.0, no gestation) | 600 | 0.0% | 0.0% | 3.628 |
| avg merge of dam + sire | 600 | 0.5% | 0.0% | 3.314 |
| child-warm (router only, 40 steps) | 640 | 5.0% | 0.0% | 3.338 |
| **child** (+150 whole-child steps) | 790 | **93.0%** | **97.0%** | **3.251** |
| control: founder on the mixture for 190 steps | 190 | 3.0% | 0.0% | 3.302 |

What it showed:
- **The parents are clean specialists.** Each one has its own skill and none of the
  other's, and keeps its TinyStories ability.
- **Nothing is inherited at birth.**
  - Neither the embryo nor a weight-average merge expresses either skill.
  - λ = 1.0 (summing both trunk deltas) is worse than λ = 0.5.
  - Router warm-up alone recovers almost nothing.
- **A short gestation brings out both skills at once.**
  - In 190 steps the child reaches 93% facts and 97% reversal, and beats the founder on
    TinyStories.
  - The founder given the same 190 steps on the same mixture learns neither. So the
    inherited experts and trunk matter.
- **Not yet compute-matched.** The child's ancestry also includes the parents' 600 steps.
  The fair comparison is `longer`: the founder on the mixture for all 790 steps. The full
  `scripts/toy_cycle.sh` run on the Mac covers it, along with `direct` and the prune arms.
