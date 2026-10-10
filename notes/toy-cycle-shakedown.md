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
