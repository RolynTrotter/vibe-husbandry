# Synthesis: what the papers say about the plan

Distilled from the per-group skims in `notes/papers/` (written by sub-agents on 2026-10-10;
numbers flagged "from memory" there should be re-checked before they go in a writeup) and
from `notes/data-plan.md`.

## The one-paragraph version

The life cycle is buildable on a 32 GB Mac, but **only at a budget 1–3 orders of magnitude
below every published result it leans on.** So v0 should be framed narrowly:
*does a child that inherits both parents' experts compose their skills better than TIES/DARE
merging or the same compute spent continuing one parent?* That is reachable. "Breeding beats
pre-training" is not testable here. Population-level questions (mate choice, selection,
diversity) should run on the toy MoE, where a generation is minutes, not hours, with
Qwen3-0.6B as a small number of confirmatory runs.

## Findings that change the plan

### Conception (01-02, 03-05)
- **One founder, then branch.** Upcycle Qwen3-0.6B to MoE-4 *once* (that checkpoint's hash is
  the `species_id`), and branch both parents from it. Then expert *k* is homologous across
  parents. Independent upcycles (or Drop-Upcycling re-init done per parent) silently break this.
  BTM shows averaging fails without a shared, partly trained seed.
- **Function-preserving init is free.** Identical expert copies plus top-k-then-softmax
  renormalisation reproduce the dense model exactly. The "upcycling dip" comes from
  unnormalised gates. (The toy MoE already renormalises over the chosen k.) Drop-Upcycling's
  r=0.5 only pays off after hundreds of billions of tokens: default r=0, ablate r=0.25.
- **Crossover is whole-expert only.** BTX blending at sub-expert granularity dropped the
  average from 34.7 to 22.2.
- **No alignment needed at conception.** Fine-tunes of one ancestor are linearly mode
  connected, and the child keeps both expert sets rather than averaging them. Alignment
  (one Hungarian assignment per SwiGLU expert) is needed only when *merging* experts at prune
  time, after many generations, or across model families.
- **Trunk merge default:** task arithmetic against the ancestor, `θ0 + λ(τA + τB)`, with λ
  from 0.5 (plain average) to 1.0. TIES (keep 20%, λ=1) and DARE (p≈0.9, rescaled) are
  baselines; with only two parents TIES has little to resolve. For generation ≥ 2, merge
  against the most recent common ancestor so deltas stay small. Simplest arm: **freeze the
  trunk in the parents**, so there is nothing to merge.
- **Router.** Keep it in fp32 (done). Balance load over the global batch. Drop the balancing
  loss during RL. BTX without a balancing loss let the code expert die.

### Gestation and RL (11, 13)
- **RL sharpens; it does not add capability.** GRPO raises maj@k and pass@1 but not pass@k.
  So the child's math + code composition has to come from conception plus router training,
  SFT, or distillation, not from RL. Report pass@k.
- **At small scale, distillation beats RL** (DeepSeek-R1, Qwen3 reports). This suggests a
  biologically pleasing gestation: **the parents teach the child.** Router warm-up, then
  distillation from each parent on its own domain, then a short GRPO polish.
- **Budget (estimated, not yet measured):**
  - Dense 0.6B: about 575 GRPO steps per 8-hour night, roughly 37k rollouts.
  - 8-expert child: about 190–290 steps per night.
  - Published RL runs use 100–1000× more. Community Qwen3-0.6B GSM8K runs report gains in
    60–180 steps, mostly format fixes.
- **RLHF:** use verifiable rewards (math answers, unit tests) only. A learned reward model
  costs memory, adds a reward-hacking surface, and we can't afford enough rollouts for it
  to matter. See `data-plan.md`.
- **LoRA recipe:** r=16 on every expert linear layer (about 49M trainable parameters), plus a
  fully trained router. Use MoE-LLaVA/CuMo staging: pre-tuned parents, then router warm-up,
  then LoRA, then RL.

### Adolescence → adult (07-10)
- **REAP saliency:** `S_j = mean over tokens routed to j of gate_j(x) · ||f_j(x)||₂`. Prune the
  lowest, delete the matching router rows, calibrate on about 1k × 2k tokens. Calibration
  **must include math and code**, or those skills collapse.
- **Red flag:** REAP was validated on 64–384 experts. At 4–8 experts, merging is competitive.
  Run the 8 → 4 step as an experiment with these arms:
  - REAP (free choice)
  - keep one allele per locus (mother's or father's expert *k*)
  - average each homologous pair
  - 4-bit quantization of the whole 8-expert child, as a same-bytes baseline.
- **Healing distillation** (forward KL on logits, Minitron) is the biggest unknown budget.
  Sweep 0 / 10M / 100M tokens.
- **Critical periods** (analogy only): keep early routing balanced across both parents'
  experts, so one parent's chromosomes aren't starved permanently. Prune only after routing
  and saliency rankings have stabilised.
- **No free adolescent.** MatFormer needs nested training from the start. Use REAP-ordered
  expert prefixes (top-6, top-4) as the adolescent sizes instead.

### Populations and wooing (06, 12)
- **Evolutionary-merge budgets assume cheap fitness calls.** EvoLLM-JP and M2N2 each ran
  about 1000 evaluations, and those are inference only. With a gestation of about 4 hours,
  the Mac does about 30 per week. That is enough to A/B one mechanism, not to measure
  population statistics.
- **Add an "embryo" tier:** conceive, then evaluate on 100–200 items with no training. This
  cheap screen decides which embryos get a gestation.
- **Log per-item correctness vectors for every genome.** M2N2-style attraction
  (complementarity) and fitness sharing need them.
- **PIMP mate choice buys diversity, not fitness.** Preferences collapse without mutation.
  Always run a random-mating control.
- **Kin gate.** Instructional fingerprinting costs about 60 examples and about 1 minute. Its
  persistent variant needs white-box access to embeddings, which conflicts with DESIGN.md's
  "without inspecting weights". It has never been tested under merging, MoE or RL. A
  probe-set interpolation barrier is a stronger compatibility check than CKA.
- **Judge.** A 0.6B judge is useless. Verify constrained forms (acrostic, meter, lipogram)
  with code, and use a 7B+ judge from another model family, scoring both answer orders.

### Scaling (14)
- Merging gets *easier* with scale. 0.6B is the hardest regime, so negative results here
  don't kill the idea.
- Checkpoint averaging (PMA) is a free baseline.
- Settings that carry over: DiLoCo's outer Nesterov (lr 0.7, momentum 0.9) for parallel
  gestation. Evolution strategies need only inference memory but millions of evaluations,
  so they are GPU-rental territory.

## Biggest red flag

**Experts specialise by token and syntax, not by domain** (Mixtral, OLMoE), and upcycled MoEs
specialise especially weakly. The "experts = chromosomes carrying skills" metaphor may not
hold at the level of individual experts. Test it early: on the toy MoE and on the first
Qwen3 parents, measure whether math- and code-trained parents' experts differ in routing
and in the effect of ablating them. If they don't, the gene unit may need to be a group
of experts per parent, or a LoRA adapter, rather than a single FFN.

## Recommended v0 sequence

0. **Done in this session.** Toy MoE training loop with cost-reporting `gestate`, and a
   dense compute-matched baseline config.
1. **Toy life cycle (r-selected regime):**
   - Founder: toy MoE-4 on TinyStories.
   - Branch two parents on different data. TinyStories vs a code or simple-math corpus
     would make specialisation visible.
   - `conceive` (cat experts, task-arithmetic trunk), `gestate` (router warm-up plus
     distillation), `prune` (the four arms above), `evaluate`.
   - All operations log cost. Run small populations here.
2. **Throughput benchmarks on the Mac:**
   - MLX vs PyTorch-MPS tok/s for Qwen3-0.6B LoRA training.
   - GRPO rollouts per hour.
   - These replace every "estimated" number above.
3. **Qwen3-0.6B v0** as in the README, with the founder/branch structure above, and
   distillation as the main gestation with GRPO as polish.
