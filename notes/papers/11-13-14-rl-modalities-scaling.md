# Notes: groups 11 (RL and fine-tuning), 13 (new modalities), 14 (scaling up)

Read via alphaXiv full text (priority 1: methods, hyperparameters and ablations; priority 2-3: abstract plus
key results). Web lookups are linked inline and are mostly self-reported community numbers, so treat them as
anecdotes. Anything marked "[estimate]" is my own arithmetic, not a measurement. Hardware assumption: one Mac
Studio, 32 GB, MLX.

Qwen3-0.6B arithmetic used below: hidden 1024, 28 layers, FFN 3072, 16 Q / 8 KV heads. Matches the README
table (4-expert adult 1.39B, 8-expert child 2.44B). The child's top-2-of-8 active parameters are about 0.86B
[estimate].

---

## Group 11: RL and fine-tuning

### 2402.03300 DeepSeekMath (priority 1)
- GRPO: no critic. For each question sample G outputs, advantage = (r_i - mean) / std within the group (outcome
  supervision, same advantage for every token). KL to the reference is added to the loss (not the reward) with
  the unbiased estimator `ref/pi - log(ref/pi) - 1`.
- RL recipe (7B Instruct -> RL): lr 1e-6, KL coef 0.04, G = 64 samples per question, max length 1024, batch
  1024, ONE policy update per exploration (so clipping is inactive). Prompts: about 144K GSM8K/MATH-style
  chain-of-thought questions from the SFT set, no new data.
- Gains: GSM8K 82.9 -> 88.2, MATH 46.8 -> 51.7 (7B); out-of-domain CMATH 84.6 -> 88.8.
- The 1.3B ablation (Fig. 5) trains thousands of steps (x-axis to 8000). I read GSM8K moving about 56 -> 66 and
  MATH about 27 -> 30 off the axes, so approximate, but that is the only small-model RL curve here.
- Key analysis for us: RL raises Maj@K but NOT Pass@K. It sharpens the output distribution rather than adding
  capability. Online RFT beats offline RFT; GRPO beats Online RFT because it also penalises wrong samples
  (+/- gradient coefficients); iterative RL helps mostly in round 1.

### 2501.12948 DeepSeek-R1 (priority 2)
- R1-Zero GRPO settings: lr 3e-6, KL 0.001, temperature 1, G = 16, max length 32k (64k later), 32 questions per
  step = 512 samples, 10.4k steps. Reference policy is reset to the current policy every 400 steps. 8192
  rollouts per step split into 16 minibatches, one inner epoch. Qwen-32B-Zero used lr 2e-6.
- Reward = rule-based accuracy + format (equal weight). They explicitly avoid neural reward models for reasoning
  because of reward hacking at scale. PRM and MCTS were reported as failed attempts. The helpful RM needed 66K
  preference pairs and the safety RM 106K prompts.
- Small models: distillation beat RL by a wide margin (Qwen2.5-32B-Zero AIME 47.0 vs R1-Distill-32B 72.6; R1-Distill-1.5B
  gets 28.9 AIME). They conclude RL-from-base "is highly dependent on underlying model capacity".
- Distill LR for 1.5B: 1e-4, 2-3 epochs on 800K samples.

### 2203.02155 InstructGPT (priority 3)
- RLHF recipe: SFT, then 6B RM, then PPO. 1.3B PPO-ptx is preferred to 175B GPT-3. KL beta 0.02, 256K episodes
  (about 31K unique prompts), batch 512. Training longer than 256K episodes did not help at 1.3B. LR above 8e-6
  diverged without the pretraining mix.
- "ptx": mix pretraining gradients (gamma 27.8) to avoid the alignment tax. Raising the KL coefficient 100x did
  NOT fix the regressions. Cross-labeler RM accuracy is only about 70-72%.
- Relevance: a replay/pretrain mix during gestation is the proven fix for forgetting. An RM needs a lot of
  preference data and is itself noisy.

### 2106.09685 LoRA (priority 2)
- W + (alpha/r) BA with B = 0 init, so the update starts at exactly zero. Merges into W with no inference cost.
- On GPT-3, adapting Wq and Wv at r = 4 matches adapting all four attention matrices; r = 1 already works.
  With a fixed parameter budget, adapting MORE matrices beats a higher rank on fewer. The MLP was not studied.
- delta-W amplifies directions that are present but not emphasised in W (about 20x).

### 2305.14314 QLoRA (priority 2)
- LoRA on ALL linear layers is required to match full fine-tuning (query/value-only does not). Rank r (8-256) is
  irrelevant once all layers are covered. Defaults: r = 64, alpha = 16, dropout 0.05-0.1 for small models, lr
  2e-4 (1e-4 for 33B+), constant schedule, grad-clip 0.3.
- LoRA parameters are tiny; activation gradients dominate memory (26 MB of adapters vs 567 MB of input
  gradients for a 7B). Gradient checkpointing matters more than rank.
- NF4 + double quantisation matches 16-bit LoRA. Unneeded for us at 0.6B (1.2 GB in bf16); relevant only at
  8B+.

---

## Group 13: New modalities

### 2401.15947 MoE-LLaVA (priority 1)
- Architecture: frozen CLIP-L/336 -> 2-layer MLP projector -> LLM (StableLM-1.6B, Qwen-1.8B, Phi-2 2.7B). Every
  other FFN becomes 4 experts, top-2, copied from the dense FFN. Linear router, balance loss alpha = 0.01.
- Three-stage "MoE-Tuning": (I) projector only (LLaVA-558K, lr 1e-3); (II) everything except the vision encoder
  (964K, lr 2e-5), which makes a dense LVLM; (III) copy FFN into experts and train ONLY the MoE layers
  (665K, lr 2e-5), 1 epoch (2 epochs overfit).
- Skipping the dense stage II is the most important ablation: GQA 58.4 vs 61.5, SQA 58.1 vs 63.1, POPE 81.9
  vs 87.0. Sparsifying and adding a modality at the same time degrades badly.
- Tuning FFN/experts only is as good as tuning everything (61.5 vs 61.3 GQA) at 75% of the time. All-layers MoE
  is worse and slower than alternating. Top-2 >> top-1; 2 experts vs 1 expert is only +1 POPE.
- Compute: 8x A800, stage III about 20 h on 8x V100-32G for the 1.8B model.
- Caveats: small models show one dominant expert (Phi-2 layers 17-23). The 7B variant was WORSE than its dense
  counterpart; authors blame having only 655K instruction samples for sparse learning.

### 2405.05949 CuMo (priority 1)
- MoE placed in the vision side: MLP connector and every CLIP FFN, co-upcycled (each expert initialised from
  the pretrained block). LLM left dense. Top-2 of 4 beat top-2 of 8 (too little data for 8).
- Training from scratch "struggled to converge" even at a lower LR. Co-upcycling fixed it. A pre-finetuning
  stage (708K ALLaVA captions, all parameters, lr 2e-6) before introducing MoE was needed for stability.
- Losses: balance 0.1 + router z-loss 0.01. LLM lr had to drop to 2e-6 (2e-5 was unstable).
- Strongest finding for us: upcycled LLM-MoE (Mistral 8x7B made from dense copies) was clearly worse than the
  real pre-trained Mixtral (MMVet 35.7 vs 40.0, SEED 70.5 vs 72.6). Copies add parameters but no new
  knowledge, while diverse pre-trained experts do.
- Compute: 8-32x A100 across the stages; 1.65M instruction samples.

### 2405.11273 Uni-MoE (priority 2)
- Modality encoders (CLIP, Whisper-small, BEATs) -> connectors (Q-Former for audio/speech) -> LLM with sparse MoE.
  Progressive training: (1) connectors only, (2) train modality-specific experts separately, (3) merge them into
  the MoE and tune with LoRA (r = 8, alpha = 16, lr 4e-5) on mixed data.
- Experts pre-tuned per modality beat identical-init experts (avg 49.2 vs 47.5-48.2). Balance loss was not
  needed with 4 experts, but helped with 8. This is the closest published analogue of "child receives
  separately-trained experts".
- Mixed-modality MoE reduced the per-modality performance bias, and more modalities helped single-modality
  tasks.

### 2407.21770 MoMa (priority 2)
- Experts are split into modality groups (4 text + 4 image). Tokens are hard-routed to their modality's group and
  learned routing happens inside the group. 3.7x FLOPs saving vs a dense baseline (text 2.6x, image 5.2x) at 1.4B.
- Upcycling from a 1-text/1-image seed after only 10k steps gave an extra 1.2x. Image experts show diminishing
  returns beyond 2-4. Expert-choice routing breaks causality at inference and needs auxiliary routers.
- Relevance: a modality is a self-contained expert GROUP, which is the cleanest unit to inherit.

### 2304.08485 LLaVA / Visual Instruction Tuning (priority 2)
- Recipe: frozen CLIP ViT-L/14 + one linear layer + LLM. Stage 1: 595K CC3M pairs, projector only, lr 2e-3,
  about 4 h on 8x A100 (7B LLM). Stage 2: 158K GPT-4-generated instructions, 3 epochs, lr 2e-5, about 10 h.
- Minimal viable vision module = linear projector + frozen encoder + about 600K captions + about 150K
  instructions. All the "knowledge" is in the frozen encoder and the LLM.

### 2405.09818 Chameleon (priority 3)
- Early fusion with VQ image tokens trained from scratch on about 9T tokens (7B = 856K A100 hours). Needs QK-norm,
  norm re-ordering and z-loss for stability. Not retrofittable onto Qwen3 without an enlarged vocabulary and
  pretraining. Not a candidate for us.

### 2212.04356 Whisper (priority 3)
- Encoder-decoder, 680K hours. Sizes: tiny 39M, base 74M, small 244M. LibriSpeech test-clean WER 5.6 (tiny.en)
  and 4.2 (base.en). Fine as a frozen speech encoder to plug into the LM (Uni-MoE uses Whisper-small). Says
  nothing about speech output. Small models show negative multi-task transfer at low compute.

### 2405.15793 SWE-agent (priority 2)
- The gain comes from the agent-computer interface, not weights: few compact commands, concise feedback, guardrails
  (lint on edit). GPT-4 Turbo 12.5% on SWE-bench, 18.0% on Lite (shell-only 11.0%). Search design ablation: -6
  points for a naive "iterative" search.
- Llama-3 8B (8k ctx) and DeepSeek-Coder were "subpar" in the agent setting. SWE-bench is far beyond a 0.6B model.
  Take the ACI principles and use toy sandboxes with state-diff rewards.

---

## Group 14: Scaling up

### 2410.03617 What Matters for Model Merging at Scale (priority 1)
- PaLM-2 1B/8B/24B/64B, 2-8 full-FT experts, Averaging / Task Arithmetic / TIES / DARE-TIES, held-in and held-out.
- (1) Merge experts built from a strong instruction-tuned base: held-in retention is much higher. (2) Larger
  models merge more easily. (3) Merged models beat multitask training on held-out when the base is strong
  and large (24B+, 6+ experts). (4) Larger models tolerate more experts: 8B non-IT retains 0.66 -> 0.39 going
  from 2 to 8 experts vs IT 0.91 -> 0.86. (5) Methods converge at scale (averaging is enough at 64B).
- For us: the smallest model they tried is 1B, we are at 0.6B. Everything points to "worst regime": use the
  post-trained Qwen3-0.6B, expect real method differences, and do not assume results transfer upward or downward.

### 2505.12082 Model Merging in Pre-training (priority 1)
- PMA = average of N checkpoints from the constant-LR phase (WSD schedule), SMA/EMA/WMA. Gains on every
  benchmark from 0.7B/7B to 20B/200B MoE and dense 411M-70B (e.g. HumanEval 31.1 -> 36.6, GSM8K +5). Merged
  stable-phase checkpoints match the annealed model, so the merge is a free LR-decay simulator.
- Best N about 10, interval scales with model/batch (about 4B tokens at 0.7B, 8B at 1.3B, 80B at 10B). Method
  hardly matters late; WMA wins early. Large intervals or early-training checkpoints hurt.
- "PMA-init" (starting SFT/CT from the merged weights) smooths grad-norm and rescues loss spikes. SFT gains were
  not reproducible at other sizes. Explicitly NOT studied for RL.

### 2311.08105 DiLoCo (priority 1)
- Inner AdamW (lr 4e-4, 1000 warmup, wd 0.1, batch 512 x 1024) for H = 500 steps, then outer Nesterov SGD on the
  averaged delta (outer lr 0.7, momentum 0.9; the setting is "very robust"). Plain SGD outer (= FedAvg) and Adam
  outer are worse. Tuned grid: outer lr 0.1-1.0, momentum 0.8-0.95, H 50-2000.
- 8 workers (non-iid shards): 15.02 PPL vs 15.30 for an 8x batch baseline, 500x less communication. Diminishing
  returns past 8 replicas. Works from a pretrained init and even from scratch. Robust to non-iid data and to
  workers dropping out. Pruning 50% of outer-gradient entries: +0.4% PPL.
- Costs data efficiency: 15.02 vs 14.72 for 8x more sequential updates.

### 2407.07852 OpenDiLoCo (priority 3)
- Replication in Hivemind: 150M matches an 8x-batch baseline (13.73 vs 13.68). 1.1B with 4 workers: H = 125
  gets 10.76 PPL (baseline 10.52), H = 500 gets 11.14, so more local steps hurt at larger batch. fp16 all-reduce
  is free. 90-95% utilisation across 3 countries.

### 2412.01152 INTELLECT-1 (priority 2)
- 10B on 1T tokens, up to 14 nodes. H = 100 inner steps (38 min), outer Nesterov lr 0.7, momentum 0.9 (same
  as DiLoCo), inner lr 7.5e-5, int8 pseudo-gradient all-reduce (400x less traffic). Compute utilisation 83% global /
  96% US. Losing 4 of 12 nodes at once caused instability and forced a checkpoint resume.
- Post-training used MergeKit (16 merges of candidate SFT/DPO runs) and DistillKit.

### 2402.07871 Scaling Laws for Fine-Grained MoE (priority 2)
- Granularity G = d_ff / d_expert. Loss follows a power law in G. G = 1 (experts = FFN size) is almost never
  optimal; compute-optimal G rises with budget (G 8-16 at 1e20 FLOPs, 32-64 at 1e24+). MoE needs more tokens than
  dense to pay off (token exponent 0.147 vs 0.127), and the MoE advantage grows with scale (about 20x savings
  at 1e20 FLOPs). Routing overhead hurts if G is too large for d_model (G = 64 at d_model = 256 declined).

### 2502.03009 Scaling Laws for Upcycling MoE (priority 2)
- Upcycling beats from-scratch MoE only while the upcycle token budget D2 is below
  D* approx 4 (N1/1e9)^(-0.7 + 0.04 log(N1/1e9)) billion tokens (4B at 1B model size). The more the dense model
  was trained (sunk cost D1), the lower the starting loss but the slower the MoE improves afterwards
  (exponent shrinks with log D1): experts that are already near-optimal copies find it hard to diversify.
- Quality improves with sparsity and active parameters, no trade-off seen up to 7B. If no pretrained dense
  model exists, train directly. Runs up to 1B dense, about 300B tokens.

### 2509.24372 Evolution Strategies at Scale (priority 1)
- Plain ES on the full parameter vector, no gradients: perturb with sigma = 0.001, evaluate each of N = 30
  perturbed models (greedy decoding, same seeds), z-score rewards, update `theta += alpha * mean(R_n eps_n)`
  with alpha = 5e-4, 500 iterations. Memory = inference only (one layer of noise at a time), no optimizer state
  or activations. Population 30 is tiny vs the 10,000+ of older ES.
- Countdown (200 training prompts, 3M sample evaluations in total): Qwen2.5-0.5B-Instruct 0.1 -> ES 14.4 (GRPO-v
  13.0, Dr.GRPO-v 13.5). Qwen2.5-1.5B 0.7 -> ES 37.3 (GRPO-v 27.8, Dr.GRPO-v 31.0). Std across seeds about 1-2
  points; ES is the more stable. RL baselines needed a per-model sweep over KL beta and lr and "did not make much
  progress" otherwise (GRPO-z with Qwen2.5-0.5B stayed at 0.3-0.5%).
- Larger N helps only slightly (GSM8K 1.5B: N = 16 -> 0.697, N = 128 -> 0.727). No KL penalty needed; less reward
  hacking in the conciseness experiment.

### 2410.11163 Model Swarms (priority 2)
- Particle-swarm search over LoRA experts (Gemma-7B): 10 SFT LoRAs expanded to N = 20 particles by pairwise
  interpolation crossover `x = t*xa + (1-t)*xb`, then velocity updates (inertia 0.1-0.3, own-best 0.1-0.5,
  global-best 0.2-0.6, repel-worst 0.01-0.1, step 0.5-1.0 decayed 0.95), K = 50 iterations, restart stuck
  particles. Utility = score on only 200 validation examples. No gradients.
- Beats 12 merge baselines by 13% on average (and SFT on the same 200 examples: GSM8K 0.459 vs 0.279).
  Crossover initialisation is the most valuable component; diversity of the starting experts matters a lot (+35%
  from least to most diverse). Cost is linear in particles x iterations; storage is 3n + 1 adapter copies.

---

## Implications for vibe-husbandry (prioritized)

1. **Measure first: spend 1 hour on a throughput benchmark before designing the experiments.** I found no
   reliable public MLX numbers for 0.6B-1.5B models with batched sampling (web search came back empty;
   [mlx-lm-lora](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora) lists Qwen3-0.6B GRPO at about 0.02 it/s and 9-20 GB
   with no chip or config attached; a vllm-mlx paper quotes up to 525 tok/s for Qwen3-0.6B on an M4 Max). Run
   `mlx_lm.benchmark` / batched generation on the actual Mac for the dense 0.6B and the 8-expert child, and
   re-fill the table below.

2. **Gestation budget (estimates).** Assumes 64 rollouts per step (8 prompts x G = 8), 384 tokens, LoRA, and the
   0.02 it/s figure as an anchor.

   | Config | Rollouts/step | Seconds/step [estimate] | Steps per 8 h | Rollouts per 8 h | Comparable to |
   |---|---|---|---|---|---|
   | Smoke test, dense 0.6B, 256 tok | 32 | 20-25 | 1100-1400 | 35-45k | pipeline debug |
   | Standard, dense 0.6B | 64 | about 50 | about 575 | about 37k | community Qwen3-0.6B runs of 60-180 steps |
   | Child 2.45B MoE (top-2/8) | 64 | 100-150 | 190-290 | 12-18k | one decent gestation per night |
   | Child + frozen reward model | 64 | +5 per step | same as above | same | feasible, but the data/hacking cost is the problem |
   | Literature, for scale | - | - | - | 5M (R1, 10.4k steps x 512); 3M (ES paper); 144K questions (DeepSeekMath) | 100-1000x more than a night |

   Rollouts/hour is therefore about 4-6k (dense) and about 1.5-2.3k (child). Everything else follows from this
   scale: a night is the community-anecdote regime, not the paper regime. Seeds are the real cost: plan for
   child + 2 parents + 3 compute-matched baselines x 2 seeds = about 10 nights, so shorten v0 runs
   (about 150-250 steps, a couple of hours) rather than plan on 8 h each.

3. **Minimum RL that gives a measurable gain on 0.6B.** Anecdotal web results (Hugging Face model cards found by
   search; I could not attribute each number to a specific card, so treat them as unverified): Qwen3-0.6B +
   GSM8K GRPO reported 8% -> 59% after 150 steps (100 held-out items), 56.3% -> 62.8% by step 180 (TRL), and
   22% -> 47% in 60 steps after SFT; much of the jump is format compliance, not reasoning. Candidate cards:
   [muskannnnn](https://huggingface.co/muskannnnn/Qwen3-0.6B-GSM8K-Reasoning),
   [Atharva1232](https://huggingface.co/Atharva1232/Qwen3-0.6B-GRPO-GSM8K),
   [x32](https://huggingface.co/x32/Qwen3-0.6B-GRPO-GSM8K-Think). For Qwen2.5-0.5B-Instruct Countdown,
   [a community GRPO card](https://featherless.ai/models/s1lv3rj1nx/countdown-qwen2.5-0.5b-grpo-lr3e6) reports about
   12%, and the TinyZero README (per search snippet) says the 0.5B base fails to learn. So: expect a few points on a
   task where the base already answers in the right format, and large jumps only from format fixes. Treat 100-200
   steps as the floor and use prompts with 20-80% pass rate (one
   [repo](https://github.com/miki-0408/GRPO) reports that filtering by within-group reward variance turned a
   200-step run from a drop into +20 points on 300 training prompts; self-reported).

4. **Verifiable rewards only for gestation.** R1 abandoned neural RMs and PRMs for reward hacking; InstructGPT
   needed a 6B RM plus a large preference set and still only got about 70% agreement; our judge is a
   0.6B-class model. Use exact-match math (GSM8K/MATH-style train splits) plus unit-test code (an MBPP-like train
   split; keep HumanEval/GSM8K test strictly held out). Keep the judge for wooing (cheap, one-shot) not for RL.
   An RM ablation is optional and later; if run, watch a held-out judge for divergence.

5. **GRPO starting recipe (my extrapolation from the papers, to be swept on 3 values).** G = 8; 8 prompts per
   step; temperature 1 (R1) or 0.8 (mlx-lm-lora default); max completion 384-512; one update per rollout batch
   (DeepSeekMath), so clipping is moot; KL beta 0.01-0.04 (DeepSeekMath 0.04, R1 0.001; set 0 to skip the
   reference pass and save time); LoRA lr about 1e-5 to 5e-5 (full-FT RL used 1e-6 to 3e-6, LoRA usually wants
   about 10x more; unverified); router lr similar. Log: reward curve, fraction of groups with zero reward variance,
   mean length (length hacking), KL, per-expert load. Note mlx-lm-lora defaults: group size 4 and epsilon 1e-4
   (clip width looks tiny; only matters with more than one update per batch, so check). Qwen3 is a hybrid
   thinking model: disable thinking or cap length, otherwise 384 tokens will truncate chains of thought and
   poison the reward (my knowledge of Qwen3, not from the papers).

6. **Report pass@k and maj@k, not just pass@1.** DeepSeekMath: RL moves maj@K, not pass@K. If the child's
   advantage over the parents is meant to be capability (math from one parent, code from the other), RL cannot
   create it. It must come from inheritance plus router learning, so the before-RL vs after-RL child eval is
   the experiment that separates the two.

7. **Compute-matched baselines to add to the list (CLAUDE.md already has three).**
   (a) Online rejection-sampling fine-tuning, which DeepSeekMath showed is the closest cheaper alternative to GRPO.
   (b) Distillation / RFT from a larger Qwen3 teacher: R1 reports distillation >> RL at small scale; it is also a
   natural "nuptial gift".
   (c) PMA-style checkpoint averaging of the last 8-10 gestation checkpoints. It is free, helps in pretraining
   (+5 on GSM8K in their runs) and is the right comparison for any "merge" claim. For LoRA parameters,
   average the merged delta BA (not A and B separately) or use EMA on the merged weights.

8. **Trainable-parameter plan for the 32 GB child (full FT does not fit, per README).** LoRA r = 16 on every
   linear layer of every expert (gate/up/down) plus the attention projections gives about 49M trainable
   parameters (about 2% of 2.45B) [estimate]. QLoRA: coverage of all linear layers matters, rank does not, and
   memory is dominated by activations, so use gradient checkpointing and do not bother with quantising the
   base at 0.6B. The router is only about 0.23M parameters (8 x 1024 x 28), so train it in full (not LoRA). Use
   a balance loss in the 1e-2 to 1e-3 range (MoE-LLaVA 0.01, CuMo 0.1 + z-loss 0.01, upcycling-law paper 1e-3);
   Uni-MoE found it unnecessary at 4 experts and useful at 8, which is exactly our child size. Add a replay of
   ancestor-style data (InstructGPT "ptx") to limit forgetting.

9. **Conception/gestation staging should copy MoE-LLaVA / CuMo / Uni-MoE.** (i) Parents are already fine-tuned
   dense-or-MoE models (the analogue of their stage II: skipping it cost 5-7 points). (ii) Warm up the router
   alone on a mixed math+code SFT set (MoE-LLaVA stage III trains only MoE layers). (iii) Then LoRA on experts
   and RL. (iv) Pre-trained diverse experts beat copies (CuMo LLM table, Uni-MoE 49.2 vs 47.5-48.2). This is the
   best published support for the project premise, so cite it in the README.

10. **Upcycling caveat for the ancestor.** The upcycling law says upcycling gains shrink as the dense sunk cost grows
    (its fit extends to about 300B tokens, Qwen3-0.6B saw tens of trillions; D* for 0.6B comes out at roughly
    6B tokens by the formula [estimate]). Qwen3-0.6B is heavily over-trained, so copied experts are near-optimal and slow to
    diversify. Parent divergence must therefore come from the branch-train step (math vs code fine-tune); do not
    rely on router noise alone to diversify experts. Fine-grained experts (G 2-4, i.e. splitting each FFN) would give
    more, smaller chromosomes and finer crossover points, and the fine-grained scaling law favours G > 1; this is
    an optional ablation, not a v0 item.

11. **Merging for conception.** Use the post-trained checkpoint as ancestor (2410.03617 finding 1). Expect
    method sensitivity at 0.6B (the "methods converge" effect appears only at large scale), so keep the TIES
    baseline and add plain averaging and task-arithmetic scaling. The DiLoCo result says plain averaging
    (outer SGD, lr 1) is inferior to outer Nesterov momentum (lr 0.7, momentum 0.9) when many deltas from a
    common init are combined. Hypothesis worth one cheap experiment: treat the average of the two parents'
    deltas from the ancestor as an outer gradient and apply it with lr 0.7 instead of 1.0, and carry momentum
    across generations.

12. **Modalities (v2+).**
    - Vision for a 0.6B LM: frozen small CLIP/SigLIP ViT-B encoder, MLP projector (LLaVA/MoE-LLaVA recipe), stage 1
      projector-only on a caption subset, stage 2 on instruction data. LLaVA used 595K + 158K samples on 8x A100
      for 14 h at 7B; for 0.6B and a 100K-caption subset I estimate on the order of 2-5 h on the Mac for stage 1
      (about 6 x 0.6e9 x tokens FLOPs at 5-10 TFLOP/s effective; unmeasured). Precompute and cache encoder features.
    - Make the vision module a hard-routed expert GROUP (MoMa style) plus projector, not experts shared with text.
      Text tokens never touch it, so text capability is bit-identical when the module is absent, and the
      genome entry `modalities[name] = (encoder, projector, experts)` is a closed, inheritable unit.
      Initialise by copying FFNs and training only the group (MoE-LLaVA stage III).
    - Voice: Whisper tiny/base as a frozen encoder plus a small connector (Uni-MoE style). Output speech is not covered.
    - Skip Chameleon-style early fusion (needs from-scratch pretraining).
    - Tool use / file system: apply SWE-agent's ACI principles; design 4-6 compact commands with concise feedback
      and linting guardrails. Use toy sandboxes (create/edit/grep/run in a throwaway directory) with a verifiable
      end-state reward and at most 3-5 turns, since multi-turn rollouts multiply rollout cost.
    - Data-hunger warning: MoE-LLaVA's 7B variant failed with 655K samples, so keep module MoE small (2 to 4
      experts, top-2) for tiny data.

13. **Scaling plan: what carries over.**
    - Carries over: the genome/operation structure; Qwen3-style upcycle + branch + mix (BTX/CuMo); pre-training-style
      checkpoint averaging (PMA verified up to 200B MoE); larger models merge easier (2410.03617), so conception
      should get easier, not harder, at scale; the RL recipe (GRPO, verifiable rewards) at lr 1e-6 to 3e-6 and
      KL 0.001-0.04; LoRA with all linear layers.
    - Needs re-tuning: merge method (converges at scale), number of GRPO steps (millions of rollouts, not tens of
      thousands), router balance loss and expert count, merge interval for PMA (scales with model and batch size).
    - DiLoCo settings for the scale phase if parents are trained on separate islands: inner AdamW, outer Nesterov
      lr 0.7 / momentum 0.9, H = 100-500 (INTELLECT-1 chose 100 at 10B; OpenDiLoCo found 125 better than 500 at
      1.1B with big batches), 4-8 workers, fp16/int8 all-reduce. Evidence is dense pretraining only; nothing here
      tests MoE gestation or RL under DiLoCo.
    - ES: viable as a memory-light "gestation" at scale because it needs inference memory only (a 2.45B child
      would fit in 5 GB), but the paper's budget is 3M sample evaluations. My estimate of 30 members x 100 prompts x
      300 tokens is about 0.9M tokens per iteration, about 10 min on the Mac [estimate], so only about 50
      iterations per night vs the 500 in the paper. Use it on the Mac only as a small sanity check; reserve it
      for rented GPUs. Its lower variance and lack of reward hacking make it attractive as a gestation variant
      to compare against GRPO.
    - Model Swarms is the cheap gradient-free option that does fit the Mac: PSO over LoRA adapters with
      interpolation crossover. 20 particles x 200 validation examples x short outputs x 50 iterations is about
      200K short generations. It also supplies a crossover-coefficient search for conception and is a natural
      "population" operator for group 06. Guard against overfitting 200 examples with a held-out test.

---

## Red flags

- **RL evidence at 0.6B is thin.** Web results are model cards with 100-item held-out sets and mixed baselines
  (format fixes versus reasoning). R1 says RL-from-base depends on model capacity; ES paper's GRPO-z at 0.5B stayed
  near 0%. Silent GRPO failure is plausible; add sanity checks.
- **RL sharpens, it does not expand** (pass@K flat). Do not attribute capability gains to gestation without pass@k.
- **GRPO sensitivity.** The ES paper's RL baselines had to be swept over (beta, lr) per model or did not learn.
  The same paper is an ES advocacy paper, so its RL baselines may be under-tuned, and its ES results use 200
  training prompts on narrow tasks (Countdown, GSM8K, conciseness).
- **Hardware numbers in the mlx-lm-lora table are unattributed**; there is no usable MLX small-model batched
  generation benchmark in the sources I could reach. All Mac timings above are estimates.
- **Length and answer-extraction hacking** (a repo noted answers growing longer when the reward grabs the last
  number). Qwen3 thinking mode can overrun the length cap.
- **Over-trained ancestor** (upcycling ossification) and the upcycling law is extrapolated well beyond its fit.
- **Merging results do not transfer across scale.** 2410.03617 spans 1B-64B and finds method differences vanish at
  scale; at 0.6B they matter. PaLM-2 experts are full-FT on T0 tasks, not RL'd LoRAs on code/math.
- **DiLoCo is less data-efficient** (15.02 vs 14.72 PPL for 8x updates), has diminishing returns past 8 workers, was
  tested on dense pretraining up to 10B, and large simultaneous worker loss destabilised INTELLECT-1.
- **PMA has not been tested on RL**, and its SFT-init gain did not replicate across model sizes.
- **Sparse multimodal needs data**: the MoE-LLaVA 7B model underperformed its dense twin; expert imbalance appears
  in small models.
- **Evaluation noise.** GSM8K test (1319 items) gives a 95% CI of about +/-2.7 points on one model; child vs
  parent differences of 1-3 points need paired evaluation and several seeds.

---

## Open questions for the humans

1. Which Mac Studio chip and memory bandwidth (M-series Max or Ultra)? It sets all the rollout-throughput
   estimates by a factor of 2 or more.
2. Thinking mode: use Qwen3-0.6B in non-thinking mode for GRPO (short answers, cheap rollouts), or allow long
   chains of thought (3-5x the rollout cost)?
3. Which RL task mix and held-out splits for v0 (GSM8K/MATH-style plus MBPP-style unit tests)? HumanEval's 164
   problems are too small to detect small gains.
4. Is a larger teacher (Qwen3-4B/8B traces) allowed as a baseline and/or a "nuptial gift"? R1 suggests it will win
   at this size.
5. How many nights per experiment and how many seeds do we commit to? The baseline matrix is about 10 gestation-equivalents.
6. Should gestation use LoRA on experts (as assumed) or a smaller full-FT variant (top half of layers only)?
7. Is a rented-GPU budget on the table for the scale phase (ES, DiLoCo-style parallel parents)? If not, the
   scaling claims stay as design only.
8. Do we want to test the "outer momentum across generations" idea and fine-grained experts (G > 1) in v0, or
   keep v0 to the README plan?
