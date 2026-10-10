# Notes: groups 07 growth, 08 pruning, 09 distillation, 10 critical periods

Read via alphaXiv (full text for priority 1, targeted pages for 2, abstracts for 3).
Reading depth is tagged [full] / [skim] / [abstract]. Anything marked ESTIMATE or
SPECULATION is mine, not from a paper.

---

## Group 08: pruning (adolescence -> adulthood)

### 2510.13999 REAP the Experts (Cerebras, ICLR 2026) [full, priority 1]
- Saliency, exactly: `S_j = (1/|X_j|) * sum_{x in X_j} g_j(x) * ||f_j(x)||_2`, where `X_j` is the set of
  calibration tokens for which expert j is in the top-k, `g_j` is the router gate value for j, and `f_j(x)` is
  expert j's output. Averaging only over tokens where j is active makes it frequency-agnostic, so rare
  specialists are not penalized. Prune the lowest S_j. Derived from an upper bound on layer reconstruction error.
- Pruning is a coordinate subspace of the original layer: router rows for dropped experts are deleted, surviving
  experts and rows are untouched. Merging ties gates (`g_i + g_j`), so the router loses independent control of
  the merged experts. The paper calls the result an "irreducible error" ("functional subspace collapse").
- Calibration: 1,024 samples packed to 2,048 tokens (about 2M tokens) for models <=110B; no fine-tuning afterwards.
  The paper stresses that calibration must be domain-matched: C4-only calibration collapses code accuracy
  (several models score 0% on code).
- Results (compression ratio is the share of experts removed per layer):
  - Qwen3-30B-A3B (128 experts): code avg 0.558 -> 0.551 at 25%, 0.541 at 50%. Merging (HC-SMoE, M-SMoE) at 50%
    falls to 0.36-0.40. Math avg 0.887 -> 0.857 at 50%.
  - GLM-4.5-Air at 50%: REAP code 0.515 vs baseline 0.580. M-SMoE 0.284.
  - Qwen3-Coder-480B and Kimi-K2: 50% pruned loses about 1% on code and keeps SWE-Bench.
  - Zero-shot multiple choice (MC): about -4% at 25% and -13% at 50% for the best methods. MC is a weak proxy for generation.
- Frequency-based pruning is bad (Qwen3-Coder-480B at 50%: 0.01 code). EAN (sum of activation norms) is the strong
  baseline. REAP is better mostly at 50%.
- Caveat: M-SMoE does better than pruning on code for low-granularity MoEs (Mixtral, Llama-4-Scout, 8-16 experts).
  Our MoE has 4-8 experts, so it is low-granularity.
- Quantization: pruning works directly on FP8 and W4A16 checkpoints with no re-quantization. Merging would need
  quantization scales reconciled. 4-bit AWQ of Qwen3-30B beats the 50%-REAP bf16 model at half the checkpoint size.

### 2604.04356 REAM: merging improves pruning (Mila/Samsung, Apr 2026) [skim, priority 2]
- REAM keeps REAP's gate-removal and saliency, but absorbs the dropped experts into the top-saliency "centroid"
  experts: the top N' experts by REAP score are centroids, and the remaining experts join their most similar
  centroid. Most groups stay singletons ("pseudo-pruning"). Similarity = gate-logit cosine + cosine of
  gate-scaled expert outputs. Neurons are aligned (Hungarian) with a combined activation + weight cost, then
  merged by a saliency-weighted average. It runs layer by layer, recomputing activations after each merged layer.
- Calibration: 3,072 x 512 tokens (about 1.6M), mixed C4/math/code. The mix is a Pareto lever:
  C4 share predicts MC score (r about 0.95) and hurts generation (GEN). Code share drives GEN. With code at 0%,
  HumanEval/LCB collapse. Best GEN mix was math+code, no C4.
- Results: Qwen3-30B at 96/128 experts: GEN 69.8 (REAM) vs 68.6 (REAP) vs 70.9 uncompressed. At 25% on
  Qwen3-80B/Coder-Next/GLM-4.5-Air, REAM beats REAP by +1.9 to +2.0 GEN points (Table 2). A 50% result is
  reported but I did not extract the numbers.
- Ablation: dropping REAP saliency for frequency costs -8.7 points, the largest single drop. Pseudo-pruning
  matters (-3.6). Sequential merging helps a little (-1.0).
- Cost: about 1.5h on one ~30GB GPU for a 30B model. No fine-tuning.

### 2402.14800 NAEE: Not All Experts are Equal [skim, priority 2]
- One-shot expert pruning for Mixtral 8x7B. For each layer it enumerates all C(n, r) expert subsets and
  picks the one with the least Frobenius reconstruction error against cached layer outputs. Feasible for
  8 experts, not for 128.
- Calibration: 128 x 2048 tokens of C4. Anything from 64 to 256 sequences works. A single sequence is worse.
- Pruning 2 of 8 experts costs -2.9 points average. Pruning 4 of 8 costs -7.1 points (67.6 -> 59.6 on 8 zero-shot
  tasks). It beats Wanda 2:4 at equal memory (59.6 vs 57.5) and is about 1.27x faster.
- Domain-specific calibration matters. C4-pruned r=6 drops GSM8K 58.6 -> 41.0 until the calibration set is
  switched to MATH. Fine-tuning after pruning (MetaMathQA, 900 steps, lr 2e-5) recovers to about the unpruned
  level. A 7-expert pruned model beats the 8-expert model on GSM8K.
- Layer-wise (independent) pruning beats progressive (layer-aware) at high pruning rate. The authors suspect
  calibration overfit.
- Dynamic skipping of the second expert when `w2 < beta*w1` (beta is the per-layer median ratio) adds about 1.05-1.1x speed for about -1 point.

### 2310.01334 MC-SMoE: Merge, then Compress [skim, priority 2]
- M-SMoE: align expert neurons by permutation (weight matching), pick "dominant" experts by routed frequency
  (global across layers), assign the others to the nearest dominant expert by router-logit cosine, then merge
  with a frequency-weighted average. MC adds low-rank + structured-sparse compression of merged experts.
- Results on Switch-base-32: 60% memory reduction with no loss. MC-SMoE gets 80% memory and 20% fewer FLOPs
  with <1% loss on most tasks. Fine-tuning plus KD after merging is used in these results.
- Frequency-weighted merging beats uniform and Fisher-weighted averaging.
- Caveat: it was tested on small fine-tuned Switch/fairseq MoEs with different training, not generative LLMs.
  The first MoE layer is skipped because it is very sensitive.

### 2506.23266 Sub-MoE [abstract, priority 3]
- K-means clusters experts by output cosine, then does a joint SVD over each cluster to share the U matrix and
  merges the V parts by frequency. Claims to beat both pruning and merging on Mixtral, DeepSeek and Qwen MoEs.
  The abstract is truncated, so I do not know the compression ratios. It was not compared against REAP here.

### 2407.14679 Minitron: Compact LMs via Pruning and Knowledge Distillation (NVIDIA) [full, priority 1]
- Dense structured pruning (MLP neurons, heads, embedding channels, layers) followed by distillation.
  Importance is activation-based, forward-pass only, from 1,024 calibration samples:
  neuron importance = sum over batch (L2 norm) and sequence (mean) of MLP activations (best aggregation was
  batch=L2, seq=mean). Layers are ranked by perplexity change or block-importance (cosine between layer
  input and output).
- Best practices from the paper:
  1. Prune the largest model and distill into smaller ones.
  2. Use one-shot importance estimation (iterating importance estimation gave identical final loss).
  3. Width pruning beats depth pruning, but only after retraining (depth looks better right after pruning).
  4. Retrain with distillation only (no CE loss). Forward KL on logits beat reverse KL, MSE and cosine.
     Intermediate-state losses help only if depth is cut a lot.
  5. For big compression ratios, prune in steps (15B -> 8B -> 4B beat one shot by +12% MMLU),
     keeping the original as teacher.
  6. Prune the final-stage checkpoint (post data-annealing).
- Token budgets:
  - 1.8B tokens (400 steps) is "lightweight retraining" and is enough to rank candidate architectures
    (rankings stabilize by about 300 steps).
  - 18.9B tokens: Minitron-8B MMLU 58.3.
  - 94B tokens: MMLU 62.8 for 8B. 100B tokens for the 4B.
  - Compared with training from scratch: 40x fewer tokens, MMLU +16%, 1.8x compute saving for the family.
- Iso-compute 4B ablation (Table 11): pruned+distilled 37.8 MMLU vs random-init 24.4 (chance) at 150B vs 400B.
- Optimizer: cosine LR 2e-4 -> 4.5e-7, batch about 4.5M tokens (1.8B tokens / 400 steps), Megatron.
- Caveat: dense models 4-15B, no MoE, no RL-trained teacher, all pretraining data.

### 2310.06694 Sheared LLaMA [skim, priority 2]
- Targeted structured pruning with learned masks and a Lagrangian to hit a given target architecture,
  plus dynamic batch loading (re-weight data domains by their loss gap to scaling-law references).
- LLaMA2-7B -> 1.3B/2.7B using 0.4B pruning tokens + 50B continued-pretraining tokens (3% of the compute
  of training from scratch; beats Pythia/INCITE/OpenLLaMA).
- Spending more of a fixed 5B budget on pruning helps (PPL 7.46 -> 7.08 from 0.2B to 1.6B pruning tokens).
- Pruning keeps code (GitHub) knowledge better than C4 knowledge, so recovery is domain-imbalanced. Dynamic
  batch loading fixes that.
- Pythia-410M -> 160M: 10B tokens roughly matches the original 160M trained on 300B tokens.

### 2305.11627 LLM-Pruner [abstract, priority 3]
- Gradient-based structural pruning of coupled structures. Recovery via LoRA in about 3 hours on 50K samples.
  Relevant as proof that LoRA healing after structural pruning is cheap, though Sheared-LLaMA reports lower loss and
  better architectures at equal tokens.

### 2301.00774 SparseGPT, 2306.11695 Wanda [abstract, priority 2]
- Unstructured / 2:4 one-shot weight pruning (SparseGPT: second-order reconstruction; Wanda: |W| x ||X||).
  Hardware-bound (no speedup on Apple silicon), and NAEE showed expert pruning beats Wanda 2:4 at equal memory on Mixtral.
  Not useful as our primary adult-shrink operator. Wanda's `|w| * ||x||` is the same idea as REAP at the weight level.

### 1803.03635 Lottery Ticket Hypothesis [full, priority 1]
- Train, prune the smallest-magnitude p% weights, reset survivors to the *initial* weights, retrain. Iterative
  pruning (20% per round on Lenet/Conv nets; 10-15% for conv layers) finds "winning tickets" at 10-20% of the original
  size (down to about 3.6% on Lenet). Gentle rounds (10-20%) beat large rounds. One-shot pruning also finds tickets, but at larger sizes.
- Reinit with random weights kills the effect. For deeper nets (VGG-19, ResNet-18), needs LR warmup
  (later work: rewinding to an early iteration instead of init).
- Caveat: unstructured pruning of small vision nets. No structured or LLM evidence here. Takeaway for us is
  methodological (iterate small steps, keep inherited weights, never reinit survivors), not a recipe.

---

## Group 09: distillation

### 1503.02531 Hinton et al., Distilling the Knowledge [abstract, priority 2]
- Soft-target (temperature) distillation of an ensemble into one model; also "specialist" models trained fast
  and in parallel, unlike a mixture of experts. Background only. Minitron used KL on logits without
  a temperature discussion in the pages I read.

### 2306.08543 MiniLLM [skim, priority 2]
- Reverse KL, `KL[q_student || p_teacher]`, optimized with policy gradient on student-sampled responses,
  plus three stabilizers: single-step decomposition, teacher-mixed sampling (alpha=0.2), length normalization;
  plus a pretraining-LM loss to preserve base ability. Ablating length norm collapses validation Rouge-L
  27.4 -> 17.4; teacher-mixed sampling matters too (22.3).
- Setup: students 120M-13B, teacher 1.5-13B, instruction following (Dolly 12.5K examples), 5,000 steps, lr 5e-6, batch 64,
  256 sampled responses per rollout, 4 inner epochs, max length 512. LLaMA-13B -> 7B took under 10 hours on 16 V100s.
- Caveat for us: this is on-policy and costs sampling. Minitron found forward KL best when retraining pruned base models.
  The two disagree on KL direction because of different regimes (SFT/instruction vs pretraining-style).

---

## Group 07: growth (gestation)

### 2409.12903 HyperCloning (Apple) [full, priority 1]
- Function-preserving width expansion of a *dense* model: the hidden size is multiplied by n and every
  hidden vector becomes n stacked copies. Linear layers: `W_D = [[W/n ...]]` blocks (symmetric form), bias repeated,
  LayerNorm/RMSNorm params repeated, positional embeddings repeated. Attention query scaling by
  `sqrt(d_S/d_D)` if head dim grows. Depth, vocabulary, number of layers are unchanged. Output logits are identical to the source model.
- Variants: symmetric > noisy symmetric (10 dB SNR) > noisy diagonal > diagonal. Noise adds little over
  noise-free symmetric.
- Results: OPT-350M->1.3B, Pythia-460M->1.4B, OLMo-1B->2.9B. 2.2x-4x faster to reach random-init accuracy and
  better final accuracy. Training budget was large (hundreds of billions of tokens on 64 GPUs).
- Observations: initial catastrophic forgetting at the start of training; cloned weights start rank-deficient
  (half the singular values are zero) but become full-rank after training. A better source checkpoint gives a better target, with
  diminishing gap over training.
- Not for MoE as published (dense width only). The relevant ideas are function-preserving init and symmetry breaking.

### 2310.07707 MatFormer [full, priority 1]
- Nested FFN: the first m_i hidden neurons form sub-model i (m = d_ff/8, /4, /2, d_ff; g=4). Each training
  step samples one granularity uniformly. Attention is shared. Optional nesting on heads or d_model.
- Mix'n'Match: choose a different granularity per layer to get hundreds of sub-models at no extra training. Heuristic:
  non-decreasing granularity with minimal slope across layers. As good as evolutionary NAS.
- 850M MatLM, 80B tokens, 256 TPU-v3: sub-models (582M-850M) beat independently trained baselines in loss and 1-shot accuracy
  and are highly consistent with the full model (useful for speculative decoding).
- Cost: wall-clock equals training all four baselines. You cannot get this free from a model not trained with nesting.
- For us: it needs nested training (continued training with random granularity sampling). It does not give a cheap
  adolescent from the already-trained Qwen3-0.6B.

### 2405.15319 Stacking Your Transformers [skim, priority 2]
- Systematic comparison of 4 growth operators x width/depth. Depthwise copy-stacking (G_stack) wins:
  1.1B +49%, 3B +54.5%, 7B +54.6% speedup (300B tokens). Widthwise operators (copy, zero, random, learned)
  gave no notable advantage over scratch (-0.7% to -17.9% speedup).
- Guideline: growth timing d is a fraction of the budget (e.g. 10B tokens for the small model), growth factor g=4.
  A random small model stacked (d=0) behaves like scratch, so the benefit comes from the trained small model, not the stacking itself.
- Caveat: pretraining-from-scratch regime only. It says width growth is weak, which is a warning for expert-count growth
  (but our extra experts come from trained parents, not copies).

### 1511.05641 Net2Net, 2110.07143 bert2BERT, 2303.00980 LiGO [abstract, priority 2-3]
- Net2Net: function-preserving wider/deeper (split neurons, identity layers). bert2BERT: extends it to Transformers,
  saves about 45-47% compute reusing half-size models. LiGO: learned linear growth operator (width x depth, Kronecker
  factorized), up to 50% savings. All need extensive pretraining budgets to show gains.

---

## Group 10: critical periods

### 1711.08856 Critical Learning Periods in Deep Networks (Achille, Rovere, Soatto) [full, priority 1]
- A temporary low-level deficit (blur: downsample to 8x8 then upsample) in the first epochs permanently hurts final
  accuracy even after 160-300 more clean epochs. If blur is not removed within the first 40-60 epochs, test error
  rises from about 6-8% to over 18%. Sensitivity to a 40-epoch window peaks around epoch 30, in the early rapid-learning
  phase, and nearly vanishes later.
- High-level deficits (vertical flip, label permutation) show no critical period. White noise is less damaging than blur.
  Deeper networks have stronger critical periods. Holds for CNNs, ResNet-18, fully connected nets (MNIST), SGD and Adam,
  fixed LR. Higher weight decay stretches and flattens the critical period.
- Fisher Information trace of the weights rises quickly, then falls (consolidation) while accuracy still improves. Sensitivity tracks
  the Fisher trace. After the peak, the network loses "information plasticity" (the layers can no longer reorganize).
  This is compared to synaptic density in macaque V1 (rise then pruning).
- Pretraining can hurt: pretraining on slightly blurred data can be worse than starting fresh.

### 2308.12221 Critical Periods Emerge Even in Deep Linear Networks [skim, priority 2]
- Two analytic settings. (1) Multi-pathway deep linear net: a temporary gating deficit on pathway B early leads pathway A to
  learn all features (winner-take-all). A mid-training deficit only affects features not yet learned. Late deficit: no effect.
  Competition is stronger with more depth. (2) Matrix completion: pretraining on a higher-rank task then switching to low-rank
  leaves a worse solution, strongest in deeper nets and low-data regimes. Critical periods depend on depth and data structure,
  not on optimizer or biology.

---

## Implications for vibe-husbandry

### Prioritized recommendations

1. **Implement REAP-style `prune(child, n_experts)` as the default adult-shrink operator, with its sibling baselines
   computed from the same pass.** One calibration pass yields REAP, EAN (`sum ||f||`), frequency, and the activations
   needed for REAM/M-SMoE grouping. Calibration is about 2M tokens (1,024 x 2,048); NAEE shows 64-128 sequences
   already suffice. Cost ESTIMATE: 2M tokens x about 1.2 GFLOP/token = about 2.5 PFLOP, so minutes on the Mac.
   The calibration mix must match the child's target tasks (math + code, not C4 only). Both REAP and REAM
   report collapse on tasks missing from calibration. Use something like 0.4 code / 0.4 math / 0.2 general.
2. **Treat the pruning budget as the main experimental variable, not a settled recipe.** The literature supports
   the *pruning > merging* claim only for high-granularity (>=64 experts) generation settings. At 4-8 experts the
   story is open (M-SMoE won on Mixtral-class code; NAEE lost 7 points averaged at 8 -> 4). Run the matrix:
   REAP vs EAN vs frequency vs M-SMoE/REAM-style merge vs homologous-pair average (see item 4), each with and
   without healing.
3. **Heal with logit-only forward-KL distillation from the unpruned child (Minitron recipe).** No CE term.
   Teacher is the frozen 8-expert child (about 4.9 GB bf16, fits next to the student). Data: the RL prompt/solution
   distribution plus a small pretraining slice. Expected token budget is the biggest unknown: Minitron used 1.8B tokens just to
   rank candidates and 94B for final models; Sheared-LLaMA used 50B; NAEE used 900 SFT steps. On a 32 GB Mac,
   ESTIMATE of 2-4K training tokens/s for a 1.4B student means 100M tokens is about 8-14 hours and 1B tokens is
   about 3-6 days. So plan for a sweep at {0, 10M, 100M, 500M} tokens and report tokens-to-recover as a cost metric.
   Watch memory: full-vocab logits (151,936 x seq x batch) for teacher and student need chunked loss computation.
4. **Exploit the species rule: the two parents' experts are homologous.** Slot i of mother and father descend
   from the same upcycled expert, so the alignment step that makes merging hard (M-SMoE/REAM permutation matching)
   is nearly free. Two natural adult-shrink operators, both going through `prune`:
   - *Selection*: keep one allele per locus (or let REAP choose, with a "one per locus" constraint as the ablation).
   - *Averaging*: merge each pair (this is the naive weight-average/TIES baseline recast per-expert).
   If selection wins, the genome interface wants per-locus provenance tracking.
5. **Add matched-size baselines that sidestep expert pruning**: (a) quantize the 8-expert child to 4-bit at about the
   same memory as the bf16 4-expert adult (REAP's appendix: 4-bit AWQ at half the checkpoint beats 50% REAP at
   bf16); (b) Minitron-style neuron (width) pruning of *every* expert to 50% width, which keeps all 8 experts and
   the router intact (SPECULATION: this preserves REAP's "independent router control" argument); (c) train the
   4-expert parent longer for the same total token count.
6. **Gestation schedule informed by critical periods (SPECULATION, mapped from CNN and linear-net evidence).**
   The trunk is already past its low-level critical period (inherited from the pretrained ancestor), but the
   *router* and the cross-parent expert mix are new and effectively start at epoch 0:
   - Phase A (first ~5-10% of gestation, router warm-up): keep the routing balanced across parents' alleles
     (load-balance loss, noise on router, or equalized gradient), keep data low-variance, and avoid format or tokenizer shifts.
     The deep-linear paper's winner-take-all result predicts that if one parent's experts win early traffic, the
     other's will never recover. Monitor the share of tokens routed to mother vs father experts.
   - Phase B (RL on math+code mix): high-level changes such as reward changes, task mix, and curriculum order are
     plastic (flip/label-permutation had no critical period), so ordering of tasks is low risk.
   - Phase C (adolescence, prune): prune *after* consolidation. Use a measurable trigger: REAP scores
     rank-stable between checkpoints (Spearman > 0.9, my threshold guess), router entropy and parent-share
     plateaued, Fisher-trace proxy (grad-norm of log-likelihood on held-out data) past its peak. Minitron found candidate
     rankings stabilize after about 300 retrain steps, which is an analogous signal.
   - Do not heavily regularize early: higher weight decay lengthened and flattened critical periods
     (Achille), i.e. it keeps plasticity open longer, which may be desirable for the router but costs speed.
7. **Prune gently, in steps, if the single-step result is bad.** Lottery-ticket (10-20% per round),
   Minitron (15B -> 8B -> 4B beats one shot by +12% MMLU) and the Mac budget all suggest 8 -> 6 -> 4 with a short
   distillation between. Counter-evidence: NAEE found layer-aware progressive pruning worse at high prune
   rates (calibration overfit), and Minitron found iterating importance estimation within a dimension gives no benefit.
   Treat as an ablation, 8->4 one-shot vs 8->6->4.
8. **Adolescence as a free intermediate model.** MatFormer-style nesting requires nested training from the start,
   so it is not free for the Qwen3 ancestor. Cheaper option: the REAP ranking itself gives a nested family
   (top-6, top-4 experts per layer) at zero training cost. Mix'n'Match's rule says to keep per-layer expert counts smooth
   (non-decreasing) rather than skewed. Treat "adolescent" as a checkpoint with 6 experts rather than a separate architecture.
9. **Skip HyperCloning/Net2Net/LiGO for v0.** They grow the width of a dense model with function preservation.
   Our child grows by *adding experts from another parent*, so function preservation is not available
   (only an approximation: initialize the father's router rows from the mother's homologous rows, plus noise).
   Stacking's finding that width growth gave no speedup is a weak warning but not directly applicable.
   Revisit if the project wants to grow depth or hidden size across generations.
10. **Quantization as maturation.** Nothing in these papers tests quantize-then-train for MoE. What we do know:
    REAP prunes FP8 and W4A16 checkpoints directly, with no re-quantization, while merging complicates group scales,
    so *prune by selection after quantization* is compatible. QLoRA (not read; from general knowledge) trains LoRA on a
    frozen 4-bit base and would shrink the child's gestation memory (the 8-expert base would be about 1.2-1.3 GB
    at 4-bit). Candidate order: gestate with LoRA on a 4-bit child, prune, then re-quantize or keep quantized.
    Needs a direct test.

### Pseudo-code: `prune(child, n_experts)` with REAP saliency

```python
def prune(child, n_keep, calib_tokens, mode="reap", per_locus=False, heal_tokens=0):
    """Genome op. Returns adult genome + cost report (flops, peak_mem, tokens_used)."""
    cost = CostMeter()
    S = {l: zeros(n_experts[l]) for l in moe_layers}      # saliency numerators
    N = {l: zeros(n_experts[l]) for l in moe_layers}      # token counts (only where active)
    with hooks(child) as h:                                # h captures, per MoE layer:
        for batch in calib_tokens:                         #   topk_idx [T,k], gate [T,k], expert_out [T,k,d]
            child.forward(batch)
            for l in moe_layers:
                idx, g, out = h.topk_idx[l], h.gate[l], h.expert_out[l]
                score = g * out.norm(dim=-1)              # g_j(x) * ||f_j(x)||_2   (REAP)
                for j in range(n_experts[l]):
                    m = (idx == j)
                    S[l][j] += score[m].sum();  N[l][j] += m.sum()
    cost.add_flops(2 * active_params(child) * calib_tokens.numel())
    # NOTE: g must be the same gate value used in the forward pass (after any top-k renormalization).
    keep = {}
    for l in moe_layers:
        sal = S[l] / N[l].clamp(min=1)                    # conditional mean, frequency-agnostic
        if per_locus:                                     # one allele per locus (mother_i vs father_i)
            keep[l] = [argmax(sal[a], sal[b]) for (a, b) in homologous_pairs[l]]
        else:
            keep[l] = topk(sal, n_keep).indices          # uniform count per layer, as in REAP
        child.experts[l] = [child.experts[l][j] for j in keep[l]]
        child.router[l].weight = child.router[l].weight[keep[l]]   # delete rows, do not merge
    adult = child
    if heal_tokens:
        adult = distill(student=adult, teacher=child_frozen, loss="forward_kl_logits",
                        tokens=heal_tokens, data=rl_prompts_mix, cost=cost)
    return adult, cost.report()
```

Variants to add behind the same signature: `mode="ean"` (`sum ||f||`, no gate), `mode="freq"`, `mode="ream"`
(centroids = top-S, others absorbed by saliency-weighted average after neuron alignment), `mode="pair_avg"`.

---

## Open questions for the humans

1. **What is the invariant for "adult size"?** Same expert count as the parents (4 per layer), same bytes (allowing a
   4-bit 8-expert adult), or same active parameters? This decides whether quantization is a fair baseline or an alternative to pruning.
2. **Is healing distillation allowed in the budget,** and how is its cost compared against gestation tokens? REAP and
   REAM are one-shot with no retraining; Minitron and Sheared need orders of magnitude more tokens than we can afford.
   We need a token budget per stage to make the compute-matched baselines meaningful.
3. **Do we need the pruned model to retain *both* parents' capabilities** (math + code), or just the best child score?
   That determines the calibration mix, and whether the prune step may narrow the genome (every
   calibration-driven prune biases the next generation toward the tasks it was calibrated on).
4. **Architecture details of the upcycled MoE.** Top-k, shared experts, whether the FFN was split or copied (Drop-Upcycling
   re-initializes part of each expert), router init. These change how redundant the experts are and whether REAP's 128-expert findings transfer at all.
5. **Must the adolescent be a usable model,** or is it just a checkpoint? If it must be usable at several sizes,
   we need nested training (MatFormer-like) from the start of gestation; otherwise REAP-ordered prefixes suffice.
6. **Teacher-forced vs on-policy distillation.** Minitron (forward KL, pretraining-style data) and MiniLLM
   (reverse KL, on-policy) disagree. Which preserves RL-acquired behavior? Needs a pilot on the actual child.
7. **MLX tooling.** Is there existing KD/logit-distillation support in mlx-lm-lora, or do we write it? The memory
   cost of full-vocab logits is the main practical risk.

## Red flags (summary)
- REAP's headline results are on 21B-1T models with 64-384 experts. Our 4-8 expert MoE is the regime where
  REAP's own paper shows merging can be competitive. Treat the 50% pruning claim as unverified for us.
- Calibration set is an implicit prior on what survives. A code-free or C4-only calibration wipes out code ability.
- Pruning evidence is mostly one-shot with *no* fine-tuning and judged by multiple choice or short code evals. MC scores
  and generation quality disagree (REAM, REAP). Always report generation benchmarks.
- Minitron/Sheared token budgets are 2-3 orders of magnitude above what the Mac can do.
- Critical-period evidence is from small CNNs and deep linear nets, trained from scratch. Mapping to a pretrained
  MoE child is analogy only.
- HyperCloning gains were measured with huge training budgets, and the method is for dense width, not MoE.
- I read the abstract only for Net2Net, bert2BERT, LiGO, SparseGPT, Wanda, LLM-Pruner, Sub-MoE and Hinton. Details above
  for those are from the abstracts.
