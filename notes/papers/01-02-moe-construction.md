# Groups 01 (MoE foundations) and 02 (building an MoE from parts): reading notes

Scope: 16 papers from `papers.tsv`. Priority-1 papers (Mixtral, OLMoE, Sparse Upcycling, BTX,
Drop-Upcycling, MergeME) were read in detail; priority 2-3 for method, key numbers and pitfalls.
All numbers below were taken from the paper text (via alphaXiv) unless marked **[mine]** (my own
arithmetic or extrapolation) or **[memory]** (not verified in this pass).

Our setting for scaling arguments: ancestor Qwen3-0.6B (28 layers, tied embeddings, QK-Norm;
pretrained on 36T tokens per the Qwen3 report), 4-expert upcycle = 1.39B params, 8-expert child =
2.45B (README table), throughput assumed 2-5k tok/s, i.e. **[mine]** 1M tokens = 3-8 min,
100M tokens = 6-14 h, 1B tokens = 2.3-5.8 days, one day = 170-430M tokens.

---

## Group 01: MoE foundations

### 2401.04088, Mixtral of Experts (P1)
- 8 SwiGLU experts per layer, top-2, softmax over the top-2 logits only (topK-then-softmax), every FFN
  layer is MoE. 47B total, 13B active. Initialized/trained details and token counts are not disclosed
  (reportedly upcycled from Mistral 7B; OLMoE says so, Mixtral does not).
- Instruct = SFT then DPO (MT-Bench 8.30). Nothing on router init or balancing losses.
- Routing analysis on The Pile (layers 0/15/31): **no domain specialization.** Expert histograms for
  ArXiv, PubMed, PhilPapers are near-identical; only DM Mathematics differs slightly.
- What the router does track is **syntax/token identity and position**: "self" in Python, indentation
  tokens, "Question" go to the same expert. Consecutive tokens share an expert far above chance:
  first-choice repeat is 14.0/27.9/22.7% (ArXiv, layers 0/15/31) vs 12.5% random; either-choice repeat
  62% at layer 15 vs 46% random.
- Takeaway: if our "math parent" and "code parent" are plain Mixtral-style experts, do not expect
  expert k of the math parent to light up on math. Heritability has to be measured, not assumed.

### 2409.02060, OLMoE (P1)
- 1.3B active / 6.9B total, 64 experts, top-8 (fine-grained), dropless token-choice, no shared expert,
  5.1T tokens. LBL weight 0.01, router z-loss 0.001, trunc-normal init std 0.02, QK-Norm, AdamW
  4e-4 peak, 4M-token batches. Data: DCLM-Baseline (bulk), StarCoder, peS2o, arXiv, OpenWebMath,
  Algebraic Stack, Wikipedia.
- MoE vs dense, same active params, 130B tokens: MoE reaches dense quality with ~3x fewer tokens, ~2x
  less wall-clock (23.6k vs 37.5k tok/s/GPU, memory/dispatch overhead).
- Granularity: 8 experts/top-1 -> 32/top-4 gives ~+10% HellaSwag/MMLU at 130B tokens; 64/top-8 adds
  only 1-2% more. Shared expert: slightly *worse* than all-routed at equal compute. Token choice
  beats expert choice per token (EC is ~20% faster but awkward for autoregressive decoding).
- **Sparse upcycling result is a warning:** upcycling OLMo-1B (already 2T tokens) into 8 experts/top-2
  and training 610B more tokens, a from-scratch MoE catches up at ~500B tokens (= only 25% of the dense
  budget, vs 120% claimed by Komatsuzaki et al.) and wins after ~600B. They blame a heavily overtrained
  dense model plus suboptimal inherited hyperparameters (no QK-Norm, non-truncated init). Adding noise
  to upcycled weights did not help (Appendix F).
- **No LBL = collapse:** in the first MoE layer, with no balancing loss, nearly all tokens go to one
  expert, a second gets some, the rest stay flat ("dead weights"). LBL helps training loss within a few
  B tokens. LBL during SFT/DPO *hurts* slightly (53.8 vs 52.8 after SFT; 57.7 vs 57.1 after DPO), and
  expert load barely changes during adaptation.
- **Router saturation:** after 1% of pretraining (5k steps, 20B tokens) ~60% of top-8 routing already
  matches the final model; ~80% by 40%. Layer 0 saturates slowest (DeepSeek also skips MoE in layer 0).
  Early routing decisions stick.
- Specialization: OLMoE-from-scratch shows **strong domain specialization** (e.g. a layer-0 expert is
  ~100% on arXiv; GitHub and arXiv share layer-7 experts) and strong **vocabulary specialization**
  (later layers specialize on the *predicted output* token; one expert ~90% on non-Latin script
  tokens, others on geography, connectors like "Then/Therefore", measurement units). Low
  co-activation, i.e. low redundancy. Mixtral (upcycled) is near-uniform by comparison; they
  hypothesize that **dense initialization limits specialization**.

### 1701.06538, Shazeer et al. 2017, Sparsely-Gated MoE (P2)
- Noisy top-k gating; importance loss + load loss (CV^2 of expert importance/load), weights 0.1 (LM) or
  0.01 (MT). Gating weights initialised to zero to get an even start.
- Without balancing losses: test ppl 39.8, max-to-mean expert load 17.8x. With either loss: ~35.6-35.7,
  load ~1.1-1.5x. Imbalance is "self-reinforcing": favoured experts train faster and get picked more.
- Up to 137B params, 65536 experts best on 100B-word corpus, 131072 degrades ("too much sparsity").
- Experts specialize on syntax/semantics (e.g. one fires for "plays a core/critical/leading role").

### 2006.16668, GShard (P3)
- Top-2 gating, expert capacity O(N/E), overflow tokens pass through the residual, aux loss
  (1/E) sum c_e/S * m_e (non-differentiable count x differentiable mean gate), random dispatch of the
  2nd expert with prob. proportional to its gate. fp32 throughout; 1T-param bf16 run hit
  numerical-stability issues. Deeper models are more sample efficient. Background only.

### 2101.03961, Switch Transformer (P2)
- Top-1 routing, aux loss alpha * N * sum f_i P_i with alpha = 1e-2 (swept 1e-1..1e-5; 1e-2 balances
  quickly without hurting CE). Capacity factor 1.0-1.25 is fine; <1% dropped tokens.
- **Stability recipe worth stealing:** compute the router in fp32 even in a bf16 model (bf16 router
  diverged: -3.78 vs -1.72 neg-log-ppl), and scale init std down 10x (0.1x init: -2.72 +/- 0.01
  vs 1.0x: -3.60 +/- 0.68 across seeds). Expert-attention layers diverge in bf16.
- 7x pretraining speedup over FLOP-matched T5-Base. Sparse -> dense distillation keeps ~30% of the
  sparse model's quality gain at up to 99% size reduction (relevant to prune + distill).

### 2401.06066, DeepSeekMoE (P2)
- Fine-grained segmentation (N experts -> mN experts of size 1/m, top-mK) plus shared experts. 16 experts
  top-2 = 120 combos; 64 experts top-8 = 4.4e9 combos.
- 2B-scale ablation at 100B tokens: GShard 2.0B -> + shared expert -> + finer segmentation, each
  improves. Disabling the shared expert: Pile loss 1.808 -> 2.414. Shared:activated-routed ratio 1:3.
  Lower redundancy among routed experts (disabling top experts hurts them more than GShard).
- Balancing: expert-level factor 0.01 at 2B, 0.001 at 16B, 0.003 at 145B. All layers except the first
  are MoE because "load balance converges slower for the first layer".
- Caveat: OLMoE and NVIDIA's upcycling study both found shared experts give no gain; Qwen3 dropped them.

### 2505.09388, Qwen3 Technical Report (P2)
- Our ancestor. Qwen3-0.6B: 28 layers, 16Q/8KV heads, tied embeddings, QK-Norm, no QKV bias, 32K ctx;
  pretrained on the same 36T-token corpus as the family (30T general + 5T reasoning/STEM + long-ctx).
  Hidden 1024 / FFN 3072 **[memory]**, consistent with README's parameter table.
- Qwen3 MoE (30B-A3B, 235B-A22B): 128 experts, top-8, **no shared experts**, **global-batch load
  balancing loss** "to encourage expert specialization" (balance over the full batch, not each
  micro-batch).
- 0.6B-Base: GSM8K 59.6 (4-shot CoT), MATH 32.4, EvalPlus 36.2, MBPP 36.6, MMLU 52.8, BBH 41.5.
  0.6B post-trained, thinking: MATH-500 77.6, LiveCodeBench v5 12.3; non-thinking MATH-500 55.2.
- Small models are post-trained by **strong-to-weak distillation** (off-policy then on-policy logit
  matching, KL), which the authors say beats RL on pass@1 *and* pass@64 at 1/10 of the GPU hours.
  The flagship's reasoning-RL stage used only 3,995 query-verifier pairs (GRPO).
- Implication: the 0.6B is already distilled/RL-polished. GRPO gains on top may be small and noisy.

### 2407.10671, Qwen2 Technical Report (P3)
- Only relevant for Qwen2-57B-A14B, **upcycled from Qwen2-7B** (+4.5T extra tokens on top of 7T):
  64 routed experts (top-8) + 8 shared; expert size 2560 vs FFN 18944. Recipe: replicate the FFN
  ceil(n*hE/hFFN) times, **shuffle parameters along the intermediate dimension** per copy, slice
  into experts, **randomly re-initialize 50% of each expert's parameters** (the origin of
  Drop-Upcycling's r = 0.5).
- NVIDIA's replication (below) found neither the shuffle nor the 50% re-init helped at 2B/110B tokens.
- Data note: relaxing the quality filter to 12T tokens did not beat the 7T set for larger models (so
  quality > volume), but Qwen2-0.5B (trained on the 12T set) still shows data scaling helps sub-billion models.

---

## Group 02: building an MoE from parts

### 2212.05055, Sparse Upcycling (P1)
- Recipe: copy every non-MoE weight; replace a *subset* of MLPs with MoE layers whose experts are exact
  copies of the dense MLP; **router randomly initialized, N(0, 0.02)**; resume training with the
  original hyperparameters/LR schedule. T5 Base/Large/XL and ViT B/L.
- Defaults: MoE in **every other layer** (6/12 for base), 32 experts, expert-choice routing capacity
  factor C = 2 (best per unit compute), top-2 + aux loss 0.01 in the decoder. Vision only: reuse
  optimizer state and renormalize combine weights (helped vision, hurt language).
- Headline numbers: ViT-B/16 +1% ImageNet 10-shot needs +58% extra compute dense vs **+13%** upcycled.
  T5-Base (Table 5): upcycling at +61% compute: C4 acc 69.78, SuperGLUE 79.23; dense at +100%: 68.53 /
  77.36; MoE-from-scratch at +61%: 69.52 / 74.45. Upcycling is the better deal at +10% to +60% of the
  original dense budget; a from-scratch MoE eventually catches up beyond ~100%.
- Ablations: randomly initialized experts (or only a subset upcycled) is much worse at limited compute.
  Small Gaussian noise on copies: no effect; large noise: hurts. More experts = bigger initial quality
  drop (32 good for >= +20% budget). Dense "depth tiling" warm start is worse than sparse upcycling.
  Gains are consistent regardless of how long the dense model was pretrained (Fig. 6).
- Data: C4 (T5), JFT-300M and ImageNet (ViT). No new data sources.

### 2208.03306, Branch-Train-Merge (P2)
- Branch k copies of a *partially trained* seed LM, train each on one domain independently, then
  ensemble outputs or **parameter-average**. Up to 1.3B per expert; 64 domains, 22.4B total params
  matches a 1.3B dense LM trained with 2.5x more compute.
- **Seed phase is mandatory for averaging:** averaging experts branched from a random init gives
  perplexities in the thousands. Best split is ~20-50% seed (train domains) / 50-80% (eval domains).
  Even a JavaScript-only seed works vs random init.
- Uniform parameter average underperforms even a dense LM (ppl 18.0 vs 14.2 at 1.3B); averaging with
  weights from the domain posterior wins (13.4). Random-split (non-domain) ensembles are *worse* than
  dense, so **partners must genuinely differ** (support for complementarity-based mate choice).
- Budget: reusing optimizer state or not "has little effect".

### 2403.07816, Branch-Train-MiX (P1)
- Pipeline: branch Llama-2-7B into math/code/Wikipedia experts (201B / 210B / 42B tokens), then build
  an MoE with each expert's **FFN as one expert** (+ the original seed as a 4th "generalist" expert),
  **average all non-FFN params** (attention, embeddings, norms), random-init router, top-2, then
  **MoE-finetune everything on a data mix for 80B tokens** (30.2% math, 40.3% code, 10.3% wiki, 19.2%
  Llama-2 pretraining data). LR 1e-4 (100 warmup, cosine to 10%), AdamW wd 0.1, batch 4M tokens, seq 4096.
  Aux LBL alpha = 0.01.
- Results (aggregate, Table 2): BTX Top-2 avg 47.9 vs dense continued training 44.5, sparse upcycling
  (data-matched) 46.3, BTM 43.4; vs Llama-2 13B 45.4. Individual experts forget badly (Llama-2 seed TriviaQA 58.5 -> math expert 37.1 /
  code expert 29.9), the MoE recovers it (57.1). BTX math 37.1 GSM8K vs
  Llemma 39.3: **close to, not above, the specialists**.
- Table 4 (MoE-ft tokens, avg score): Top-2 at 10B tokens 34.6, 40B 35.9, 80B 37.3. Router type matters
  (Switch top-1 24.7 at 10B tokens, sample-top-1 33.0, soft routing 35.8).
- **Freezing the expert FFNs during MoE-ft changes nothing** (avg 34.7 vs 34.6/34.7); the stage mainly
  trains the router and the averaged attention.
- **Splitting/blending experts destroys quality:** blending (each MoE expert = chunks from all domain
  FFNs) avg 22.2 vs 34.7; splitting each expert into 2 chunks (8 experts, top-2) 28.0, top-4 34.5.
  FFN neurons from different branches do not recombine at sub-expert granularity.
- Balancing: without LBL the Code expert is dead (HumanEval routing -> ~0) and Math dominates; with LBL
  the Code expert returns and dominates code/math tasks. Trade-off: LBL moves HumanEval 19.5 -> 27.4
  but GSM8K 34.6 -> 29.8. GSM8K routes mostly to Code and generalist experts, MATH to the Math expert.

### 2410.07524, Upcycling LLMs into MoE, NVIDIA (P2)
- Ablations on Nemotron-2B (110B tokens = 10% of pretraining) and Nemotron-4 15B (final run 1T tokens).
  Upcycling beats continued dense training by 1.1% LM loss at equal tokens; at 15B/1T: MMLU 67.6
  (E8 top-2) / 66.2 (granular E8G8T8) vs 65.3 dense, val loss 1.306 / 1.320 vs 1.377.
- **Router math for function-preserving init:** with softmax-then-topK, expert outputs are scaled
  down (probs ~1/E each, no renormalization), so the MoE != dense at step 0. topK-then-softmax is exactly
  function-preserving for identical copies, but loses router magnitude information. They found
  softmax-then-topK better in the end *if* paired with **weight scaling** of expert W1, W2 (e.g. factor
  4 for 64 experts top-8 fine-grained; also helped coarse 8-expert top-1: 1.5% lower loss).
- Fine-grained upcycling needs "virtual group" init (shard the FFN G ways, replicate E times, router
  weights shared within each group so top-k picks one replica of every shard).
- Aux loss 1e-2 to 1e-3 best; no aux loss -> dead experts and early plateau; z-loss 1e-3 no effect;
  batch up to 4M tokens fine; LR schedule matters (details on pp. 6-9 not captured here). 64 experts
  best, 256 slightly worse ("experts were all copies"). **Qwen2-style shuffle + 50% re-init: no gain.**
  Shared experts: no gain. Higher top-k helps (E8 top-2 beats top-1: 1.358 vs 1.380 loss).

### 2502.19261, Drop-Upcycling (P1)
- Method: copy the FFN to every expert, then for each expert independently sample a fraction r of
  *intermediate-dimension indices*, and re-initialize those columns of W_gate, W_up and the matching
  rows of W_down from N(mu, sigma) using the statistics of the selected weights (same index set across
  the three matrices). Router random. Attention/embeddings copied. Fraction of identical parameters
  shared by all k selected experts ~ (1-r)^k.
- Setup: Llama/Mixtral-style, 8 experts, dropless top-2; 8x152M, 8x1.5B, 8x3.7B; dense seeds trained
  1T tokens, MoE continued 500B tokens (r ablations 150B). Data: LLM-jp corpus v3 (Japanese/English/code).
- **r = 0.5 best** in loss and downstream at both scales; 0.1/0.25 look good early but lose later;
  0.75/1.0 waste the dense knowledge; r = 0 (naive upcycling) "struggled to improve significantly
  beyond the dense model" (slowest learning-curve slope). Table 1 (8x1.5B avg): FS 34.5, NU 37.4,
  RNU 39.1, DU r=0.5 **40.3**, BTX 38.6 (BTX used +300B tokens). DU 8x3.7B (5.9B active) matches a
  13B dense at ~1/4 of the FLOPs; beats 3.7B dense trained 2T tokens.
- **Crossover timing:** Drop-Upcycling overtakes the others only "after some training" in a 500B-token
  run; the advantage is described for "hundreds of billions of tokens or more".
- Naive upcycling + a *global* load-balancing loss -> collapse to two experts selected in every layer;
  layer-wise balancing mitigates this but with no loss/perf difference. DU and from-scratch models show
  domain-specialized routing (Japanese/English/code experts) in either setup.

### 2502.00997, MergeME (P1)
- Setup is BTX-like at 1B (1B base model; code/math/knowledge experts, 100B-tokens-per-expert-equivalent
  cost, then 40B tokens MoE-ft, 448 "B-param x B-token" cost vs 1258 for upcycling on 340B tokens).
- (1) Replace averaging of non-FFN params by **TIES or DARE on task vectors**: avg score 12.52 / 12.86
  vs BTX 11.72 (MoE upcycled from scratch on 340B tokens: 16.05). The gain is largest early in MoE-ft,
  and both route more tokens to domain experts (GSM8K math-expert routing 0.28 -> 0.35 / 0.46).
- (2) Training-free routing: **sequence-level perplexity routing** (softmax of top-k of 1/PPL under
  each expert model), plus **keeping each expert's attention separate** (not averaged): 8.08 vs random
  routing 6.92, vs Dare dense merge 7.11. Router-only fine-tuning of the averaged model is poor
  (7.92 vs BTX 11.72 after full MoE-ft), consistent with the averaged attention needing to adapt.
- (3) Heterogeneous experts via projectors + sequence-level router (not needed for same-species
  breeding; relevant for the Qwen x Granite stretch goal). Task-vector cosine similarity between
  math and code experts is low for attention as well as FFN (up to ~0.5), so "attention is shared
  across domains" is not guaranteed. A math expert from a different tokenizer got *less* routing than
  expected; they suggest LBL.
- Limitation: only 1B models and 3 domains; load-balance loss not used.

### 2406.12034, Self-MoE (P3)
- Experts = LoRA modules (rank 8, alpha 16, lr 3e-4, 3 epochs, batch 32, all linear layers, <0.3% of
  params each), trained on **self-generated synthetic data** (100 seeds -> 5K examples per domain; 20K
  total); one shared linear router (top-1/top-2, softmax-then-topK) trained only on the pooled data
  with experts frozen. Hours on one A100-80GB.
- Gemma-7B: avg 47.8 -> 54.3 (MMLU +7.2, BBH +5.0, GSM8K +10.0, HumanEval +3.7); beats instance
  merging (52.4), TIES (47.9), DARE (44.3). Single-domain specialization hurts other domains
  (knowledge expert: BBH 56.1 -> 41.7); the routed mixture avoids that. Works on 5 base LLMs (+6-8 pts).
  Using all experts every time is worse than top-k (noise).
- Why it matters: total adapter budget (~24M tokens) fits our hardware; the router-with-frozen-experts
  idea is the cheapest conception test.

### 2403.13257, MergeKit (P2)
- Tooling: linear/SLERP/task arithmetic/TIES/DARE/breadcrumbs; lazy out-of-core tensor loading (runs on a
  laptop) and a DAG scheduler. `mergekit-moe` builds a "FrankenMoE" from dense models with router rows
  from **prompt-based hidden-state heuristics** ("hidden"), cheap embedding init, or **random gates
  for sparse upcycling**. LGPL: study only, per CLAUDE.md. Useful as a reference for the genome IO
  layer (lazy loading is exactly what `conceive` needs at scale).

---

## Implications for vibe-husbandry

### What the evidence says about the regime we can reach
The cheapest published "MoE helps" results use tens of billions of tokens (MergeME 40B, BTX 10B
ablation / 80B main, NVIDIA 110B, Drop-Upcycling 150-500B), and the pretrained dense seeds saw 2T+
tokens. Qwen3-0.6B saw 36T tokens (~60,000 tokens/param), the most overtrained ancestor in this
literature; OLMoE explicitly says overtrained dense checkpoints leave little room for upcycling gains.
**[mine]** Scaling by tokens/param: BTX's MoE-ft is ~1.4-11 tokens per param (10-80B / 7B), i.e.
0.9B-7B tokens for us (3-60 days at 2-5k tok/s). Realistic per-child budget is **100-300M tokens
(6-25 h)**, 1-2 orders of magnitude below the smallest published setting. So:
- Do **not** expect a general-LM-loss win from 2x experts in this budget. The reachable win is
  *capability composition*: child >= each parent on the union of their specialties, without the
  forgetting seen in single-domain specialists (BTX, Self-MoE), and beating TIES/DARE merging. BTX's
  evidence says "close to specialists, far better than them elsewhere", not "better than specialists".
- Effect sizes will be small relative to seed variance for a 0.6B model; plan multiple seeds per arm.

### Prioritized recommendations
**P0, correctness of conception**
1. **Make homology explicit.** Do the upcycling (and any re-init) *once*, at the species founder, and
   branch both parents from that same MoE-4. Expert k in parent A and expert k in parent B then share an
   origin and attention/FFN parameters are in the same basin (BTM: averaging only works from a shared
   partially-trained seed). Index-matching replaces Hungarian alignment within a species; keep
   Hungarian only as a control or for cross-founder experiments. If each parent upcycled independently
   with its own random Drop-Upcycling mask, homology would be silently broken.
2. **Function-preserving init + unit tests.** Use identical expert copies and a renormalized top-k
   (topK-then-softmax, or Qwen3-style `norm_topk_prob`): output == dense output exactly, so there is no
   upcycling dip (the dip in the papers comes from unnormalized softmax weights/capacity dropping/
   re-init, not from copying). Add tests: (a) founder MoE == dense Qwen3-0.6B to float tolerance;
   (b) child with router masked to parent A's experts (+A's attention) == parent A. Child doubling
   experts with top-2 renormalization keeps the output scale; softmax-over-8-then-topK would shrink
   it (NVIDIA needed weight scaling for that).
3. **Router for 2x experts.** Test arms: (i) fresh N(0,0.02) or zero (Shazeer) router;
   (ii) concat of parents' router rows with per-parent logit offset calibrated so each parent gets ~50%
   of routing mass on a mixed calibration set (parents' logits live on different scales and top-k
   competes across parents); (iii) hidden-state-of-prompts init as in mergekit-moe; (iv) training-free
   PPL routing (MergeME) as a baseline. Then a **router-only warmup with all experts frozen** (BTX
   shows frozen experts cost nothing; OLMoE shows early routing decisions harden within ~5k steps),
   with explicit balancing, before unfreezing anything.
4. **Keep the expert as the atomic crossover unit.** BTX: mixing chunks of different branches' FFNs drops
   avg from 34.7 to 22.2 (splitting into half-size experts: 28.0). No neuron-level or sub-expert
   recombination in `conceive` until Hungarian/OT alignment is shown to fix this; keep each layer's
   experts together with their parent's layer path where possible.

**P1, gestation recipe on a 32 GB Mac**
5. **Train less than you think.** Per BTX Table 5, freezing experts is free, so train router +
   (LoRA on) attention/trunk first; add expert LoRA only for a final phase. Throughput budget
   **[mine]**: router warmup 20-50M tokens (~3-7 h); gestation mixture CPT/SFT 100-300M tokens; RL after.
   Use the BTX data mixing rule: sample every parent's domain data *plus generalist replay*
   (BTX: 19% original pretraining data) or the child forgets (single experts lost 20 pts of TriviaQA).
6. **Keep a generalist expert in each parent.** BTX's seed-model-as-expert recovers general ability.
   Let one of each parent's 4 experts stay (near) un-fine-tuned. Then pruning 8 -> 4 can keep one
   generalist + 3 specialists.
7. **Use half the layers for MoE** (Sparse Upcycling: every other layer; best 6/12 on ViT-B; DeepSeek keeps
   layer 0 dense; OLMoE shows layer 0 saturates slowest). **[mine]** MoE in 14 of 28 layers puts the
   8-expert child at ~0.6 + 7 x 0.132 = **~1.5B params** (vs 2.45B) and the founder MoE-4 at ~1.0B,
   which brings full fine-tuning of a *trunk + router* closer to the 32 GB budget. README currently
   assumes all 28 layers; consider this a v0 decision.
8. **Drop-Upcycling ratio: default r = 0 at the founder, r = 0.25 as an ablation, 0.5 as the
   long-budget arm.** The r=0.5 win needs hundreds of B tokens; at <1B tokens it just throws away
   knowledge (r >= 0.75 hurt even at 150B). Parent divergence through domain training already supplies
   diversity. NVIDIA found Qwen2's shuffle + 50% re-init gave no benefit at 110B tokens, OLMoE found
   noise gave none. If r > 0, apply the mask once at the founder (item 1).
9. **Merge the non-FFN trunk smartly, cheaply.** Compare plain average (BTX), TIES/DARE on task vectors
   (MergeME: +7-10% relative avg early in MoE-ft), and separate per-parent attention (MergeME's
   training-free best). These need no training, so run the full grid. The README's "average or
   task-arithmetic" plan is supported.
10. **Gestation objective:** try parent-as-teacher logit distillation next to GRPO. Qwen3 reports
    distillation beating RL for small models at 1/10 of the GPU hours, and the two parents are natural
    teachers. Keep GRPO as the "growing up" stage but expect small gains on a 0.6B that is already
    RL/distillation-polished (consider starting from Qwen3-0.6B-Base).

**P1, load balancing (router suddenly has 2x experts)**
11. LBL: Switch form `alpha * N * sum_i f_i P_i`, alpha = 0.01 (0.001-0.01 range works; NVIDIA says
    1e-2 to 1e-3 best); router in **fp32** (Switch); z-loss 1e-3 optional (no measured gain); smaller init
    for any new weights. Compute f_i and P_i over the **accumulated global batch** (Qwen3's global-batch
    LBL); micro-batches on a Mac are tiny and per-micro-batch balancing forces uniform use, which fights
    specialization. Drop-Upcycling saw collapse to 2 experts with a global LBL on a naive upcycle, so
    log per-layer load.
12. **Collapse monitors** to log every N steps: per-layer expert load entropy, dead-expert count,
    and the share of routing mass going to each parent's experts (the "stepchild starvation" check).
    BTX shows the router will happily park a whole expert (Code) until LBL revives it; the child's
    router may favour whichever parent's logits are larger. Alarm if one parent gets <10-15% of mass.
13. LBL trade-off is real: BTX LBL +7.9 on HumanEval, -4.8 on GSM8K. Consider LBL at warmup/CPT and **no
    LBL during RL** (OLMoE: no-LBL adaptation scored higher and load distribution did not move much).
14. Keep layer 0 dense or give it a looser balance target; avoid expert-choice routing (inference
    mismatch for autoregressive generation); no shared experts in v0 (OLMoE, NVIDIA, Qwen3 agree).

**P2, evaluation and heritability**
15. **Baselines** (CLAUDE.md demands compute-matched): (a) dense Qwen3-0.6B continued on the same token
    count and mixture; (b) sparse-upcycled 8-expert MoE trained directly on the mixture (BTX "CM");
    (c) TIES and DARE of the two dense parents; (d) BTX 4-way analogue (generalist + A + B experts);
    (e) child from a **random pair** vs a **wooed** pair (BTM: random splits are worse than dense, so
    complementarity should show a measurable effect); (f) parents-as-teachers distilled dense 0.6B.
16. **Measure what is inherited.** Do BTX/OLMoE-style routing histograms by domain and by token, plus
    per-parent routing mass per task (BTX Table 6/Fig. 3 style). Expect token/syntax-level routing
    (Mixtral, Shazeer); domain-level routing only appears when the expert was trained on domain data in
    isolation (BTX) or when trained from scratch with fine granularity (OLMoE, Drop-Upcycling).
17. **Cheap proxy genome:** also run a LoRA-genome variant (Self-MoE: LoRA experts on a shared FFN, ~1%
    params, 20K synthetic examples, router-only training) for quick debugging of `woo/conceive/prune`
    interfaces; it fits our token budget by 1-2 orders of magnitude. The "chromosome = adapter" version
    keeps the same genome interface.
18. For prune + adult distillation, Switch's sparse -> dense distillation (~30% of gain retained at
    99% shrink) and Qwen3's on-policy KL distillation are the closest precedents.

### Red flags and tensions
- **Budget gap** (above): published gains are in a 100x larger token regime on much less overtrained seeds.
- **Overtrained ancestor:** OLMoE's upcycling result (scratch catches up after 25% of dense budget) is
  the worst case, and Qwen3-0.6B is more overtrained than OLMo-1B.
- **Experts are mostly token/syntax specialists** (Mixtral, Shazeer 2017, OLMoE vocabulary results).
  "Math parent's experts" may not carry "math" in any inspectable sense; heritability evidence will
  come from performance, not from routing maps. Domain structure shows up only with isolated
  training (BTX) or non-upcycled fine-grained MoE (OLMoE).
- **Upcycled MoEs specialize poorly** (OLMoE on Mixtral; Drop-Upcycling on naive upcycle). Our whole
  stack is upcycled. Mitigation is Drop-Upcycling, but its benefit is a long-budget effect.
- **Pruning tension (my synthesis):** the more specialized/less redundant the child's experts are
  (fine-grained, Drop-Upcycling, DeepSeekMoE "lower redundancy"), the more it hurts to prune 8 -> 4;
  the more redundant (coarse, naive upcycled), the cheaper pruning is but the less a doubled expert set
  contributed. We should measure expert redundancy (OLMoE co-activation, DeepSeekMoE disable-top-
  expert test) before and after gestation.
- **Sub-expert recombination fails** (BTX blend/split), so only whole-expert inheritance is supported by
  evidence.
- **Domain-specialist ceiling:** BTX/MergeME children approach but do not beat specialists in-domain
  (GSM8K 37.1 vs 39.3; MergeME MoE merges < from-scratch MoE upcycling 12.9 vs 16.1 at 40B vs 340B tokens).
  Success criterion should be "union of skills, no forgetting, beats TIES/DARE", not "beats the best parent".
- **Router-only fine-tuning is not enough with averaged attention** (MergeME 7.92 vs 11.72 for full
  MoE-ft), though BTX's frozen-FFN result says training the trunk + router suffices; always train some
  trunk parameters (LoRA on attention) during gestation.
- **bf16 router instability** (Switch, GShard): keep router logits/softmax in fp32 on MLX/MPS.

---

## Open questions for the humans
1. **Success criterion:** is the v0 claim "child beats TIES/dense-continuation on the union of math+code
   benchmarks" (reachable), or any general-LM-loss improvement (probably not reachable below ~10B
   tokens)? What wall-clock budget (days) per pairing and per seed is acceptable?
2. **Ancestor checkpoint:** Qwen3-0.6B (post-trained, thinking/non-thinking, distilled) or
   Qwen3-0.6B-Base? Base is more plastic and avoids thinking-mode quirks; post-trained is already
   RL-polished (little GRPO headroom).
3. **Founder structure:** confirm both parents branch from one shared upcycled MoE-4 (homologous
   experts). Is cross-founder breeding (independent upcycles) in scope for v0 (then alignment is needed)?
4. **Genome granularity:** full-FFN experts, or LoRA adapters as the chromosome for v0? And MoE in all 28
   layers (README) or every other / upper half (memory, Sparse Upcycling)?
5. **Routing spec:** top-k = 2 of 4 (doubles FFN FLOPs vs dense; active ~0.86B) or top-1 (iso-FLOP,
   but BTX/NVIDIA show top-2 clearly better)? Router normalization (topK-then-softmax, preserves
   function) vs softmax-then-topK (needs weight scaling)?
6. **Gestation objective:** is GRPO a hard requirement, or may we substitute/compare parents-as-teachers
   distillation (Qwen3 evidence)? Which rewards/verifiers and datasets (GSM8K, MATH, MBPP/HumanEval
   with unit tests) are approved?
7. **Data:** which open corpora may we pull? Suggested minimal set mirroring BTX/OLMoE: OpenWebMath +
   AlgebraicStack (math), StarCoder subset (code), DCLM/FineWeb-Edu slice for generalist replay,
   Wikipedia; SFT sets like MetaMathQA/CodeFeedback. Any licensing or disk-size limits?
8. **Adult definition:** prune 8 -> 4 with at least one expert from each parent forced to survive, or
   free selection by REAP-style saliency? How is "heritability" scored?
9. **Number of seeds / statistical power:** at 0.6B, differences will be a few points; are 3+ seeds per
   arm affordable, or should we pre-register a smaller set of arms (e.g., r in {0, 0.25}, router init in
   {random, concat+calibrated})?
