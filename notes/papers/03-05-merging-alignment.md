# Notes: weight merging (03), alignment/meiosis (04), cross-architecture (05)

Scope: groups 03, 04, 05 of `papers.tsv`. Priority-1 papers were read in method and results; priority 2-3 from abstract plus key result.
Convention: "(paper)" = stated in the paper; "(inference)" = my extrapolation to vibe-husbandry, not tested by the paper; "(memory)" = from my background knowledge, verify before relying on it.

---

## Group 03: weight-space merging

### 2212.04089 Editing Models with Task Arithmetic (P1)
- Task vector tau = theta_ft - theta_pre. Merge = theta_pre + lambda * sum(tau_i). Negation (lambda<0) removes a skill; addition builds multi-task models.
- Averaging two fine-tunes equals task arithmetic with lambda = 0.5 (paper, footnote). So "plain average" is the lambda=0.5 special case, not a different family.
- lambda is picked on held-out data over a grid {0, 0.05, ..., 1.0} with a single shared lambda. Optimal lambda varies a lot per experiment. Roughly 0.3-0.5 was near-optimal on CLIP with 2-8 task vectors. The recommended no-validation default is 0.4 (also quoted by TIES).
- Adding two vectors keeps 98.9% of fine-tuned accuracy (CLIP). With 8 tasks, normalized accuracy tops out near 0.91 versus 0.994 for joint multi-task training. The gap grows with the number of tasks.
- Accuracy of the weight-averaged model tracks accuracy of the logit ensemble (Pearson 0.99). Different fine-tuning seeds change little.
- Larger models edit better. The best subset of task vectors is often not "all of them".
- Needs a shared pretrained init and the same architecture (the paper says so; it merges only shared weights when heads differ).

### 2306.01708 TIES-Merging (P1)
- Three steps on task vectors: (1) Trim: keep the top-k% by magnitude per task vector, zero the rest. (2) Elect sign: per parameter, take the sign of the summed trimmed values (total mass). (3) Disjoint mean: average only the entries that agree with the elected sign, ignoring zeros. Then theta = theta_init + lambda * tau_m.
- No-validation recipe (paper): **k = 20% kept, lambda = 1** (because it takes a mean, not a sum). Search found lambda 0.9-1.1 equivalent. Range tested for lambda: 0.8-1.8; k in {10, 20, 30}.
- Results, fixed-hyperparameter recipe (no validation set), T5-base / T5-large / ViT-B/32 / ViT-L/14: TIES 69.7 / 74.4 / 72.4 / 86.0, task arithmetic (lambda 0.4) 73.2 / 73.5 / 60.4 / 83.3, averaging 65.9 / 59.6 / 65.8 / 79.6. So **task arithmetic beat TIES on T5-base (73.2 vs 69.7)** and TIES wins elsewhere. With a validation set TIES leads by 0.7-3.6 points everywhere (e.g. T5-base 73.9 vs 73.2). Out-of-domain: TIES beats the best baseline by 1.0 (T5-base) and 4.4 (T5-large) points. These are 7-11-task merges, not 2.
- Ablation: dropping scaling or the disjoint mean costs the most (-2.5 and -1.9 on T5-base). Trim and elect each cost about 1.4.
- Key mechanism claims: only about 20% of the delta matters (top-20% keeps full task performance). Sign conflicts survive trimming and **grow with the number of models merged**, even for same-task models. Flipping the signs of the top-20% parameters is catastrophic; flipping the bottom 80% is harmless.
- Oracle sign vector (from a multitask model) gets 72.0 vs 66.4 for TIES. That shows how much sign resolution leaves on the table.
- Limitations stated: relies on a common init and architecture; still below multitask training. (Inference: with only two parents, sign conflicts are rarer, so the TIES edge over averaging should be smaller than in these 7-11-task settings. The paper's Fig. 13 shows the methods converging at 2-3 tasks.)

### 2408.07666 Model Merging in LLMs, MLLMs, and Beyond (survey, P1)
- Taxonomy: pre-merging (linearized finetuning, architecture transformation, **weight re-basin/alignment**) and during-merging (basic average / task arithmetic, weighted, sparse and low-rank subspace like TIES/DARE, optimization-based, routing-based including MoE-style dynamic merging, post-calibration).
- Cited head-to-head on Llama-3.1-8B (5 text-generation tasks; normalized accuracy, individual experts = 100): pretrained 60.0, weight avg 81.1, **task arithmetic 84.8**, Fisher 80.3, RegMean 80.6, DARE 78.7, TIES 81.4. (Read from a figure; treat as approximate.) Takeaway: on LLMs simple task arithmetic was the strongest of the classic methods. On ViTs TIES/subspace methods are better.
- "Larger models are generally easier to merge" (ViT-L vs ViT-B). The performance gap to individual experts widens as the number of merged tasks grows.
- On alignment: Jordan et al. report that matching methods depend on normalization layers (BatchNorm/LayerNorm) being present; Crisostomi et al. note pairwise permutations are not cycle-consistent, so multi-model alignment should be optimized jointly.
- Routing-based merging (SMEAR, Twin-Merging, WEMoE) is the closest analogue to a router over inherited experts. Cheap to read for the router design.
- No single method wins; choose by data availability and compute.

### 2311.03099 DARE / "Language Models are Super Mario" (P2)
- Drop a random fraction p of delta parameters, rescale the rest by 1/(1-p), then add the sparse delta to the base. Plug-in before task arithmetic / TIES. Rescaling is essential (drop-only gets much worse as p grows).
- Works because SFT deltas are tiny (typically < 0.002) and redundant: p = 0.9 is fine on 13B models; p = 0.99 only on 70B. Tolerable p grows with model size (7B and 13B fail at 0.99). Qwen3-0.6B is far smaller, so expect a lower safe p; sweep it.
- **Fails if the delta is large**: WizardCoder-Python (derived from CodeLlama, which had 500B extra tokens) vs the Llama-2 base has |delta| > 0.01, and dropping just 10% of its deltas takes HumanEval from 63.4 to 0.0.
- Merging results (Llama-2-13B family, scaling in [0.5, 1.0], TIES keep ratio in {0.5, 0.7, 0.9}): DARE helps TIES more than task arithmetic. Example: LM+Code with TIES gets HumanEval 0.0 without DARE and 18.3 with DARE, while task arithmetic is already stable without it (28.7 / 26.8 on LM+Math at HumanEval).
- Precondition stated: source models must each be well fine-tuned, from the same backbone.

### 2203.05482 Model Soups (P2)
- Averaging fine-tunes of one pretrained init with different hyperparameters often lands in one basin. Greedy soup (add a model only if held-out accuracy does not drop) beats the best single model: 81.03 vs 80.38 (CLIP ViT-B/32), 90.94 vs 90.78 (ViT-G).
- **Uniform soup can collapse** when ingredients are bad or fine-tuned with large lr: error barriers appear between solutions. On GLUE text tasks, uniform soups were far worse than the best model (BERT-large MRPC 15.8 vs 88.8; T5-large CoLA 0.0 vs 61.7) while greedy soups were safe.
- Soup accuracy tracks ensemble accuracy for moderate/small lr; high lr breaks it.
- Practical warning: tied input/output embeddings need care when averaging in place. (Qwen3-0.6B ties them, memory; verify.)

### 2111.09832 Fisher-weighted averaging (P3)
- Merge by per-parameter Fisher information weights (Laplace approximation to each posterior). Needs gradients and labeled data. TIES paper found it gave little gain with many checkpoints and cost high memory. Treat as an optional baseline only.

### 2212.09849 RegMean / Dataless Knowledge Fusion (P3)
- Closed-form per-linear-layer merge from input-activation Gram matrices (X^T X). Beat Fisher on language-model merges. TIES paper notes it needs storing statistics the size of the model. Cheap for 0.6B if we want a data-aware baseline; can be applied expert-by-expert.

### 2312.06795 Model Breadcrumbs (P3)
- Sparse task vectors with both tails removed (drop negligible and outlier large deltas), claimed to need no per-task hyperparameter retuning as tasks are added. Another sparsify-then-sum option. Low priority.

---

## Group 04: alignment ("meiosis")

### 2209.04836 Git Re-Basin (P1)
- Premise: after accounting for hidden-unit permutations, SGD solutions lie in one basin (linearly mode connected, LMC). Barrier = max_a L(interp) - mean(L(A), L(B)).
- Three matchers: **activation matching** (permutation maximizing correlation or OLS fit between activations, one pass over data, a per-layer linear assignment problem); **weight matching** (maximize inner product of weights across all layers; a sum of bilinear assignments, NP-hard, solved by layer-wise coordinate descent; needs no data; seconds to a few minutes); **straight-through estimator** (learned permutation; best but very expensive). Weight and activation matching perform similarly; weight matching is orders of magnitude faster.
- Results: zero-barrier LMC for independently trained MLPs on MNIST and for wide ResNet-20 on CIFAR-10 (first demonstration). ImageNet ResNet-50 at 1x width only reached about 51% top-1 and a 67% barrier reduction, not zero.
- **Width matters**: wider models have smaller barriers after matching; narrow models don't connect. **LMC is emergent**: models are not connected near init; it appears gradually during training. LMC is a property of SGD, not architecture (adversarial counterexample exists).
- One-pass greedy matching (OT-fusion style) did far worse than iterating to convergence on ResNet-50 (about 7% vs 51%).
- Normalization layers and residual connections are handled by the implementation; non-permutation invariances (rescaling) exist but are not handled.
- Nothing in the paper is about fine-tunes of a shared pretrained model. It is about independent training from scratch. That is the regime we are NOT in.

### 2305.03053 ZipIt! (P1)
- Merges models trained on **disjoint tasks** from different inits, without retraining. Two ideas: (1) allow merging features **within** a model as well as across (a "merge matrix" over the concatenated feature set, greedy highest-correlation pairing without replacement); (2) **partial zip**: merge only up to layer k and leave later layers as separate heads. Correlation is computed on a small unlabeled sample; about 100 images (CIFAR) or 640 images (ImageNet) were enough.
- CIFAR-10 (5+5), ResNet-20 x4: joint accuracy permutation 58.4, Git Re-Basin 46.2, **ZipIt 20/20 layers 79.1**, ZipIt 13/20 83.8, ensemble 87.4. On ImageNet 200+200, full zip is poor for everything (8.6%); partial zip 10/50 reaches 60.9 vs ensemble 63.3.
- Greedy matching is ~1.1 s versus ~11 min for optimal and loses ~0.6 points. Mean cross-model neuron correlation falls with depth (0.50 -> 0.37 -> 0.27), so late layers diverge most.
- Relevance: (a) it is the template for **pruning by merging** (merge redundant units inside one model), which fits the adolescence -> adult step; (b) it shows that when parents are specialists (math vs code), forcing one-to-one fusion of late layers hurts, so keep them separate; (c) it needs enough width to hold both skills.
- Limits: CNNs only; no transformers.

### 2003.10306 Safe Crossover of Neural Networks Through Neuron Alignment (P1)
- Names the **competing conventions** problem: the same function has many weight layouts, so naive crossover of parents breaks offspring. Fix: functionally align neurons first, then recombine. The paper explicitly compares this to homologous chromosome pairing in meiosis.
- Neuron = activation vector over a batch. Two matchers: pairwise cross-correlation with bipartite matching (Hungarian) or semi-matching, and CCA/SVCCA (one pair per canonical direction, SVD to 256 directions on MNIST to avoid spurious correlations).
- Setup: 200 pairs of from-scratch MLPs (MNIST: 1x512; CIFAR-10: 3x100). Naive 50/50 crossover on CIFAR-10 drops two ~40%-accuracy parents to ~10% (chance). Aligned crossover had lower loss than naive in 193/200 (MNIST) and 200/200 (CIFAR-10) trials. PwC beat CCA on MNIST (neurons ~ single digits); CCA better on CIFAR-10 (distributed features).
- Alignment is cheap (single pass of activations plus an SVD/LAP).
- Caveats: toy MLPs from random init; large hidden layers are needed or neurons have no counterpart ("networks with small hidden layers... may not learn the same features, rendering any sort of crossover meaningless"); tested only on MLPs.
- Cites NEAT's **historical markings**: only neurons with the same origin recombine. Our shared-ancestor slot IDs play that role.

### 2602.05495 Transport and Merge (P1)
- OT-based **cross-architecture** merge: record activations of target and source LLM on a small calibration set, build a correlation cost (1 - Pearson) between channels for every (target layer, source layer) pair, solve entropic OT (Sinkhorn, eps 0.1; 0.03 for GSM8K; 200 iterations) for the feature transport Q, then a second OT for layer-to-layer mixing P.
- Fuse: W_target + alpha * mask * (sum_m P[l,m] Q_out W_source Q_in^T - W_target). Only the top-k most active neurons per layer/projection are fused (default **k = 128**); fusion strength alpha in [0,1]. Optional "residual-frozen adaptation": freeze the transported delta, train the base weights, fold back.
- Experiments: LLaMA-3 8B source into ~1B targets (Malaysian, Indonesian, Thai, Cantonese). Example: MalayMMLU Humanities 42.3 -> 46.2 (no adaptation) -> 48.8 (adapted); Indonesian average 38.4 -> 39.2 -> 39.3; Cantonese CMMLU 25.3 -> 27.4. Gains are 1-6 points, modest.
- Needs only a small input set; one-time cost O(sum n_l n'_m) per Sinkhorn iteration.
- Relevance: later (Qwen x Granite) experiment. Not needed in v0. Also a hint for how to transplant a few "most active" experts or neurons between different-width models.

### 1912.05671 Linear Mode Connectivity and the Lottery Ticket Hypothesis (P2)
- Instability analysis: branch a network at step k, train two copies with different SGD noise, check the interpolation barrier. Result: all but LeNet are unstable at init, but become **stable (linearly connected outcome) early**: ~3% of training for ResNet-20, ~20% for ResNet-50.
- After the stability point, copies stay LMC even though they end up far apart in L2 (distance between copies is a quarter to half of the whole training path length).
- Takeaway for us: two parents branched from a well-trained ancestor are far past the stability point, so LMC is expected, even though their weights are far apart. This does not apply to a re-initialized or heavily re-trained expert.

### 1910.05653 Model Fusion via Optimal Transport (OT Fusion, P2)
- Layer-wise OT alignment (activation or weight based) then averaging; handles different widths; no retraining. Greedy single pass.
- Data points: MNIST MLP with 4 models trained from the **same random init** on skewed data: vanilla average 80.6%, OT average 93.6%, ensemble 95.0%. CIFAR-10 VGG11 from different inits: vanilla average ~17%, OT ~86% before fine-tuning. Vanilla averaging "fails to retrain" for >2 different-init models, while OT then fine-tuned beats the individual models.
- Caveat: "same init" here is a random init, not a pretrained checkpoint, so it says alignment can help even without different inits when merging several diverged models. It does not contradict the pretrained fine-tune case.

### 2405.07987 Platonic Representation Hypothesis (P2)
- Representations across models converge with scale and competence ("all strong models are alike"), measured with mutual nearest-neighbor kernel alignment. Absolute alignment is low (max 0.16 on a 0-1 scale across modalities).
- Authors tried CKA first and found it gave a "very weak trend of alignment" even within a modality; they switched to mutual-kNN. (Also: Bansal et al. found stitching reveals what CKA cannot.) So CKA is a weak discriminator.
- Supports expecting same-family models to be compatible, but gives no operational merge test.

### 2106.07682 Revisiting Model Stitching (P3)
- A trainable affine layer between frozen bottom of A and top of B. Good same-architecture networks can be stitched with no accuracy drop, even across training methods; more data/width/training gives better "plug-in" representations. Argues stitching reveals what CKA cannot. Possible alternative compatibility test (stitch parent A's residual stream into parent B's experts), but it needs training a layer.

---

## Group 05: cross-architecture fusion (later-phase)

### 2401.10491 FuseLLM (P2)
- Distillation-based, not weight-merging: align tokens across tokenizers (minimum edit distance), run each source LLM to get next-token distributions, fuse them (MinCE = take the source with lowest cross-entropy on that text; better than average-weighted AvgCE), and continue-train a target LLM with L = 0.9 CLM + 0.1 fusion loss.
- Sources: Llama-2 7B, OpenLLaMA 7B, MPT 7B; target Llama-2 7B; data MiniPile (1.8B tokens, one epoch, ~33 h on 8x A100-40GB).
- Gains: BBH +5.2%, commonsense +1.25%, MultiPL-E +6.4% relative. Beat ensemble and weight-merge on a perplexity test of three same-architecture domain models (0.7034 vs 0.7112 for weight merging, 0.7262 for ensemble).
- Not feasible at this scale on a 32 GB Mac; the idea (fuse next-token distributions, MinCE) is cheap in principle for 0.6B models, if the tokenizers matched (they do for Qwen3-only lineages).

### 2410.13699 Unconstrained Model Merging (P2)
- Homogeneous: CMA-ES (via mergekit) tunes layer-group coefficients for task arithmetic (every 5 layers share a weight) or TIES (every 10 layers share density and weight); 30 search iterations on ~1,000 samples from GSM8K / MBPP / MMLU.
- DeepSeek-Math-RL x DeepSeek-Coder-Instruct (the same base): task arithmetic gave GSM8K-CoT 86.2 and HumanEval 50.0; TIES gave 82.3 and 45.1, and produced code with extraneous output. The searched TIES densities came out at 0.73-1.0, i.e. the search chose to keep almost everything. They attribute TIES's weakness to **sign conflicts between math and code deltas**, which both lean on shared reasoning parameters.
- That is exactly our v0 pairing (math parent x code parent), a warning that TIES's sign election can hurt when both parents move the same parameters in different directions.
- Heterogeneous: FuseLLM-style distribution fusion with a "pivot" model; math is harder to transfer than code, so use the math model as pivot.

### 2402.16107 FuseChat / FusionChat (P3)
- Two stages: pairwise knowledge fusion of heterogeneous chat LLMs into same-structure targets, then weight-merge the targets with weights from each model's parameter change ratio. Shows the fuse-then-merge pattern; not needed now.

### 2404.13322 MergeNet (P3)
- Learned parameter adapter that queries low-rank parameters of a source model and maps them into a target of a different architecture; trained jointly. Heavy; not relevant before cross-species work.

---

## Implications for vibe-husbandry

Ordered by priority.

1. **Trunk merge default: task-vector average with the ancestor as base (= task arithmetic with lambda 0.5), implemented in `conceive()` as `theta_0 + lambda*(tau_A + tau_B)`.**
   Reasoning: task arithmetic is the strongest classic method on LLMs in the survey comparison and in the DeepSeek math x code case (2410.13699). With just two parents, TIES's advantage (sign conflicts and redundancy grow with the number of models) is small, and sign election hurt in the math x code case. Keep `lambda` a recorded parameter (sweep {0.5, 0.6, 0.7, 0.8, 1.0}); averaging shrinks deltas, so a per-generation compensation may be needed (see 7).
   Baselines to implement, all on the same code path:
   - plain average (lambda 0.5);
   - task arithmetic lambda in [0.5, 1.0] picked on a held-out set;
   - **TIES: keep top 20%, lambda 1.0** (paper's no-validation recipe; also try 50/70/90% as in DARE's grid);
   - **DARE + task arithmetic / DARE + TIES: p = 0.9 first, then back off to 0.7 / 0.5** (small model, so lower p is likely needed; rescale 1/(1-p) is mandatory);
   - greedy-soup-style gate (item 4).
   TIES/DARE need to be a baseline for every compared arm because CLAUDE.md demands TIES as a compute-matched baseline. Apply it slot-wise: expert k of parent A with expert k of parent B, plus averaged trunk and averaged router.

2. **Do not align at conception; do check for the need.** Both parents descend from the same upcycled ancestor, branch from a well-trained pretrained model, and fine-tuning deltas are small. The literature (TIES, ZipIt related work, Model Soups, Frankle LMC) says such fine-tunes lie in one basin and merge without permutation. Permutation work (Git Re-Basin, OT fusion, ZipIt) is for independently trained models, which we exclude by the species rule.
   - **The child keeps both parents' experts, so it never has to average experts.** Competing conventions only bite where weights are actually mixed: the trunk, the router, and any expert-merging inside `prune()`.
   - The species rule's slot IDs act like NEAT historical markings (Safe Crossover cites them): expert slot k in A and in B share an origin, so identity pairing is the natural homology. Record `origin_slot` in each expert.
   - **Use one upcycled ancestor checkpoint and branch from it.** If Drop-Upcycling's random re-initialization were done separately per parent, the two parents' experts would not be homologous and everything above fails. (Inference.)

3. **Where alignment will matter, and what it costs (a single LAP, not NP-hard).** (a) `prune()` if experts are merged rather than dropped (ZipIt-style merge within a model); (b) lineages that diverge over many generations or with high-lr RL; (c) cross-species. For a SwiGLU FFN, the FFN hidden-unit permutation is an exact symmetry (permute rows of gate/up and columns of down together). Because both ends of that permutation attach to the residual stream, which is fixed, the problem reduces to **one linear assignment per expert** (no coordinate descent needed), solvable with `scipy.optimize.linear_sum_assignment` on a 3072x3072 matrix in about a second (inference from Git Re-Basin's weight-matching formulation; Qwen3-0.6B FFN width from memory, verify). Attention-head permutation is similar but needs GQA group constraints.

4. **Cheapest alignment/compatibility checks, in order of cost, to put in `woo()`/`conceive()`:**
   - (a) **Interpolation barrier on a probe set** (the operational definition of compatibility): evaluate loss/accuracy of the merged model at alpha in {0.25, 0.5, 0.75} on about 256 probe prompts; compare to the endpoints. One forward pass per alpha. This is what Frankle, Git Re-Basin and Soups all use. The Soups-style gate: if the merged model is worse than the better parent by more than a threshold, retry with lower lambda, or fall back to an alternative (greedy soup recipe).
   - (b) **Data-free weight statistics**: per-layer task-vector norm ratio ||tau_A||/||tau_B||, cosine similarity between tau_A and tau_B, and TIES-style sign-agreement fraction after top-20% trim. Free, and tracks interference.
   - (c) **Hungarian weight-matching gain** per expert pair: run LAP on FFN neurons; report (i) fraction of the matching that is the identity, (ii) cost(matched)/cost(identity). If the matching is near-identity and the gain is about 1, no alignment is needed. If not, align and re-run (a).
   - (d) Activation correlation of FFN hidden units on about 100-640 tokens/images (ZipIt: a few hundred samples are enough). Used to confirm (c) or when weight matching is ambiguous.
   - **CKA is a weak gate**: the Platonic paper found it a very weak trend even within modality, and Bansal et al. say stitching reveals what CKA cannot. The DESIGN.md shibboleth gate (CKA above a threshold on residual-stream activations) should be necessary-not-sufficient; pair it with check (a). Consider mutual-kNN alignment (Platonic paper) as a more sensitive alternative.

5. **Gestation freezes the trunk, so trunk merge quality is not repairable later.** The README plan freezes attention and embeddings during gestation. Then a bad trunk average cannot be fixed by RL. Add a LoRA on attention (or unfreeze the last layers) as an ablation arm. Another baseline arm: **freeze the trunk in the parents' branch training too (trunk = ancestor in both)**; then the trunk merge is the identity and the whole question disappears, at some capability cost. (Inference.)
   Also note experts were trained on parent-specific hidden states; the child's averaged trunk changes the expert inputs (the "co-adaptation" shift). Quantify with the probe-set barrier on each parent's domain.

6. **Merge base choice for later generations.** Use the most recent common ancestor (MRCA) of the two parents as theta_0 for task vectors rather than the species root, which keeps deltas small and sparse. DARE explicitly fails once deltas reach about 0.03 (continued-pretraining scale) versus 0.002 (SFT scale), and TIES/DARE assume small, redundant deltas. Genome `lineage` already stores the information needed. (Inference.)

7. **Track divergence across generations.** Averaging two parents' trunk deltas shrinks them (TIES: interference reduces parameter magnitude). Under repeated conception the trunk regresses toward the ancestor unless lambda is raised. Log per-generation: trunk ||tau|| vs MRCA, the barrier from check (a), sign-agreement from check (b), and the Hungarian identity fraction. If the identity fraction falls or the barrier rises with generation, that is the trigger to switch on alignment. Evidence available: Soups (high-lr fine-tunes get barriers), Frankle (LMC persists once stable, even at large L2 distance), Git Re-Basin (LMC is emergent and width-dependent; barrier appears with narrow models). None of these test repeated merge cycles; that is an open experiment and arguably a contribution.

8. **Use plain-average-then-recover as the safest path, and treat merge sign conflicts as the main math x code risk.** In 2410.13699 TIES on math+code lost to task arithmetic and the evolved density stayed near 1. Our v0 parents have exactly this combination. Pre-register that TIES might lose, and report it.

9. **Practical cautions.** Tied embeddings (Qwen3-0.6B ties input/output embeddings, memory): average once, not in place twice. Greedy-soup evaluation set should be held out from both gestation RL data and final eval. DARE must be applied to deltas from the base (not to full weights; dropping full weights collapses the model). Use `float32` accumulation when averaging bf16 deltas to avoid rounding away small deltas (inference).

10. **Cross-species (later).** Transport and Merge (OT on activations, top-k 128 neurons, alpha) is the weight-space route; FuseLLM/FusionChat (distribution fusion with MinCE) is the distillation route. FuseLLM cost (33 h on 8x A100 for 1.8B tokens) suggests distillation belongs on rented GPUs; the OT method needs only a small calibration set and may run locally for 1B-scale models. Gains in both are modest (1-6 points). Defer.

---

## Open questions for the humans

1. Should parent branch training update the trunk at all? Freezing it (trunk = ancestor) eliminates trunk merging entirely; updating it gives parents more capacity but needs the merge strategies above. Which arm is primary for v0?
2. Is the 4-expert upcycled ancestor created once (one checkpoint, hashed as `species_id`) and shared by all parents? Drop-Upcycling's random re-init must happen once, before branching, or experts lose homology.
3. Parents are trained with RL/SFT on a 0.6B model. Do we know the typical delta magnitude (e.g. median |tau| per layer)? This decides whether DARE is usable (needs roughly < 0.002 per DARE paper) and what safe p is at this scale.
4. What barrier threshold counts as incompatible? I suggest "merged probe loss worse than the better parent by more than X nats or Y points on the probe set", but X/Y needs a calibration run on a few known-good and known-bad pairs.
5. Do we want to merge experts at prune time (needs alignment, ZipIt/REAM-style) or only drop them (REAP-style, no alignment)? That decides whether the alignment code is needed in v0 at all.
6. Is it acceptable that the shibboleth gate uses a stronger test than CKA (probe-set barrier / mutual-kNN)? CKA alone appears to be too permissive.
7. Merge base for generation >= 2: MRCA (my recommendation) or species root? Needs a decision before the lineage format is frozen.
8. Does anyone have evidence on RL-specific deltas (e.g. how sparse GRPO updates are)? None of the papers in these groups cover RL fine-tune deltas; our parents are RL-trained, so DARE/TIES assumptions (small, redundant SFT-style deltas) are unverified for them.
