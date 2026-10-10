# Notes: groups 06 (evolution/populations) and 12 (wooing/mate choice)

Read 2026-10-10 via alphaXiv full text. Priority-1 papers read in depth; priority 2-3 abstract plus key result.
"Not recovered" means the number was not in the text I read, not that the paper lacks it.
Items marked (mine) are my own inference or arithmetic, not claims from the papers.

---

## Group 06: evolution and populations

### 2403.13187 Evolutionary Optimization of Model Merging Recipes (Sakana, "EvoLLM-JP") [P1]
- Two search spaces: parameter space (per-layer DARE-TIES density/weight, 7B models) and data-flow space (which layers
  to chain, T = 192 inference steps, scaling matrix W).
- Parameter-space budget: CMA-ES (Optuna), population = 4 + floor(3 ln n_params) (about 9 for 3 models x 2 params),
  **1000 trials**. Fitness = accuracy on **1069** translated GSM8K items, greedy decoding, zero-shot. Final test is a
  disjoint 250-item MGSM-JA set.
- Data-flow budget: CMA-ES (EvoJAX), **population 128 x 100 generations**, batches of 200 items, 200 held out for validation.
- Result: merged 7B reaches MGSM-JA 50.0 vs 38.8 for the best parent. A 10B layer-stacked model gets 73.0 on JCoLA. Plain
  TIES / DARE-TIES / passthrough baselines are clearly worse, so the search itself earns the gain.
- Cost is only visible via the follow-up: MERGE3 (2502.10436) estimates the 1000-trial run at ~4e6 TFLOPs, **over a month on one RTX 4090**.
- Source models were already specialists (Japanese-continued-pretrain at >10k GPU-hours). The evolution is cheap only
  relative to making the parents, and nothing is trained inside the loop.
- Adding up to 8 irrelevant models degraded results only mildly. The search is robust to junk parents.

### 2508.16204 M2N2: Competition and Attraction Improve Model Fusion (Sakana) [P1]
- **Archive (population):** MNIST 20 models (CPU, 10 runs: ~1h for merge methods vs ~15h for CMA-ES). LLM experiment archive 15,
  seeded with 2 expert LLMs plus a warm-up of at most 50 random merges. No mutation for LLMs.
- **Budget:** LLM experiment = **1000 fitness evaluations**, 4x H100, ~24h per method. Eval set: first 1319 GSM8K train items
  plus 100 WebShop tasks (separate 1319 / 100 test sets).
- **Operator:** pick two parents, random split point and mix ratio, SLERP on each side, insert child if it beats the
  *worst* archive member (steady-state, one child at a time).
- **Competition (implicit fitness sharing):** each eval item j has capacity c_j (1 for binary scoring). A model's fitness is
  sum_j s_j * c_j / (z_j^alpha + eps), where z_j is the population's total score on item j. Models that solve items nobody
  else solves are protected, with no hand-made diversity metric. Smaller archives win early but converge to worse optima,
  so archive size should scale with eval budget. alpha=0 removes competition and behaves like plain GA.
- **Attraction (this is the mate choice):** parent 1 is drawn by competition-weighted fitness. Parent 2 is drawn with
  probability proportional to sum_j (c_j / (z_j+eps)) * max(s_j(B) - s_j(A), 0). That is "strong where A is weak, weighted
  toward under-contested items".
- **Evidence for each component (LLM, GSM8K+WebShop avg):** full 63.49; no attraction 63.04 (+/-1.14, **within noise**); no
  split-point 61.15; GA 60.05; MAP-Elites 60.14; CMA-ES 49.06. On MNIST-from-pretrained, attraction clearly helps
  throughout. So attraction is well-motivated but only weakly shown on LLMs.
- **Red flag:** merged model scores GSM8K 40.2 vs 74.2 for the math parent. The "win" is a compromise between skills,
  not a model that keeps both.
- Diversity diagnostics worth copying: *coverage* (fraction of eval items solved by at least one member) and per-item
  outcome entropy. GA collapses coverage and entropy early; M2N2 keeps them high.

### 2503.01155 GENOME: Nature-Inspired Population-Based Evolution of LLMs [P1]
- Population of **N=10 LoRA individuals** (gemma-2-2b-it, 10 Tulu-v2 domain experts), **K=10 iterations**. Crossover rate
  0.3, individual mutation rate 0.3, gene mutation rate 0.2, sigma 0.001, elite ratio alpha (not recovered).
- Init: random convex combination of two expert LoRAs. Crossover: fitness-weighted linear blend of two parents chosen with
  probability proportional to fitness. Selection: top-alpha elites plus fitness-proportional resampling.
  GENOME+ adds "succession" (PSO-like experience vector that pulls toward global best, away from global worst) and a top-3 ensemble.
- **Fitness = 200 validation items per task**, test ~1000. Same 200-item budget for all baselines.
- **Wall-clock** (single GPU, vLLM, N=10): GENOME 3600s (4090) / 1600s (A100) on MMLU-Pro; 4000 / 2600s on MATH; GENOME+
  5000-6000s on the 4090. So **about 1-1.7 hours per full evolution run** on a 24 GB card with a 2B model.
- Scales to N=40 with continued gains (to 40 LLMs). Ablation: selection and ensemble matter most. Gains vs Model Swarms avg
  +10.75%. Absolute accuracies stay low (MATH 15.6-16.8%).
- Mate choice here is plain fitness-proportional, with no complementarity.

### 2410.14735 CycleQD (Sakana, ICLR 2025) [P3]
- MAP-Elites over merged Llama3-8B-Instruct models. K=3 tasks (MBPP, DB, OS). Each generation uses one task as quality
  and the others as behavior descriptors, cycling the roles. 1200 generations. Compute not recovered.
- Crossover = task-vector merge with random (possibly negative) weights from N(mu, sigma). Mutation = random scaling of
  the SVD components of the child's task vector (stays in task-relevant directions; Gaussian noise overfit).
- Avg 52.4 vs 51.6 for NSGA-II merging and 47.0 for fine-tuning on all data. The edge over NSGA-II is small.
- Takeaway for us: SVD-subspace mutation is a better way to perturb routers/experts at conception than Gaussian noise.

### 1711.09846 Population Based Training (DeepMind) [P2]
- The closest structural analogue to *gestation*: evolution acts on training runs, not on finished models. Each member trains;
  every 2k-5k steps (or 1-10M agent steps) a member below the bottom 20% **copies weights and hyperparameters** from a
  random top-20% member ("exploit"), then perturbs hyperparameters x0.8/x1.2 ("explore"). Truncation beat binary
  tournament.
- Population sizes 10-80; **<10 gave high variance and worse results, 20-40 was enough**. Total compute = population x one
  run; the claim is beating random search at *equal compute*, not beating one long run.
- Appendix: can run semi-serially on one machine with partial synchrony (advance everyone N steps, then exploit/explore).

### 2505.11427 Mergenetic (library) and 2502.10436 MERGE3 (the method it implements) [P2]
- MERGE3 cuts fitness cost ~**50x (about 2% of the data)** by (1) random subset, (2) IRT latent-ability estimation of the
  endpoint models on the full set, (3) a merge-aware estimator (merged ability = linear mix of parent abilities; only the
  mixing coefficients are fitted from the subset).
- Concrete settings: **population 25, 7 generations, 20 items per language (or 100 items for JA-GSM8K)**, NSGA-II or GA.
  Matches the 1000-trial/1069-item EvoLLM-JP result on JA-GSM8K on one consumer GPU.
- Estimator accuracy: absolute error roughly 4-10 points at 10-100 samples. Uniform random subsampling worked as well as
  clustering-based subsampling.
- Library is built on mergekit + pymoo + lm-eval-harness (CUDA-oriented; mergekit post-0.1.0 is BSL licensed). Not MLX.

### Model Swarms 2410.11163 (not in papers.tsv, read for the "Model Swarms-like" question) [reference]
- PSO over LoRA experts: pool of 10 experts expanded to **N=20** by random pairwise interpolation, up to **K=50 iterations**,
  utility = **200 validation items**, test 1000. Peak storage = 3N+1 adapters.
- Ablation: the crossover-initialised pool is the single most useful ingredient (MMLU 0.583 vs 0.504 without).
- **Red flag in the appendix:** results are "best-found of 200-1000 random hyperparameter runs", picked on the same 200
  validation items. The nominal cost is a large underestimate, and the validation-selected winner is optimistic.
- SFT on the same 200 items did worse than the swarm (GSM8K 0.279 vs 0.459), so search beats naive fine-tuning in the
  tiny-data regime.

### NEAT (Stanley & Miikkulainen 2002; not on arXiv; from memory, verify before citing) [reference]
- **Historical markings:** every new structural gene gets a global innovation number. Crossover lines genes up by number,
  with no topology analysis. This is the standard answer to the competing-conventions problem and is exactly what the
  README wants from "homologous pairing".
- **Speciation:** compatibility distance delta = c1*E/N + c2*D/N + c3*Wbar (excess genes, disjoint genes, mean weight difference of
  matching genes) against a threshold. **Explicit fitness sharing** (f / species size) protects new structures while they
  are still being tuned. Offspring slots are allocated per species. Mating is almost always within species.
- Paper-scale populations were 150+ with tens of generations on small control tasks. Evaluation was cheap, which is
  not our regime.

---

## Group 12: wooing and mate choice

### 2303.17441 All You Need Is Sex for Diversity (PIMP vs random mate vs standard) [P1]
- **Encoding:** each individual has two chromosomes, the solution tree and a **preference tree** (an "ideal partner") built
  from the same function/terminal set. The preference is never scored for fitness. It is only evaluated at mate selection.
- **Mate choice:** parent 1 (chooser) wins a size-5 fitness tournament. Parent 2 (courter) is the closest of 5 *random*
  candidates to the chooser's ideal, comparing phenotypes, i.e. the preference acts as the fitness function for the
  candidates. Both chromosomes go through crossover and mutation, so preferences are inherited and mutated.
- **Setup:** pop 100, 1500 generations, 30 runs, 3 symbolic-regression problems, mutation 5% / 10%.
- **Results:** performance differences vs random-mate and standard were **not significant**. Diversity was:
  unique solutions after 1500 gens PIMP 84-88%, random mate 77-85%, standard 61-68%. Root-node convergence avoided in 5-12/30 runs
  (PIMP) vs 3-9 (random) vs 0-5 (standard). Roles (chooser, courter, both) self-segregate.
- Reading: most of the diversity gain comes from putting only one parent under fitness pressure, with the preference
  adding a smaller increment on top.

### 2504.06110 On the Dynamics of Mating Preferences in GP [P1]
- Tracks preference evolution under different mutations. **Without subtree mutation (which grows trees) preferences
  collapse to a single node** within a few hundred generations, driven by crossover bias and absence of selection on
  preferences, not by self-reinforcement. Preferences do not sustain themselves.
- Diversity benefit persists anyway (courters stay diverse and low-fitness while choosers carry the fitness pressure), and PIMP
  yields smaller, less bloated trees. On a real dataset (Diabetes) PIMP did not beat tournament selection on fitness.
- Cost: sustained preferences roughly **doubled runtime** (larger preference trees to evaluate).
- Reading: the "heritable preference" is fragile and set-up dependent. The asymmetry of roles does the work.

### 2306.05685 Judging LLM-as-a-Judge (MT-Bench / Chatbot Arena) [P1]
- GPT-4 judge agrees with humans **>80%**, same as human-human. Biases measured: **position** (GPT-4 consistent after
  swapping in only 65% of cases; GPT-3.5 46%; Claude-v1 24%, mostly picking the first answer), **verbosity** (a padded "repetitive list"
  fools Claude-v1 and GPT-3.5 91.3% of the time, GPT-4 8.7%), **self-enhancement** (GPT-4 +10% win rate for itself,
  Claude-v1 +25%), and weak grading of math/reasoning (14/20 failures default, 3/20 with a reference answer).
- Mitigations: swap order and only count agreement, few-shot (GPT-4 consistency 65% to 77.5%), reference-guided grading.
- **Small judges:** zero-shot Vicuna-13B was unusable (consistency 11-16%, 22-79% unparseable outputs). Fine-tuned on 20K
  arena votes as a 3-way classifier it reached 65% consistency and 85.5% agreement on non-tie votes (GPT-4: 87%), 56.8% with ties.
- Implication: a 0.6B judge is not viable. A classifier-style head or logit-scored judge of 7B+ with order swapping might be.

### 2401.12255 Instructional Fingerprinting (IF) [P2]
- A publisher implants (key, response) pairs as a backdoor: keys are long random secrets (classical Chinese, Pokemon names in
  Japanese, random vocab tokens) plus the word "FINGERPRINT"; the response is a fixed string.
- **Cost:** n=10 pairs (n=1 works), **60 training instances** (10 pairs + 50 Flan regularisation). ~**1 minute on one A100 for 13B**.
  Verification = n generations and check exact match (FSR). Models tested 6B-13B (LLaMA, Mistral, Pythia, GPT-J, mT5).
- **Robustness:** adapter variant (IF-adapter) keeps **100% FSR** after fine-tuning on 52k Alpaca etc. for 3 epochs, no harm on
  SuperGLUE. The plain-SFT / embedding variants kept only **~32-48% FSR** with the simple template (better with the dialogue
  template and conditional loss p(y|x), numbers not extracted).
- **Caveats:** the adapter variant's verification needs the private F-Adapter plus the suspect's embedding matrix (white-box). Only
  the SFT variant is black-box. Not tested against merging, MoE upcycling, pruning or RL. Third-party registry needed to prevent overclaim.

### 2502.00706 Model Provenance Testing for LLMs [P1]
- Black-box. Query model g and candidate parent f (plus a **control set of unrelated pretrained models**) on T random
  short prompts. Statistic = fraction of prompts where **first output tokens match**. One-sided z-test of sim(f,g) > sim(c,g) for every
  control c, Holm-Bonferroni corrected, alpha=0.05.
- **Scale:** >600 HF models, 30M-4B parameters, 2 benchmarks (BENCH-A 100 derived models/10 parents; BENCH-B 383/57). **Precision 90-95%,
  recall 80-90%**. T of 500-2000 prompts (1000 typical). Precision *drops* as T grows (it starts flagging similar-but-unrelated
  models); recall rises. Entropy-selected prompts cut online queries 4-6x without accuracy loss.
- Cost for us: first-token only, so ~1000 single forward passes per model: seconds on a 0.6B model.
- Limits: tested on fine-tuned derivatives, not on merges/MoE/pruned children. Non-adaptive adversary only. Needs
  controls; it gives a yes/no with a bounded false-positive rate, not a graded relatedness.

### 1509.08158 Handicap principle implies dimorphic ornaments [P2]
- Minimal dynamics model: total potential = (1-s)*individual (peaked at a_opt, rising with health) + s*social (rewards
  ornaments larger than the population mean, exponent gamma). Even identical individuals split into two morphs
  (large and small ornament). Ornament size stays an honest signal of health (Kendall tau 0.91-1.00).
- Conditions: individual cost must dominate at large size (else unbounded growth, gamma >= 2), and social benefit must
  dominate somewhere (else the ornament is "false"). Both morphs sit above the lone optimum, so the population pays a survival cost.
- Use for us: displays need a *bounded, real* cost, and preference strength has a stable range. Expect a bimodal show-off / modest split.

### 2605.13750 Co-evolution of Costly Signaling and Cooperation [P3]
- Agent-based sim (PD, Snowdrift, Stag Hunt; well-mixed and lattice). Signals persist because of the cooperative response
  they elicit, not their production cost. In PD/SD they spin in a "signaling war" where a successful signal attracts
  exploitation and turns over. Signals with no fixed meaning can acquire it.
- Use for us: a fixed display rubric will be learned and gamed; meaning erodes. Argues for per-chooser, heritable, rotating rubrics.

---

## Implications for vibe-husbandry (prioritized)

**1. Budget by evaluation type, not by "generation". Gestation is 100-1000x more expensive per fitness call than anything in these papers.**
All six evolutionary papers get hundreds to thousands of fitness calls because each call is an inference-only eval of a merged
model (seconds to minutes). Only PBT has training inside the loop, and PBT wants 20-40 members. Back-of-envelope (mine):
if one gestation is ~4 h and the Mac is usable ~70% of a week, that is **about 30 gestations per week**: roughly 6-8 adults, 3-4
generations, 3-4 matings per generation. That is enough to demonstrate and A/B one mechanism, not enough for population
statistics. So:
- Run population-dynamics questions (wooing policy, preference inheritance, speciation, archive size) at **toy scale**
  (50-150M from-scratch MoE per the README), where pop 20-40 and 50+ generations are affordable.
- Use Qwen3-0.6B only for a handful of end-to-end child/baseline comparisons, as paired comparisons rather than population statistics.
- Add a **cheap "embryo" tier**: conceive (no gradients) and evaluate on the 100-200-item proxy set, then spend gestation only on
  the top candidates. Also run the compute-matched EvoLLM-JP-style baseline (CMA-ES over layerwise merge coefficients) here.
  This tier is what the wooing score should be validated against.

**2. Fitness protocol.**
- Proxy set of **100-200 items per task** (what GENOME/Model Swarms/MERGE3 used). Binomial standard error at n=200 and p=0.5 is
  ~3.5 points, so differences under ~7 points are not resolvable (mine). Do not select on gaps smaller than that.
- Keep a **frozen, never-searched test set**. Model Swarms' best-of-200-1000-hyperparameter-runs reporting is the failure mode to
  avoid. Count every hyperparameter run in the cost ledger.
- **Log the per-item correctness vector for every genome.** It is free, and it powers attraction, fitness sharing, coverage,
  IRT-style estimation (MERGE3) and the complementarity term of the wooing score.
- Report eval FLOPs and judge FLOPs next to gestation FLOPs. MERGE3 shows fitness evaluation dominates evolutionary merging cost.

**3. Implement M2N2 attraction and competition as the chooser's complementarity and survivor-selection rule.**
It maps almost one-to-one onto "complementarity" in DESIGN.md and costs nothing once per-item vectors exist. Keep it
ablatable: on LLMs the attraction ablation was within noise, so earn it with a random-mate arm and a fitness-proportional arm.
Scale archive size with the number of fitness evals (small archives win early, lose late). Track coverage and
per-item entropy as the diversity metrics.

**4. PIMP-style heritable preference: cheap, but expect diversity, not performance.**
Cheap mapping: chooser's preference = a small heritable vector over the behaviour space (per-item correctness profile or
output embedding on the probe set), mutated with small noise. The chooser samples ~5 random adults and picks the closest
to its ideal (or the most complementary to itself). It costs almost nothing. Build in the lessons from 2504.06110:
(a) only the chooser is under fitness selection, the asymmetry is what produces diversity; (b) preferences collapse unless there is
ongoing mutation pressure, so monitor preference entropy; (c) always run a **random-mate control**, because it gets
most of the diversity benefit; (d) do not claim a fitness benefit.

**5. Kin gate: use the cheap tests, but test their robustness first.**
- Implant IF keys (10 pairs, 60 examples, about a minute) into the **ancestor before the first branch**. Check FSR (10 generations)
  at every conception and every generation. This proves descent from the fingerprinted ancestor, not weight compatibility, so keep the CKA/probe check.
- Add a continuous relatedness measure: first-token agreement on ~500-1000 random prompts between parents (provenance-test statistic),
  usable for outbreeding / incest avoidance. The paper uses it as a yes/no test. Using it as a graded distance is my extension.
- **Unknown, and the first experiment to run:** whether fingerprints survive averaging attention at conception, expert-LoRA
  GRPO, and REAP-style pruning. None of these papers test merging, MoE or RL. A 1-hour toy run (imprint, branch, conceive,
  short GRPO, prune, re-check FSR) settles it.
- Practical gate = species_id hash (free, exact) + IF check + agreement/CKA threshold. In a single-ancestor project provenance
  testing adds little (everything is kin) except as a graded distance and for guarding against claimed-but-false lineage in a multi-party setting.

**6. Display judging: verify by code first, LLM judge second.**
Constrained forms (meter, acrostic, lipogram, valid-Python sonnet) are mostly programmatically checkable. Use code for the
hard constraints and an LLM judge only for the residual quality score. For the LLM judge:
- Not the 0.6B chooser; not zero-shot tiny models (Vicuna-13B zero-shot: 11-16% swap-consistency, 22-79% unparseable).
- Use a 7B-class instruct model run as a classifier or logit-scored comparison (not free-text), both orders, count only consistent
  verdicts, reference/rubric in the prompt. Pick a **different model family from the Qwen ancestor** to reduce self-enhancement bias (mine).
- Calibrate once against ~50 human labels. Single-answer scoring or comparison to a fixed reference avoids quadratic pairwise cost.
- Expect gaming: rotate or make rubrics chooser-specific and heritable (2605.13750), and keep display costs bounded but real (1509.08158).

**7. Borrow NEAT's historical markings and speciation almost verbatim.**
Upcycled experts start as copies of the ancestor's FFN, so every expert has a natural **origin ID** (layer, expert index, branch).
Pair homologous experts by origin ID at conception (free) and compare against Hungarian matching on activations.
The "compatibility distance" = mean weight/task-vector distance of homologous experts (plus counts of added/pruned experts). Use it
for (a) species/niche clustering with **fitness sharing** so a specialist lineage is not out-competed by generalists
before it matures, and (b) mating within niche with a small cross-niche rate.

**8. PBT-style selection inside gestation.**
If memory allows 2-3 sibling children of one couple, apply PBT's truncation exploit/explore at checkpoints (copy LoRA + router
state from the best sibling, perturb LR / gift size). Adapters are small. PBT's own warning (<10 members is high variance)
means treat this as hyperparameter robustness, not as a population.

**9. Small perturbation advice.** Perturb routers/experts at conception in the SVD subspace of the child's task vector (CycleQD),
not with isotropic Gaussian noise. Initialise any search population by random interpolation of parents (Model Swarms ablation: largest single contributor).

**10. Baseline hygiene.** In every comparison include (i) train-longer at matched FLOPs including eval and judge FLOPs, (ii) TIES merge,
(iii) direct MoE training, (iv) an EvoLLM-JP-style CMA-ES merge. M2N2's LLM result shows the merged model gave up most of the
specialist's skill (GSM8K 40 vs 74). Report per-skill retention, not just the average.

---

## Open questions for the humans

1. **What does one gestation actually cost** (wall-clock on the Mac, peak memory with router + LoRA only)? Everything in item 1 hinges on it. Is a
   small, toy-scale population study acceptable as the main evidence for wooing/selection, with Qwen3-0.6B as validation?
2. **Is success = diversity (coverage, entropy) or fitness?** The mate-choice literature supports the former and gives little evidence for the latter.
3. **Shibboleth threat model:** is a white-box check acceptable? The strong, persistent IF variant needs the suitor's embeddings,
   which contradicts "without anyone inspecting weights" in DESIGN.md. The black-box SFT variant persists much less.
4. **Judge:** local 7B-class judge (cost: memory and time on the Mac) vs an external API judge? DESIGN.md says "fixed external judge model"; which one,
   and is cost per mating a concern?
5. **Test discipline:** who owns the frozen held-out test set, and may it ever feed back into selection?
6. **Is incest avoidance a goal?** With one ancestor every model is kin, so a graded relatedness measure is needed to make outbreeding meaningful.
7. **Fisherian runaway:** what observable counts as runaway (preference drift decoupled from child fitness)? Needs a metric before the experiment.
8. **Which NEAT facts do you want cited?** I wrote them from memory; someone should check the 2002 paper before they go into a writeup.
