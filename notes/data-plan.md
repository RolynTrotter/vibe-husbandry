# Data plan for vibe-husbandry v0

Scope: training, RL, eval, calibration and courtship-display data for (a) a from-scratch toy MoE
(~20-100M params) and (b) the Qwen3-0.6B -> 4-expert upcycled MoE -> math/code parents -> child -> pruned adult loop.
Budget: one Mac Studio, 32 GB unified memory (see README hardware table).

**How this was verified (2026-10-10).** Every dataset/model ID below returned HTTP 200 from
`https://huggingface.co/api/datasets/<id>` (or `/api/models/<id>`). Row counts come from
`datasets-server.huggingface.co/size`. Token counts are *my measurements*: I tokenized 300 random rows per
dataset with the actual `Qwen/Qwen3-0.6B` `tokenizer.json` and extrapolated (mean tokens/example x rows).
Licences are the HF `license:` tag; "(none)" means the repo has no licence tag and the licence must be read
from the dataset card or the upstream source before release. Everything about Mac speed is a
back-of-envelope estimate, not a measurement (marked "est."); the first job of the data pipeline is a
10-minute throughput calibration run that replaces those numbers.

IDs that did **not** resolve and are therefore not recommended: `hails/mmlu_pro` (use `TIGER-Lab/MMLU-Pro`),
`openai/gsm8k_platinum` (use `madrylab/gsm8k-platinum`), `Skywork/Skywork-SynPref-40M` (Skywork-Reward-V2's
preference data is not retrievable via the API; v0.2 80K set exists).

---

## 0. Decisions that shape everything else

1. **Non-thinking mode everywhere in v0.** Qwen3-0.6B defaults to long `<think>` traces. Its own report lists
   MATH-500 55.2 (non-thinking) vs 77.6 (thinking), at roughly an order of magnitude more tokens per answer.
   On a Mac, RL and eval must use `enable_thinking=False` (SFT targets get an empty `<think>\n\n</think>`
   prefix, or use the Base model, see Open Questions) and cap generations at 512 tokens (GSM8K-level) /
   1024 tokens (MATH-level).
2. **Export the upcycled genome as `Qwen3MoeForCausalLM`** (config fields `num_experts`,
   `num_experts_per_tok`, `moe_intermediate_size`, `decoder_sparse_step`, `mlp_only_layers`, `norm_topk_prob`).
   `mlx-lm` already ships `mlx_lm/models/qwen3_moe.py` (checked), HF transformers has the same class, and
   `mlx_lm.evaluate` / lm-evaluation-harness then work with zero custom model code. This is a data-plan
   issue because it decides whether the eval suite in section 4 is plug-and-play.
3. **Sanity anchors.** Qwen3 tech report (arXiv 2505.09388), Qwen3-0.6B-Base: MMLU 52.81 (5-shot), GSM8K 59.59
   (4-shot CoT), MATH 32.44 (4-shot CoT), EvalPlus 36.23 (0-shot), MBPP 36.60 (3-shot). Post-trained
   Qwen3-0.6B non-thinking: MATH-500 55.2, IFEval strict-prompt 54.5, LiveCodeBench v5 3.6.
   Our harness must reproduce these within a few points *before* any breeding result is believed.
4. **Qwen3 almost certainly saw GSM8K/MATH/MBPP-style data in pretraining.** Absolute gains from
   "specialisation" SFT on these sets will be small; the experiment's signal is *relative* (child vs parents vs
   TIES vs train-longer). Use `madrylab/gsm8k-platinum` (cleaned labels) alongside GSM8K, and plan a
   fresh-problem eval (LiveCodeBench) for finalists.
5. **Decontaminate.** Before using any SFT/RL set, run a cheap 13-gram overlap check against GSM8K test,
   MATH-500, HumanEval, MBPP test (script: hash all 13-grams of test prompts/solutions, drop train rows that hit).

---

## 1. Toy from-scratch MoE (20-100M params)

### 1.1 Corpora

| HF id | Split / config | Size | Approx tokens (Qwen3 tok.) | Licence | Purpose | Recommended subset |
|---|---|---|---|---|---|---|
| `roneneldan/TinyStories` | train / validation | 2,119,719 / 21,990 stories (1.9 GB txt) | ~390-430M (184-202 tok/story measured) | cdla-sharing-1.0 | Main toy pretraining text; fluent small models are known to train on it | 1 epoch (~400M) for 100M model; 100-200M for 20-50M model. Hold out the official validation split for loss curves |
| `HuggingFaceTB/cosmopedia` | `stories` (also `auto_math_text`, `khanacademy`, `openstax`, `wikihow`, ...) | 4,992,964 rows in `stories` | ~2.6B in `stories` (528 tok/row) | apache-2.0 | Optional richer/synthetic prose if TinyStories saturates | 50-100M tokens only if needed |
| `HuggingFaceFW/fineweb-edu` | `sample-10BT` (28.5 GB parquet, 14 files) | ~10B GPT-2 tokens | n/a (do not download all) | odc-by | Realistic web text for generality; too big for toy | 1-2 parquet shards (~50-100M tokens) as a "general" domain, optional |
| `HuggingFaceTB/smollm-corpus` | `fineweb-edu-dedup`, `cosmopedia-v2`, `python-edu` | 190M / 39M / 7.7M rows | huge | odc-by | Pre-curated alternative to FineWeb-Edu | **Avoid `python-edu`**: it stores Software Heritage blob ids, content must be fetched separately |
| `Salesforce/wikitext` | `wikitext-103-raw-v1` | 1.8M rows train | ~100M words | cc-by-sa-3.0 / gfdl | Held-out perplexity for forgetting checks on toy | validation + test only (~1 MB) |
| `OpenCoder-LLM/opc-sft-stage2` | `educational_instruct` | 118,278 rows | ~16M (136 tok/row) | mit | Short Python functions: **toy "code" domain** (parent B at toy scale) | all of it (16M tokens), or 50k rows |
| (generated) | synthetic arithmetic / string tasks | unlimited | 20-50M | n/a | **Toy "math" domain** with exact verifiers; also the toy's RL environment | generate 20-50M tokens, e.g. `37+58=95`, multi-digit add/sub/mul, 2-step word problems |

`karpathy/tiny_shakespeare` (tiny, 1 MB, no licence tag) exists and is useful only as a 5-minute smoke test.

**Toy domain recommendation.** Parent A = stories + synthetic arithmetic, Parent B = stories + short Python
(`educational_instruct`). Both start from the same toy ancestor (TinyStories pretrain, with a 10-20% replay of both
domains so the shared vocabulary/embeddings cover them), exactly mirroring the Qwen v0 structure. Both domains have
cheap programmatic verifiers (arithmetic equality; `assert` tests), so toy GRPO uses the same reward code path as v0.

### 1.2 Tokenizer

* **Do not use the Qwen3 tokenizer unmodified.** Qwen3's vocab is 151,936 entries (config); a 50-100M model with
  d_model 512 would spend 78M parameters on a tied embedding table. The embedding would dominate the toy.
* **Measured:** 3,000 TinyStories stories (606k tokens) touch only 7,618 distinct Qwen3 token ids and the curve
  was still rising slowly, so the full corpus likely uses ~10-20k.
* **Option A (recommended): train a BPE of 8,192 entries** (HF `tokenizers`, byte-level, a few minutes) on a
  mix of the toy corpus (TinyStories + code + arithmetic). Keep Qwen's special tokens/chat template layout
  (`<|im_start|>`, `<|im_end|>`) so the chat/RL code paths are identical to v0. Fertility on stories is similar to
  Qwen's, slightly worse on code. Token budgets above become roughly +10-25%.
* **Option B: remap Qwen3's vocab to the ids observed in the toy corpus** (~15k). Keeps the *same tokenization*
  as the 0.6B pipeline (useful to test pipeline code), but the toy then has no real species relation to Qwen anyway.
* Whichever is chosen, the toy and its descendants form their own "species" (hash the tokenizer into
  `species_id`); do not try to transplant toy experts into Qwen3.

---

## 2. Parent specialisation data (SFT)

Target: **~20M tokens per parent plus 10-15% general replay**, 1-2 epochs => 25-50M tokens processed per parent
(~1-4 h on a 0.6B/1.4B model with LoRA or partial full-FT; est. 3-5k tok/s).
Cap sequence length at 1,024 tokens (drop longer rows). Format in the Qwen3 chat template, loss on the response only.

### 2.1 Parent A: math

| HF id | Split | Rows | Tokens (est.) | Licence | Notes | Take |
|---|---|---|---|---|---|---|
| `openai/gsm8k` | `main` train / test | 7,473 / 1,319 | 1.3M (179 tok) | mit | Grade-school CoT. The `socratic` config adds sub-question style | all train (1.3M), 2 epochs |
| `DigitalLearningGmbH/MATH-lighteval` | train / test | 7,500 / 5,000 | 1.5M (206 tok) | mit | MATH train with `level`, `type`; answers in `\boxed{}` | all train (1.5M) |
| `meta-math/MetaMathQA` | train | 395,000 | ~94M (237 tok) | mit | Rewritten GSM8K+MATH questions (GSM/MATH train only; no test leakage). Best cost/benefit for a small model | **40k rows ~9.5M** |
| `AI-MO/NuminaMath-CoT` | train | 859,494 | ~406M (473 tok) | apache-2.0 | Broader (olympiad, synthetic); includes GSM8K/MATH train solutions | **10-15k rows ~5-7M**, filter to <=1,024 tokens |
| `nvidia/OpenMathInstruct-2` | `train_1M` (also 2M, 5M, full 13.9M) | 1,000,000 | ~459M (459 tok) | cc-by-4.0 | Llama-3.1-405B-generated CoT; cleaner than most | **10k rows ~4.5M** |
| `AI-MO/NuminaMath-TIR` | train | 72,441 | ~52M (717 tok) | apache-2.0 | **Do not use for Parent A**: solutions embed Python tool code, which would leak the code specialisation into the math parent and muddy the experiment | skip |
| `open-r1/Mixture-of-Thoughts` | `math` | 93,733 | ~532M (5,671 tok) | (none) | R1-style long reasoning traces; far too long for v0 | skip for v0 |

Recipe A (~20M tokens): GSM8K 7.5k + MATH 7.5k + MetaMathQA 40k + NuminaMath-CoT 12k + OpenMathInstruct-2 10k
~= 77k examples.

### 2.2 Parent B: code

| HF id | Split / config | Rows | Tokens (est.) | Licence | Notes | Take |
|---|---|---|---|---|---|---|
| `OpenCoder-LLM/opc-sft-stage2` | `educational_instruct` | 118,278 | ~16M (136 tok) | mit | Short, mostly Python, has `testcase` (asserts) and `entry_point` fields so it doubles as RL data (section 3) | **40k rows ~5.4M** (hold the rest for RL) |
| `OpenCoder-LLM/opc-sft-stage2` | `evol_instruct` | 111,183 | ~63M (571 tok) | mit | Harder, longer | **12k rows ~6.8M** |
| `ise-uiuc/Magicoder-OSS-Instruct-75K` | train | 75,197 | ~37M (487 tok) | mit | Realistic snippets from OSS; multi-language, filter to Python | **8k rows ~3.9M** |
| `theblackcat102/evol-codealpaca-v1` | train | 111,272 | ~58M (520 tok) | apache-2.0 | Also the default REAP calibration set (section 5) | **hold out for calibration**; optionally 4k rows ~2M for SFT |
| `sahil2801/CodeAlpaca-20k` | train | 20,022 | ~1.6M (82 tok) | cc-by-4.0 | Very short, low-noise | all (1.6M) |
| `m-a-p/CodeFeedback-Filtered-Instruction` | train | 156,526 | ~87M (556 tok) | apache-2.0 | Multi-language; alternative to Magicoder | optional |
| `google-research-datasets/mbpp` | `full`: train 374 / validation 90 / prompt 10 / **test 500** | 474 trainable | <0.1M | cc-by-4.0 | **Eval set.** Do not SFT on it; use `prompt` split only as few-shot | 0 for training |
| `nvidia/OpenCodeInstruct` | train | 1.4M | big (~5 GB parquet) | cc-by-4.0 | Large, has execution feedback; overkill | skip for v0 |

Recipe B (~20M tokens): educational_instruct 40k + evol_instruct 12k + Magicoder-Python 8k + CodeAlpaca 20k
~= 80k examples. Filter to Python-only to match HumanEval/MBPP.

### 2.3 General replay (both parents and the child, 10-15% of tokens)

Purpose: keep chat ability and reduce forgetting, and keep the two parents *comparable* on general ability.

| HF id | Rows | Tokens | Licence | Take |
|---|---|---|---|---|
| `HuggingFaceH4/ultrachat_200k` (`train_sft`) | 207,865 | ~257M (1,237 tok/dialogue; multi-turn) | mit | 2.5k dialogues truncated to 1,024 ~2.5M |
| `allenai/tulu-3-sft-mixture` | 939,343 | not measured | odc-by | alternative incl. some instruction-following and math/code; optional |
| `HuggingFaceTB/smoltalk` | ~1M+ (`all`) | ~890M (889 tok) | (none; read card) | alternative; short convos |

### 2.4 Licences and provenance caveats (all SFT sets above)
Most are synthetic outputs of third-party LLMs (GPT-3.5/4, Llama-3.1-405B and so on), so redistributing trained
weights may inherit terms of service. Fine for private research; read each card before releasing checkpoints.

---

## 3. RL prompts with verifiable rewards (RLVR)

Budget: GRPO group size 8 x 16 prompts/step x ~300 generated tokens = ~38k generated tokens/step. At est. 800-1,500
tok/s batched decode on 0.6B-1.4B that is ~0.5-1 min of generation plus the update, so **300 steps is ~5-10 h and
consumes only ~4,800 prompts per run (single pass)**. A pool of 5-10k prompts per domain is plenty; more variety is
about *difficulty calibration*, not quantity.

**Difficulty filtering matters more than size.** A 0.6B model gets zero reward on competition problems, and GRPO
has no gradient when all samples in a group score the same. Pre-filter by sampling 4-8 completions per candidate
with the ancestor and keep prompts with pass-rate strictly between 0 and 1 (DAPO "dynamic sampling"). Cost for
3k candidates x 8 samples x 300 tokens = 7M tokens (~1.5-3 h est.); do it once per ancestor and reuse for all
lineages (this is also a fair, shared "nuptial gift" input).

### 3.1 Math prompts

| HF id | Split | Rows | Prompt tokens | Licence | Answer format | Take |
|---|---|---|---|---|---|---|
| `openai/gsm8k` | train | 7,473 | ~60 | mit | integer after `####` | all, after difficulty filter; easiest source of signal |
| `allenai/RLVR-GSM` | train / test | 7,473 / 1,319 | n/a | mit | Tulu-3 RLVR format (messages + ground_truth) | same content as above, use whichever format the trainer likes |
| `DigitalLearningGmbH/MATH-lighteval` | train | 7,500 | ~80 | mit | `\boxed{}`; has `level` 1-5 and `type` | levels 1-3 (~3.5k), then 4 as the model improves |
| `allenai/RLVR-MATH` | train | 7,500 | n/a | (none) | RLVR format | alternative to the above |
| `agentica-org/DeepScaleR-Preview-Dataset` | train | 40,315 | ~83 | mit | `answer` field | **hard** (AIME/AMC/Omni-MATH): use only after difficulty filter; expect <15% usable by 0.6B |
| `open-r1/DAPO-Math-17k-Processed` | `en` 14,116 / `cn` 3,282 / `all` 17,398 | 17,398 | ~96 | (none; upstream apache-2.0) | integer answers | hard; filter. **Do not use `BytedTsinghua-SIA/DAPO-Math-17k` raw**: it lists 1,791,700 rows (apache-2.0) because the 17k prompts are duplicated ~100x |
| `Skywork/Skywork-OR1-RL-Data` | `math` 105,055 / `code` 14,057 | 119k | n/a | (none) | curated RL set from the Skywork-OR1 work | optional, large; code half is large on disk (2 GB) and likely stdin/stdout style |

Reward: `math-verify` (PyPI `math-verify` 0.9.0, Apache-2.0, HF-maintained) for symbolic equivalence on `\boxed{}`
answers + 0.1 format reward. Same function serves MATH-500 eval.

### 3.2 Code prompts with unit tests

| HF id | Split | Rows | Tokens | Licence | Test form | Take |
|---|---|---|---|---|---|---|
| `OpenCoder-LLM/opc-sft-stage2` | `educational_instruct` | 118,278 | ~16M total | mit | list of `assert ...` strings + `entry_point` | the 78k rows not used in SFT -> RL pool; **best licence + easiest harness** |
| `KodCode/KodCode-Light-RL-10K` | train | 10,000 | ~2.1M prompt, 6.6M with solution+tests | **cc-by-nc-4.0** | pytest file (`from solution import f`), `test_info`, difficulty and pass-rate columns | use pass-rate columns to select medium-difficulty rows; non-commercial |
| `KodCode/KodCode-V1` | train | 100k-1M (size tag) | big | cc-by-nc-4.0 | same | only if Light-RL-10K runs dry |
| `google-research-datasets/mbpp` | train+validation+prompt | 474 | tiny | cc-by-4.0 | 3 asserts per task | too small for RL; keep for eval and few-shot |
| `likaixin/APPS-verified` | train | 4,211 | ~1.5M | (none) | stdin/stdout `input_output` JSON | competitive-programming style: too hard for v0; optional later |
| `codeparrot/apps` | train/test | ~10k | n/a | mit | script-based loader | viewer-unsupported (script dataset); use `APPS-verified` instead |
| `open-r1/verifiable-coding-problems-python`, `PrimeIntellect/verifiable-coding-problems`, `BAAI/TACO`, `deepmind/code_contests` | | 10k-100k+ | large | mixed / (none) | stdin/stdout | later scaling, not v0 |

Reward: fraction of tests passed (or all-or-nothing, 1.0 / 0.0) + small format reward (one fenced Python block),
negative for timeout. Mix math:code at 50:50 for the child's gestation; parents' own RL (if any) uses their domain only.

### 3.3 Running code rewards safely on a Mac

Threat model: our own small model plus buggy test files, not an adversary, but GRPO *will* find exploits
(`sys.exit(0)`, `os._exit(0)`, monkeypatching `assert`, reading test files, fork bombs, infinite loops,
filling the disk, deleting files). Layers, cheapest first:

1. **Process isolation per execution:** `subprocess` with `python -I`, a fresh temp dir per sample, a minimal env
   (no tokens, no `HOME`), stdin closed, new process group, wall-clock timeout (2-5 s) with group kill.
   `resource.setrlimit`: `RLIMIT_CPU`, `RLIMIT_FSIZE`, `RLIMIT_NPROC`. **Do not rely on `RLIMIT_AS` on macOS**
   (not reliably enforced; verify on the target machine); enforce memory by polling RSS in the parent and killing.
2. **OS sandbox:** `sandbox-exec` with a profile that does `(deny network*)`, allows reads of the interpreter and
   stdlib, and allows writes only under the temp dir. Deprecated by Apple and the profile language is
   undocumented, but reported working on macOS 15 (jmmv.dev write-up, 2019; a 2026 note); test your own OS version
   and keep a CI test "does this sandbox block `curl`, a write to `~`, and `socket`".
3. **Stronger option: a Linux container** (Docker via Colima/OrbStack/Docker Desktop) with `--network none
   --read-only --tmpfs /work --memory 512m --pids-limit 64 --cpus 1 --cap-drop ALL`. Container start costs
   ~0.3-1 s, so run a pool of long-lived worker containers and `exec` into them rather than one container per sample.
   (Apple's own `container` CLI is a possible alternative on newer macOS; I could not verify it, so not recommended yet.)
4. **Anti-hacking in the harness:** keep tests out of the model-visible prompt; run model code and the test
   driver as separate processes or at least print a **random success nonce** only after all tests have run, and
   require the nonce in stdout (so `os._exit(0)` earns nothing); parse pytest's junit XML rather than trusting the
   exit code; scrub env and close extra fds before `exec`ing model code; reject solutions that import `os`/`subprocess`/`socket`/`ctypes`
   via an AST pre-check as defence in depth only (not as the security boundary).
5. **Throughput:** 128 executions/step (16 prompts x 8) at <=5 s each over 8 workers = <=80 s worst case,
   typically ~10-20 s; fine, but run it concurrently with the next generation batch if the trainer allows.
6. **Eval code execution** (lm-evaluation-harness `humaneval`/`mbpp` have `unsafe_code: true` in their YAML,
   require `--confirm_run_unsafe_code`; HF `code_eval` also wants `HF_ALLOW_CODE_EVAL=1`) executes model code in-process
   forks with weak protection. Run the whole eval command inside layer 2 or 3.

---

## 4. Evaluation sets

### 4.1 Target capabilities (parents' specialties)

| HF id | Split | Rows | Licence | lm-eval task | Notes |
|---|---|---|---|---|---|
| `openai/gsm8k` | `main` test | 1,319 | mit | `gsm8k` (5-shot default; `gsm8k_cot` variants exist) | Report strict + flexible extract. Use `--apply-chat-template` for the chat model |
| `madrylab/gsm8k-platinum` | test | 1,209 | mit | none in harness (verify) | Re-annotated GSM8K test; same format, run with own loop |
| `HuggingFaceH4/MATH-500` | test | 500 | (none; subset of MATH, mit) | **not in lm-eval** (`minerva_math`/`hendrycks_math` use the full 5k test and Minerva prompts) | Own loop: chat prompt, `\boxed{}`, `math-verify`. Qwen reference 55.2 non-thinking |
| `DigitalLearningGmbH/MATH-lighteval` | test | 5,000 | mit | n/a | Larger, lower-variance alternative; ~10x the cost |
| `openai/openai_humaneval` | test | 164 | mit | `humaneval` (`unsafe_code: true`, pass@1 greedy) | 164 items => +-4pp standard error: needs paired comparisons |
| `evalplus/humanevalplus` | test | 164 | apache-2.0 | use PyPI `evalplus` 0.3.1 (Apache-2.0) | stricter tests; HumanEval+ |
| `google-research-datasets/mbpp` | `full` test 500 / `sanitized` test 257 | 500 | cc-by-4.0 | `mbpp` (3-shot, `unsafe_code: true`) | the lm-eval prompt is a base-model style |
| `evalplus/mbppplus` | test | 378 | apache-2.0 | `evalplus` | MBPP+ |
| `livecodebench/code_generation_lite` | releases by date | <1k per release | cc | n/a | Fresh/uncontaminated problems for finalists; **very large on disk** (46 GB repo), fetch one release file only |
| `bigcode/bigcodebench` | v0.1.x | ~1.1k | apache-2.0 | n/a | later |

### 4.2 General-capability regression suite (detect catastrophic forgetting)

| HF id | Split | Rows | Licence | lm-eval task | Subset for gating |
|---|---|---|---|---|---|
| `cais/mmlu` | `all` test (57 subjects) | 14,042 | mit | `mmlu` | 20 questions/subject = 1,140 (fixed seed, stratified) |
| `TIGER-Lab/MMLU-Pro` | test | 12,032 | mit | `mmlu_pro` | optional; at 0.6B it is near floor |
| `Rowan/hellaswag` | validation | 10,042 | (none) | `hellaswag` | first 1,000 |
| `allenai/ai2_arc` | ARC-Challenge test 1,172 / ARC-Easy test 2,376 | 3,548 | cc-by-sa-4.0 | `arc_challenge`, `arc_easy` | full ARC-C |
| `allenai/winogrande` | `winogrande_xl` validation | 1,267 | (none) | `winogrande` | full |
| `google/IFEval` | train (the eval prompts) | 541 | apache-2.0 | `ifeval` | full; doubles as constrained-writing capability check (section 6) |
| `truthfulqa/truthful_qa` | validation | 817 | apache-2.0 | `truthfulqa_mc2` | optional |
| `allenai/openbookqa`, `ybisk/piqa` | | | unknown | `openbookqa`, `piqa` | optional (licence "unknown") |

Statistics: with n = 1,000 and accuracy ~0.5 the standard error is ~1.6pp, so the gate should be
"**drop > 3pp vs ancestor on 2+ of the 5 regression tasks, or > 2pp on the average**", and compare models on the *same
fixed items* (paired differences are much tighter than independent CIs). HumanEval (164) and MATH-500 (500) have
SE of ~3.8pp and ~2.2pp; do not read differences below those as real without multiple seeds/pairing.

### 4.3 How lm-evaluation-harness fits (and doesn't)

* Install `lm-eval` (PyPI 0.4.13, MIT). On Mac use **`mlx_lm.evaluate`** (in mlx-lm 0.32.0; wraps
  `lm_eval.simple_evaluate` through a registered `mlxlm` model). Flags confirmed in its source: `--tasks`, `--limit`,
  `--num-shots`, `--batch-size` (default 8), `--apply-chat-template`, `--chat-template-args '{"enable_thinking":false}'`,
  `--max-tokens`, `--confirm-run-unsafe-code`, `--output-dir`.
* Loglikelihood tasks (MMLU, HellaSwag, ARC, Winogrande) are cheap prefill-only work; generation tasks (GSM8K,
  HumanEval, MBPP, IFEval) dominate wall-clock. MATH-500 needs our own loop because the harness has no MATH-500
  task (confirmed by listing `lm_eval/tasks`).
* Few-shot and chat template change numbers by 5-15pp; pin one config per task in `eval/suite.yaml` and never
  compare numbers across configs.

### 4.4 Wall-clock for a 0.6B-1.4B model on the Mac (est., assumes ~4k tok/s prefill, 600-1,500 tok/s batched decode, non-thinking)

| Task | Work | Est. time |
|---|---|---|
| GSM8K 1,319, 5-shot (~900-token prompts, ~200 generated) | 1.2M prefill + 0.26M decode | 8-12 min (0-shot chat: ~5) |
| MATH-500, non-thinking, <=1,024 tokens | ~0.3M decode | 5-10 min (thinking mode: 40+ min, avoid) |
| HumanEval 164 / MBPP 500 | tiny | 1-2 / 3-5 min plus execution |
| MMLU 5-shot full 14,042 | ~10M prefill | ~40 min; 1,140-item subset ~3-4 min; 0-shot full ~8 min |
| HellaSwag full 10,042 (4 endings) | ~4M | ~15 min; first 1,000 ~1.5 min |
| ARC-C + Winogrande | | ~2 min |
| IFEval 541 | ~0.16M decode | ~3 min |

* **Tier 0 (smoke, every checkpoint, <=5 min):** GSM8K 200, HumanEval, MMLU 300, own toy verifiers.
* **Tier 1 (stage gates, ~25-35 min):** GSM8K full, MATH-500, HumanEval, MBPP, MMLU subset 1,140, HellaSwag 1,000,
  ARC-C, Winogrande, IFEval.
* **Tier 2 (finalists, ~2-3 h):** everything full, + HumanEval+/MBPP+, + 3 seeds for sampled metrics, + LiveCodeBench slice.

---

## 5. Calibration data

### 5.1 Expert pruning (REAP-style, 8 -> 4 experts)

What REAP actually does (arXiv 2510.13999, read from the paper): for models up to ~110B it uses
**1,024 random samples packed to 2,048 tokens (~2.1M tokens)**; domain-matched calibration is critical: **generic C4
calibration "collapses" code accuracy (some pruned models produce no coherent output)** while C4 is fine for
multiple-choice tasks. Cerebras's reference repo (`CerebrasResearch/reap`) defaults to `theblackcat102/evol-codealpaca-v1`
and uses `open-r1/Mixture-of-Thoughts` (math/code/science) for reasoning checkpoints.

Recommended for v0 (~1M tokens, 500 x 2,048): chosen to *mirror the deployment distribution of the adult*:

| Share | Source (held out from SFT/RL) | Tokens |
|---|---|---|
| 30% math | held-out GSM8K/MATH/MetaMathQA rows with reference responses (+ some model self-generated responses) | 300k |
| 30% code | `theblackcat102/evol-codealpaca-v1` (REAP default) + `educational_instruct` held-out | 300k |
| 25% general chat | `HuggingFaceH4/ultrachat_200k` rows | 250k |
| 10% constrained writing | the programmatic poetry/IFEval-style prompts from section 6 with verified-good outputs | 100k |
| 5% web text | `HuggingFaceFW/fineweb-edu` shard slice | 50k |

Checks: (a) **split-half stability**: expert saliency ranking from two disjoint halves should correlate (Spearman >= 0.95);
if not, double the calibration set. (b) **Domain ablation** (math-only / code-only / mixed calibration) as a first-class
experiment, because "which parent's experts survive" depends on it. (c) Calibrate on the *adolescent after gestation*,
using the same chat template/response masks as RL.
Pruning needs forward passes with hooks only, ~1M tokens is a few minutes on the Mac.

Also needed: **expert-alignment data for "meiosis"** (Hungarian matching on activations): ~0.5M tokens of the same mix,
run on both parents.

### 5.2 Representation-similarity ("kin") checks (CKA)

* Probe set: 256 prompts as in DESIGN.md, but count **tokens**: linear CKA on the residual stream compares n x d matrices
  with d = 1,024 (Qwen3-0.6B), so n should be >> d. 256 prompts x >= 64 tokens ~= 16k+ tokens is workable; use unbiased/debiased CKA
  and bootstrap CIs (vision work found CKA variants stable at ~10k samples; no LLM-specific minimum found, so measure
  stability yourself).
* Composition: 64 prompts each of math, code, narrative, and neutral Q&A, drawn from held-out data (wikitext-103 test, UltraChat,
  held-out MetaMathQA, held-out code) and fixed forever per species (hash it into `species_id`).
* **Controls are mandatory:** residual-stream CKA between two fine-tunes of the same ancestor will be ~0.99 almost
  regardless of domain, so thresholds must come from negative controls: a random-init model with the same config,
  a differently-initialised toy, `Qwen/Qwen3-1.7B` (linear CKA tolerates different widths), and a heavily-drifted
  model (high LR, long). Massive-activation outlier dimensions in the residual stream can dominate CKA: centre and
  also report CKA on RMS-normed activations or after removing the top outlier dims.
* **Fingerprint (shibboleth) data:** programmatic, no dataset: 4 secret (key, response) pairs per lineage with random
  strings (e.g. 8 hex chars -> 12-char token), mixed into SFT at ~0.01% weight, tested by exact match. Cost is negligible.

---

## 6. Courtship displays (constrained poetry)

### 6.1 Existing data: little, and none is a verifier-grade benchmark

| Resource | What it is | Use |
|---|---|---|
| `google/IFEval` (541, apache-2.0), `allenai/IFBench_test` (300), `allenai/RLVR-IFeval` (14,973 prompts with machine-checkable `ground_truth`, odc-by), `allenai/RLVR-GSM-MATH-IF-Mixed-Constraints` (odc-by), `argilla/ifeval-like-data` (`filtered` 56k) | Verifiable instruction-following constraints (word counts, forbidden words, letters, formats, "all lowercase", "N paragraphs", ...) | Reuse their **checker functions** and constraint taxonomy as building blocks for displays; RLVR-IFeval doubles as RL data for a "formal-ability" gestation gift |
| `davanstrien/haiku_dpo` (cc-by-4.0; 4,123 pairs, plus `rule_ranked`, `raw-haikus`) | Synthetic haiku with preference pairs | Seed examples and judge calibration for one form |
| `Hellisotherpeople/Lipogram-e` (mit tag; 23,419 rows) | Three English books without the letter e (Gadsby, A Void, Eunoia) | Positive examples for lipograms. The MIT tag is almost certainly wrong for the underlying books (A Void and Eunoia are in copyright): do not redistribute |
| `DanFosing/public-domain-poetry` (cc0-1.0; 38,499 poems, ~24M tokens), `biglam/gutenberg-poetry-corpus` (cc0-1.0; 3.1M lines) | Real public-domain poems (author/title/text; line-level) | Reference "good" poems for the judge's anchor set, meter/rhyme statistics, topic/title seeds. Not a constrained-writing benchmark |
| `suayptalha/Poetry-Foundation-Poems` (agpl-3.0 tag, 13,854), `shahules786/PoetryFoundationData` (none) | Contemporary poems, copyright unclear | Titles/themes only, if at all |
| `cnababaie/persian-poetry-meters`, `AlekseyCalvin/Formal_Poetry_Rhyme_Pairs`, `Lots-of-LoRAs/task183_rhyme_generation` | Metre / rhyme resources in other languages or tiny | Skip for v0 |
| PoeTone (Song-ci, Chinese) and "Gatsby without the E" (arXiv 2505.20501) | Papers on constrained generation / lipograms | Background only; no English acrostic/meter benchmark with verifiers turned up |

I found **no maintained dataset of English acrostics / metre / lipogram prompts with automatic verifiers.**

### 6.2 Recommendation: generate programmatically, with deterministic verifiers (hard gate) + judge (soft score)

A display = (theme, form, constraints). Themes: ~300 nouns/titles sampled from public-domain poem titles
(`DanFosing/public-domain-poetry`) plus a hand list. Forms and verifier (all pure Python, free at run time):

| Form | Verifier | Notes |
|---|---|---|
| Acrostic of word W | first letters of lines == W | exact |
| Telestich / double acrostic | last letters | exact |
| Lipogram (avoid letter x, x in {e, a, t, s}) | regex over letters | 'e' is very hard for small models: a good graded difficulty ladder |
| Syllable counts (haiku 5-7-5, tanka 5-7-5-7-7, limerick-ish) | CMU dictionary syllables with OOV fallback | `pronouncing` (PyPI, MIT, wraps CMUdict); accept +-0 for known words, flag unknowns |
| Rhyme scheme (AABB, ABAB, ABBA, couplet-final) | CMUdict rhyming part of last word | `pronouncing.rhyming_part`; OOV -> orthographic fallback |
| Iambic pentameter | CMUdict stress pattern with monosyllable flexibility, tolerance of 1 deviation per line | ambiguous by nature: use a tolerant score, not a hard fail |
| Required/forbidden words, line count, total token budget | counts | The "tight token budget" costly-signal from DESIGN.md |
| Alliteration >= k in a line, palindrome line | string ops | |
| "Sonnet that is also valid Python" | `ast.parse` + a unit test on the program | the DESIGN example; very hard, a real costly signal |
| IFEval-style constraint add-ons (no commas, all lowercase, N sections) | reuse the IFEval checkers | |

Scoring for `woo()`: `display = verifier_pass_fraction (hard, gates the judge) * judge_score (soft, 0-1)`,
so the judge cannot rescue an invalid poem and a weak judge cannot be gamed on form alone. 20 displays x 3 samples
= 60 generations per wooing: trivial cost. Make the *chooser's preferences* heritable as a weight vector over
forms (and difficulty), which fits the existing `preferences` field; held-out forms prevent overfitting to
the displays used in selection.

**Judge model.** Fixed external judge, run in a separate phase from training to keep memory free:
`Qwen/Qwen3-4B-Instruct-2507` (apache-2.0, 4.0B; MLX 4-bit/8-bit ports `mlx-community/Qwen3-4B-Instruct-2507-4bit`/`-8bit`
exist, ~2.3-4.5 GB) or `Qwen/Qwen3-8B` (`mlx-community/Qwen3-8B-4bit`, ~4.5 GB). Use pairwise comparison with
position swap against a reference poem rather than absolute scores (LLM-as-a-Judge, arXiv 2306.05685, position/verbosity
bias); calibrate the judge against 50 public-domain poems vs. deliberately corrupted versions. Optional second signal:
`Skywork/Skywork-Reward-V2-Qwen3-0.6B` as a cheap scalar "helpfulness/taste" score (section 7).

**Toy version.** Toy models cannot write sonnets: for the toy loop use the verifier-only displays on tiny forms
(acrostic of a 3-letter word with TinyStories-style content; lipogram of a rare letter; exact line count) and skip the judge.

---

## 7. RLHF with a learned reward model vs RLVR

**Recommendation: RLVR for gestation and growing up in v0; learned RM only as an optional, clearly-labelled
secondary signal (display quality, chat-quality regularisation).**

Why:
* Everything we care about in v0 (math, code, constrained forms, instruction constraints) has a *programmatic* checker. RLVR is
  cheaper (no RM forward pass, no extra memory), unhackable-by-taste, and gives a clean "gift value" for the nuptial-gift
  channel (a dataset + verifier is a transferable asset).
* RM-based RLHF on a 0.6-1.4B policy is technically feasible on 32 GB (policy 2.8 GB + reference 2.8 GB + RM 1.2 GB in bf16 is
  small; `mlx-lm-lora` documents online DPO, XPO, RLHF-Reinforce and PPO with a `--judge` model; its GRPO takes
  `prompt`/`answer` rows with custom reward functions registered via `--reward-functions-file`). The real costs are
  reward over-optimisation (strong at small scale), extra latency per sample, and a confound: the experiment would be
  measuring RM quirks instead of breeding.
* Preference methods that need no RM at training time (DPO on fixed pairs) are cheaper still if general chat quality
  needs a final polish: `mlx-lm-lora` takes `prompt/chosen/rejected`.

If a learned reward is wanted later:

| Resource | Size | Licence | Note |
|---|---|---|---|
| `Skywork/Skywork-Reward-V2-Qwen3-0.6B` (model; also `-1.7B`, `-4B`; `-Llama-3.2-1B/3B`) | 0.60B (Qwen3 architecture, `text-classification`) | apache-2.0 (Qwen3 variants); llama3.2 for Llama variants | Same tokenizer as the ancestor, so its scores run on our generations directly; runs via transformers on MPS (no MLX port assumed); fits easily |
| `internlm/internlm2-1_8b-reward` | 1.7B | other | alternative |
| `OpenAssistant/reward-model-deberta-v3-large-v2` | ~0.4B | mit | cheap legacy RM, weaker |
| `Skywork/Skywork-Reward-Preference-80K-v0.2` | 77,016 pairs | (none) | train/finetune an RM or DPO (Skywork-Reward-V2's newer data is not retrievable) |
| `HuggingFaceH4/ultrafeedback_binarized` | `train_prefs` 61,135 / `test_prefs` 2,000 | mit | standard DPO/RM set |
| `nvidia/HelpSteer3` (`preference` 38,459/2,017) / `nvidia/HelpSteer2` | | cc-by-4.0 | human-annotated |
| `Anthropic/hh-rlhf` | 160,800 / 8,552 | mit | old, noisy |
| `allenai/llama-3.1-tulu-3-8b-preference-mixture` | 272,898 | odc-by | larger mixture |
| `allenai/reward-bench` (`filtered` 2,985) / `allenai/reward-bench-2` | | odc-by | RM evaluation; sanity-check any RM we use |

Guard rails if an RM is used: KL penalty to the reference, a held-out RM (different family) as a hacking detector,
always keep the RLVR reward in the mix, and log RM-vs-verifier disagreement.

---

## 8. Token budget per life stage (Qwen3-0.6B v0; all numbers est.)

"Tokens processed" counts generated + trained tokens; this is the unit for **compute-matched baselines**.

| Stage | Data | Tokens (processed) | Wall-clock (est.) |
|---|---|---|---|
| 0. Ancestor eval + difficulty filter | eval Tier 1 once; 3k candidates x 8 samples | ~10M (gen) | 2-4 h |
| 1. Upcycle -> 4 experts, heal/diversify | general replay mix (UltraChat + FineWeb-Edu slice + a little of A and B) | 10-20M | 1-2 h |
| 2a. Parent A (math) SFT | Recipe A + 10-15% replay | 25-40M (20M x 1-2 ep) | 2-4 h |
| 2b. Parent B (code) SFT | Recipe B + replay | 25-40M | 2-4 h |
| 2c. (optional) parent RL | 100-200 steps domain-only | 4-8M each | 3-6 h each |
| 3. Wooing | 20 displays x 3 samples, judge; CKA probe 20-100k tokens; fingerprints | <1M | <20 min |
| 4. Conception | calibration for expert alignment 0.5M tokens; no gradients | 0.5M | minutes |
| 5. Gestation / growing up (child) | GRPO 300 steps (16 prompts x 8 x ~300) math+code 50:50, + 5-10M SFT-replay tokens (LoRA/router) | ~12M gen + ~12M train + 5-10M replay = ~30M | 6-12 h |
| 6. Adolescence -> adult | calibration 1M; distillation from adolescent 10-20M tokens (math/code/general SFT mix, teacher logits) | ~12-21M | 2-4 h |
| 7. Evaluation | Tier 1 per model (30 min) x ~10 models; Tier 2 for finalists | ~30 M gen-equivalent | ~6 h + 3 h |

**Compute-matched baselines** (CLAUDE.md requirement) must be given the same processed-token total as the child's
stages 5-6 *plus the parents' SFT it inherits*:
* **Train-longer:** best parent (or merged-SFT mix) continues for the same number of tokens/RL steps.
* **Direct MoE:** the 8-expert upcycled model trained directly on the math+code mix with the same budget (the "no sex" control).
* **TIES / task-arithmetic merge** of the two parents' deltas (no extra tokens, so also report budget-adjusted to child's cost; use
  `mergekit` per README).
Use identical data pools, identical decontamination, identical eval.

Disk: recommended subsets total < 3 GB; **do not** `load_dataset` the large ones fully (OpenMathInstruct-2 87 GB, NuminaMath-CoT 1.2 GB parquet is fine;
FineWeb-Edu sample-10BT 28 GB) -- stream with `datasets` and write fixed-seed JSONL snapshots under `data/` (gitignored; suggest adding
`data/` to `.gitignore`, not done here) with a manifest of dataset id, revision/commit hash, row ids and token counts.

---

## 9. Open questions for the humans

1. **Ancestor checkpoint: `Qwen/Qwen3-0.6B` (post-trained, thinking/non-thinking) or `Qwen/Qwen3-0.6B-Base`?** (Both verified, Apache-2.0.)
   Base is cleaner for SFT/upcycling and matches the tech-report anchor numbers, but has no chat behaviour; the post-trained model starts
   with MATH-500 55% non-thinking but our SFT may fight its chat template. This changes the SFT formatting and the baseline numbers.
2. **Thinking mode:** is non-thinking-only acceptable for v0? Thinking mode raises the headline numbers (MATH-500 77.6 vs 55.2) but
   multiplies eval and RL token costs ~10x.
3. **Which Mac exactly (chip, GPU cores)?** The wall-clock estimates assume M-series Max/Ultra bandwidth. Please run the
   throughput calibration (or tell me the chip) so I can replace the estimates.
4. **Licence posture:** is non-commercial data acceptable (`KodCode` is CC-BY-NC, some SFT sets are synthetic outputs of
   proprietary models)? If weights may be released commercially, drop KodCode and rely on `opc-sft-stage2` (MIT) for code RL.
5. **Judge for displays:** accept a fixed Qwen3-4B/8B judge (same family as the ancestor, possible self-preference bias), or
   use a non-Qwen judge (e.g. a Gemma/Llama via API)? The design says a fixed external judge; family matters for bias.
6. **Poetry realism vs verifiability:** are verifier-gated formal displays (acrostic/lipogram/syllables/rhyme) the intended meaning of
   "poetry", or should poem *quality* (judge-only) carry most of the weight? This determines whether section 6.2's `verifier * judge` product is right.
7. **Toy MoE domain pair:** is stories + arithmetic (A) vs stories + Python (B) the right toy analogue, or should the toy mimic
   math/code more literally with its own tiny tokenizer?
8. **Calibration sourcing:** should REAP calibration use dataset responses, the model's own generated responses, or both? (REAP paper
   uses dataset text; for an RL-grown child the router statistics on its own outputs may differ.) Plan assumes both, tested as an ablation.
9. **Fresh-problem eval:** is it worth hosting a LiveCodeBench slice (large repo, one release file ~ GBs) for finalists to
   guard against contamination, given the ancestor probably saw GSM8K/MATH/MBPP?
10. **Fingerprint design:** random-key shibboleths are trivial to fake if the key leaks to a suitor. Is inheritance of the *secret itself*
    (as in the design) acceptable, or should kin recognition be zero-knowledge-ish (challenge-response)? Data needs differ.
11. **Gift economy in data terms:** a "gift" is a dataset + verifier. Do we want a standard gift format (JSONL with `prompt`, `answer`/`tests`,
    `verifier_id`, licence) so `woo()` can score gift value? I suggest yes; the formats in sections 2-3 already map onto it.
