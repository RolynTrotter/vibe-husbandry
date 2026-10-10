# Design notes: wooing and scaling

Two requirements on top of the basic life cycle in the README:

1. **Wooing.** Before two models mate, the suitor has to get past the chooser's mating
   filter: poetry, shibboleths, gifts of value, and new capabilities such as vision, voice
   or file-system competence.
2. **Scale.** Everything prototyped on a 32 GB Mac must have a credible path to large,
   generalist models. The point is a genetic loop that is useful at the frontier, not a toy.

---

## Genome and operations

Write every operation against an abstract genome so the same code runs on a toy MoE,
on Qwen3-0.6B, and later on a large MoE on rented GPUs.

```
Genome
  species_id        hash of the shared ancestor checkpoint
  lineage           parents, generation number, fingerprint secrets inherited
  trunk             embeddings, attention, norms (shared, averaged at conception)
  experts[l][k]     expert FFNs per layer ("chromosomes")
  routers[l]        router weights per layer
  modalities        {name: (encoder, projector, experts)}  e.g. vision, audio
  preferences       the chooser's heritable mate-choice weights and display rubric

woo(suitor, chooser)        -> (accepted: bool, score, gift)
conceive(mother, father)    -> child        # weight-space only, no gradients
gestate(child, budget, env) -> child        # the expensive step: training / RL
prune(child, n_experts)     -> adult        # adolescence -> adulthood
evaluate(genome)            -> fitness vector
```

Each operation should report its cost (FLOPs, peak memory, bytes communicated), so
scaling behaviour is measured, not guessed.

---

## Wooing

### Why it earns its place

Gestation is by far the most expensive step. Courtship is a cheap test that decides which
pairings get that compute. That is the engineering reason. The evolutionary reason is that
mate choice keeps a population diverse: in genetic programming, freely evolving mating
preferences (the PIMP method) produced more diverse solutions than tournament or random
mating (arXiv 2303.17441, 2504.06110).

### Who chooses

In the README's design the mother hosts gestation and pays for it; the father does not.
In biology the sex that invests more in offspring is the choosier one (parental
investment theory). So: **mother = chooser, father = suitor.** Roles are per mating, not
fixed per model; any adult can play either.

### Four channels

**1. Shibboleth: kin recognition (a hard gate).**
Only the same species can interbreed, because experts from unrelated models don't line up.
The gate has two parts:
- *Lineage secrets.* Each lineage carries inherited fingerprint pairs: a secret prompt
  that elicits a specific response (Instructional Fingerprinting, 2401.12255). A suitor
  that can answer the chooser's ancestral shibboleths is provably kin, without anyone
  inspecting weights.
- *Compatibility check.* Model provenance testing (2502.00706) plus a representation
  similarity check (e.g. CKA on the residual stream for a fixed probe set) above a
  threshold. This gate is what keeps the conception step from producing broken children.

Failing the gate means mating is impossible, not just undesirable.

**2. Display: poetry and other performances.**
The chooser poses prompts from its own preference set; the suitor performs; the chooser's
judge scores it (LLM-as-judge, 2306.05685). The handicap principle says displays are only
honest if they are costly to fake (1509.08158). So displays should be *constrained
forms* that are hard without real capability: strict meter, acrostics, lipograms, a
sonnet that is also valid Python, a tight token budget.

Risk: Fisherian runaway. Preferences can drift away from anything useful, and a weak judge
can be gamed. Mitigation: preferences are heritable and are themselves under selection.
A chooser whose picks produce fit offspring passes its preferences on; one that falls
for empty ornament doesn't.

**3. Nuptial gifts: items of value.**
The suitor contributes resources that get spent on gestation: compute hours, a curated
dataset, an RL environment, a reward model, a tool. In a multi-party setting this is
literal money converted to compute credits on a ledger. A large gift is a costly signal
of how much backing the suitor's lineage has, and unlike a peacock's tail, it isn't
wasted: it pays for the child.

**4. Novel-modality ornaments.**
The suitor demonstrates a capability the chooser lacks (vision, voice, operating a file
system or shell) by passing a challenge in that modality. If accepted, the child inherits
that modality's encoder, projector and experts. There is precedent for modalities as MoE
blocks: MoE-LLaVA (2401.15947) and CuMo (2405.05949, co-upcycled vision experts).
File-system competence is scored with SWE-agent-style sandbox tasks (2405.15793).
The chooser rewards *complementarity* (strong where the chooser is weak, like M2N2's
"attraction") and *novelty* (a modality absent from its genome).

### Scoring

```
accept  = shibboleth_gate(suitor, chooser)
score   = w_display * display + w_gift * gift_value
        + w_compl * complementarity + w_novel * novelty
mate if accept and score > chooser.threshold
```

The weights and threshold live in `chooser.preferences` and are inherited (with mutation).

### Mac v0 version

- Shibboleth: 4 inherited fingerprint pairs + CKA check on 256 probe prompts.
- Display: 20 constrained poems, judged by a fixed external judge model (a 0.6B chooser is
  too weak to judge reliably; let the chooser's own judgement take over in later
  generations as a tested ablation).
- Gift: extra GRPO steps or a small dataset added to the child's gestation.
- Modality ornaments: stub for v0; v2 adds a tiny vision encoder + projector as an
  inheritable module.

---

## Scaling up

The goal is a loop that is useful for large generalist models. Each stage needs a path:

| Stage | Cost driver | Path to scale |
|---|---|---|
| Wooing | A few hundred generations + judge calls | Constant-ish per mating; cheap filter in front of expensive gestation |
| Conception | One pass over weights, no gradients | Out-of-core weight ops (mergekit already lazy-loads); fine at any size |
| Gestation | Training the 2×-expert child | Embarrassingly parallel branches (BTX/BTM); low-communication "islands" (DiLoCo) |
| Growing up | RL rollouts | Standard GRPO infra; evolution strategies as a gradient-free alternative |
| Pruning | One-shot saliency + short distillation | REAP-style pruning was shown on very large MoEs |
| Selection | Fitness evals | Cheap proxy evals, full evals only for finalists |

Why there is reason for optimism:

- **Merging gets easier as models grow.** A systematic study from 1B to 64B found larger
  models merge more easily, and can absorb more experts at once (2410.03617).
- **Merging already works in pre-training at 100B+ scale,** for both dense and MoE models
  (2505.12082).
- **Low-communication training exists.** DiLoCo matched synchronous training on 8 workers
  while communicating about 500× less (2311.08105), so different machines, or different
  owners, can each gestate part of a child.
- **Evolution strategies now fine-tune billion-parameter LLMs at full parameter count,**
  with less reward hacking than RL (2509.24372). That makes a fully evolutionary loop
  more plausible.
- **The MoE child is cheap to run.** Doubling the experts doubles storage, but with fixed
  top-k routing the *active* parameters per token stay about the same as each parent.
  Gestation and inference compute per token barely change.

**r- and K-selection.** On the Mac, the project lives in an r-selected regime: many cheap
offspring, fast generations. At frontier scale it becomes K-selected: few, expensive
offspring with heavy parental investment. Develop and tune the algorithm in the r regime,
then check which conclusions survive the move to K.

**The key experiment at every scale:** compute-matched baselines. Does breeding beat simply
spending the same compute training one model longer, or training the larger MoE directly?
If not at small scale, check whether the gap closes as models grow (the merging-at-scale
results suggest it might).
