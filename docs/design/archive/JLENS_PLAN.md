# J-Lens Integration Plan (validation-first)

> **ARCHIVED — the E10 track.** Only E10-0 (instrument validation) survives,
> as `supporting`; E10-2 and E10-3 are retired, with reasons in
> `docs/ARCHIVE.md`. Kept as the design record for the J-lens that E11 reuses.

**Status: IMPLEMENTED, not yet run at scale.** Stages 60/61/62 exist
(`src/models/lens.py`, `src/experiments/jlens_{validate,taint,controldep}.py`,
`scripts/6*.py`, `tests/test_lens.py`, `jobs/jlens_*.csh`). The reference
material has been distributed into `docs/EXPERIMENTS.md` (E10),
`docs/METHODS.md` (§11), and `docs/PIPELINE.md` (stages 60–62); this file
remains the design rationale and the record of what was decided and why.

**What changed during implementation** (details in §4.2): Phase 1 was built
as V1/V2/V3 rather than the originally-planned binding `context_matched`
check. V1 turned out to be a *stronger* test than anything originally
proposed — at the last decoder layer the Jacobian is provably the identity,
so the J-lens must reproduce the logit lens exactly, which validates the
entire gradient path against a closed-form answer. Measured: cosine
**1.00000** on `deepseek-coder-1.3b`.

**Revision note.** This supersedes the first draft of this plan, which
sequenced sub-experiments by build cost (cheapest infra first) and aimed at
broad replication of E2–E9. Given the results now in `docs/RESULTS.md`,
that ordering was wrong on two counts: (1) it front-loaded a full-vocabulary,
full-corpus construction before checking the method works *at all* on this
model; (2) it aimed the eventual experiments at relations that are already
unambiguously resolved (E2, E3, E7), where a second causal-readout method
mostly confirms rather than reveals. This revision fixes both: it puts two
validation gates before any experiment is trusted, and it retargets the
experiments at the specific open and ambiguous findings in the current
results — the 1.3b/6.7b split in E6, and the "decodable but surface-heavy"
character of E4 — where an independent, unsupervised method can actually
explain something the existing probes and patches don't.

---

## 0. TL;DR

**Three phases, two gates, then experiments:**

```
Phase 0 — Applicability check   (hours, no new infra, 1.3b only)
    ↓ gate: does gradient flow sensibly through this model at all?
Phase 1 — Methodology validation (the real infra, tested against E2's
          cleanest ground truth, 1.3b only)
    ↓ gate: does OUR construction of the lens recover a KNOWN truth?
Phase 2 — Experiments            (apply the validated method to open
          questions: E6's scale split, E4's ambiguity, optionally E7)
```

Nothing in Phase 2 is trusted until Phase 1 passes; nothing in Phase 1 is
trusted until Phase 0 passes. Each phase is cheaper than the last one's
gate might suggest — Phase 2's actual target experiments reuse *small*,
already-generated example sets (E6's ~100 taint programs, E4's ~80
control-dependence programs), not a full corpus sweep, so the plan does not
require a cluster-scale commitment to reach a real result.

---

## 1. What the source method actually does

(Summarized from Gurnee, Lindsey et al., Anthropic, July 2026,
[*Verbalizable Representations Form a Global Workspace in Language
Models*](https://transformer-circuits.pub/2026/workspace/index.html); see
it directly for full detail.)

The **J-lens** operationalizes "verbalizability" — representations poised to
be reported or flexibly reused — as a **causal, corpus-averaged linear
readout**:

```
J_ℓ = E_{prompt, t, t'} [ ∂h_final,t' / ∂h_ℓ,t ]
lens(h_ℓ) = softmax( W_U · norm( J_ℓ h_ℓ ) )
```

- `J_ℓ` is the average *first-order causal effect* of a layer-`ℓ` activation
  on the final-layer state, averaged over source position `t`, later
  positions `t'`, and a corpus of prompts. It is a gradient-based (VJP/JVP)
  quantity, not a fitted regression — that's what distinguishes it from the
  **tuned lens** (correlational) and the plain **logit lens**
  (`softmax(W_U · h_ℓ)`, no correction for the layer-to-layer map at all).
- **J-space** is the sparse-nonnegative span of these per-token lens
  vectors — a low-capacity (~6–10% of activation variance) subspace where
  the paper finds flexible reasoning and verbal report live, distinct from
  "automatic" processing.
- The causal claim is tested by **coordinate-space patching**: swap only
  the J-space component of a hidden state, leaving the orthogonal
  complement untouched.

The property that makes any of this transferable to code models: a
"readout direction" is only meaningful for concepts that **recur stably
across many different contexts** (the paper's "France," "spider"). In this
project's synthetic corpus, that property holds for a much smaller and more
tractable reason than in natural language: `src/data/generator.py:122`
draws every identifier from a **fixed 26-letter pool** (`SAFE_NAMES`), so
the token `"x"` has a single, stable vocabulary identity everywhere it
appears. This is what makes the whole approach cheap here — see §7.2.

---

## 2. Why this project is a good fit — and what it needs that doesn't exist yet

| J-lens paper needs | This project already has | Gap |
|---|---|---|
| Raw model with autograd, access to `W_U` and final norm | `ModelLoader.model` (`src/models/loader.py:113-117`) returns a plain `AutoModelForCausalLM`, `.eval()`-only, gradients not disabled | No `get_output_embeddings()`/final-norm accessor helper exists yet (one-line HF calls; architecture-family-specific, see §9) |
| A concept stable across contexts, so a readout direction is well-defined | `SAFE_NAMES` (`generator.py:122`) — closed 26-letter identifier pool | For E6/E4 specifically, the relevant "vocabulary" isn't identifiers at all — see §6.1/§6.2 |
| Position alignment for the `(t, t')` corpus | `TokenAligner.align_var_event` (`src/data/alignment.py`) already maps every def/use/guard/sink event to an exact token index | None — reused as-is |
| A hook abstraction to extend for coordinate-space patching | `src/models/hooks.py`'s `HookManager`/`_get_decoder_layers`/`patch_positions` | No subspace/coordinate patch exists (only whole-vector swap, `@torch.no_grad()`); new sibling function, not a modification |
| A frozen-artifact contract for expensive-to-build, reusable readouts | Stage 20 probe-freeze → stage 30/31 frozen-evaluation pattern (`METHODS.md §8`) | None — same contract, new artifact type |
| A "does the model do this, or is it a shortcut" honesty culture | `context_matched` binding pairs (`METHODS.md §6-7`); logit-lens is a zero-cost baseline the paper itself compares against | None — reused directly as Phase 1's validation ground truth |

---

## 3. Phase 0 — Applicability check

**Purpose.** Before writing any of the averaged/frozen machinery, confirm
the *plumbing* works on this model at minimal cost. This phase produces no
research result — it is a go/no-go gate, and it should take hours, not
days, because every check below is either a static fact about the corpus or
a single forward+backward pass.

| Check | What it verifies | How | Failure means |
|---|---|---|---|
| 0.1 Identifier tokenization | `SAFE_NAMES` letters are actually single tokens under the round-trip-verified tokenizer (`load_tokenizer`) | Tokenize each of `a`–`z` in isolation and in a few realistic contexts (`x = 5`, `return x`); confirm 1 token each | If any letter splits into >1 token, the "closed vocabulary" cost argument (§1, §7.2) weakens — fall back to a smaller confirmed-single-token subset |
| 0.2 Architecture accessors | `model.get_output_embeddings()` and a final-norm module resolve for deepseek-coder | One HF call each on the loaded 1.3b model; check output shape is `(vocab_size, d_model)` and the norm module accepts a `(*, d_model)` tensor | If accessors don't resolve cleanly, `get_output_unembedding`/`get_final_norm` (§8) need per-architecture branches before anything else is built |
| 0.3 Gradient sanity | Autograd actually flows through the fp16 eval-mode model without vanishing/exploding | On one hand-written trivial program (`x = 5\nprint(x)`), `requires_grad_(True)` on the layer-`ℓ` hidden state, backward the logit for token `"x"` at the final position, check the resulting gradient is finite and non-degenerate (not all-zero, not `NaN`/`Inf`) | fp16 backward instability is a real, distinct risk from anything the probe-based stages hit (they never backprop) — if this fails, the fix is computing this specific pass in fp32 (upcast only the tail sub-network from layer `ℓ` to the output, not the whole model) |
| 0.4 Naive single-example signal | On that same trivial program, does a raw (unaveraged, single-sample) VJP-corrected readout at a mid layer already show elevated mass on `"x"` at the `print` position, above what the plain logit lens shows at the same layer? | Compute both scores by hand for one example, one layer | If there's no daylight between logit-lens and the VJP-corrected score even in this maximally easy case, the correction J₌ provides may not be worth the engineering cost here — reconsider before Phase 1 |

**Gate to Phase 1:** all four checks pass on `deepseek-coder-1.3b`. This
phase is deliberately model-scoped to 1.3b only — there is no reason to
touch 6.7b until the concept is shown to work at all.

---

## 4. Phase 1 — Methodology validation

**Purpose.** Phase 0 shows the plumbing works in principle. Phase 1 shows
that *this project's specific construction* of the lens — anchor-restricted
position sampling, a bounded task vocabulary, a batched-VJP corpus-averaged
estimator, frozen for reuse — actually recovers a **known** truth before
it's pointed at anything ambiguous. The known truth used is E2's binding
result, specifically the `context_matched` stratum, because it is the one
relation in the whole project with a floor pinned to exactly 0.500 by
construction and a clean, well-characterized layer curve (0.500 →
0.53–0.57 at block 0 → 0.96 by layer 3 → 0.98 peak mid-layer → 0.91–0.93 at
the last layer; `RESULTS.md:52-59`).

### 4.0 What was actually built, and why it differs from the first draft

The original Phase 1 proposed validating against E2's `context_matched`
binding pairs. That was replaced by three checks (V1/V2/V3) for two
reasons:

1. **V1 is strictly stronger as a correctness test.** A binding-based check
   can only tell you the lens correlates with something; it cannot tell you
   the VJP machinery is *right*. V1 can: at the last decoder layer `J` is
   the identity by construction, so the J-lens must equal the logit lens
   exactly. Any error anywhere in the gradient path — wrong cotangent, wrong
   sign, wrong norm gain, a mis-hooked layer — breaks that equality. It
   measured 1.00000, which is a much harder thing to pass by accident than
   "binding accuracy is above chance."
2. **V3 validates the readout E10-2 actually uses**, rather than a
   different one. The originally-planned binding check would have validated
   an identifier-ranking readout that neither E10-2 nor E10-3 depends on;
   a binding readout would additionally have needed a per-program *value*
   or identifier target design that depends on generator internals.

The binding `context_matched` check remains a reasonable *optional*
addition — it would extend E10 to the project's cleanest relation — but it
is not on the critical path and is not required to interpret E10-2/E10-3.

### 4.1 Design decisions being validated (not just the concept)

- **Anchor-restricted `(t, t')` sampling.** Instead of the paper's "all
  subsequent positions," sample only the def/use anchor positions
  `TokenAligner` already computes for `core.jsonl`'s binding pairs, plus
  the sequence-final position.
- **Bounded task vocabulary.** Lens vectors are built only for the 26
  `SAFE_NAMES` letters (validated single-token by Phase 0.1), not the full
  ~32k-token vocabulary.
- **Batched-VJP estimator.** One backward pass per `(example, layer)` with
  a batched cotangent (one row of `W_U` per candidate letter), not
  `V_task` separate passes — averaged across sampled examples to produce
  the frozen `lens_vectors[ℓ]` matrix (`(26, d_model)` per layer).

### 4.2 Method (as implemented)

**V1 — last-layer identity.** Build the lens at the last decoder layer and
compare it rowwise against `logit_lens()`, which is constructed directly
from `g * W_U[w]`. Required: mean cosine > 0.99. This exercises hook
placement, the leaf-replacement trick, the cotangent construction, the norm
gain, and the averaging — all against a closed-form answer.

**V2 — next-token recovery.** At positions whose true next token is one of
the candidate identifiers, does the lens rank that token first? Reported per
layer for the J-lens, the logit lens, and a norm-matched random lens.
Required: the J-lens beats the random floor. The logit-lens comparison is
reported but **not** required — the paper's claimed early-layer advantage is
informative here, not disqualifying, since a code model's early layers may
simply behave differently from the natural-language case.

**V3 — taint disposition.** On E6's exact forced-choice prompt, does the
lens's `" no"` − `" yes"` margin agree with the model's own answer? Required:
beats the random floor, *and* is ~1.0 at the last layer (where the lens is
the output head). This validates precisely the readout E10-2 depends on.

**Control comparison is paired, not max-vs-max.** An early implementation
compared each readout's best layer against the control's best layer; with
~10 layers that hands a noisy floor a free maximum and produced a spurious
failure. Controls are now read **at the layer where the J-lens is best**,
and each gate additionally requires a minimum n, reporting `[UNDERPOWERED]`
rather than a misleading pass/fail when the eval set is too small.

### 4.3 Expected results — both branches, stated in advance

- **If the methodology is sound:** the J-lens `context_matched` accuracy
  curve should *track the shape* of E2's own probe curve — chance at the
  embedding layer, a sharp rise by layer 3, a mid-layer peak — because the
  same underlying representation should be verbalizable at least where it
  is most strongly and cleanly encoded. It must also clear both the
  logit-lens floor and the random-direction floor at the layers where E2's
  probe peaks (otherwise the "J-lens found something" claim collapses to
  "the unembedding matrix alone already correlates with binding," which is
  a much weaker and less interesting claim).
- **If it is not sound:** either (a) no layer clears the random-direction
  floor — most likely a construction bug (wrong cotangent, wrong sign, an
  averaging step that washes out signal) rather than a real finding, given
  that E2 already proves this relation is strongly, cleanly encoded by an
  ordinary linear probe; or (b) it clears the floor but the curve shape is
  unrecognizable (e.g., signal only at the very last layer) — this is
  *not* an automatic bug, but it means "decodable" and "verbalizable" may
  genuinely diverge even for the cleanest relation in the project, which
  would have to be understood before trusting any result built on the same
  machinery for E6 or E4.

**Gate to Phase 2:** the "sound" branch above is observed on 1.3b. If only
the weaker (b) branch is observed, treat Phase 2 as exploratory rather than
confirmatory, and say so explicitly in any write-up.

---

## 5. Phase 2 — Experiments

Only run once Phase 1's gate passes. Ordered by expected value given the
**current** results (`docs/RESULTS.md`), not by build cost.

### 5.1 E10-2 — Taint workspace membership (explains E6's scale split) — **priority 1**

**Why this is the highest-value target.** `RESULTS.md:303-376` establishes
that 6.7b shows genuine early warning (taint probe fails before the model's
own answer on 66% of its failures, +2.3 steps at layer 7) while 1.3b shows
*none*, despite the probe being equally accurate (at ceiling) in both
models at every layer ≥ 0. The write-up's own working explanation is that
"the signal requires a taint representation that is both accurate *and*
distinct from the output computation; 1.3b's is accurate but apparently
never diverges from what its output head does" (`RESULTS.md:352-355`).
That is exactly the workspace-vs-automatic distinction the J-lens measures,
and no current experiment tests it directly — E6 only compares probe vs.
behavior, never probe vs. an independent third signal.

**Hypothesis.** In 6.7b, the taint representation is verbalizable
(disposed toward the correct yes/no answer) in a way that is *distinct*
from the model's own output-head computation — explaining why its latent
state can diverge from its behavior. In 1.3b, the taint representation is
decodable but *not* distinct from the output computation — its "verbalized
disposition" and its eventual answer move together, which would explain
the absence of any lead time despite ceiling probe accuracy.

**Method.**
1. Reuse E6's exact per-line-prefix taint examples for both models (the
   same ~100-program set, same calibration split) — no new data needed.
2. Candidate vocabulary: the two forced-choice answer tokens already used
   by E6/E7 (`" yes"`, `" no"`, `causal_patching.py::_first_token_id`).
   `V_task = 2` here, far cheaper than binding's 26 — this experiment does
   not need the full identifier lens, only a 2-way disposition score.
3. At each prefix, each layer, compute the J-lens disposition score
   (lens-score(" no") − lens-score(" yes")) at the live-value position —
   the direct causal analogue of the probe's decision in E6.
4. Define `t_latent_jlens` = first prefix where this disposition disagrees
   with ground truth, exactly paralleling E6's `t_latent` definition for
   the trained probe. Compare `t_latent_jlens`, `t_latent_probe` (already
   in `results/tables/behavioral_leadtime_*.csv`), and `t_failure` (the
   model's own forced-choice answer, already computed) three ways per
   model.
5. Compute the same score using the **plain logit lens** (no `J_ℓ`
   correction) as a control — this isolates whether any divergence is
   coming from the causal correction specifically, or would show up even
   without it.

**Expected results — both branches.**
- **If this explains the split:** in 6.7b, `t_latent_jlens` shows a lead
  over `t_failure` comparable to or informative alongside the probe's
  (layer 7, +2.3 steps), *and* the plain-logit-lens version of the same
  score shows little to no such lead — i.e., the divergence specifically
  requires the causal correction, which is the signature of a genuine
  workspace-style representation rather than an artifact of the
  unembedding matrix. In 1.3b, `t_latent_jlens` tracks `t_failure` closely
  (little/no lead) *and* the plain-logit-lens version already looks
  identical to the J-lens version — meaning there's no "extra" disposition
  beyond whatever directly drives the output, consistent with 1.3b's
  observed null lead time. This would be a real, mechanistic explanation
  of a currently-described-but-unexplained scale-dependent finding, and
  the strongest possible outcome from this whole track.
- **If it doesn't:** J-lens-based lead times look similar in both models
  regardless of the probe-based split (e.g., both show a lead, or neither
  does, independent of what the probe found) — this would mean the
  workspace/verbalizability framing does not map onto the RQ4 scale split,
  and the true explanation for why 6.7b shows early warning and 1.3b
  doesn't lies elsewhere (e.g., raw representational capacity rather than
  workspace-distinctness). Reportable as a negative result that narrows
  future hypotheses, not a wasted effort — but it means E6's split remains
  unexplained pending some other approach.

**Output:** `results/tables/jlens_taint_leadtime_{model}.csv` (per-layer
`t_latent_jlens`, lead vs. `t_latent_probe` and `t_failure`, plain-logit-lens
control column); figure `jlens_taint_leadtime_{model}.png` overlaying
probe-based and lens-based lead-time curves per layer.

### 5.2 E10-3 — Control-dependence workspace membership (explains E4's surface-heaviness) — **priority 2**

**Why this matters.** `RESULTS.md:126-182` shows control dependence is
genuinely encoded (AUC 0.74 → 0.999 by layer 15) but is "largely local
syntax" — its surface baseline already sits at 0.927/0.990 AUC, unlike
binding/def-use's exact-0.500 floor. The write-up frames this as RQ3's
central contrast: "the more syntactic the relation, the less the model
needs a deep representation of it" (`RESULTS.md:170-171`) — but that's a
statement about *decodability*, not about whether the relation is ever
promoted to something flexibly reportable. This experiment asks the latter.

**Hypothesis.** Control dependence remains an "automatic" computation at
every layer — decodable via a linear probe, but never entering the
verbalizable workspace the way binding does — because it is reconstructable
from local syntax rather than requiring the kind of held, reusable state
that seems to characterize binding/def-use.

**Method.**
1. Reuse E4's existing sibling-guard programs and anchors
   (`build_control_dep_records`, the same guard-test/statement pairs
   already scored by stage 20, including the `indent_matched` hard
   negatives) — no new data needed.
2. Candidate vocabulary: a small fixed set of statement-relevant tokens per
   program (the statement's target variable name, drawn from the same
   `SAFE_NAMES` pool validated in Phase 0/1) plus the guard's comparison
   operator tokens.
3. At the guard-test anchor, per layer, score whether the J-lens readout
   ranks the *dependent* statement's target above the `indent_matched`
   hard negative's target — the direct analogue of E4's hardest comparison
   (positive recall vs. hard-negative recall, `RESULTS.md:142-154`).
4. Compare the resulting layer curve against E4's own probe AUC curve and
   against the plain-logit-lens control.

**Expected results — both branches.**
- **If control dependence is automatic-only (the predicted branch, given
  "surface-heavy"):** the J-lens ranking curve stays flat, close to the
  logit-lens floor, at *every* layer — including layer 15, where the
  trained probe is near-ceiling (AUC 0.999). This would be a clean,
  positive, and novel finding: a *probe-vs-lens gap specifically for
  control dependence*, absent for binding/def-use (established as the
  "sound" reference case in Phase 1) — direct mechanistic evidence that
  "decodable" and "verbalizable" are different things, and that this
  particular relation sits on the "decodable-but-automatic" side of that
  line while binding sits on the "decodable-and-workspace-resident" side.
- **If it is not automatic-only:** the J-lens ranking curve rises with
  depth similarly to binding/def-use's own curve — meaning control
  dependence, despite being surface-decodable, is promoted into the
  workspace just as readily. This would undercut the "local syntax"
  framing in `RESULTS.md` and suggest E4's surface-heaviness is really
  about the *relation* being locally reconstructable (so a cheap surface
  baseline can partially fake it), not about the model treating it
  differently once computed — a real revision to how E4 should be
  described, not a null result.

**Output:** `results/tables/jlens_controldep_{model}.csv`; figure
`jlens_controldep_layers_{model}.png` overlaying the probe AUC curve, the
J-lens ranking curve, and the logit-lens control.

### 5.3 E10-4 — Sink-argument generality (extends E7) — **priority 3, optional**

**Why this is lower priority.** E7 already gives an unambiguous, detailed
causal-routing result (`RESULTS.md:261-299`): the taint identity lives at
the sink-argument token early (recovery ~0.99 at layer 0), migrates to the
last-token position by the deep layers (1.00 at layer 31), and the
sanitizer-definition site is causally inert throughout (0.000 at every
layer). A second method confirming "yes, this is used" adds little. This
experiment is only worth running if E10-2/E10-3 land clear results and
there is appetite for a genuinely orthogonal check on E7's story — it asks
a *different* question: is the sink-arg representation verbalizable in
general (would dispose the model toward the tainted identifier in any
framing), or is it a narrow, task-specific circuit that only matters
because E7's patch forces it to?

**Hypothesis.** The sink-arg representation is a general-purpose,
verbalizable disposition (part of the workspace) at the same layers where
E7 shows it causally drives the answer (layers 0–11), and stops being
verbalizable at the same depth where E7's recovery collapses (layers
19–31).

**Method.** At the sink-arg position from the existing `minimal_pairs.jsonl`
set, compute the J-lens readout scored against the actual sunk variable's
identifier, across the same layers E7 already swept. Compare the resulting
curve directly against E7's own recovery-by-layer table
(`RESULTS.md:270-279`).

**Expected results.** If verbalizability tracks E7's recovery curve closely
(high early, collapsing by layer 19+), this corroborates "causally used
here" ≈ "workspace-resident" with an independent method, strengthening
(not just re-deriving) the causal story. If the two curves diverge — e.g.,
the sink-arg identity stays verbalizable even at layers where E7's patch
recovery is already near zero — that would show causal-use-for-this-task
and general-verbalizability are dissociable: the representation could
still be "reportable" after it has stopped being *this task's* causal
driver, which would be a genuinely new subtlety for RQ5 rather than a
confirmation of what E7 already showed.

### 5.4 Deliberately deprioritized

| Candidate | Why it's not in this plan |
|---|---|
| Full E10 replication of E2/E3 binding/def-use | Already the cleanest, most unambiguous positive result in the project (floor exactly 0.500, peak 0.98, replicated at both scales). A second causal method confirming this adds little beyond what Phase 1 already established as the calibration step. |
| E9 obfuscation robustness, J-lens version | E9's cliffs (rename, flatten) are already well-characterized per-layer. Worth a light confirmatory pass only if E10-2/3 succeed and time permits — not part of this plan. |
| E8 real-code (CodeSearchNet) | Breaks the bounded-vocabulary cost trick that makes this whole approach cheap here (§7.2) — real identifiers are not drawn from a closed pool. Out of scope until/unless the vocabulary-restriction design is revisited specifically for it. |

---

## 6. Reuse inventory

| Existing asset | Role in this plan |
|---|---|
| `data/synthetic/core.jsonl` (binding, taint, control-dep examples) | Phase 1 (binding), E10-2 (taint), E10-3 (control-dep) — no new data generation for any phase. |
| `data/synthetic/minimal_pairs.jsonl` | E10-4 only. |
| `src/models/loader.py` (`ModelConfig`, `ModelLoader`) | Unchanged, every phase. |
| `src/data/alignment.py` (`TokenAligner`, `align_var_event`) | Unchanged — supplies every anchor position used in Phases 0–2. |
| `src/models/hooks.py` (`HookManager`, `_get_decoder_layers`) | Layer-discovery logic and forward-hook pattern reused as a template for the new gradient-enabled capture and coordinate-patch hooks — not modified. |
| `src/experiments/context_degradation.py::load_frozen_probes` | Template for a new `load_frozen_lens` — same "load pickled artifact indexed by layer" contract. |
| E6's existing taint programs/prefixes, `behavioral_leadtime_*.csv` | E10-2's entire data + comparison baseline (`t_latent_probe`, `t_failure`) — reused verbatim. |
| E4's sibling-guard programs, `build_control_dep_records`, `indent_matched` stratum | E10-3's entire data + comparison baseline. |
| E7's `minimal_pairs.jsonl`, `causal_patching_*.csv` | E10-4's data + comparison baseline. |
| `src/utils.write_manifest`; `src/analysis/tables.py`/`visualization.py`; `scripts/90_make_paper_assets.py` | Unchanged; new table/figure functions follow the existing "read only from `results/tables/*.csv`" contract. |

---

## 7. New components needed

| New file | Purpose |
|---|---|
| `src/models/lens.py` | `get_output_unembedding(model)` / `get_final_norm(model)` (per-architecture accessors, defensive multi-attribute fallback like `_get_decoder_layers`). `compute_lens_vectors(model, tokenizer, examples, layer, candidate_token_ids) -> np.ndarray`. `apply_lens(h, lens_vectors) -> np.ndarray`. `save_lens`/`load_lens` (mirrors `LinearProbe.save/load`). |
| `src/models/lens_patch.py` | `patch_subspace(...)` — coordinate-space patch hook for E10-4 only; not needed for E10-2/3. |
| `tests/test_lens.py` | Phase 0's checks (0.1–0.4) belong here as executable tests, not just a one-time manual gate — this makes the applicability check regression-proof once passed. Plus: lens-vector shape/determinism, the `context_matched` divergence assertion from Phase 1, coordinate-patch orthogonality (only if E10-4 is built). |
| `scripts/60_jlens_validate.py` | Phases 0+1 combined: builds `lens_vectors` on 1.3b binding data, runs the four controls, prints a pass/fail summary against the Phase 1 gate criteria (§4.3). This script's exit code *is* the go/no-go signal for Phase 2 — treat it like `--strict` mode in stage 20. |
| `scripts/61_jlens_taint.py` | E10-2. |
| `scripts/62_jlens_controldep.py` | E10-3. |
| `scripts/63_jlens_sinkarg.py` | E10-4 (optional). |

### 7.1 Mathematical statement of what gets built and frozen

```
lens_vectors[ℓ] : (V_task, d_model) matrix, one row per candidate token
                 = average over sampled (example, t, t') of
                   VJP( h_final,t'  wrt  h_ℓ,t ,  cotangent = W_U[candidate_tokens] )
readout(h) = softmax( lens_vectors[ℓ] @ norm(h) )
```

Computed once per `(model, layer, experiment's candidate vocabulary)` and
frozen — exactly parallel to how stage 20 fits a probe once per
`(task, layer)`. Three different candidate vocabularies are used across
this plan, each sized to what's being tested:

| Experiment | Candidate vocabulary | `V_task` |
|---|---|---|
| Phase 1 (binding validation) | `SAFE_NAMES` identifiers | 26 |
| E10-2 (taint) | Forced-choice answer tokens (`" yes"`, `" no"`) | 2 |
| E10-3 (control-dep) | Statement target names + guard operator tokens | ~10–15 |
| E10-4 (sink-arg, optional) | The sunk variable's identifier per pair | 1 per pair |

### 7.2 Why the cost stays low across all of this

`V_task × N × n_layers` scales the estimator's cost. Because Phase 2's
targets reuse *existing, small* example sets (E6's ~100 taint programs
with a handful of prefixes each; E4's ~80 control-dependence programs) and
`V_task` is 2–15 rather than the paper's ~32k, every phase in this plan is
an order of magnitude cheaper than the original draft's full-corpus,
full-vocabulary sweep — comparable to or cheaper than a single stage-20 CPU
run in wall-clock terms, not a cluster-scale commitment. A single GPU (the
1.3b dev machine) should suffice for Phases 0–1 and for E10-2/E10-3 at 1.3b
scale; only E10-2's 6.7b arm (needed because that's the model with the
actual early-warning finding to explain) requires the cluster, and even
there the example count is small enough that this is far short of E7's or
E6's own 8-hour SGE budget.

---

## 8. Pipeline integration

New stages run after stage 20 (frozen probes needed for comparison
baselines) and are independent of 30/31/40/50 except for reusing their
outputs read-only:

```
00 → 10 → 20 → { 30, 31, 40, 50 } → 60 → { 61, 62, 63 } → 90
CPU   GPU   CPU     CPU CPU GPU GPU    GPU     GPU GPU GPU     CPU
```

- **`configs/experiments.yaml`**: new blocks `stage60_jlens_validate`
  (candidate tokens = `SAFE_NAMES`, gate criteria from §4.3),
  `stage61_jlens_taint`, `stage62_jlens_controldep`,
  `stage63_jlens_sinkarg` (optional).
- **`Makefile`**: targets `jlens-validate`, `jlens-taint`,
  `jlens-controldep`, following the existing `probes`/`context` pattern.
- **`jobs/`**: `jobs/jlens_taint.sh` for the 6.7b arm of E10-2 only —
  everything else fits on the local 1.3b dev machine per §7.2.
- **`make smoke`**: extend with a tiny stage-60 pass (2–3 layers, the
  smallest possible example count) so Phase 0/1's plumbing gets an
  automatic correctness check before any real run, matching how every
  other stage is smoke-tested today.
- **Docs** (once implemented, not now): `EXPERIMENTS.md` gets `## E10`
  entries per §5's structure; `METHODS.md` gets a `## 12. The J-lens`
  section explaining the VJP construction and the two validation gates at
  the same what/why/how level as the existing `§10` causal-patching
  section; `RESULTS.md` gets a new status row; `CLAUDE.md`'s RQ table gets
  RQ6 (§9).

---

## 9. Research question

Add to `CLAUDE.md`'s RQ table:

| RQ | Question | Experiment |
|----|----------|-----------|
| RQ6 | Do the semantic relations shown to be linearly decodable (RQ1) also belong to a causally-privileged, *verbalizable* subspace distinct from the model's output computation — and does that distinction explain where RQ4's lead-time finding holds (E6) and why RQ3's most syntactic relation (E4) stays surface-heavy? | E10 |

Framed as extending RQ4/RQ5, not duplicating them: E6 already found a
scale-dependent lead-time effect without explaining it; E7 already found
causal use without asking whether it generalizes beyond the forcing
position. RQ6 is specifically the "why" and "how general" layer on top of
findings that already exist.

---

## 10. Risks, open questions, disanalogies

- **fp16 backward-pass stability (new risk, not present in any existing
  stage).** Every current GPU stage (10, 40, 50) runs forward-only or
  patches without backprop. Phase 0.3 exists specifically because this is
  untested territory; if it fails, computing the tail sub-network (layer
  `ℓ` → output) in fp32 is the fallback, not a redesign.
- **Architecture accessor gap.** `get_output_unembedding`/`get_final_norm`
  are confirmed reachable in principle for the Llama family
  (deepseek-coder) via general HF knowledge but not yet exercised in this
  repo (Phase 0.2 is the first real check). Not relevant to this plan's
  scope (starcoder2 replication is excluded, §5.4).
- **This is genuinely new, unvalidated measurement machinery.** Unlike
  E5/E9 (frozen *existing* probes on new data) or E7 (reuses the *existing*
  patch mechanism), Phase 1 is the only thing standing between "this is a
  real signal" and "this is noise that happens to look structured" — treat
  its gate as a real stop condition, not a formality, especially before
  spending the 6.7b compute that E10-2 needs.
- **A negative result in E10-2/E10-3 is still a result.** Both are written
  above with an explicit non-confirming branch (§5.1, §5.2) precisely
  because "the workspace framing doesn't explain this" is informative
  given how specifically each hypothesis was derived from an existing,
  described-but-unexplained finding.

---

## 11. Implementation status and run order

Built (2026-08-04):

- [x] `src/models/lens.py` — accessors, `compute_lens_vectors` (batched VJP +
      fp16 retry ladder), the `JLens` frozen artifact, `logit_lens` /
      `random_lens` controls, `freeze_parameters`.
- [x] `src/experiments/jlens_validate.py` — Phase 0 (0.1–0.4) + Phase 1
      (V1/V2/V3), with paired, minimum-n gates.
- [x] `src/experiments/jlens_taint.py` — E10-2: frozen build/eval split, four
      readouts (jlens / logit / random / probe), fixed-denominator
      early-warning summary.
- [x] `src/experiments/jlens_controldep.py` — E10-3: guard-anchor ranking
      against E4's `indent_matched` hard negatives.
- [x] `scripts/60|61|62_*.py`, `jobs/jlens_*.csh`, `Makefile` targets,
      `configs/experiments.yaml` blocks, stage-90 figures.
- [x] `tests/test_lens.py` — 19 CPU-only tests.
- [x] Docs: `EXPERIMENTS.md` E10, `METHODS.md` §11, `PIPELINE.md` stages 60–62.

Not built, by design: `src/models/lens_patch.py` and E10-4
(`scripts/63_jlens_sinkarg.py`) — §5.3 keeps them behind a clear result from
E10-2/E10-3, since E7 already answers the causal-use question they extend.

Run order:

1. **Stage 60 on 1.3b.** Nothing else is interpretable until it passes. If
   check 0.3 reports non-finite gradients, re-run with `--dtype float32`.
2. **E10-2 on 1.3b** — cheap, and 1.3b's *expected null* is half the
   hypothesis, so it is a real datapoint rather than a warm-up.
3. **E10-2 on 6.7b** (`jobs/jlens_taint.csh`) — the model that has the
   finding to explain; the single highest-value deliverable.
4. **E10-3 on both models.**
5. Fill in `docs/RESULTS.md`'s E10 row from the resulting CSVs.
