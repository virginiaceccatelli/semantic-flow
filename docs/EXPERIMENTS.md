# Experiments

Each experiment states its hypothesis, method, controls, metrics, and output
files. Ground truth always comes from static analysis of the program
(`src/graphs/`) or from *executing* it (E11), aligned to token positions via
AST spans (`src/data/alignment.py`) — never by matching token strings.

This file is organized by **current role**, not by number. The registry that
the pipeline itself reads is `results/STATUS.yaml`; this file is its prose
counterpart, and the two are meant to be kept in step.

1. [Active foundation](#1-active-foundation) — E2, E3
2. [Active J-space binding experiments](#2-active-j-space-binding-experiments) — E11 (stages 70–74)
3. [Supporting / appendix experiments](#3-supporting--appendix-experiments) — E1, E4, E5, E7, E8, E9, E10-0
4. [Archived experiments](#4-archived-experiments) — E6, E10-2, E10-3
5. [Instrument validation](#5-instrument-validation-not-a-result) — E12 (stages 80–88, parked)
6. [Binding interchange](#6-binding-interchange-e13-stages-100107) — E13 (stages 100–107, active)

Shared metric definitions:

- **accuracy / f1 / auc** — mean over grouped CV test folds (no source example
  in both train and test).
- **selectivity** — accuracy − control accuracy, where the control retrains the
  identical probe on labels shuffled *within* each source example.
  Selectivity ≈ 0 means the probe exploits dataset regularities.
- **converged** — every sklearn fit finished within tolerance; results with
  `converged=False` are not reportable.
- **cluster bootstrap** — resampling whole *base programs* with replacement.
  E11 reports no interval any other way (`src/analysis/bootstrap.py`).

---

# 1. Active foundation

The results the project stands on. Both have a surface floor pinned to exactly
0.500 by construction, which is what separates them from everything in §3.

## E2 — variable binding (lexical vs semantic identity)

**Hypothesis.** Mid layers encode *which definition an identifier occurrence
refers to*, beyond surface name identity.

**Method.** Pairwise probe on `[h_i; h_j; h_i−h_j; |h_i−h_j|]`. Positives:
occurrence pairs sharing a reaching definition. Negative strata, each reported
separately:

| stratum | what it isolates |
|---|---|
| `same_name_diff_binding` | same surface name, different binding (shadowing). A purely lexical probe fails here — but local *context* can still leak. |
| `diff_name` | easy negatives (capped at 3× positives) |
| `distance_matched` | controls for token-distance shortcuts |
| `context_matched` | **the** test: designed (def, use) pairs from program pairs that are token-identical except one rebinding token. Anchor windows and distance are identical across the pair while the label flips, so NO surface feature is informative; both programs share one CV group. |

**Surface-shortcut baseline.** Stage 20 additionally fits a probe on windowed
token ids (±3 around each anchor) + bucketed anchor distance — no hidden states
(`features="surface"`, `layer=-1` rows). On `context_matched` it is ~0.5 by
construction.

**Decision rule.** If held-out accuracy on `context_matched` ≈ chance while
overall accuracy is high, the model tracks surface form, not bindings. The gap
between this stratum and the surface baseline, by layer, is the central
"lexical vs semantic" figure (`binding_strata_*.png`).

Stage: `make probes MODEL=...` (20). Output: rows `task=binding` in
`static_probes_*.csv`.

## E3 — def-use edges (data flow)

**Hypothesis.** A def→use edge between two positions is linearly decodable,
degrading with token distance.

**Method.** Directed (definition, use) pairs; positives from the reaching-def
DFG; same negative strata as E2; per-distance-bucket held-out accuracy (buckets
0–10, 10–50, 50–200, 200+).

Stage: 20. Output: `defuse_distance_*.png` (layer × distance heatmap).

---

# 2. Active J-space binding experiments

## E11 — is the selected value routed into causally reusable coordinates?

**Research question.** When a code model resolves variable binding, does it
route the *selected* value into J-lens coordinates that downstream computation
can causally reuse?

**Framing.** The J-lens is treated as a causal, output-aligned coordinate
system — `v_w = J_ℓ^T (g·W_U[w])`, the direction at layer ℓ whose component
pushes the model's own output head toward token `w`. E11 makes no claim about
reportability, verbalizability, or a workspace; `docs/LEGACY_RESULTS.md`
records why that framing was dropped.

**Why it is the right next question.** E2/E3 show binding is decodable, which
is compatible with the representation being a faithful shadow of a computation
happening elsewhere. E7 attempted the causal question and cannot separate
semantic state from transported surface difference (§3). E11's intervention is
built to make that separation: it edits two coordinates and leaves the rest of
the state alone, at a position where the two programs are token-identical, and
demands that one edit produce a *different* correct answer under each of
several downstream operations.

### E11-0 — counterfactual pairs (stage 70, CPU)

**Method.** A one-token mutation of the inner definition's name flips which
value the marked use selects, while both values occur in both programs:

```python
# case 0007                      # case 0007
x = 3                            x = 3
def f():                         def f():
    y = 7                            x = 7
    return x * 2 + 1                 return x * 2 + 1
assert f() ==   → 7              assert f() ==   → 15
```

Templates: `global_shadow`, `call_frame` (the operation lives in a callee, so
routing crosses a call boundary), `padded_shadow` (filler between mutation and
use). Operation families: affine, multiply/subtract, threshold comparison,
modulus/parity, list indexing. Each **base** carries several families over the
same two values — the unit of the cross-operation test and of the cluster
bootstrap.

**Invariants**, enforced at generation and re-checked in `tests/test_jspace.py`:

| invariant | why it matters |
|---|---|
| exactly one differing token, at the inner definition's *name* | the marked use itself must be identical |
| equal token length | every probed position is the same index in both programs |
| mutation never adjacent to the use | no local window can leak the label |
| answers distinct, single-token, **disjoint from both values** | otherwise an answer token and a distractor value token are the same lens row, and readout and swap are circular |
| ground truth by execution, cross-checked against the operation's Python function | catches a template that binds the wrong variable |

Positions recorded per pair: `pre_def`, `def_source`, `def_target`, `mutation`,
`use`, `answer`. The calibration/test split is assigned here, grouped by base
and stratified by template, and stored in the data file so every stage agrees.

Output: `data/synthetic/jspace_pairs_{model}.jsonl`.

### E11-1 — the frozen lens (stage 71, GPU) — a GATE

**Method.** One J-lens per layer, built from a **held-out generic Python
corpus** (CodeSearchNet by default), never from evaluation programs, with broad
source positions `t` and randomly sampled future readout positions `t' ≥ t`.
Candidate vocabulary: the ten digits plus every value and answer in the pair
file.

**Stability.** Three independent build samples per layer. Two numbers, because
they can come apart: rowwise cosine (do the *directions* agree?) and
margin-sign agreement on held-out states (do the *decisions* agree?). The gate
reads the second. A layer that fails it is reported but must not carry a claim.

**Validation.** V1 (at the last decoder layer `J` is the identity, so the
J-lens must reproduce the logit lens) and V2 (next-token recovery against the
Gram-matched floor) come from `jlens_validate` unchanged.

**Required checks:** `V1_last_layer_equals_logit_lens`,
`V2_beats_gram_matched_floor`, `lens_stable_across_seeds`. Stage 71 exits
non-zero if any fails; 72/73 are not interpretable until it passes.

Output: `results/jspace/{model}/lenses/*.pkl`,
`jspace_lens_{stability,validation,checks}.csv`.

### E11-2 — bound-value readout (stage 72, GPU)

**Hypothesis.** At the marked use, the frozen J-lens ranks the *bound* value
above the distractor, and the ranking **flips** with the counterfactual.

**Metric.** The paired counterfactual margin reversal, not accuracy. With
`m = score(v_source) − score(v_target)` measured at the same position in both
programs, a reversal requires `m > 0` in the source-binding program *and*
`m < 0` in its mutation. A readout with any per-token bias — preferring small
numbers, the first-mentioned literal, the token it just saw — produces the same
margin in both and scores zero. Accuracy and candidate rank are reported too,
as the weaker numbers.

**Readouts**, all on identical hidden states: `jlens`, `logit`, `gram_random`
(same norms *and* angles as the real lens; only the directions are arbitrary),
and `probe` — a linear probe trained on the calibration split. The probe is the
incumbent, not a floor: E2/E3 already show supervision recovers binding, so the
interesting outcome is whether the unsupervised lens matches it.

**Positions.** All six, so the answer to "where does the selected value become
legible" is measured rather than assumed. `pre_def` precedes both definitions
and is the position-level floor.

**Behaviour.** The same forward passes score the model's own forced choice
between the two answers. Reported for every example; the "both counterfactuals
correct" subset is labelled and summarized alongside, never instead.

Output: `jspace_readout{,_summary,_contrasts}.csv`, `jspace_behaviour.csv`.

### E11-3 — the coordinate swap (stage 73, GPU)

**Hypothesis.** Exchanging the two value coordinates at the marked use moves
the output toward the answer implied by the swapped-in value.

**Method.** With `V = [v_source, v_target]` and `c = V⁺h`,
`h_patched = h + V(swap(c) − c)`. Applied in both directions, at individual
layers and short bands of consecutive probed layers (each layer sees the
previous one's effect). Scored as a paired shift in
`logP(answer implied by the other value) − logP(answer bound here)` against the
same program's clean run.

Three algebraic properties, each a unit test: only the two coordinates change;
the operator is an involution; identical directions give *exactly* the zero
edit, so the same-value control is provably inert.

**The key test.** The same value swap must move each operation family toward
*its own* answer. An intervention that steers the answer token cannot produce a
different correct token per family from one edit, so the summary reports the
per-family minimum, and `all_families_positive` is what the go/no-go reads.

**Controls.**

| variant | if it matches `jlens_value`, the finding is… |
|---|---|
| `logit_value` | the unembedding matrix, not the causal correction |
| `gram_random` | any 2-d subspace of the same shape |
| `noop_same_value` | numerical noise (the edit is provably the zero vector) |
| `jlens_answer` | direct answer steering, not an intermediate value |
| `whole_state` | not a control — the reference ceiling for this position |
| `pre_def` position | position, not subspace: nothing bound can be routed there yet |

**Two structural zeros**, kept in the output as free correctness checks rather
than suppressed: an edit at the *last* decoder layer at any position before the
answer cannot move the logits (nothing downstream mixes it in), and
`whole_state` at `pre_def` is zero because the two programs are token-identical
before the mutation, so the "counterfactual" state there is the same state. A
nonzero value in either cell means positions or hooks are wrong.

Output: `jspace_swap{,_summary,_by_operation,_contrasts}.csv`.

### E11-4 — pre-registered go/no-go (stage 74, CPU)

Recommend the full 6.7b run only if all three hold on the 1.3b pilot (200
pairs, two operation families, four layers):

1. behavioural balanced accuracy ≥ 0.75;
2. the bound-value J-lens readout beats the Gram-matched random control
   (paired cluster-bootstrap CI lower bound above zero);
3. the coordinate swap produces a positive paired logit shift (CI lower bound
   above zero).

The layer and the intervention site are chosen on the **calibration** split and
recorded before the test numbers are read. Output:
`results/jspace/{model}/go_no_go.{yaml,md}`.

### Running it

```bash
make jspace-pilot                                   # local, 1.3b
jobs/jspace_pilot.csh                               # cluster, 1.3b
setenv MODEL deepseek-coder-6.7b; jobs/jspace_full.csh   # cluster, full run
```

---

# 3. Supporting / appendix experiments

Real and reported; they constrain the picture rather than carry it.

## E1 — lexical token type (sanity baseline)

**Hypothesis.** Token-type identity is near-perfectly decodable at every layer.
This validates the extraction and probing machinery; failure here means a
pipeline bug, not a finding.

**Method.** Multiclass linear probe on single hidden states; labels from
`classify_token`. **Expected:** > 0.95 accuracy from early layers.

Stage: 20. Output: rows `task=lexical_token_type` in `static_probes_*.csv`.

## E4 — control dependence (supporting, not central)

**Hypothesis.** Whether a statement executes under a guard is encoded in the
pair (guard-expression state, statement state).

**Method.** Positives: (guard `test`/`iter` expression anchor, statement anchor)
for statements inside the guard's body/orelse, computed by AST walk with
nesting. Negatives: same-program statements outside the guard, including the
`indent_matched` hard stratum (a statement in a *sibling* guard's body at the
same nesting depth).

**Why not central.** Its surface baseline is already 0.927 / AUC 0.990, unlike
E2/E3 whose floor is pinned to exactly 0.500. Control dependence is largely a
local syntactic relation, so this result is best read as the contrast that
makes E2's isolation meaningful.

Stage: 20.

## E5 — context degradation

**Hypothesis.** Semantic relation recovery degrades as filler separates
definition from use, and degrades *differently* by filler type: prose and dead
code (inert) < lexically similar decoys < shadowing scopes < competing updates.

**Method.** **Frozen** E2/E3 probes from stage 20 — never retrained — applied to
variants where a token-counted filler block (0–1000 tokens, measured with the
real tokenizer) is inserted between the tracked def and use. Ground truth is
recomputed from each variant's own source.

Stage: `make context MODEL=...` (30). Output: `context_degradation_*.csv`,
`context_{task}_*.png`.

## E7 — causal patching (preliminary causal evidence)

**What it measures.** Length-matched pairs (identical except the sink
argument); patch clean→corrupted at each probed layer × position and measure
logit-diff recovery `(ld_patched − ld_corr) / (ld_clean − ld_corr)`.

**Retired claim.** That it isolates *semantic use*. `sink_arg` is the only place
the two programs differ, so patching there transports the surface difference
along with any semantic state; the `sanitizer_def` null has no positive control
at that position; and late-layer `last_token` recovery forces the answer
trivially. Reasoning in `docs/LEGACY_RESULTS.md`.

**What it still supports.** The causal locus of the decision migrates from the
sink-argument token to the last-token position across the middle of the
network — a reproducible description of where the decision becomes committed.

Stage: `make patching MODEL=...` (50). Output: `causal_patching{,_summary}_*.csv`,
`patching_recovery_*.png`.

## E8 — real-code generalization (with its limitation)

**Method.** Stages 10+20 run unchanged on ~200 ast-parseable CodeSearchNet
functions (fixed-seed sample). Report synthetic vs real accuracy/selectivity
side by side per task and layer.

**Limitation, reported with the result.** Real identifiers are informative, so
the embedding layer starts at ~0.96 AUC and no stratum pins the surface floor to
chance. E8 shows the decoder transfers to naturalistic inputs; it does **not**
show that the semantic component specifically transfers. The fix —
context-matched pairs built by mutating real functions — is open item 1 in
`docs/RESULTS.md`.

Stages: 10, 20 on `data/real/csn_python_200.jsonl`.

## E9 — obfuscation robustness

**Hypothesis.** If the model represents program *semantics* rather than surface
form, frozen E2/E3 probe accuracy should survive semantics-preserving
obfuscation; probes riding lexical shortcuts should collapse already at pure
renaming.

**Method.** Tigress-inspired obfuscation ladder implemented natively for Python
in `src/data/obfuscation.py`. Cumulative levels, each variant
**execution-verified** observationally equivalent to its base:

| level | name | transformation |
|---|---|---|
| 0 | normalize | ast round-trip only — shared formatting baseline |
| 1 | rename | consistent alpha-renaming of all locals |
| 2 | opaque | + dead branches under opaque predicates |
| 3 | encode | + mixed boolean-arithmetic encoding |
| 4 | flatten | + control-flow flattening into a state machine |

Frozen E2/E3 probes are evaluated on the variants; ground truth is rebuilt from
each variant's own source. All levels of a base are kept or dropped together.

Stage: `make obfuscation MODEL=...` (31). Output:
`obfuscation_robustness_*.csv`, `obfuscation_levels_*.png`.

## E10-0 — J-lens implementation validation (stage 60)

**Hypothesis.** None — this is a gate, not a result, and it exits non-zero on
failure.

**Checks.** Phase 0 (applicability): candidate identifiers are single tokens;
the unembedding and final-norm accessors resolve; autograd reaches an
intermediate activation with finite gradients; the Jacobian correction is not a
no-op. Phase 1: **V1** the last-layer identity against the logit lens; **V2**
next-token recovery beating a norm-matched random floor; **V3** the yes/no
disposition readout (retained for completeness — it passed at n=10, too small
to carry weight).

Controls are compared **at the J-lens's own best layer**, not max-against-max.

**Why it survives while E10-2/E10-3 do not.** It is a statement about the
instrument, not about the model, and E11 reuses it unchanged.

Stage: `make jlens-validate MODEL=...` (60). Output:
`jlens_validation{,_checks}_{model}.csv`.

---

# 4. Archived experiments

Claims withdrawn; data, figures and manifests preserved; stage commands
unchanged. `scripts/90_make_paper_assets.py` skips these by default and
regenerates them with `--include-archived`. Full reasoning for each:
`docs/LEGACY_RESULTS.md`.

## E6 — behavioural lead time (stage 40)

**Retired claim.** That latent taint-state corruption precedes behavioural
failure, and that the effect is scale-dependent.

**Why.** The original positive was measured on a constant responder (balanced
accuracy 0.500 in both models). With a fixed prompt and an analytic null, a
no-model `position` baseline scores +0.113 excess while a 99%-accurate probe
scores −0.010: within readout families, early-warning rate and accuracy
correlate at r ≈ −0.9, so the metric rewards unreliability rather than
anticipation. The negative is not a finding either — a metric that cannot
separate anticipation from unreliability does not become trustworthy when it
returns a null.

Still runnable: `make leadtime MODEL=...`; `scripts/41_leadtime_floors.py`
re-applies the floors to any stage-40 run without a GPU.

## E10-2 — taint workspace membership (stage 61)

**Retired claim.** That the taint state is "verbalizable", and that this
explains E6's scale difference.

**Why.** It inherits E6's broken metric, and the effect it was built to explain
did not survive its own floors. Its lasting contribution is diagnostic: it
produced the r = −0.905 measurement that condemned the early-warning statistic,
and it showed a norm-matched random direction outscoring every real readout.

Still runnable: `make jlens-taint MODEL=...`.

## E10-3 — control dependence, J-lens (stages 62, 63)

**Retired claim.** That control dependence is decodable but not verbalizable —
a probe/lens dissociation.

**Why.** The null itself was carefully defended (temporal confound identified
and conditioned out; no layer differs from chance after Bonferroni correction
with a cluster bootstrap). But `guard_var` is a recency/identity control, not a
*relational* one, so "not verbalizable" cannot be distinguished from "this
readout cannot express relations at these positions" — and `next_ident`, the
control that was supposed to be sharp, failed for a mechanical anchoring
reason. The relation it interrogates is also the most syntactic one in the
suite (E4's surface baseline is 0.927), which is where such a dissociation
would matter least.

Still runnable: `make jlens-controldep MODEL=...`;
`scripts/63_controldep_temporal.py` applies the temporal split and corrected
tests retrospectively, no GPU required.

---

# 5. Instrument validation (NOT a result)

## E12 — latent store transitions (stages 80–88)

**Status: implemented, not run. Claims nothing.**

**Question.** Can we reliably identify and interchange a computed,
**text-absent** program value in a pretrained code model, such that downstream
computation correctly *transforms* the installed value?

**Why it is not a finding.** Causal state interchange on a learned low-rank
subspace is established method — DAS, Othello-GPT, and variable binding in
symbolic programs (`arXiv:2505.20896`) all do a version of it. What this
project owns that the field does not is the construction-pinned surface floor.
E12 builds and checks the instrument; `docs/design/E13_DIRECTIONS.md` is what a
pass licenses. Full design: `docs/design/E12_PLAN.md`. Commands:
`docs/RUNBOOK_E12.md`.

**The data.** Token-aligned triples whose tracked value has no token:

```python
def f():                    # counterfactual: a = 2  (one differing token)
    a = 1
    b = 4                   # irrelevant variable — the twin mutates this instead
    c = a + 4               # 5 / 6, absent from the text of EVERY program
    d = c + 3               # 8 / 9
    return d
assert f() ==
```

Four operation families over the same `c` (`add`, `sub_from`, `double_sub`,
`mod`), ≥3 per base, so one edit must imply a different correct answer in each
and one family can be held out.

**The critical readout.** After installing the counterfactual's `c` at the
injection anchor, the frozen decoder at the *next* statement reads one of:

| bin | value | what it would mean |
|---|---|---|
| `stale` | 8 | the edit did nothing |
| `copied` | 6 | the value was carried, the operator was not applied |
| `transformed` | 9 | the program's own next statement ran on the installed value |
| `other` | — | noise / off-manifold |

`copied` is why the endpoint is internal: answer-token steering and
carry-without-composition both predict it; only a transition predicts
`transformed`.

**The gates.** Each stage refuses to run (exit 2) unless its prerequisites
passed; `--override-gate REASON` is permitted and recorded permanently.

| gate | stage | asserts |
|---|---|---|
| G0 | 81 | trace, reference interpreter and stored labels agree; invariants hold |
| G1 | 82 | balanced accuracy ≥ 0.75 overall, ≥ 0.70 per retained family |
| G2 | 84 | the text-absent value decodes above measured lexical/control-task baselines |
| G3 | 85 | frozen-decoder transfer is measurable, with a live text-present control |
| G4 | 86 | whole-state interchange yields `transformed` — the ceiling *and* the aliveness check |
| G5 | 87 | low-rank interchange ≥ 50% of the ceiling, clears six controls, transfers to a held-out operation |

**Controls (G5).** `random_rank`, `random_norm` (matched on removed norm, not
rank), `noop` (provably the zero edit), `irrelevant` (the unread-literal twin),
`pre_def` (position, not subspace), and `held_out_family` — the decisive one,
since a subspace encoding the answer cannot transfer to a family mapping the
same value to a different answer.

**Limitation, stated with the result.** G2 has no construction-pinned floor:
the value is a deterministic function of the visible text, so an executing
baseline scores 1.0. It is a precondition, not a result.

Stages: `make store MODEL=...` (80–88), `make store-pilot`,
`jobs/store_{pilot,full}.csh`.

---

# 6. Binding interchange (E13, stages 100–107)

**Status: implemented, not run. The active direction.**

**Question.** Does a low-rank, magnitude-free interchange at the site where a
variable binding is resolved transport *which definition is in scope*, rather
than a token or an answer direction? This is the question `paper/main.tex`
§Discussion declares open.

**The data.** Four programs per base — binding × value assignment — all
token-identical except one character:

```python
# arm ab, source (outer binding) -> a      # arm ba, source (outer binding) -> b
x = a                                      x = b
def f():                                   def f():
    y = b       # target: `x = b`              y = a       # target: `x = a`
    return x                                   return x
assert f() ==                              assert f() ==
```

**The identification.** Install the target run's state into the source run at
the marked use. In arm `ab` the answer must move **a → b**; in arm `ba` the same
intervention must move it **b → a**. Fit the alignment on `ab`, read the claim
on `ba`. A subspace encoding "the token b", or "the answer", scores positive on
`ab` and **negative** on `ba`. Only one encoding which definition is in scope
survives both.

E11 could not build this: with an arithmetic operation between the value and the
answer it had to forbid `answer == value` to avoid circularity, and paid with a
capability requirement. Here the answer IS the bound value and the arm crossing
breaks the circularity instead — so there is **no arithmetic anywhere**, which
is exactly the coupling that sank E12.

**The gates.** Each stage refuses to run (exit 2) on a failed prerequisite.

| gate | stage | asserts |
|---|---|---|
| H0 | 101 | execution and a **scope-aware** reference interpreter agree; invariants hold, including the arm crossing |
| H1 | 102 | the model returns the bound variable — ≥ 0.85 overall **and** ≥ 0.75 per cell |
| H2 | 104 | the binding is decodable at the use anchor above the measured surface baseline (E2's `context_matched`, replicated here) |
| H3 | 105 | whole-state interchange flips the answer in **both** arms — the ceiling, and the proof H5 is testable |
| H4 | 106 | low-rank interchange beats matched controls on the **training** arm |
| H5 | 106 | the same subspace transfers to the **held-out** arm, where an explicit `answer_direction` fails |

**Controls.** `whole_state` (ceiling, per arm), **`answer_direction`** (the
positive control for the falsification: must pass `ab`, must fail `ba`),
`random_rank`, `random_norm` (matched on removed norm), `noop` (provably zero),
and the `def_source` site (a structural zero — the programs are token-identical
before the mutation).

**Do not claim** H4 alone: without H5 it is E11 again, and E11's own go/no-go
read NO-GO. Full design, outcome table and literature position:
`docs/design/E13_PLAN.md`.

Stages: `make binding MODEL=...` (100–107), `make binding-pilot`,
`jobs/binding_{pilot,full}.csh`.

---

## Models & replication

| Role | Model | Where |
|---|---|---|
| Development / smoke / E11 pilot | deepseek-coder-1.3b | local MPS or cluster |
| Main results / E11 full run | deepseek-coder-6.7b | cluster GPU |
| Architecture replication (optional) | starcoder2-3b | cluster GPU |

All experiments are model-agnostic through `--model`; probed layers per model
live in `configs/models.yaml`, and per-stage settings in
`configs/experiments.yaml`.
