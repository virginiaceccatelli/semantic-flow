# Experiments

Each experiment states its hypothesis, method, controls, metrics, and output
files. Ground truth always comes from static analysis of the program
(`src/graphs/`), aligned to token positions via AST spans
(`src/data/alignment.py`) — never by matching token strings.

Shared metric definitions:

- **accuracy / f1 / auc** — mean over grouped CV test folds (no source example
  in both train and test).
- **selectivity** — accuracy − control accuracy, where the control retrains
  the identical probe on labels shuffled *within* each source example.
  Selectivity ≈ 0 means the probe exploits dataset regularities, not the
  hidden state.
- **converged** — every sklearn fit finished within tolerance; results with
  `converged=False` are not reportable.

---

## E1 — lexical token type (sanity baseline)

**Hypothesis.** Token-type identity (keyword/identifier/literal/…) is
near-perfectly decodable at every layer. This validates the extraction and
probing machinery; failure here means a pipeline bug, not a finding.

**Method.** Multiclass linear probe on single hidden states; labels from
`classify_token`. **Expected:** > 0.95 accuracy from early layers.

Output: rows `task=lexical_token_type` in `static_probes_*.csv`;
figure `layers_accuracy_*.png`.

## E2 — variable binding (lexical vs semantic identity)

**Hypothesis.** Mid layers encode *which definition an identifier occurrence
refers to*, beyond surface name identity.

**Method.** Pairwise probe on `[h_i; h_j; h_i−h_j; |h_i−h_j|]`. Positives:
occurrence pairs sharing a reaching definition (binding id). Negative strata,
each reported separately:

| stratum | what it isolates |
|---|---|
| `same_name_diff_binding` | same surface name, different binding (shadowing). A purely lexical probe fails here — but local *context* can still leak (see `context_matched`). |
| `diff_name` | easy negatives (capped at 3× positives) |
| `distance_matched` | controls for token-distance shortcuts |
| `context_matched` | **the** test: designed (def, use) pairs from program pairs that are token-identical except one rebinding token. Anchor windows and distance are identical across the pair while the label flips, so NO surface feature is informative; both programs share one CV group. |

**Surface-shortcut baseline.** Stage 20 additionally fits a probe on windowed
token ids (±3 around each anchor) + bucketed anchor distance — no hidden
states (`features="surface"`, `layer=-1` rows). This is the floor every
hidden-state probe must beat; on `context_matched` it is ~0.5 by construction.

**Decision rule.** If held-out accuracy on `context_matched` ≈ chance
while overall accuracy is high, the model tracks surface form, not bindings.
The gap between this stratum and the surface baseline, by layer, is the
paper's central "lexical vs semantic" figure (`binding_strata_*.png`).

## E3 — def-use edges (data flow)

**Hypothesis.** A def→use edge between two positions is linearly decodable,
degrading with token distance.

**Method.** Directed (definition, use) pairs; positives from the reaching-def
DFG; same negative strata as E2; per-distance-bucket held-out accuracy
(buckets 0–10, 10–50, 50–200, 200+).

Output: `defuse_distance_*.png` (layer × distance heatmap).

## E4 — control dependence

**Hypothesis.** Whether a statement executes under a guard is encoded in the
pair (guard-expression state, statement state).

**Method.** Positives: (guard `test`/`iter` expression anchor, statement
anchor) for statements inside the guard's body/orelse — computed by AST walk
with nesting (a statement is dependent on all enclosing guards). Negatives:
same-program statements outside the guard (before it, or after the join
point). ~50% of binding programs include a branch for this purpose.

## E5 — context degradation

**Hypothesis.** Semantic relation recovery degrades as filler separates
definition from use, and degrades *differently* by filler type: prose and
dead code (inert) < lexically similar decoys < shadowing scopes < competing
updates (which genuinely change the reaching definition).

**Method.** **Frozen** E2/E3 probes from stage 20 — never retrained — applied
to variants where a token-counted filler block (sizes 0–1000 tokens, measured
with the real tokenizer) is inserted between the tracked def and use. Ground
truth is recomputed from each variant's own source, so `competing_update`
tests whether the model *updates* its state, while the inert fillers test
pure distance.

**Metrics.** Frozen-probe accuracy per (task, layer, filler_type, size).
Size 0 is the reference point. Output: `context_degradation_*.csv`,
`context_{task}_*.png`.

## E6 — behavioral lead time

**Hypothesis.** Latent taint-state corruption (probe decodes wrongly)
precedes behavioral failure (model answers the taint question wrongly):
lead_time > 0.

**Method.** Taint programs carry per-line ground truth
(`metadata.line_labels`). For each line-prefix: (a) the frozen taint-state
probe decodes "is the live value tainted?" from the last-token hidden state —
threshold calibrated on a held-out 30% calibration split; (b) the model
answers the same question as a yes/no forced choice via continuation
log-probs. `t_latent` / `t_failure` = first prefix where (a) / (b) is wrong.

**Why this is not circular** (the old version was): the probe signal is a
linear readout of the residual stream trained on ground truth; the behavior
signal is the model's own output head. They can disagree, and the direction
of disagreement is the finding.

**Metrics.** Lead-time distribution, fraction positive, bootstrap CI (2000
resamples). Output: `behavioral_leadtime{,_summary}_*.csv`, `leadtime_*.png`.

## E7 — causal patching (encoding vs use)

**Hypothesis.** If the taint relation is truly *used*, restoring the clean
run's residual state at the semantically critical position should move the
corrupted run's answer toward the clean answer.

**Method.** Length-matched pairs (identical token sequences except the sink
argument: clean sinks the sanitized variable, corrupted sinks the raw one).
Patch clean→corrupted at each probed layer × position:

| position | role |
|---|---|
| `sink_arg` | the differing tokens — the critical site |
| `sanitizer_def` | where the sanitized value is bound |
| `last_token` | the readout position — **reported separately**: patching here at late layers trivially forces the answer and is not evidence of semantic use |

**Metrics.** logit-diff recovery
`(ld_patched − ld_corr) / (ld_clean − ld_corr)` with `ld = logP(no) − logP(yes)`;
answer flip rate; causal class per (layer, position) using the frozen taint
probe: `encoded_and_used` (probe decodes both sides correctly ∧ recovery>0.5),
`encoded_but_unused`, `not_encoded`.

Output: `causal_patching{,_summary}_*.csv`, `patching_recovery_*.png`.

## E8 — real-code generalization

**Hypothesis.** E2/E3 probe accuracy transfers from synthetic programs to
real Python within a modest gap; a large gap means the probes fit generator
artifacts.

**Method.** Stages 10+20 run unchanged on ~200 ast-parseable CodeSearchNet
functions (fixed-seed sample). Report synthetic vs real accuracy/selectivity
side by side per task and layer.

## E9 — obfuscation robustness

**Hypothesis.** If the model represents program *semantics* rather than
surface form, frozen E2/E3 probe accuracy should survive semantics-preserving
obfuscation; probes riding lexical shortcuts should collapse already at pure
renaming. The transformation-based counterpart to E5: E5 stresses the
representations with *distance*, E9 with *surface form*.

**Method.** Tigress-inspired (tigress.wtf) obfuscation ladder implemented
natively for Python in `src/data/obfuscation.py` (Tigress itself is C-only).
Cumulative levels of increasing difficulty, each variant **execution-verified**
observationally equivalent to its base (`func()` output compared):

| level | name | transformation |
|---|---|---|
| 0 | normalize | ast round-trip only — shared formatting baseline |
| 1 | rename | consistent alpha-renaming of all locals (isolates lexical reliance, RQ3) |
| 2 | opaque | + dead branches under opaque predicates (provably false for all ints, e.g. `v*v % 4 == 3`) with decoy assignments |
| 3 | encode | + mixed boolean-arithmetic encoding (`a+b → (a^b)+((a&b)<<1)`, `c → (c^m)^m`) |
| 4 | flatten | + control-flow flattening into a while/state-machine with shuffled state ids |

**Frozen** E2/E3 probes from stage 20 — never retrained — are evaluated on the
variants; ground truth is rebuilt from each variant's own source (same
contract as E5). All levels of a base are kept or dropped together, so level
curves compare identical base-program sets.

**Metrics.** Frozen-probe accuracy per (task, layer, obf_level). Level 0 is
the reference; per-level deltas attribute degradation to each transformation
class. Output: `obfuscation_robustness_*.csv`, `obfuscation_levels_*.png`,
`obfuscation_{task}_*.png`.

---

## Models & replication

| Role | Model | Where |
|---|---|---|
| Development / smoke | deepseek-coder-1.3b | local MPS |
| Main results | deepseek-coder-6.7b | cluster GPU |
| Architecture replication (optional) | starcoder2-3b | cluster GPU |

All experiments are model-agnostic through `--model`; probed layers per model
live in `configs/models.yaml`.
