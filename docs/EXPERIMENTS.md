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

## E10 — J-lens: is it verbalizable, not just decodable?

E1–E9 all ask whether a relation can be *read out* of the hidden state by a
supervised probe. E10 asks a different question with an unsupervised,
gradient-based readout — the **Jacobian lens** of Gurnee, Lindsey et al.
(2026) — built from the model's own output head rather than fit against
static-analysis labels:

```
J_l     = E[ d h_final,t' / d h_l,t ]           (averaged over a corpus)
v_w     = J_l^T (g * W_U[w])                    (one lens vector per token)
score_w = v_w . h                               (up to a positive scale)
```

Because `v_w` is a causal derivative of the model's own output, a high score
means the state is *disposed to make the model say* `w` — "verbalizable" —
which is a strictly stronger property than being linearly decodable. Full
rationale, cost analysis, and expected results: `docs/JLENS_PLAN.md`.
Implementation: `src/models/lens.py`, `src/experiments/jlens_*.py`.

**Scale caveat.** Scores drop a shared positive factor (`1/rms(J h)`), so
only *within-position* comparisons are claimed — rankings, argmax, and the
sign of a score difference. Every E10 metric below is rank- or sign-based.

### E10-0 — validation gate (stage 60)

**Hypothesis.** None — this is a gate, not a result. Stages 61/62 are not
interpretable unless it passes, so it exits non-zero on failure.

**Checks.** Phase 0 (applicability): candidate identifiers are single tokens;
the unembedding and final-norm accessors resolve; autograd reaches an
intermediate activation with finite gradients; the Jacobian correction is
not a no-op. Phase 1 (methodology): **V1** at the last decoder layer `J` is
the identity, so the J-lens must reproduce the logit lens *exactly* — this
exercises the whole VJP path against a closed-form answer; **V2** next-token
recovery beats a norm-matched random floor; **V3** the yes/no taint readout
(the one E10-2 depends on) beats that floor and matches the model's own
answer at the last layer.

Controls are compared **at the J-lens's own best layer**, not
max-against-max, so a noisy floor does not get a free maximum over ~10 draws.
Gates additionally require a minimum n and report `[UNDERPOWERED]` otherwise.

Output: `jlens_validation_{model}.csv`, `jlens_validation_checks_{model}.csv`;
figures `jlens_validation_{nexttoken,disposition}_{model}.png`.

### E10-2 — taint workspace membership (stage 61) — the priority experiment

**Hypothesis.** E6 found early warning in 6.7b but not 1.3b, and
`docs/RESULTS.md` attributes that to 1.3b's taint state being accurate but
"never diverging from what its output head does". If so, the *verbalizable*
readout should show a lead in 6.7b and none in 1.3b — explaining a finding
E6 could only describe, since E6 compares a probe against behaviour and
never against an independent third signal.

**Method.** E6's line-prefix stepping, unchanged. Three signals per prefix:
`t_latent_probe` (frozen stage-20 probe), `t_latent_jlens` (lens yes/no
margin), `t_failure` (forced choice). Lenses are built on the calibration
split **only** and frozen before any test prefix is scored; every readout —
including the lens — gets a threshold calibrated on that same split, so
neither side has a handicap.

**Controls.** The **logit-lens** variant is decisive: if it shows the same
lead, the effect is the unembedding matrix rather than the causal
correction, and the finding does not stand. A norm-matched **random** lens
is the floor.

**Floors — identical to stage 40's, and it delegates to the same summary
function so the two experiments mean the same thing.** A behavioural-signal
sanity check (balanced accuracy; a constant responder posts a healthy-looking
raw accuracy equal to the base rate), a no-model `position` readout, a
norm-matched `random` readout, a `constant_readout` flag, and an **analytic
null** — for a readout with per-prefix error rate ε whose errors are
independent of the model's state, P(errs before step k) = 1−(1−ε)^(k−1).

**Metric.** `early_warning_excess` = observed − analytic null. The raw rate
cannot support a claim: it rises with a readout's error rate whether or not it
carries information (E6 measured r = −0.96 between the two within readout
families), which is why a no-model baseline outscores a 99%-accurate probe
there.

Output: `jlens_taint{,_summary,_prefixes,_sanity}_{model}.csv`; figures
`jlens_taint_excess_{model}.png` (the claim-bearing one) and
`jlens_taint_earlywarning_{model}.png` (raw rate, for contrast).

### E10-3 — control dependence: verbalizable or automatic? (stage 62)

**Hypothesis.** E4 showed control dependence is decodable (AUC 0.999 by mid
layers) but "largely local syntax" (surface baseline already 0.927). If it
is genuinely automatic, the lens ranking should stay near its floor at
*every* layer even where E4's probe is at ceiling — a probe/lens
dissociation absent for binding and def-use.

**Method.** At the guard-expression anchor (E4's `pos_i`), compare the lens
score of a control-dependent statement's target variable against that of an
`indent_matched` statement's target — E4's hard negative, a statement in a
*sibling* guard's body at the same nesting depth. Chance is exactly 0.5: the
two targets are interchangeable neutral variables. Readout positions `t'`
are sampled generically from positions after the guard, never at the
labelled statements, so the lens never sees the labels it is scored against.

**Positive controls (required to read the null).** A flat `control_dep` curve
alone is only absence of evidence — it cannot separate "control dependence is
not verbalizable" from "this readout reads nothing at these positions". Two
controls therefore run at the *same anchors* with the *same* two-alternative
readout, scored against another variable present in the program:

| `comparison` | ranks | why it should succeed |
|---|---|---|
| `guard_var` | the variable the guard tests, vs another present variable | the model has just read it |
| `next_ident` | the next identifier occurring after the anchor, vs another present variable | pure local continuation, which V2 shows the lens does well |

`next_ident` is the sharp one: its positive is usually the *same* token as
`control_dep`'s, and only the negative differs (a distant variable rather than
the sibling guard's target). If `next_ident` succeeds where `control_dep`
fails, the gap isolates the guard→statement binding rather than the readout.

Output: `jlens_controldep{,_summary}_{model}.csv` (with a `comparison` column);
figures `jlens_controldep_{stratum}_{model}.png` and
`jlens_controldep_dissociation_{model}.png`.

---

## Models & replication

| Role | Model | Where |
|---|---|---|
| Development / smoke | deepseek-coder-1.3b | local MPS |
| Main results | deepseek-coder-6.7b | cluster GPU |
| Architecture replication (optional) | starcoder2-3b | cluster GPU |

All experiments are model-agnostic through `--model`; probed layers per model
live in `configs/models.yaml`.
