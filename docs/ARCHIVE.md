# Archive — what was retired or abandoned, and why

Everything in this project that was tried and did not survive. Nothing here is
deleted: every raw CSV is still in `results/tables/`, every figure in
`results/figures/`, every manifest in `results/manifests/`, and every stage
command still runs. What is withdrawn is the **claim**, not the data.

This file exists because the retirements are the project's methodological
content. Four intervention designs were attempted before the current one, and
each failed for a *different* reason that constrained the next. Read in order,
they are an argument about what a causal claim in interpretability requires.

| | Attempt | Failed because | Lesson carried forward |
|---|---|---|---|
| 1 | **E7** whole-state activation patching | the informative position is the only place the two programs differ, so the patch transports the input | intervene only where the inputs agree |
| 2 | **E10-2 / E10-3** output-aligned readout | the positive control was an *identity* control where the test was *relational* | a null needs a positive control matched **in kind** |
| 3 | **E11** rank-2 coordinate swap | the site's dose-response is 18x convex, so the edit was below its effective causal dose | ... and matched **in scale** |
| 4 | **E12** latent store transitions | the design made two-step arithmetic the load-bearing capability for a question about program state | do not couple the semantic question to an unrelated capability |
| — | **E6** behavioural lead time | the metric rewarded unreliable readouts | a metric that cannot separate signal from noise is not trustworthy in either direction |

Current status of every experiment: `results/STATUS.yaml`.
Currently supported findings: `docs/RESULTS.md`.
What each experiment does: `docs/EXPERIMENTS.md`.

---

## 2026-08-13 — E13's H5 discriminator moved from the margin to the argmax

**A gate criterion was changed after seeing data.** That is the thing this
project is most careful about, so the change, its justification, and the numbers
under *both* rules are recorded here in full. Judge it yourself.

**What changed.** H5's third condition — the explicit `answer_direction`
control must FAIL on the held-out arm — was evaluated on `delta_ld`, the
two-way logit margin. It is now evaluated on `says_installed`, the
full-vocabulary argmax, against the arm-to-arm ratio `whole_state` achieves on
the same rows.

**Why this is implementing the pre-registration rather than relaxing it.** The
module docstring of `src/experiments/binding_interchange.py`, written before any
6.7B run, states: *"`delta_ld` is positively biased and must not be gated on
alone… Every row therefore also records `says_installed`, the full-vocabulary
argmax, which a disruption cannot produce systematically, and the gates read
that."* No gate evaluator ever read it. The stated design and the implementation
disagreed from the start, and stages 106 and 108 have reported different H5
verdicts on every run because of it.

**Why the margin is the wrong metric here specifically.** H1 is 1.000 on this
corpus, so the clean distribution is confident and `logP(own)` sits far above
`logP(installed)`. Any edit that merely disrupts the state regresses both toward
the middle and *raises* `delta_ld` with nothing transported. This is not
hypothetical: the control also knocked the model off both candidate tokens on
6.4% of held-out rows, which is disruption, not transport.

**The 6.7B numbers under both rules** (site `use`, layer 8, rank 1, 280 held-out
bases). Only condition 3 differs:

| | margin `delta_ld` | argmax `says_installed` |
|---|---|---|
| `das_binding` transfers to `ba` | +9.009 [8.933, 9.089] = 188% of ceiling — **passes** | 100.0% = 114% of ceiling — **passes** |
| `answer_direction` on `ab` | +2.322 [2.157, 2.482] — works | 27.9% — works |
| `answer_direction` on `ba` | +0.335 [0.208, 0.456], interval clears zero — scored **"did not fail"** | 4.3%; arm ratio 0.154 against transport's 1.025 — **fails** |
| **H5 verdict** | **FAIL** (on condition 3) | **PASS** |

**What the new rule is, precisely.** A control fails on the held-out arm if its
`ba`/`ab` argmax ratio is below `MIN_TRANSFER_FRACTION` (0.50) of the ratio
`whole_state` achieves. The reference is measured on the same rows rather than
chosen, and the coefficient is the threshold already pre-registered for H5's
first condition — no new free parameter was introduced.

**What would have made this illegitimate**, and did not happen: inventing a
threshold that the observed 4.3% happens to clear; switching metrics on
condition 1 as well, where the verdict is unchanged either way (188% of ceiling
on the margin, 114% on the argmax); or making the change without leaving the
old numbers on the record.

Runs predating `says_installed` still evaluate under the old rule — the fallback
is explicit in `evaluate_gate_h5` and pinned by a test.

---

## The retired frame: "computes, uses, but does not report"

The previous `docs/RESULTS.md` closed with a single synthesizing claim:

> The model computes program semantics, causally uses some of them, and
> reports none of them.

with a global-workspace reading layered on top — that the J-lens measures
*verbalizability*, that relations absent from it are computed "outside the
workspace", and that binding is anticipatory in a way the output head does not
reflect.

**Why it was retired.** The claim is a conjunction of three parts, and each
part rested on an experiment that has since failed its own controls:

| part | rested on | what happened |
|---|---|---|
| "computes" | E2, E3 | **survives** — it is the one part that holds, and it is now the foundation in `docs/RESULTS.md` |
| "causally uses" | E7 | the design cannot separate semantic use from transported surface difference (below) |
| "reports none of them" | E6, E10-2, E10-3 | all three nulls turned out to be uninformative rather than negative (below) |

Two of the three legs were absence-of-evidence results from experiments whose
positive controls were too weak to license reading the absence. A conjunction
inherits the weakness of its weakest conjunct, and "does not report" was
carrying most of the interpretive weight while being the least supported piece.

There is a second, structural reason. "Verbalizable" and "reportable" are
claims about what a model *would say if asked*, and the J-lens does not measure
that: it measures a first-order causal projection of a hidden state onto the
output head. Those coincide at the last layer, where the lens *is* the output
head, and they are not the same thing anywhere else. Reading a mid-layer lens
null as "the model cannot report this" imports an interpretation the instrument
does not support. The active direction (E11) keeps the same instrument and
drops that reading entirely: the J-lens is treated as a **causal,
output-aligned coordinate system**, and the question is whether downstream
computation reuses those coordinates — which is a question interventions can
answer.

---

## E6 — behavioural lead time (archived)

**Retired claim.** That latent taint-state corruption precedes behavioural
failure (`lead_time > 0`), and that the effect is scale-dependent — present in
6.7b, absent in 1.3b.

### What actually happened, in order

**1. The original positive was measured on a constant responder.** Under the
bare prompt, both models answered the same token to every prefix of every
program:

| Model | answer | raw accuracy | **balanced accuracy** |
|---|---|---:|---:|
| 1.3b | always "no" | 0.220 | **0.500** |
| 6.7b | always "yes" | **0.780** | **0.500** |

6.7b's 0.780 is exactly the base rate of `tainted=1`, so the behavioural signal
looked healthy under the check anyone would run. The generator always emits the
taint source on line 2, so `tainted=1` at the first evaluable prefix; a model
that always says "yes" is therefore wrong only at and after the sanitizer,
which places `t_failure` mid-program on exactly the sanitized programs and
hands a "lead" to any readout that errs before them. The entire reported scale
split was produced by the two models' opposite constant biases. No model
computation entered the behavioural side of either number.

**2. The prompt was fixed, and only one arm survived.** Few-shot demonstrations
*and* naming the variable lift 6.7b to balanced accuracy 0.857
(`scripts/diagnose_taint_prompt.py` sweeps the four variants and documents that
neither ingredient works alone). 1.3b cannot do the task under any prompt —
balanced accuracy 0.471 — so lead time there is **undefined**, not zero.

**3. With floors in place, the metric itself turned out to be broken.** Stage
40 now reports every readout against an analytic null: for a readout with
per-prefix error rate ε whose errors are independent of the model's state,
P(errs before step *k*) = 1−(1−ε)^(k−1).

| Readout | 6.7b excess over the null |
|---|---:|
| `position` — no model at all, "tainted iff step ≤ 3" | **+0.113** |
| `random` — norm-matched random direction | +0.005 … +0.067 |
| `probe` — trained, ~99% accurate | **−0.010** |

A baseline that knows nothing but how many lines into the program it is scores
higher than a 99%-accurate probe, and not one of the probe's positive cells
survives Bonferroni correction. Within readout families, early-warning rate and
accuracy correlate at r ≈ −0.9: **the metric rewards unreliability, not
anticipation.**

**Why the *negative* is not a finding either.** A metric that cannot
distinguish anticipation from unreliability does not become trustworthy when it
returns a null. The honest summary is that E6 as designed cannot answer its
question, on either model.

**What is still worth knowing.** `readout_never_wrong` is 19/19 (6.7b) and
49/49 (1.3b): on every example where the model answered wrongly, the probe
decoded the taint state correctly at every prefix. That is a real measurement,
but it is also what a 99%-accurate probe does by definition, and the inference
from it ("represented but not used") needs the comparison that failed above.

**Preserved:** `behavioral_leadtime{,_summary,_prefixes}_{model}.csv`,
`behavioral_sanity_{model}.csv`, `leadtime_{model}.png`,
`leadtime_excess_{model}.png`. `scripts/41_leadtime_floors.py` re-applies the
floors to any stage-40 run without a GPU.

---

## E10-2 — taint workspace membership (archived)

**Retired claim.** That the taint state is "verbalizable", and that its
workspace membership explains E6's scale difference.

**Why.** E10-2 was built to explain an E6 effect that did not survive its own
floors, and it inherits E6's metric wholesale. It also produced the measurement
that condemned that metric — across all 40 (layer, readout) cells,
early-warning rate is predicted by readout unreliability at Pearson
r = −0.905 (p = 1.1×10⁻¹⁵) — and a norm-matched random direction carrying no
information posts a *higher* mean early-warning rate (0.634) than the J-lens
(0.481), the logit lens (0.373) or the trained probe (0.354).

So the experiment is informative about the *method* and uninformative about the
*model*: it establishes that the early-warning statistic cannot support a
claim, which is why E6 is archived alongside it, and it establishes nothing
about whether taint is verbalizable.

**Preserved:** `jlens_taint{,_summary,_prefixes,_sanity}_{model}.csv`,
`jlens_taint_excess_{model}.png`, `jlens_taint_earlywarning_{model}.png`.

---

## E10-3 — control dependence: "decodable but not verbalizable" (archived)

**Retired claim.** That control dependence is decodable (E4: AUC 0.999) but not
verbalizable (J-lens at chance at every layer) — a probe/lens dissociation.

**The measurements themselves held up, and were carefully done.** At guard
anchors the J-lens ranks the guard's own variable above another present
variable at 0.813 by the last layer, while the control-dependence comparison
sits within ±0.02 of chance everywhere. A below-chance tail turned out to be a
temporal confound (the `indent_matched` negative precedes the anchor half the
time, so recency favours the wrong answer); conditioning on the matched subset
removed it completely, and after Bonferroni correction with a cluster bootstrap
over source programs, **no layer differs from chance** — a clean, well-defended
null. `scripts/63_controldep_temporal.py` reproduces that analysis.

**Why it is archived anyway.** Two reasons, neither about the execution:

1. **The positive control does not license the inference.** `guard_var` shows
   the readout is alive and identifier-sensitive at those anchors. It does not
   show that a *relational* fact would be readable there if one were present —
   it is a recency/identity control, not a relational one. `next_ident`, which
   was supposed to be the sharp control, failed for a mechanical reason (the
   guard anchor is the trailing literal of the guard expression, so the "next"
   identifier is several tokens downstream past the colon and the indent). With
   no relational positive control, "not verbalizable" and "this readout cannot
   express relations at these positions" remain indistinguishable.
2. **The relation it interrogates is the wrong one to build on.** E4's own
   surface baseline is 0.927 — control dependence is largely local syntax (see
   `docs/RESULTS.md`). A dissociation between "decodable" and "verbalizable" is
   least interesting exactly where decodability is mostly syntactic.

The one cell that came close (1.3b L7, 0.576 matched, binomial p = 0.0023) is
not significant under the cluster bootstrap (95% CI [0.495, 0.644]) and does
not replicate at 6.7b (0.472, the other side of chance). It was claimed nowhere
then and is claimed nowhere now.

**Preserved:** `jlens_controldep{,_summary}_{model}.csv`,
`jlens_controldep_dissociation_{model}.png`,
`jlens_controldep_{stratum}_{model}.png`.

---

## E7 — the "isolates semantic use" claim (retired; the experiment is not)

**E7 itself is preserved and reclassified as `supporting` /  preliminary causal
evidence.** Its numbers are in `docs/RESULTS.md`. What is retired is the
interpretation that it isolates *semantic use* of the taint relation, and in
particular the reading that "the model represents the sanitization and does not
route it to the answer".

Three reasons:

1. **`sink_arg` is where the two programs' tokens differ.** It is the only
   place they differ — that is how the pair is constructed. Patching there
   transports the surface difference along with any semantic state, so the ~1.0
   recovery at early layers is exactly what a pure input-restoration effect
   would produce. The design cannot separate the two.
2. **The `sanitizer_def` null is uncontrolled.** Recovery is 0.000 at every
   layer, and the conclusion drawn was that the sanitization site is causally
   inert. But there is no positive control at that position — nothing
   establishes that *any* intervention there could have moved the output — so
   the null is absence of evidence at one hand-picked token.
3. **The `last_token` column is not evidence of anything semantic**, which the
   original write-up already flagged: patching the readout position at late
   layers forces the answer trivially.

What survives is the routing observation itself: the causal locus of the
decision migrates from the sink-argument token (recovery 0.99 at layer 0) to
the last-token position (1.00 at the final layer), crossing over around the
middle of the network. That is a real and reproducible description of *where*
the model's decision becomes committed. It is preliminary because the
intervention is a whole-state replacement, which changes everything the
position encodes at once.

**This is the specific gap E11 was designed to close.** Its coordinate swap
edits two coordinates of a state and leaves the rest of it alone, at a position
where the two programs are token-identical, and requires the same edit to
produce a *different* correct answer under each of several downstream
operations. Those three properties are exactly what E7 lacks.

**Preserved:** `causal_patching{,_summary}_{model}.csv`,
`patching_recovery_{model}.png`. Stage 50 is unchanged and still runs.

---

## Smaller claims that went with the frame

- **"Binding is anticipatory."** Never separately tested. It travelled with the
  E6 lead-time result and has no independent support; E11 does not need it and
  does not make it.
- **"E10 shows the J-lens measures reportability in code models."** What E10-0
  shows is narrower and still stands: the implementation is correct (V1 is
  exact at the last layer) and the Jacobian correction recovers next-token
  content the plain logit lens cannot (+0.15 / +0.18 top-1 pre-final layer).
  That is a statement about the instrument, and it is the part E11 reuses.
- **"The 1.3b/6.7b split in E6 is a scale effect."** Retired: the 1.3b arm was
  structurally forced (wrong at the first evaluable prefix on 100% of
  programs, so a positive lead is arithmetically impossible) and the 6.7b arm
  was uncontrolled.

---

## E11 — J-space coordinate swap (NO-GO; the use-position null retracted)

**Not archived — reported, and read narrowly.** E11 is kept in
`results/STATUS.yaml` because its numbers appear in the paper. What is recorded
here is that it did not pass its own pre-registration, which the headline
sentence can obscure.

**Both go/no-go files read NO-GO** (`results/jspace/6.7b-5fam/go_no_go.md`,
`go_no_go_answer.md`):

| criterion | use position | answer position |
|---|---|---|
| `behavioural_balanced_accuracy` (≥ 0.75) | FAIL 0.706 | FAIL 0.706 |
| `readout_beats_random_control` | FAIL +0.056 [−0.007, +0.117] | PASS +0.257 |
| `swap_moves_logits_toward_swapped_value` | FAIL +0.001 [−0.002, +0.004] | PASS +0.141 |
| `swap_is_specific_to_the_value_subspace` | FAIL | **FAIL −0.016 [−0.024, −0.009]** |
| cross-operation, all families positive | False | False |

**Retracted: the use-position null.** A dose-matched control added after the
run showed the site's response to small edits is strongly convex — efficiency
rises **18×** from the smallest dose to the largest, and a push along the
*known-correct* direction at 2% of ‖h‖ produces 0.002 nats with an interval
covering zero, the same as the value swap at 3.7%. No two-dimensional edit is
large enough to test the question at that site.

Without that control this would have been reported as a clean null: a passing
readout positive control, four subspace controls at the same magnitude, and a
site potent enough to flip 22% of answers when replaced wholesale. **It is the
most instructive failure in the project**, and it is why every subsequent design
either has no dose parameter or measures the site's response curve first.

**Also not attributable to the Jacobian correction.** The plain logit lens is
more efficient at the same site (2.35 vs 1.82), and at the last layer the two
are equal by construction. What survives is a claim about output-aligned value
directions in general, not about the J-lens.

**Outstanding:** the `probe_basis` control never ran — stage 72 was not re-run
before stage 73, so no frozen probes were on disk and the variant was silently
skipped rather than refused. That failure is why every later stage is hard-gated
and refuses to run on a missing prerequisite.

**Preserved:** `results/jspace/`, `jspace_{lens,readout,behaviour,swap}_*.csv`.

---

## E12 — latent store transitions (parked before any claim)

**Never claimed anything.** E12 was built as instrument validation: can a
computed, **text-absent** program value be identified and interchanged such that
downstream computation *transforms* it? It is parked at its behavioural gate.

**Why it was tried.** E11's swapped values are literals in the program text, so
its surviving claim is about output-aligned *token* directions. Tracking a value
that appears nowhere in the text removes that escape route — in
`a = 1; c = a + 4; d = c + 3`, the value of `c` has no token, so a direction
carrying it cannot be a token-presence direction.

**Why it is parked.** Text-absent-because-computed *forces arithmetic*. The
design made two chained arithmetic steps the load-bearing capability for a
question about program state. On 1.3B:

- balanced accuracy **0.418 — below chance** on a two-alternative forced choice;
- the correct answer was the argmax on **6.3%** of prompts, against **10%** for
  a uniform random digit;
- two of four operation families sat at **exactly 0.500**.

That exact-0.500 pattern is the tell. A simulated model doing **no computation
at all** — picking whichever candidate is numerically closer to the head literal
— reproduces it precisely:

```
proximity to head (no computation)  overall 0.494  add 0.500  double_sub 0.500
observed, deepseek-coder-1.3b       overall 0.418  add 0.500  double_sub 0.500
```

On a monotone operation the two candidates straddle the anchor symmetrically, so
a pure proximity rule scores exactly chance. Two of four families therefore had
**no computational headroom in the metric at all**, independent of model size.

**The prediction was available in advance.** Arithmetic in language models is
implemented by a sparse set of pattern-matching heuristic neurons that do not
chain (Nikankin et al., https://arxiv.org/abs/2410.21272). A two-step chain was
the wrong thing to require.

**What is kept.** All code, gates and diagnostics still run; the G1 triage
(`scripts/89_store_diagnose.py`) separates a constant responder, a format that
elicits no digit, a model answering the *intermediate* instead of the final
value, and a genuine capability limit. Design and full post-mortem:
`docs/design/archive/E12_PLAN.md`; commands:
`docs/design/archive/RUNBOOK_E12.md`.

**Lesson carried forward, and it is the one that produced E13.** Do not couple
the semantic question to a capability that is not the phenomenon of interest.
E13 requires *no arithmetic anywhere* — the model returns a variable — and gets
its falsification from a value-assignment factorial instead.
