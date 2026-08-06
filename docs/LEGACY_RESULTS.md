# Legacy results — the retired interpretation

This file holds the interpretation this project used to run on, and the reason
each part of it was retired. It exists so the retirement is auditable: the raw
data behind every claim below is still in `results/tables/`, the figures are
still in `results/figures/`, and every manifest is still in
`results/manifests/`. What changed is what we say the data shows.

Nothing here is deleted, and everything here is still reproducible — the stage
commands are unchanged, and `python scripts/90_make_paper_assets.py
--include-archived` regenerates the archived figures. What archived means is
narrow and specific: **the claim is withdrawn, so the asset no longer
regenerates into the default figure set the paper draws from.**

Current status of every experiment: `results/STATUS.yaml`.
Current supported findings: `docs/RESULTS.md`.

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
