# E13 binding interchange — deepseek-coder-6.7b

## What this experiment asks

This report tests whether a learned rank-1, magnitude-free interchange transports which definition is in scope. Read the gates in order: data validity and baseline behavior come first, then decodability and a whole-state ceiling, followed by training-arm and held-out-arm intervention tests. Only the complete gate pattern licenses a causal conclusion.

**Verdict: NOT SUPPORTED**

Does a low-rank, magnitude-free interchange at the site where a variable binding is resolved transport WHICH DEFINITION IS IN SCOPE, rather than a token or an answer direction?

## Identification

The same binding flip demands opposite token movements in the two value assignments. The alignment is fitted on arm `ab` and the claim is read on arm `ba`, where an answer direction is refuted rather than confounded.

## Gates

- **PASS** `H0` (the four-program factorial verifies: invariants, alignment, execution truth) — 1.0000 of 400 bases agree with an independent interpreter and satisfy every invariant, including the arm crossing that makes the held-out test a falsification (threshold 0.999). Per check: semantics_agree 1.0000, arms_crossed 1.0000, mutation_distance_ok 1.0000, anchors_ordered 1.0000, values_distinct 1.0000, tokens_distinct 1.0000
- **PASS** `H1` (the model returns the correctly bound variable (behavioural accuracy)) — overall 1.000 [1.000, 1.000] against 0.85; weakest cell ab_source 1.000 against 0.75
- **PASS** `H2` (which definition is in scope is decodable at the use anchor) — best layer 8: binding decodable at 1.000 (selectivity 0.524) against a MEASURED surface baseline of 0.500; margin +0.500. Thresholds 0.8 and 0.1. The floor is pinned by construction here: the anchor token is identical across the counterfactual and the mutation is outside the baseline's window.
- **PASS** `H3` (whole-state interchange flips the answer — the ceiling, per arm) — site use, layer 8 (both chosen on calibration): ab: +4.781 [+4.683, +4.878], flip rate 0.857; ba: +4.799 [+4.694, +4.903], flip rate 0.879 (thresholds: CI above 0.0, flip rate 0.25). Both arms must be measurable or an H5 null says nothing.
- **PASS** `H4` (low-rank interchange beats matched controls on the TRAINING arm) — ab @ use L8 r1: +9.029 [+8.952, +9.108] = 189% of the whole-state ceiling +4.781 (threshold 50%); controls cleared: True; edit moved 0.479 of ||h||
- **FAIL** `H5` (the same subspace transfers to the HELD-OUT value assignment, where an answer direction cannot) — ba @ use L8 r1: das_binding +9.009 [+8.933, +9.089] = 188% of the held-out ceiling (threshold 50%); discriminator — answer_direction ab +0.001 [-0.000, +0.001] (passes: False), ba +0.001 [+0.000, +0.002] (fails: False)

## Diagnostic

The subspace did not transfer to the held-out value assignment. Read the answer_direction rows FIRST: if that control also passes on `ba`, the discriminator is broken and no verdict is licensed. If it fails on `ba` as designed and das_binding fails too, the learned subspace is an answer direction — which is a real, reportable negative and exactly what E11 could not establish.

Re-run after fixing: `write it up — both outcomes are reportable; see docs/RESULTS.md R13`

## Do not claim

- that the model 'understands' scope — the claim is transport at one site
- anything about real code, other languages, or other model families
- that H4 alone supports the conclusion; without H5 it is E11 again
- a null from H5 without checking that answer_direction failed on `ba` too
