# Results

What this project currently establishes, what it does not, and what is still
running. Nothing here is a summary of intent: every row is a measurement that
exists in `results/tables/*.csv`, and every claim is paired with the control
that could have falsified it.

Three rules for reading this file:

- The machine-readable registry is `results/STATUS.yaml`, which
  `scripts/90_make_paper_assets.py` reads to decide which figures to regenerate.
- Claims this project **used to make and has withdrawn** are in
  `docs/ARCHIVE.md`, each with its reason. The data behind them is preserved.
- What each experiment does and why: `docs/EXPERIMENTS.md`.

---

## The one-paragraph version

Variable binding and def–use structure **are** represented in DeepSeek-Coder
1.3B and 6.7B, above a floor that is pinned to exactly 0.500 by construction
rather than estimated. That representation is **built** in the first few
transformer blocks, **robust** to distance and to identifier renaming in the
middle layers, and **fragile** exactly where the underlying scope or control
structure gets harder. Whether the model **causally uses** it now has its
first affirmative answer: in E13 a rank-1, magnitude-free interchange transports
*which definition is in scope* into both value assignments of a 2x2 — including
the one it was never fitted on, where a token or answer-direction account demands
the opposite movement. Three earlier intervention designs were retired or parked
for nameable reasons, and those retirements are not incidental: they are the
project's methodological content, and E13's design is what survived them. One
baseline is still outstanding before the claim can be written at full strength.

---

## Status at a glance

| Exp | What it tests | 1.3B | 6.7B | Status | Verdict |
|---|---|:--:|:--:|---|---|
| **E2** binding | binding, surface-proof | ● | ● | **foundation** | decodable from mid layers over a 0.500 floor |
| **E3** def-use | def→use edges | ● | ● | **foundation** | decodable, mild distance decay |
| E1 token type | lexical baseline | ● | ● | supporting | ceiling at the embeddings, as designed |
| E4 control dep | guard→statement | ● | ● | supporting, **not central** | decodable, but its surface floor is already 0.927 |
| E5 context | robustness to filler | ● | ● | supporting | survives length; collapses under interference |
| E9 obfuscation | semantics-preserving edits | ◐ | ● | supporting | robust to renaming mid-layer; breaks on flattening |
| E8 real code | CodeSearchNet transfer | ● | ● | supporting | transfers, with a stated limitation |
| E7 patching | causal, raw | ● | ● | supporting | **preliminary only**; the "isolates use" claim is retired |
| E10-0 J-lens | instrument validation | ● | ● | supporting | V1 exact; the Jacobian correction is real |
| E11 J-space | is the value causally reused? | ● | ● | **NO-GO** | see below — reported, not claimed |
| E12 store | text-absent value transfer | ● | ☐ | **parked** | behavioural gate failed at 0.418 |
| **E13** binding interchange | is the *binding* transported? | ☐ | ☑ | **H0–H5 pass** | rank-1 interchange installs the binding's value in BOTH arms (100%/100%); pending the `mean_difference` baseline |
| E6, E10-2, E10-3 | — | — | — | archived | `docs/ARCHIVE.md` |

Legend: ☐ not run · ◐ dev model only · ◑ partially run · ● run

---

# 1. Established: the representation exists and is built with depth

## E2 — variable binding

The claim rests on one control. A probe can score 100% on "are these two tokens
the same variable?" by reading the token strings, so every binding pair has a
**`context_matched`** partner: a second program that is token-identical except
the single character that flips the binding. The correct answer flips; nothing
observable about the text does.

| `context_matched` accuracy | 1.3B | 6.7B |
|---|---:|---:|
| surface baseline (token ids + distance, no model) | 0.500 | 0.500 |
| embedding layer (−1, token identity only) | 0.500 | 0.500 |
| block 0 (first transformer layer) | 0.570 | 0.531 |
| layer 3 | 0.961 | 0.914 |
| **peak (mid layers)** | **0.984** (L7) | **0.984** (L11–15) |
| last layer | 0.930 (L23) | 0.914 (L31) |

Three phases, each saying something different:

1. **Nothing at the input.** Both floors are *exactly* 0.500 — by construction,
   and confirmed in the data. The binding information is not in the tokens; it
   has to be built.
2. **Built in the first few blocks**, reaching ~0.91–0.96 by layer 3 and
   plateauing near 0.98 through the middle. That is early for a relation
   requiring scope resolution.
3. **Partly shed near the output** (~0.91–0.93), consistent with the final
   layers reorganising toward next-token prediction.

**Only `context_matched` is a clean headline.** The other strata sit at ~0.99
from block 0 because the token strings already separate them — the surface
baseline scores 0.78–0.94 on them too.

**Cross-scale.** The two models agree on shape and differ only where a scaling
account predicts: 6.7B does slightly less work in block 0 and holds its peak
longer — the same relative depth, stretched. The surface-baseline and
embedding rows are numerically identical across models, which they must be
since neither involves the model; that identity doubles as a corpus-integrity
check.

## E3 — def-use edges

Same design, same floors, same profile: peak ~0.99 at layers 7–11 with honest
decay by distance. The hardest bucket (50–200 tokens apart) holds at
**0.96–0.99** against ~0.99 for nearby pairs, so the model tracks def-use links
across real distance rather than adjacency.

---

# 2. Established: what the representation is made of

Both results below use **frozen** probes — fitted once on base programs, never
refitted on a variant — so a change in accuracy is a change in the model's
state, not in the probe.

## E5 — distance is cheap, interference is not

6.7B binding accuracy at 500 inserted filler tokens:

| filler | what it adds | acc | reading |
|---|---|---:|---|
| `comment_prose` | inert English | **0.921** | length is almost free |
| `dead_code` | unreachable statements | 0.794 | mild |
| `lexical_decoy` | similar-looking fresh names | 0.795 | mild |
| `competing_update` | rebinds *other* variables | 0.859 | moderate |
| `scope_shadow` | reuses the *tracked* names | **0.570** | **severe** |

At 1000 tokens `scope_shadow` reaches chance (0.498) while every other filler
stays above 0.70. Per layer: under `scope_shadow`, block 0 is the *most* stable
part of the network while the middle layers — the ones doing the binding work —
collapse. **The interference lands on the computation, not on a lookup.**

## E9 — renaming is survivable mid-layer; flattening is not

6.7B binding, best-layer accuracy per cumulative level:

| level | transform | best layer |
|---:|---|---:|
| 0 | normalize | ~1.000 |
| 1 | + rename every local | **0.897** (L11) |
| 2 | + opaque predicates | 0.857 |
| 3 | + MBA arithmetic | 0.846 |
| 4 | + control-flow flatten | **0.750** |

The layer breakdown is the finding: renaming pushes the *embedding and block-0*
probes **below chance** (0.29–0.33) — those layers keyed on identifier strings
and renaming actively misleads them — while mid layers 7–15 hold at 0.85–0.90.
Opaque predicates and rewritten arithmetic barely register, because they do not
change which definition reaches which use. Control-flow flattening is the true
limit.

**Together, E5 and E9 describe one failure surface.** The representation is
robust to how far apart things are and to what they are called, and it fails
when the scope or control structure it is a representation *of* becomes harder.
That is what one wants from a computed relation rather than a positional
heuristic — and it is a first, coarse map of when a tool built on these
representations should not be trusted.

---

# 3. Supporting, with limitations stated

## E4 — control dependence is decodable, but largely local syntax

| control_dep, best layer | positive recall | hard-negative recall |
|---|---:|---:|
| surface baseline (no model) | 0.959 | 0.676 |
| hidden — 1.3B (L11) | 0.981 | 0.873 |
| hidden — 6.7B (L15) | **0.995** | **0.923** |

The hidden state dominates on both classes at once, so the gap is not a
threshold artifact (aggregate AUC 0.990 → 0.999). **But the surface floor is
already 0.927**, unlike binding and def-use whose floor is pinned to exactly
0.500. A statement's guard is usually its nearest enclosing `if`.

This is reported as **the contrast that makes E2's isolation meaningful**, not
as a finding about representation. It is also the evidence that the project's
criterion for "semantic" excludes things.

## E8 — transfers to real code, but does not transfer the isolation

| 6.7B, aggregate AUC | surface | embedding | peak | last |
|---|---:|---:|---:|---:|
| binding | 0.673 | 0.962 | **0.978** (L7) | 0.913 |
| def-use | 0.590 | 0.958 | **0.979** (L3) | 0.907 |

Hidden states beat the surface baseline by +0.31/+0.39 AUC at the same relative
depth as synthetic, which rules out a pure generator-template explanation.

**The limitation, stated plainly.** In real code identifiers are genuinely
informative, so the embedding layer starts at 0.96 and no stratum pins the floor
to chance. E8 shows *the whole decoder transfers to naturalistic inputs*; it
does **not** show that the semantic component specifically transfers. E2's
isolation still rests on synthetic programs.

## E7 — preliminary causal evidence only

6.7B mean recovery: `sink_arg` 0.99 at layer 0 → 0.00 at layer 31;
`last_token` −0.01 → 1.00; `sanitizer_def` 0.000 everywhere.

**Supported:** the causal locus of the decision migrates from the sink-argument
token to the last-token position across the middle of the network, crossing over
near where E2's binding curve plateaus.

**Retired:** that it isolates *semantic use*. See `docs/ARCHIVE.md`.

## E10-0 — the J-lens implementation is correct

| check | 1.3B | 6.7B | reading |
|---|---:|---:|---|
| V1 — J-lens vs logit lens at the last layer | **1.0000** | **1.0000** | `J` is provably the identity there, so this must be 1.0 — a closed-form check of the whole gradient path |
| V2 — next-token top-1 (chance 0.038) | 0.633 | 0.650 | the lens reads real content |
| V2 advantage over the logit lens, pre-final | **+0.150** | **+0.183** | the Jacobian correction recovers content the logit lens cannot |

Instrument validation, not a result about the model. *Caveat:* V3 passed at
n=10, too small to carry weight; V1 and V2 are the load-bearing checks.

---

# 4. The open question: is the representation causally used?

This is the project's centre of gravity and it is **not settled**. Four designs
have been attempted. The honest summary of each:

## E11 — reported, but formally a NO-GO

`results/jspace/6.7b-5fam/go_no_go.md` and `go_no_go_answer.md` both read
**Verdict: NO-GO**:

| check | use position | answer position |
|---|---|---|
| behavioural balanced accuracy (≥ 0.75) | FAIL 0.706 | FAIL 0.706 |
| readout beats the random control | FAIL | PASS +0.257 |
| swap moves logits toward the swapped value | FAIL +0.001 | PASS +0.141 |
| **swap is specific to the value subspace** | FAIL | **FAIL −0.016 [−0.024, −0.009]** |
| cross-operation, all families positive | False | False |

**What can be said.** At the readout position the value-coordinate swap reaches
46% of the efficiency of an ideal same-norm push while two matched-norm controls
reach zero, and it is positive in both operation families the model computes
reliably. Something output-aligned is causally reused near the output.

**What cannot.** That it is the Jacobian correction — the plain logit lens is
*more* efficient at the same site, and the specificity check fails. And the
use-position null is **retracted**: a dose-matched control showed the site's
response to small edits is 18× convex, so no two-dimensional edit is large
enough to test the question there.

E11's numbers are reported in the paper because the *failure* is informative.
They are not claimed as a positive result.

## E12 — parked, and why

The behavioural gate failed at **0.418 balanced accuracy on 1.3B — below
chance** — with the correct answer as argmax on 6.3% of prompts against a 10%
uniform floor. Two of four operation families sat at *exactly* 0.500, which a
simulation showed a model doing **no computation at all** reproduces by picking
whichever candidate is numerically closer to the head literal.

The design coupled a question about program state to two chained arithmetic
steps. That is a design error, not a finding about code models. Code and gates
are kept and runnable; nothing is claimed.

## E13 — H0–H5 all pass (6.7B)

**Gates passed so far** (6.7B, 400 base programs):

| gate | result |
|---|---|
| **H0** generation and independent ground truth | **PASS** — 400/400 bases; all six invariant checks at 1.0000, including the arm crossing |
| **H1** the model returns the bound variable | **PASS** — 1.000 overall, 1.000 in the weakest cell |
| **H2** the binding is decodable at the use anchor | **PASS** — 1.000 against a measured surface floor of 0.500 |
| **H3** whole-state interchange flips the answer, per arm | **PASS** — ab +4.781 [+4.683, +4.878], ba +4.799 [+4.694, +4.903], flip rate 0.857; both structural zeros exactly 0.00e+00 |
| **H4** low-rank interchange beats matched controls on the training arm | **PASS** — +9.029 [+8.952, +9.108]; `das − random_norm` +8.126 [+8.020, +8.225], `das − random_rank` +9.033, `das − noop` +9.029 |
| **H5** the same subspace transfers to the held-out arm | **PASS** — +9.009 [+8.933, +9.089]; transfer ratio 0.998 against `whole_state`'s 1.004 |

H1 at 1.000 and H2 at 1.000 are worth pausing on: with no arithmetic anywhere,
6.7B resolves these bindings perfectly, and which definition is in scope is
perfectly decodable at the use anchor against a floor pinned to 0.500. That is a
cleaner replication of E2's isolation than E2 itself, on a corpus built for
intervention.

**H4 and H5 both pass.** A rank-1, magnitude-free interchange at the use anchor
(layer 8), fitted on arm `ab` alone, makes the model emit the value the
*installed binding* selects on **100.0% of held-out rows in both arms** — 280
base programs, 560 rows per cell, cluster bootstrap over bases. The outcome is
the full-vocabulary argmax rather than the logit margin, because `delta_ld` is
positively biased at ceiling accuracy and any disruption inflates it.

| variant | `ab` emits installed | `ba` emits installed | edit fraction |
|---|---:|---:|---:|
| **`das_binding`** (rank 1) | **100.0%** | **100.0%** | 0.479 |
| `whole_state` (the entire donor state) | 85.7% | 87.9% | 0.805 |
| `answer_direction` (J-lens, norm-matched) | 27.9% | 4.3% | 0.479 |
| `random_norm` (dose-matched random) | 1.1% | 0.7% | **0.538** |
| `random_rank` / `noop` / raw unembedding | 0.0% | 0.0% | 0.018 / 0 / 0.479 |

All 14 machinery checks pass: structural zeros exactly 0.00e+00, alignment
orthonormal to 4.07e-07, ceiling alive in both arms, and the model emits a
non-candidate token on 0.0% of rows.

**What this refutes, rather than merely fails to support.** A *disruption*
account has to explain why the dose-matched random subspace, which is
**over**-dosed at 0.538 of ‖h‖ against the treatment's 0.479, produces the
installed answer on 1.1% of rows against 100%. An *answer-direction* account has
to explain why the explicit answer direction attenuates 6.9× across the arms
while the treatment does not attenuate at all — and why it pushes the model
off-candidate on 9.1% of rows where the treatment never does. This is the
falsification E11 could not construct, because with arithmetic between the value
and the answer it had to forbid `answer == value` to avoid circularity.

**Two things remain open, and the first gates how the claim may be written.**
The learned direction sits at |cos| **0.673** from the mean donor−host difference
— substantially aligned, not identical. A cosine cannot say whether the optimiser
earned the rest, so a closed-form `mean_difference` baseline has been added and
stage 106 needs one re-run. If that baseline also transports, the honest claim
narrows to *a single fixed direction carries the binding* — still a result, and a
cleaner one, but a different sentence. Second, a rank-1 edit outperforming the
whole-state patch (100% vs 86%) has a plausible explanation — the full patch
installs the driving component *and* components that fight it — that is **not**
independently demonstrated. Until both are settled this reads "a rank-1
subspace", never "a learned abstraction".

---

# 5. What this project does not claim

- Not that code models "understand" programs. Every claim is a decoding or
  intervention result at named sites under named controls.
- Not that binding is causally used *in general*. E13 shows a rank-1 interchange
  transports the binding at one site, in one layer, in one model, on one
  synthetic construction. E7, E10, E11 and E12 each failed to establish even
  that, for a different recorded reason.
- Not that the E13 subspace is a *learned* abstraction rather than the
  difference-in-means direction. |cos| is 0.673 and the closed-form baseline has
  not yet been run.
- Not that the isolation transfers to real code. E8 shows the decoder
  transfers; the 0.500 floor exists only in synthetic programs.
- Not that control dependence is a semantic result — its floor is 0.927.
- Not that E11's readout-position effect passed. Both go/no-go files read NO-GO
  and the specificity check failed.
- Not that the 0.500 floor is pinned against *every* computable text feature. It
  is pinned against the stated surface baseline (±3 token ids plus bucketed
  distance). A cross-position string-equality baseline is outside that window
  and is an open item.

Withdrawn claims, with reasons: `docs/ARCHIVE.md`.

---

# 6. Open items

1. **Run the `mean_difference` baseline** (one stage-106 re-run, one extra
   variant, no backward pass). The learned direction is at |cos| 0.673 from the
   mean donor−host difference; if the closed-form direction transports too, the
   E13 claim must be written as "a fixed direction" rather than "a learned
   subspace". This is the only thing standing between E13 and a paper claim.
2. **Explain, or bound, the rank-1 edit beating the whole-state patch**
   (100% vs 86% at 60% of the edit norm). The available account — the full patch
   installs components that fight the driving one — is plausible and untested. A
   reviewer will ask; better to answer it first.
3. **Context-matched pairs on real code** — the highest-value follow-up for the
   foundation. It would upgrade E8 from a transfer check to a like-for-like
   replication of E2's isolation. Build by mutating real functions; 150
   candidate sites already exist in the CodeSearchNet corpus.
4. **A cross-position string-equality surface baseline** in stage 20. The
   current baseline cannot represent "the inner definition's name equals the
   use's name", which is the feature a lexical adversary would use. CPU-only,
   about an hour. Better to build it than to have a reviewer build it.
5. **E8 stratum sizes** — `static_probes.csv` records per-stratum accuracy but
   not per-stratum *n*. If `same_name_diff_binding` on real code is a handful of
   pairs, the claim weakens from "fails" to "underpowered".
6. **Report E5/E9 at peak rather than layer-averaged** — the averages hide the
   strongest findings (renaming fools layer 0 but not layer 11).
7. **E4 re-anchoring** — guard variable and statement target instead of the
   span's trailing literal. CPU-only re-run.

Raw data of record: `results/tables/*.csv` (one row per measurement). Rendered
summaries: `results/tables/md/*.md`. Figures: `results/figures/`. Regenerate
with `python scripts/90_make_paper_assets.py` (`--include-archived` rebuilds the
retired ones too).
