# Results

**Do code LLMs build a representation of program semantics that behaves like a
computation rather than a surface trick?** For **variable binding and def-use
structure** the answer is yes, at both scales, against floors that are pinned
to chance by construction. That is the foundation, and it is the only thing
here claimed as established.

Everything else in this file is either **supporting** — real, reported, and
constraining rather than carrying the argument — or **active**: the current
direction, designed but not yet run.

Two rules for reading this document:

- The status of every experiment is recorded in `results/STATUS.yaml`, which is
  also what `scripts/90_make_paper_assets.py` reads to decide which figures to
  regenerate.
- Claims this project used to make and has withdrawn are in
  `docs/LEGACY_RESULTS.md`, each with the reason. The data behind them is
  preserved; only the interpretation is retired.

Raw data of record: `results/tables/*.csv` (one row per measurement). Rendered
summaries: `results/tables/md/*.md`. Figures: `results/figures/`. Regenerate
with `python scripts/90_make_paper_assets.py` (add `--include-archived` to
rebuild the retired ones too).

| Exp | What it tests | 1.3b | 6.7b | Status | Verdict |
|-----|---------------|:----:|:----:|---|---|
| E2 binding | variable binding, surface-proof | ● | ● | **foundation** | Decodable from mid layers over a 0.500 floor |
| E3 def-use | def→use edges | ● | ● | **foundation** | Decodable, mild distance decay |
| E1 token type | lexical baseline | ● | ● | supporting | Ceiling at the embeddings, as designed |
| E4 control dep | guard→statement | ● | ● | supporting (not central) | Decodable, but the surface floor is already 0.927 |
| E5 context | robustness to filler | ● | ● | supporting | Survives length; collapses under interference |
| E7 patching | causal, raw | ● | ● | supporting | **Preliminary** causal evidence; see the retired claim |
| E8 real code | CodeSearchNet transfer | ● | ● | supporting | Transfers, with a stated limitation |
| E9 obfuscation | semantics-preserving edits | ◐ | ● | supporting | Robust to renaming mid-layer; breaks on flatten |
| E10-0 J-lens | instrument validation | ● | ● | supporting | V1 exact; the Jacobian correction is real |
| **E11 J-space** | **is the bound value causally reusable?** | ☐ | ☐ | **active** | Not yet run |
| E6, E10-2, E10-3 | — | — | — | archived | See `docs/LEGACY_RESULTS.md` |

Legend: ☐ not run · ◐ dev model (1.3b) · ● main model (6.7b)

---

# 1. Validated foundation

## E2 — variable binding is genuinely encoded

The claim rests on one control. A probe can score 100% on "are these two tokens
the same variable?" by reading the token strings. To rule that out, every
binding pair has a **`context_matched`** partner: a second program that is
*token-identical* except for the single character that flips the binding, so
the correct label flips while every surface cue stays put.

| `context_matched` binding accuracy | 1.3b | 6.7b |
|---|---:|---:|
| Surface baseline (token ids + distance, no model) | 0.500 | 0.500 |
| Embedding layer (−1, token identity only) | 0.500 | 0.500 |
| Block 0 (first transformer layer) | 0.570 | 0.531 |
| Layer 3 | 0.961 | 0.914 |
| **Peak (mid layers)** | **0.984** (L7) | **0.984** (L11–15) |
| Last layer | 0.930 (L23) | 0.914 (L31) |

*Figures: `binding_strata_{model}_core.png`, `layers_accuracy_{model}_core.png`.*

Three things happen, in order:

1. **Nothing is there at the input.** Both floors sit at *exactly* 0.500 — not
   approximately, but by construction, and confirmed in the data. The binding
   information does not exist in the tokens; it has to be built.
2. **The model builds it in the first few blocks**, reaching ~0.91–0.96 by
   layer 3 and plateauing near 0.98 through the middle.
3. **It is partially shed near the output** (~0.91–0.93), consistent with the
   final layers reorganizing toward next-token prediction.

**Only `context_matched` is a clean headline number.** The other strata
(`diff_name`, `distance_matched`) sit at ~0.99 from block 0 because the token
strings already separate them — the surface baseline scores 0.78–0.94 on them
too. The `same_name_diff_binding` stratum is a diagnostic rather than a result:
at the embedding layer it scores 0.001 (the probe sees identical names and
confidently guesses "same binding"), and by layer 3 it is at 0.99.

**Cross-scale.** The two models agree on shape and differ only where a scaling
story predicts: 6.7b does slightly less work in block 0 (0.53 vs 0.57) and
holds its peak longer (L11–19 vs L7–11) — the same relative depth stretched
over a deeper network. The surface-baseline and embedding rows are numerically
identical across models, which doubles as a corpus-integrity check.

## E3 — def-use edges

Same design, same floors, and the same profile: peak ~0.99 at layers 7–11 with
an honest decay by distance. The hardest bucket (def and use 50–200 tokens
apart) stays at **0.96–0.99** against ~0.99 for nearby pairs, so the model
tracks def-use links across real distance rather than adjacency.
*(Figure: `defuse_distance_{model}_core.png`.)*

---

# 2. Active: E11 — J-space binding routing

> **When a code model resolves variable binding, does it route the selected
> value into J-lens coordinates that are causally reusable by downstream
> computation?**

E2 and E3 establish that binding is *decodable*. Decodable is compatible with
the representation being a faithful shadow of a computation that happens
somewhere else entirely. E11 asks the next question with an intervention, and
treats the J-lens strictly as what it is: a **causal, output-aligned coordinate
system**, `v_w = J_ℓ^T (g·W_U[w])`, the direction at layer ℓ whose component
pushes the model's own output head toward token `w`. No claim about
reportability, verbalizability, or a workspace is made or needed — see
`docs/LEGACY_RESULTS.md` for why that framing was dropped.

**Status: designed, tested on CPU, not yet run on a model.** There are no E11
numbers in this repository. What follows is the design and the pre-registered
criteria, so that the pilot's outcome is interpretable either way.

## The data (stage 70)

Token-aligned counterfactual pairs. A one-token mutation of the inner
definition's *name* flips which value the marked use selects, while both values
occur in both programs:

```python
# case 0007                      # case 0007
x = 3                            x = 3
def f():                         def f():
    y = 7                            x = 7
    return x * 2 + 1                 return x * 2 + 1
assert f() ==   → 7              assert f() ==   → 15
```

Enforced at generation and re-checked in `tests/test_jspace.py`: one differing
token; equal token length so every probed position is the same index in both;
the mutation is never the marked use and never adjacent to it; both answers are
single tokens, distinct, and **disjoint from both values** (otherwise an answer
token and a distractor value token would be the same lens row and every
downstream number would be circular); ground truth from *executing* the
program, cross-checked against the operation's own Python function.

Three templates (`global_shadow`, `call_frame` — where the operation lives in a
callee, so routing must cross a call boundary — and `padded_shadow`) and five
operation families (affine, multiply/subtract, threshold, modulus, list
indexing). Each *base* carries several families over the same two values, which
is what makes the causal test falsifiable.

## The instrument (stage 71, a gate)

One frozen J-lens per layer, built from a **held-out generic Python corpus**
(CodeSearchNet), never from the evaluation programs, with broad source
positions and randomly sampled future readout positions. Three independent
build samples per layer measure stability, on directions (rowwise cosine) and
on decisions (margin-sign agreement on held-out states) — a layer whose lens is
unstable cannot carry a claim about that layer however large its effect looks.
V1 (last-layer identity with the logit lens) and V2 (next-token recovery) are
E10-0's checks, reused unchanged.

## The readout (stage 72)

At each probed position — before the definitions, at each definition, at the
mutation, at the marked use, at the answer — rank the bound value's lens row
against the distractor's. The claim-bearing metric is the **paired
counterfactual margin reversal**: the margin must be positive in one program
*and* negative in its one-token mutation. A readout that prefers small numbers,
or the first-mentioned literal, or the token it just saw, gives the same margin
in both and scores zero reversals. Four readouts on identical hidden states:
J-lens, logit lens, a **Gram-matched** random control (same norms *and* same
angles, so only the directions are arbitrary), and a probe trained on the
calibration split — the incumbent, not a floor.

## The intervention (stage 73)

At the marked use, with `V = [v_source, v_target]` and `c = V⁺h`:

```
h_patched = h + V (swap(c) − c)
```

Only the two value coordinates change; the orthogonal complement is untouched;
the operator is an involution; and identical directions give *exactly* the zero
edit, so the same-value control is provably inert rather than approximately so.
Applied in both directions, at individual layers and short layer bands, and
scored as a paired shift in `logP(answer implied by the other value) −
logP(answer bound here)` against the same program's clean run.

**The falsification test.** The same value swap must move each operation family
toward *its own* answer — `x=3` implies 7 under `2x+1`, 1 under `x>4`, 0 under
`x%2`, something else under `tbl[x]`. An intervention that steers the answer
token cannot do that, so the summary reports the per-family minimum, not the
pooled mean. Controls: logit-lens subspace, Gram-matched random subspace,
same-value no-op, irrelevant position (`pre_def`), whole-state counterfactual
patch (the ceiling, not a control), and a direct answer-token swap.

## Method rules, enforced in code

- Lens-building corpus, calibration split and test split are disjoint;
  `counterfactual_pairs.assert_disjoint` is called by every stage and is unit
  tested against a deliberate leak.
- The calibration/test split is assigned in the data file, grouped by base
  program and stratified by template, so all stages agree by construction.
- The layer and the intervention site are selected on **calibration only** and
  recorded in the manifest before the test number is read.
- No example is dropped for being answered wrongly. Every example is reported;
  the "both counterfactuals correct" subset is labelled and summarized
  *alongside*, never instead.
- All intervals are cluster bootstraps grouped by base program, and all control
  comparisons are paired on the same rows.
- Complete per-example data, including clean and patched logits, is saved.

## Pre-registered go/no-go (stage 74)

Recommend the full 6.7b run only if all three hold on the 1.3b pilot (200
pairs, two operation families, four layers):

1. behavioural balanced accuracy ≥ 0.75;
2. the bound-value J-lens readout beats the Gram-matched random control
   (paired CI lower bound above zero, at the calibration-selected layer);
3. the coordinate swap produces a positive paired logit shift (CI lower bound
   above zero, at the calibration-selected site).

The verdict lands in `results/jspace/{model}/go_no_go.{yaml,md}`.

---

# 3. Supporting results

These are reported and constrain the picture; none of them carries the
argument.

## E1 — lexical token type (machinery check)

Peaks at **1.000 accuracy at the embedding layer** with selectivity ~0.88–0.90
in both models. Expected, not a finding: token type is a pure lexical property,
best decoded before context is added. It confirms the extraction and probing
machinery, and gives the contrast for E2 — lexical features are readable from
the embeddings, semantic relations are not and only appear after computation.

## E4 — control dependence: encoded, but largely local syntax

The first attempt was invalid (surface baseline 1.000: control-dependent
statements were the only ones indented under an `if`). Rebuilt with
sibling-guard programs and an `indent_matched` hard negative — a statement in a
*different* guard's body at the same nesting depth — the honest picture is
mixed, and identical at both scales:

| control_dep, best layer | positive recall | hard-neg recall |
|---|---:|---:|
| Surface baseline (no model) | 0.959 | 0.676 |
| Hidden — 1.3b (L11) | 0.981 | 0.873 |
| Hidden — 6.7b (L15) | **0.995** | **0.923** |

The hidden state dominates on both classes at once, so the gap is not a
threshold artifact. Aggregate AUC 0.990 (surface) → 0.999 (6.7b).

**But the surface floor is already 0.927**, unlike binding and def-use whose
floor is pinned to exactly 0.500. A statement's guard is usually its nearest
enclosing `if`, so token windows plus distance get most of the way there.
Control dependence is a largely local, syntactic relation, and this result is
best read as the contrast that makes E2's isolation meaningful — not as a
finding about representation. That is why it is classified supporting but
**not central**.

*Caveat:* probing anchors fall on each span's last token, which here are
integer literals. Re-anchoring on the guard variable and statement target is a
CPU-only stage-20 re-run and remains open.

## E5 — survives length, collapses under interference

Frozen E2/E3 probes on programs padded with five filler types, sized by real
tokenizer counts (0 → 1000 tokens).

**6.7b binding accuracy at 500 filler tokens:**

| Filler type | What it adds | Acc @500 | Reading |
|---|---|---:|---|
| `comment_prose` | inert English | **0.921** | Length is almost free |
| `dead_code` | unreachable statements | 0.794 | Mild |
| `lexical_decoy` | similar-looking fresh names | 0.795 | Mild |
| `competing_update` | rebinds other variables | 0.859 | Moderate |
| `scope_shadow` | reuses the tracked names | **0.570** | **Severe** |

At 1000 tokens `scope_shadow` drives binding to 0.498 — chance — while every
other filler stays above 0.70. The representation degrades exactly when the
*semantic task* gets harder, not when the context gets longer. A per-layer
detail sharpens it: under `scope_shadow`, block 0 is the most stable (flat
~0.75) while the mid layers — the ones doing the binding work — collapse. Both
scales show the same ranking.

## E7 — causal patching (preliminary)

Activation patching on length-matched minimal pairs, measuring logit-diff
recovery.

**6.7b, mean recovery:**

| Layer | `sink_arg` | `last_token` | `sanitizer_def` |
|---:|---:|---:|---:|
| 0 | **0.99** | −0.01 | 0.00 |
| 7 | 0.71 | 0.07 | 0.00 |
| 15 | 0.24 | 0.31 | 0.00 |
| 23 | 0.05 | 0.76 | 0.00 |
| 31 | 0.00 | **1.00** | 0.00 |

1.3b replicates the pattern (`sink_arg` ≈ 1.0 at layers 0–3, `last_token` 1.00
at L23).

**What this supports:** the causal locus of the decision migrates from the
sink-argument token to the last-token position across the middle of the
network, crossing over near where the E2 binding curve plateaus. That is a
reproducible description of where the decision becomes committed.

**What it does not support**, and what was retired: that it isolates *semantic
use*. `sink_arg` is the only place the two programs differ, so patching there
transports the surface difference along with any semantic state; the
`sanitizer_def` null has no positive control at that position; and late-layer
`last_token` recovery forces the answer trivially. Full reasoning:
`docs/LEGACY_RESULTS.md`. E11's coordinate swap exists to close exactly this
gap.

## E8 — transfers to real code, with a limitation

Stages 10+20 re-run unchanged on ~200 `ast`-parseable CodeSearchNet functions.
Read AUC, not accuracy (accuracy is threshold-dependent and peaks at the
embedding layer here):

| 6.7b, aggregate AUC | Surface | Embedding (−1) | **Peak** | Last layer |
|---|---:|---:|---:|---:|
| binding | 0.673 | 0.962 | **0.978** (L7) | 0.913 (L31) |
| def-use | 0.590 | 0.958 | **0.979** (L3) | 0.907 (L31) |

1.3b matches closely (binding 0.980 at L3, def-use 0.975 at L3). Hidden states
beat the surface baseline by +0.31/+0.39 AUC, AUC rises above the embedding
layer to an early-middle peak at the same relative depth as synthetic, and
declines toward the output.

**The limitation, stated plainly.** In real code identifiers are genuinely
informative — `self._cache` and `result` look different — so the embedding
layer starts at 0.96 rather than 0.500 and no stratum pins the surface floor to
chance. E8 therefore shows that *the whole decoder transfers to naturalistic
inputs*; it does **not** show that the semantic component specifically
transfers. E2's isolation still rests on synthetic programs. The
`same_name_diff_binding` stratum looks bad on real code (0.095→0.494) but is
class-conditional recall with no per-stratum AUC recorded, so it is weaker
evidence than it appears.

The fix — context-matched pairs built by *mutating* real functions — is open
item 1 below, and 150 candidate sites already exist in this corpus.

## E9 — robust to renaming mid-layer, broken by flattening

Frozen E2/E3 probes on a five-level, cumulative, **execution-verified**
obfuscation ladder.

**6.7b binding, best-layer accuracy per level:**

| Level | Transform | Layer-avg | **Best layer** |
|---:|---|---:|---:|
| 0 | normalize | 0.974 | ~1.000 |
| 1 | + rename identifiers | 0.704 | **0.897** (L11) |
| 2 | + opaque predicates | 0.712 | 0.857 |
| 3 | + MBA encoding | 0.728 | 0.846 |
| 4 | + control-flow flatten | 0.572 | **0.750** |

The layer breakdown is the finding: renaming pushes the *embedding and block-0*
probes below chance (0.29–0.33) — those layers keyed on identifier strings and
are actively fooled — while mid layers 7–15 hold ~0.85–0.90. Opaque predicates
and MBA arithmetic barely register, because they do not change which definition
reaches which use. Control-flow flattening is the true limit: the frozen probes
encode binding relative to the surrounding control structure, and removing that
scaffold breaks transfer. Both models trace nearly identical ladders.

## E10-0 — the J-lens implementation is correct, and the correction is real

| Validation | 1.3b | 6.7b | Reading |
|---|---:|---:|---|
| **V1** J-lens vs logit lens at the last layer | **1.0000** | **1.0000** | `J` is provably the identity there, so this must be 1.0 — a closed-form check of the whole gradient path |
| **V2** next-token top-1 (chance 0.038) | 0.633 (L19) | 0.650 (L27) | the lens reads real content |
| V2 random floor | 0.000–0.133 | 0.000–0.050 | |
| **V2 advantage over the logit lens, pre-final** | **+0.150** | **+0.183** | the Jacobian correction recovers content the logit lens cannot |

This is instrument validation and it is what E11 reuses. The last row is a
genuine positive: the Jacobian correction works in a code LLM, so an E11 null
would not be "the method doesn't work here". *Caveat:* V3 (taint disposition)
passed at n=10, too small to carry weight; V1 and V2 are the load-bearing
checks.

---

# 4. What this project no longer claims

Withdrawn, with reasons, in `docs/LEGACY_RESULTS.md`:

- **"Computes, uses, but does not report."** Two of its three legs rested on
  absence-of-evidence results whose positive controls could not license reading
  the absence.
- **E6 behavioural lead time.** The original positive was measured on a
  constant responder; with a working signal and an analytic null, a no-model
  position baseline outscores a 99%-accurate probe, because the metric rewards
  unreliability. The negative is not a finding either.
- **E10-2 taint verbalizability.** Inherits E6's metric, and the effect it was
  built to explain did not survive.
- **E10-3 "decodable but not verbalizable".** A well-defended null, but with no
  *relational* positive control it cannot be told apart from a readout that
  cannot express relations at those positions.
- **E7 "isolates semantic use".** The design conflates transported surface
  difference with semantic state; the experiment is kept as preliminary causal
  evidence.

---

# 5. Open items

1. **Context-matched pairs on real code** — the highest-value follow-up for the
   foundation: it upgrades E8 from a transfer check to a like-for-like
   replication of E2's isolation, and settles whether the low
   `same_name_diff_binding` recall is a threshold artifact. Build by mutating
   real functions: given a def of `v` at line *i* and a use at *k*, rename an
   interposed assignment target at *j* (*i*<*j*<*k*) from `w` to `v`. Same
   invariants as E2 (single differing token, unchanged anchors and distance,
   `v` used in (*i*, *j*] so def *i* stays live). Measured yield: 150 candidate
   sites across 61 of 200 functions. `_Renamer` in `src/data/obfuscation.py` is
   the transform to reuse. Note the corpus is then mutated real code, not
   pristine CodeSearchNet.
2. **E8 stratum sizes** — `static_probes.csv` records per-stratum accuracy but
   not per-stratum *n*. The `same_name_diff_binding` count on real code must be
   measured before that negative goes in a paper; if it is a handful of pairs,
   the claim weakens from "fails" to "underpowered". CPU-only.
3. **Report E5/E9 at peak/per-layer** rather than layer-averaged — the averages
   hide the strongest findings (rename fools layer 0 but not layer 11).
4. **E4 re-anchoring** — guard variable and statement target instead of the
   span's trailing literal. CPU-only stage-20 re-run.
5. **E1 lexical AUC logs as 0.000** — a multi-class reporting artifact, not a
   fit failure. Emit `NaN` so the column is not misread.
6. **Environment provenance** — the archived E6 layer sweep ran under the `uq`
   env against probe checkpoints pickled by `semflow`; E7's numbers predate
   that move. Same data and seeds, but exact reproducibility would want both
   stages re-run in one environment.
7. **E11 pilot** — run `jobs/jspace_pilot.csh` and read
   `results/jspace/deepseek-coder-1.3b/go_no_go.yaml`.
