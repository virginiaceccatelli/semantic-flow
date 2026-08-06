# Results

**Do code LLMs internally represent program semantics, and does that
representation behave like a real computation rather than a surface trick?**
Across two models — `deepseek-coder-1.3b` (24 blocks) and the main-results
`deepseek-coder-6.7b` (32 blocks) — the answer is **yes for variable binding
and def-use structure**, with the causal and robustness experiments telling a
consistent story about *where* that information lives and *when* it breaks.

Raw data of record: `results/tables/*.csv` (one row per measurement). Rendered
summaries: `results/tables/md/*.md`. Figures: `results/figures/` (`png` to view,
`pdf` for the paper). Regenerate everything with
`python scripts/90_make_paper_assets.py`.

Status legend: ☐ not run · ◐ dev model (1.3b) · ● main model (6.7b) · ✗ ran but not interpretable

| Exp | What it tests | 1.3b | 6.7b | Verdict |
|-----|---------------|:----:|:----:|---------|
| E1 token type | lexical baseline | ● | ● | Decodable from embeddings alone (as expected) |
| E2 binding + strata | variable binding | ● | ● | **Positive** — decodable, surface-cue-proof |
| E3 def-use + distance | def→use edges | ● | ● | **Positive** — decodable, mild distance decay |
| E4 control dep | guard→statement | ● | ● | **Positive but surface-heavy** — hidden beats the hard stratum (0.92 vs 0.68), but control dep is largely locally decodable; replicates across scale |
| E5 context degradation | robustness to filler | ● | ● | Survives length; collapses under interference |
| E6 lead time | latent vs behavioral failure | ✗ | ● | **Negative (6.7b), undefined (1.3b).** 6.7b: no early warning — probe excess −0.010 vs a *no-model position baseline* at +0.113. 1.3b: behavioural signal at chance (balanced acc 0.471), so lead time is undefined there. The original positive was measured on a constant responder |
| E7 causal patching | is it *used*? | ● | ● | **Positive** — information routes across layers; sanitizer site is causally inert |
| E8 real code | CodeSearchNet transfer | ● | ● | **Transfers** — ~0.90 acc / 0.98 AUC vs 0.67 surface, same mid-early layer peak; but real code can't isolate the semantic component |
| E9 obfuscation | semantics-preserving edits | ◐ | ● | Robust to renaming mid-layer; breaks on flatten |
| E10 J-lens | verbalizable vs decodable | ● | ● | **Method transfers; both experiments null.** Lens validated (V1 exact, next-token top-1 0.65 vs 0.05 random) — but taint shows no verbalizable lead beyond a random floor, control dependence sits at chance at every layer. **E10-2 also invalidates E6's early-warning metric** (see below) |

All ten experiments have now run at both scales. **E8** confirms the probes
transfer to real Python with the same layer signature, and **E4 is valid and
replicates across scale** (its hard-negative control was rebuilt).

**E10 (J-lens)** produced two changes. A positive one: the Jacobian-lens method
*transfers to code models* — it passes its closed-form correctness check exactly
and recovers next-token content the logit lens cannot. And a corrective one: it
supplied the floors E6 never had, which dissolved E6's early-warning claim
entirely. E6 has since been re-run with a working behavioural signal and three
floors; **there is no early warning in either model, and a baseline using no
model at all scores higher on the metric than a 99%-accurate probe.**

Read together, E6, E7 and E10 converge on one picture: **the model computes
program semantics, causally uses some of them, and reports none of them.** The
taint state is decoded correctly at the exact moment the output is wrong (E6),
the sanitizer site is causally inert (E7), and nothing is verbalizable (E10).

---

## The core result: binding and def-use are genuinely encoded (E2, E3)

The central claim of the project rests on one control. A probe can hit 100%
accuracy on "are these two tokens the same variable?" simply by reading the
token strings — same name, same variable. To rule that out, every binding pair
has a **`context_matched`** partner: a second program that is *token-identical*
except for the single character that flips the binding, so the correct label
flips while every surface cue stays put. If the probe still separates them, it
must be reading something the model computed, not the text.

**It does.** Two independent floors confirm no shortcut is available, and the
hidden states clear both by a wide margin:

| `context_matched` binding accuracy | 1.3b | 6.7b |
|---|---:|---:|
| Surface baseline (token ids + distance, no model) | 0.500 | 0.500 |
| Embedding layer (−1, token identity only) | 0.500 | 0.500 |
| Block 0 (first transformer layer) | 0.570 | 0.531 |
| Layer 3 | 0.961 | 0.914 |
| **Peak (mid layers)** | **0.984** (L7) | **0.984** (L11–15) |
| Last layer | 0.930 (L23) | 0.914 (L31) |

*Figures: `binding_strata_{model}_core.png`, `layers_accuracy_{model}_core.png`.*

**How to read this curve.** Three things happen, in order:

1. **Nothing is there at the input.** The surface baseline and the embedding
   layer both sit at *exactly* 0.5 — chance. This is not an approximation; it is
   guaranteed by construction and confirmed in the data. The binding information
   simply does not exist in the tokens; it has to be *built*.
2. **The model builds it in the first few blocks.** Accuracy jumps from ~0.53 at
   block 0 to ~0.91–0.96 by layer 3 and plateaus near 0.98 through the middle of
   the network. Binding is computed early and cheaply, then held.
3. **It is partially discarded near the output.** Both models decline in the
   last third (to ~0.91–0.93). This is expected and meaningful: the final layers
   reorganize the representation toward next-token prediction, so an abstract
   fact like "these are the same variable" is no longer the priority once it has
   been used.

**Def-use edges (E3) behave identically** — peak ~0.99 at layers 7–11, with a
mild, honest decay by distance. Even the hardest bucket (def and use 50–200
tokens apart) stays at **0.96–0.99**, versus ~0.99 for nearby pairs. The model
tracks def-use links across real distance, not just adjacency.
*(Figure: `defuse_distance_{model}_core.png`.)*

### Why the other strata are less interesting (and one is a trap)

The per-stratum table shows most negatives are easy — `diff_name` and
`distance_matched` sit at ~0.99 from block 0 onward — because the token strings
already separate them. Those are *not* evidence of semantic encoding; the
surface baseline scores 0.78–0.94 on them too. **Only `context_matched` is a
clean headline number**, and it is the one quoted above.

The `same_name_diff_binding` stratum is a useful diagnostic: at the embedding
layer it scores **0.001** — the probe, seeing only identical names, confidently
guesses "same binding" and is always wrong. By layer 3 it is at 0.99. That
transition is the clearest single illustration that context, not spelling, is
doing the work.

### Cross-scale replication

The two models agree on the *shape* and disagree only in the details that a
scaling story would predict:

- 6.7b does slightly **less** binding work in block 0 (0.53 vs 0.57) but holds
  its peak **longer** (plateau L11–19 vs L7–11) — the same relative depth,
  stretched across a deeper network.
- The surface-baseline and embedding rows are **numerically identical** across
  the two models (same corpus, same tokenizer, no model in the loop). This is a
  built-in integrity check: it confirms the two runs share ground truth and that
  layer −1 really is context-free.

---

## E1 lexical baseline: read with care

**E1 (token type)** peaks at **1.000 accuracy at the embedding layer (−1)** with
high selectivity (~0.88–0.90) in both models. This is the *expected* control,
not a finding: token type is a pure lexical property, so it is best decoded
before any context is added. It confirms the machinery works and gives the
contrast for E3's thesis (RQ3) — **lexical features are readable from the
embeddings; semantic relations are not, and only appear after computation.**
`taint_state` is likewise at ceiling with ~0.5 selectivity; it is fine as the
*input* to E6/E7 but is not a standalone result.

---

## E4: control dependence is encoded — but it is also largely local syntax

The first E4 attempt was invalid: its surface baseline scored **1.000**, because
control-dependent statements were the only ones indented under an `if`, so token
windows plus distance separated them trivially. The corpus was rebuilt with
**sibling-guard programs** and a hard **`indent_matched`** negative stratum — a
statement sitting inside a *different* guard's body at the *same* nesting depth,
using neutral variables unrelated to the guard. Pooled across the sibling guards,
positives and these hard negatives overlap in both indentation and distance, so
the indentation shortcut is neutralized.

With that control in place, the honest picture is **more mixed than
binding/def-use**, and it is **the same at both scales**. The clean, threshold-
proof comparison holds *both* class recalls at once — the hidden probe must catch
the control-dependent pairs **and** reject the same-depth hard negatives:

| control_dep, best layer | positive recall | hard-neg (`indent_matched`) recall |
|---|---:|---:|
| Surface baseline (no model) | 0.959 | 0.676 |
| Hidden — 1.3b (L11) | 0.981 | 0.873 |
| Hidden — 6.7b (L15) | **0.995** | **0.923** |

The hidden state dominates surface on **both** classes simultaneously, so the gap
is not a decision-threshold artifact (a biased probe would trade one recall for
the other — which is exactly what the layer −1 = 1.000 figure is). Aggregate:
AUC 0.990 (surface) → 0.997 (1.3b) / **0.999** (6.7b). The surface baseline is
**numerically identical across the two models** (positive 0.959, hard-neg 0.676)
— it is model-free, so this doubles as a corpus-integrity check confirming both
runs share the same fixed ground truth.

Reading it two ways:

- **Hidden state carries genuine control-dependence structure.** By layer it is a
  clean, balanced separation: aggregate **AUC climbs 0.74 (embeddings) → 0.999
  (L15)**, positive recall **0.48 → 0.99**, selectivity **0.14 → 0.39**, on the
  same rise-then-plateau depth profile as binding. On the hard `indent_matched`
  stratum the hidden probe recovers **0.923** of the non-dependent statements
  versus **0.676** for surface — a real **+0.25** margin that no local cue
  supplies.
- **But control dependence is substantially surface-decodable.** Unlike binding
  and def-use, whose surface floor sits at *exactly 0.500* on token-identical
  pairs, the E4 surface baseline is already at **0.927 / AUC 0.990**. Control
  dependence is a largely *local, syntactic* relation — a statement's guard is
  usually its nearest enclosing `if` — so token windows plus distance get most of
  the way there. This is the RQ3 contrast made concrete: **the more syntactic the
  relation, the less the model needs a deep representation of it.**

Caveat for the write-up: with these templates the probing anchors fall on each
span's last token (the documented "last token integrates the span" convention),
which here are integer literals (`… > 50`, `… + K`); their mid-layer hidden
states still integrate the full guard/statement (AUC 0.999), but a future pass
could anchor on the guard variable and statement target for a cleaner readout.
The **layer −1 = 1.000** figure on `indent_matched` is *not* a leak — it is a
single-class-recall artifact of a threshold-biased embedding-layer probe
(AUC there is only 0.743, identical in both models). Both scales trace the same
rise-then-plateau curve (1.3b peaks L11, 6.7b L15 — the same relative depth),
so E4 now replicates across scale.

---

## E5: representations survive length but collapse under interference

Frozen E2/E3 probes were evaluated on programs padded with five kinds of filler,
sized by **real tokenizer counts** (0 → 1000 tokens). The question is whether
long context per se erodes the representation, or whether only *semantically
relevant* interference does. The answer is unambiguous — it is interference.

**6.7b binding accuracy at 500 filler tokens, by filler type:**

| Filler type | What it adds | Acc @500 tok | Interpretation |
|---|---|---:|---|
| `comment_prose` | inert English text | **0.921** | Length is almost free |
| `dead_code` | unreachable statements, fresh names | 0.794 | Mild — extra code costs a little |
| `lexical_decoy` | similar-looking fresh names | 0.795 | Mild — surface distractors |
| `competing_update` | code that rebinds other vars | 0.859 | Moderate |
| `scope_shadow` | code that reuses the tracked names | **0.570** | **Severe** |

At 1000 tokens `scope_shadow` drives 6.7b binding to **0.498 — pure chance**,
and def-use to 0.59. Every other filler type is still well above 0.70.
*(Figures: `context_binding_{model}.png`, `context_defuse_edge_{model}.png`.)*

**The interpretation (RQ2/RQ3).** A thousand tokens of comments barely dent the
probe, so the representation is not fragile to distance or context length. What
destroys it is `scope_shadow` — filler that forces genuine scope resolution by
reusing the very names being tracked. In other words, the representation degrades
exactly when the *semantic task itself* gets harder, which is what you would want
from a real binding computation rather than a positional heuristic. A per-layer
detail sharpens this: under `scope_shadow`, block 0 is the *most* stable
(flat ~0.75) while the mid layers — the ones doing the real binding work — are
the ones that collapse. The interference lands on the computation, not the
lookup. Both model sizes show the same ranking.

---

## E9: robust to renaming in the middle layers, broken by control-flow flattening

Frozen E2/E3 probes were evaluated on a five-level, cumulative,
**execution-verified** obfuscation ladder (each level provably preserves program
behavior). The layer-averaged summary is blunt; the per-layer picture is the real
finding.

**6.7b binding, best-layer accuracy per level:**

| Level | Transform | Layer-avg | **Best layer** | Reading |
|---:|---|---:|---:|---|
| 0 | normalize (reformat) | 0.974 | ~1.000 | Formatting is irrelevant |
| 1 | + rename identifiers | 0.704 | **0.897** (L11) | The big cliff — and where it lands matters |
| 2 | + opaque predicates | 0.712 | 0.857 | Adds almost nothing over rename |
| 3 | + MBA encoding | 0.728 | 0.846 | Adds almost nothing |
| 4 | + control-flow flatten | 0.572 | **0.750** | The second, harder break |

*(Figures: `obfuscation_levels_{model}.png`, `obfuscation_{task}_{model}.png`.)*

**Two breaks, two lessons.**

- **Renaming is the first cliff, but the layer breakdown rescues the story.**
  The *average* fall to ~0.70 hides a split: at the embedding/block-0 layers,
  renaming pushes the probe **below chance (0.29–0.33)** — those early layers
  keyed on the identifier strings and are actively fooled — while the **mid
  layers (7–15) hold ~0.85–0.90**. So early layers carry name-based features and
  mid layers carry something closer to structural binding. This is the same
  early-lexical / mid-semantic division E1 and E5 point to.
- **Opaque predicates and MBA arithmetic barely register** (0.71–0.73): junk
  branches and rewritten expressions don't disturb binding, because they don't
  change *which definition reaches which use*.
- **Control-flow flattening is the true limit.** Once the control structure is
  dissolved into a dispatch loop, even the best layer only reaches ~0.75 and the
  average sits at 0.57. The frozen probes encode binding *relative to the
  surrounding control structure*; remove that scaffold and transfer largely
  fails. This is the honest boundary of how abstract the representation is.

Both models trace nearly identical ladders — the finding replicates across scale.

---

## E7: the information is causally used, and it moves across the network

Activation patching on **length-matched minimal pairs** (identical except the
sink argument) measures logit-diff recovery: how much of the model's output
flips when a single position's activations are swapped. This is the causal
counterpart (RQ5) to the correlational probes above.

**6.7b, mean recovery (fraction of output flip explained):**

| Layer | `sink_arg` | `last_token` | `sanitizer_def` |
|---:|---:|---:|---:|
| 0 | **0.99** | −0.01 | 0.00 |
| 3 | 0.91 | 0.01 | 0.00 |
| 7 | 0.71 | 0.07 | 0.00 |
| 11 | 0.50 | 0.15 | 0.00 |
| 15 | 0.24 | 0.31 | 0.00 |
| 19 | 0.04 | 0.65 | 0.00 |
| 23 | 0.05 | 0.76 | 0.00 |
| 31 | 0.00 | **1.00** | 0.00 |

*(Figure: `patching_recovery_{model}.png`.)*

**This is textbook information routing.** Early on, the taint identity lives at
the **sink-argument token** — patching it there recovers ~all of the behavior
(0.99 at layer 0). Across the middle of the network the causal locus **migrates
to the last-token position**, which fully controls the decision by layer 31.
The crossover (~layer 15) matches where the E2 binding curve is at its plateau:
the model has finished *computing* the relation and is now *moving it into place*
for the readout.

**1.3b shows the same routing** — `sink_arg` dominates early (≈1.0 at layers
0–3) and `last_token` takes over late (0.90 at L19, **1.00 at L23**), with
`sanitizer_def` at **0.000 at every layer**. The causal picture replicates
across scale.

The third column is the quiet bombshell: patching **`sanitizer_def` recovers
nothing at any layer (0.000 throughout)**. Overwriting the sanitizer's
definition never changes the output. The model's taint decision does not route
through the sanitization site at all — which sets up, and is confirmed by, E6.

---

## E6: there is no early warning — and a no-model baseline "detects" more of it

E6 asks whether the taint probe's internal state degrades *before* the model's
answer goes wrong (RQ4). The answer, once the experiment is given a working
behavioural signal and three floors, is **no** — and the way the original
positive result arose is the more useful finding.

### The original result was measured on a constant responder

Under the bare prompt (`is the current value tainted?`) **both models answered
the same token to every prefix of every program**:

| Model | answer | raw accuracy | **balanced accuracy** |
|---|---|---:|---:|
| 1.3b | always "no" | 0.220 | **0.500** |
| 6.7b | always "yes" | **0.780** | **0.500** |

6.7b's 0.780 is the trap: it is exactly the base rate of `tainted=1`, so the
signal looks healthy under the check anyone would actually run. Balanced
accuracy — the floor that catches it — is 0.500 for both.

**The two opposite biases produce E6's entire "scale split" mechanically.**
The generator always emits the taint source on line 2, so `tainted=1` at the
first evaluable prefix for every program. Therefore:

- **1.3b always says "no"** → wrong at the first prefix → `t_failure = 2` on
  100% of programs → a positive lead is *arithmetically impossible* → the
  reported "no early warning at any layer".
- **6.7b always says "yes"** → wrong only where `tainted=0`, i.e. at and after
  the sanitizer → `t_failure` lands mid-program on exactly the sanitized
  programs (32 of 70) → any probe erring before the sanitizer scores a "lead"
  → the reported "66% of failures, +2.3 prefixes".

No model computation entered the behavioural side of either number.

### Fixing the prompt: only one variant works, and only on 6.7b

`scripts/diagnose_taint_prompt.py` sweeps four prompts. Only **few-shot
demonstrations *and* naming the variable** (`is the value of \`v2\` tainted?`)
lifts 6.7b out of degeneracy — balanced accuracy **0.857**. Neither ingredient
alone does anything. **1.3b cannot do the task under any prompt**: the
stage-40 run of record measures balanced accuracy **0.471** (says-tainted rate
0.734 — no longer constant, but no better than chance), and stage 40 flags the
signal `usable=False`. Lead time is therefore **undefined** for 1.3b, not
"zero": `t_failure` is noise, so nothing can be early or late relative to it.
That is a capability result, not a representational one, and the 1.3b rows in
the tables below should be read as unmeasurable rather than negative.

### With a working signal and three floors: no early warning

Stage 40 now reports every readout against an **analytic null** — for a
readout with per-prefix error rate ε whose errors are independent of the
model's state, the chance of erring before step *k* is 1−(1−ε)^(k−1). Only
`early_warning_excess` (observed − null) can support a claim.

Only 6.7b has a usable behavioural signal, so only its column is a
measurement; 1.3b's is shown for completeness and is undefined (see above).

| Readout | **6.7b excess** | 1.3b (undefined) |
|---|---:|---:|
| **`position`** — no model at all, "tainted iff step ≤ 3" | **+0.113** | +0.080 |
| `random` — norm-matched random direction | +0.005 … +0.067 | +0.011 |
| **`probe`** — trained, ~99% accurate | **−0.010** | −0.023 |

**A baseline that knows nothing but how many lines into the program it is
scores higher on early warning than a 99%-accurate probe.** The trained probe
averages *negative* excess in both models; not one of its positive cells
survives Bonferroni correction (best: 6.7b L15, 4/19 vs null 0.062, p=0.027
against α=0.0019; 6.7b L31 is a *degraded* probe whose error 0.1718 is within
0.3% of a constant predictor's 0.1741).

The metric rewards unreliability, not anticipation — which is why the
uninformative readouts win it.

**The probe is nonetheless reading taint, not depth.** Its per-prefix error is
0.005–0.027 against the position floor's 0.233 — beating it by 10–50×. The
position confound (r = −0.57 between depth and label) explains the *random*
readout's apparent competence, not the probe's.

### What the experiment does show

At most layers `readout_never_wrong` is **19/19 (6.7b)** and **49/49 (1.3b)**:
on every example where the model answers wrongly, the probe decoded the taint
state **correctly at every prefix**. The latent state is right while the output
is wrong.

That converges with **E7**, where patching `sanitizer_def` recovered nothing at
any layer. Two independent methods — a linear probe and activation patching —
agree: **the model represents the sanitization and does not route it to the
answer.** Read together with E10, the pattern across this project is a model
that computes program semantics, uses some of them, and reports none of them.

*(Figures: `leadtime_{model}.png`. Raw: `behavioral_leadtime{,_summary,_prefixes}_{model}.csv`,
`behavioral_sanity_{model}.csv`. `scripts/41_leadtime_floors.py` re-applies the
floors to any stage-40 run without a GPU.)*


## E8: the probes transfer to real code, with the same layer signature

E8 re-runs stages 10+20 unchanged on ~200 `ast`-parseable CodeSearchNet
functions. **The probes transfer.** Binding and def-use are decodable from real
function bodies at ~0.90 accuracy, far above the model-free surface baseline,
and the *shape* of the layer curve matches the synthetic result.

**Use AUC, not accuracy, to read this.** Accuracy is threshold-dependent and
peaks at the embedding layer here; AUC is not, and it tells a different and more
informative story:

| 6.7b, aggregate AUC | Surface | Embedding (−1) | **Peak** | Last layer |
|---|---:|---:|---:|---:|
| binding | 0.673 | 0.962 | **0.978** (L7) | 0.913 (L31) |
| def-use | 0.590 | 0.958 | **0.979** (L3) | 0.907 (L31) |

| 1.3b, aggregate AUC | Surface | Embedding (−1) | **Peak** | Last layer |
|---|---:|---:|---:|---:|
| binding | 0.673 | 0.962 | **0.980** (L3) | 0.907 (L23) |
| def-use | 0.590 | 0.959 | **0.975** (L3) | 0.908 (L23) |

Three things replicate from the synthetic corpus: hidden states beat the surface
baseline by a wide margin (**+0.31 / +0.39 AUC**); AUC **rises above the
embedding layer to an early-middle peak** (L3–L7, the same relative depth as
synthetic); and it **declines toward the output** (−0.07). The rise-then-shed
profile — information built in the early blocks, held, then reorganized for
next-token prediction — is present in real code too, just compressed, because
the embedding layer already starts at 0.96 instead of 0.500.

**Why the embedding layer starts so high — and why that is expected.** In real
code, identifiers are genuinely informative: `self._cache` and `result` are
different variables and *look* different, so token identity alone predicts most
binding pairs. The synthetic `context_matched` design deliberately annihilates
that cue (both floors sit at exactly 0.500) in order to isolate the part the
model must *compute*. Real code cannot do that by construction, so the two
experiments measure different things: **E2 isolates the semantic component; E8
tests whether the whole decoder transfers to naturalistic inputs.** A high
embedding baseline on real code is a property of real code, not a failure of
the model.

**The one stratum that looks bad, and how much weight it carries.**
`same_name_diff_binding` — identically spelled occurrences binding to different
definitions — sits below chance at every layer (6.7b 0.095→0.494; 1.3b
0.082→0.516), while the surface probe scores 0.767. That is worth reporting, but
it is **weaker evidence than it appears**, for the reason already documented for
E4's layer −1 = 1.000: these per-stratum figures are **class-conditional recall
on the negatives, and no per-stratum AUC is recorded**. A probe whose threshold
leans toward "same binding" whenever the spelling is identical will score low
here regardless of what its hidden states encode — which is exactly what the
aggregate AUC of 0.978 says is happening. Distinguishing "the representation is
absent" from "the threshold is placed against this stratum" needs a
context-matched control, which this run does not have.

**What E8 does and does not license.**

- **Supported:** binding and def-use structure is linearly decodable from real
  Python at ~0.90 accuracy / ~0.98 AUC, hugely above surface features, with the
  same layer profile as the synthetic corpus. The findings are not a generator
  artifact. Both scales agree closely, and the model-free surface baseline is
  numerically identical across them (the usual corpus-integrity check).
- **Not supported:** that the *semantic* component specifically transfers. On
  real code the lexical and semantic contributions cannot be separated, because
  no stratum here pins the surface floor to chance. E2's isolation result still
  rests on synthetic programs.
- **The fix is available** (open item 1): context-matched pairs can be built from
  real functions by mutating them — 150 candidate sites already exist in this
  corpus. That would turn E8 into a like-for-like test of E2 rather than a
  transfer check, and is the single highest-value addition to this experiment.

---

## E10: the J-lens transfers to code models — and both experiments come back null

E1–E9 ask whether a relation can be *read out* of the hidden state by a trained
probe. E10 asks the stronger question with an **unsupervised** readout built
from the model's own output head — the Jacobian lens
([Gurnee, Lindsey et al. 2026](https://transformer-circuits.pub/2026/workspace/index.html)),
`v_w = J^T(g·W_U[w])` where `J = E[∂h_final/∂h_ℓ]`. A high score means the state
*disposes the model to say* `w`, which is strictly stronger than being
decodable. Method: `docs/METHODS.md` §11. Design: `docs/JLENS_PLAN.md`.

### The machinery works, and the method transfers (stage 60)

All required gates passed on both models. Two results matter:

| Validation | 1.3b | 6.7b | Reading |
|---|---:|---:|---|
| **V1** J-lens vs logit lens at the last layer | **1.0000** | **1.0000** | `J` is provably the identity there, so this *must* be 1.0 — a closed-form check of the entire gradient path |
| **V2** next-token top-1 (chance 0.038) | 0.633 (L19) | 0.650 (L27) | the lens reads real content |
| V2 random floor | 0.000–0.133 | 0.000–0.050 | |
| **V2 J-lens advantage over logit lens, pre-final layer** | **+0.150** (L−1) | **+0.183** (L19) | the paper's central claim, reproduced in a code model |

The last row is a genuine positive finding: **the Jacobian correction recovers
content the logit lens cannot, in a code LLM** — the first replication of that
claim outside natural-language models that this project is aware of. It also
means the nulls below are *not* "the method doesn't work here".

*(Figures: `jlens_validation_nexttoken_{model}.png`.)*

**One caveat on the gate.** V3 (taint disposition) passed at n=10, which is too
small to carry weight — its cells are all 0.0 or 1.0. The load-bearing
validation is V1 (exact) and V2 (n=60, huge margins), not V3.

### E10-2 (taint): null — and it breaks the early-warning metric itself

This was the priority experiment: E6 found early warning in 6.7b and not 1.3b,
and hypothesized 6.7b's taint state is "distinct from what its output head
does". Running the J-lens, logit-lens, frozen probe, and a **norm-matched
random lens** through E6's own stepping gives:

**6.7b — early-warning rate, P(readout wrong first | model wrong), n=32:**

| Layer | J-lens | logit | probe | **random** |
|---:|---:|---:|---:|---:|
| 0 | 0.906 | 0.000 | 0.000 | **0.000** |
| 3 | 0.594 | 0.000 | 0.125 | **0.812** |
| 7 | 0.000 | 0.000 | 0.656 | **0.812** |
| 11 | 0.188 | 0.000 | 0.219 | **0.719** |
| 15 | 0.781 | 0.812 | 0.500 | **1.000** |
| 19 | 0.000 | 1.000 | 0.500 | **0.906** |
| 23 | 1.000 | 0.188 | 0.281 | **0.719** |
| 31 | 0.562 | 0.562 | 0.281 | **0.750** |
| **mean** | **0.481** | 0.373 | 0.354 | **0.634** |

**A random direction carrying no information produces *more* apparent early
warning than any real readout.** The J-lens beats its own random floor at only
3 of 10 layers. There is no verbalizable-taint signal here.

**Why**, and this is the substantive finding: across all 40 (layer, readout)
cells, early-warning rate is almost perfectly predicted by how *unreliable* the
readout is —

> **Pearson r = −0.905 (p = 1.1×10⁻¹⁵)**, Spearman −0.907, between
> "fraction of examples the readout never got wrong" and early-warning rate.

A readout that is wrong often is wrong *early*, mechanically. The statistic
rewards inaccuracy, not anticipation. `docs/RESULTS.md` already flagged this for
E6's layer −1; the random control shows it is not a layer −1 quirk but the
metric's general behaviour.

**This forces a re-reading of E6 (RQ4).** Two problems, both now measurable:

1. **6.7b.** This run reproduces E6's headline exactly — the probe at layer 7
   fails first on 21/32 (0.656). But the random floor at that same layer is
   **0.812**. E6's number does not clear a control it was never tested against.
2. **1.3b.** E6 reports "no early warning at any layer" as evidence for a
   scale-dependent effect. In fact **the 1.3b model answers wrongly at the very
   first evaluated prefix (t=2) on 100% of test programs**, so a positive lead
   is *arithmetically impossible* for any readout — which is why J-lens, logit,
   probe and random all score exactly 0.000 at every layer. The 1.3b "null" is a
   floor artifact, not a fact about its representations.

So the E6 scale split is not established: one arm is uncontrolled, the other is
structurally forced. This is the most consequential thing E10 produced.

*(Figures: `jlens_taint_earlywarning_{model}.png`.)*

### E10-3 (control dependence): a clean, well-powered null

At each guard-expression anchor, does the lens rank a control-dependent
statement's target variable above an `indent_matched` one's — E4's hard
negative, a statement in a *sibling* guard's body at the same depth? Chance is
exactly 0.5, since the two targets are interchangeable neutral variables.
n = 808 comparisons per cell (SE = 0.018).

| Model | J-lens range across layers | best J-lens cell | logit | random |
|---|---|---|---|---|
| 1.3b | 0.457 – 0.517 | 0.517 (L7) | 0.457–0.507 | 0.479–0.509 |
| 6.7b | 0.387 – 0.510 | 0.510 (L0) | 0.387–0.500 | 0.465–0.511 |

**Not one cell exceeds chance** at Bonferroni-corrected significance in either
model. The best J-lens result in either model (0.517) is inside the 95% CI of
chance (±0.034). Meanwhile E4's *trained probe* reaches AUC 0.999 on this same
relation at mid layers.

That contrast is the point: **control dependence is decodable but shows no
verbalizable signature** — consistent with the E4 reading that it is largely
local syntax the model reconstructs on demand rather than a fact it holds in a
reportable form. The three cells that *are* significant (6.7b L27 logit 0.415,
L31 J-lens and logit 0.387) sit **below** chance, i.e. the deepest layers
systematically prefer the *non*-dependent target — a next-token surface bias at
the readout layers, not evidence of the relation.

*(Figures: `jlens_controldep_indent_matched_{model}.png`.)*

**Honest limit on this null.** It rules out one operationalization, not the
concept: the lens asks "is the model disposed to *name* this variable", and a
model could represent control dependence in a reportable form without being
disposed to emit the dependent statement's target identifier at the guard. E10-2
is less exposed to this objection — there the candidate tokens are the yes/no
answer the model actually emits — which is why E10-2's metric finding is the
more portable result.

---

## What the results say, in one paragraph

On **synthetic** programs, both `deepseek-coder` models linearly encode
**variable binding** and **def-use structure** in a way no surface cue can
explain: on token-identical program pairs the probe rises from a hard 0.500
floor to a ~0.98 mid-layer peak, replicated at both scales at the same relative
depth. That representation is **built in the first few blocks, held through the
middle, and shed near the output**; it is **robust to inert length and
formatting** but **collapses under scope-shadowing interference and control-flow
flattening**, i.e. exactly when the underlying semantic task gets harder. Causal
patching shows the information is **really used** and physically **routes from
the sink-argument token (layers 0–11) to the last token (layers 19–31)**. The
probes **transfer to real Python (E8)** — ~0.90 accuracy / 0.98 AUC on
CodeSearchNet functions against a 0.67 surface baseline, with the same
rise-to-an-early-middle-peak-then-decline layer profile — so none of this is a
generator artifact; what real code cannot do is *isolate* the semantic component,
since natural identifiers are themselves predictive and no stratum there pins the
surface floor to chance. What these representations are **not** is
*verbalizable*: the J-lens (E10) reproduces its own validation benchmarks in
this setting — including recovering next-token content the logit lens cannot —
yet finds **no** workspace signature for either taint or control dependence,
with control dependence flat at chance (0.39–0.52, n=808/cell) at every layer
where the trained probe reaches AUC 0.999. The picture that emerges is a model
that **computes and uses** program semantics without holding them in a
reportable form. Finally, **RQ4 gets a clear negative on the only model that can do the task**:
with a working behavioural signal and three floors, 6.7b's taint probe shows
*no* early warning (excess −0.010) while a no-model position baseline scores
+0.113 on the same metric. 1.3b's forced choice is at chance (balanced accuracy
0.471), so its lead time is undefined rather than negative. What the probe does show is that the taint state is
decoded **correctly** on every prefix of every example where the model answers
wrongly — the representation is right while the output is wrong.

## Open items before the paper

All ten experiments have run at both scales; nothing is blocked on compute.
**Item 0 is new and supersedes the old E6 items** — it is a correctness problem,
not a reporting one.

0. ~~**E6's early-warning claim needs to be withdrawn or rebuilt**~~ **DONE.**
   The prompt was fixed (few-shot + named variable; `scripts/diagnose_taint_prompt.py`
   documents why both are needed), and stage 40 now ships three floors:
   balanced-accuracy sanity on the behavioural signal, a norm-matched random
   readout, a no-model `position` baseline, and an analytic null. Verdict: no
   early warning in either model. `scripts/41_leadtime_floors.py` re-applies the
   floors to any stage-40 run without a GPU. **Remaining (optional):** the taint
   corpus still has depth→label correlation r = −0.57, which is why the *random*
   readout looks competent; the probe beats the position floor by 10–50× so its
   result is unaffected, but a re-generated corpus with randomized initial taint
   state and re-taint transitions would remove the objection entirely.

1. **Context-matched pairs on real code** — the highest-value follow-up: it would
   upgrade E8 from a transfer check to a like-for-like replication of E2's
   isolation result, and would settle whether the low `same_name_diff_binding`
   recall is a threshold artifact or a real gap. Build them by *mutating* real
   functions
   rather than generating programs: given a def of `v` at line *i* and a later
   use at line *k*, rename the target of an interposed assignment at *j*
   (*i*<*j*<*k*) from `w` to `v`. The two sources are token-identical except that
   one token, the anchors and their distance are unchanged, and the (def *i*,
   use *k*) label flips — so the surface baseline is pinned to 0.500 exactly as
   in E2. Requirements: `w` and `v` must tokenize to the same length (single
   token is cleanest); the mutated token must sit outside the ±3-token anchor
   windows; and `v` must be used somewhere in (*i*, *j*] so def *i* stays live in
   the DFG in both variants (the synthetic generator's "early use" line does this
   job). Ground truth is rebuilt per variant, exactly as E5/E9 already do.
   **Measured yield on the current corpus: 150 candidate sites across 61 of the
   200 functions (30%)** before the tokenizer length constraint — comparable to
   the 80 synthetic pairs. `_Renamer` in `src/data/obfuscation.py` is the
   transform to reuse. Caveat: this produces *mutated* real code (the rename can
   break runtime behavior, which is irrelevant for static ground truth but means
   the corpus is no longer pristine CodeSearchNet), and if identifiers have to be
   normalized to single tokens to lift yield, real identifier distribution is
   traded for real structure — worth reporting both arms.
2. **E8 stratum sizes** — `static_probes.csv` records per-stratum *accuracy* but
   not per-stratum *n*. The `same_name_diff_binding` count on real code must be
   measured before the E8 negative result goes in a paper: if that stratum is
   only a handful of pairs, the claim weakens from "fails" to "underpowered".
   A builder-side count on `csn_python_200.jsonl`, no GPU needed.
3. **E6 layer-7 robustness** — the headline (21/32 early, +2.3 steps) rests on a
   single calibration split (`calib_frac=0.3`, seed 42) and one threshold per
   layer. Re-run across a few seeds to get a CI on the early-warning *rate*, not
   just on `mean_lead`.
4. **E6 reporting** — prefer the `latent first / model wrong` rate over the
   shipped `frac_positive_lead`+`mean_lead` columns, which condition on
   `n_both_fail` and so change denominator across layers. Consider adding that
   ratio to the summary CSV in `src/experiments/behavioral_leadtime.py`.
5. Report E5/E9 at **peak/per-layer** rather than layer-averaged — the averages
   hide the strongest findings (e.g. rename fools layer 0 but not layer 11).
6. **E4 optional polish** — re-anchor on the guard variable / statement target
   instead of the span's trailing literal (a CPU-only stage-20 re-run).
7. **E1 lexical AUC is logged as 0.000** — a multi-class reporting artifact, not
   a fit failure (all lexical fits now report `converged = True`). Worth emitting
   `NaN` instead so the column isn't misread.
8. **Provenance note for E6:** the layer sweep ran under the `uq` env
   (Python 3.10 / sklearn 1.7.2) against probe checkpoints pickled by the older
   `semflow` env (Python 3.11 / sklearn 1.9.0), via `micromamba run -n semflow`.
   E7's patching numbers predate that move. Same data, same seeds, but if a
   reviewer asks for exact reproducibility both stages should be re-run in one
   environment.
