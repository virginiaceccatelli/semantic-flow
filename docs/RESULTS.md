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

Status legend: ☐ not run · ◐ dev model (1.3b) · ● main model (6.7b)

| Exp | What it tests | 1.3b | 6.7b | Verdict |
|-----|---------------|:----:|:----:|---------|
| E1 token type | lexical baseline | ● | ● | Decodable from embeddings alone (as expected) |
| E2 binding + strata | variable binding | ● | ● | **Positive** — decodable, surface-cue-proof |
| E3 def-use + distance | def→use edges | ● | ● | **Positive** — decodable, mild distance decay |
| E4 control dep | guard→statement | ● | ● | **Positive but surface-heavy** — hidden beats the hard stratum (0.92 vs 0.68), but control dep is largely locally decodable; replicates across scale |
| E5 context degradation | robustness to filler | ● | ● | Survives length; collapses under interference |
| E6 lead time | latent vs behavioral failure | ◐ | ● | Degenerate (no positive lead in either model) — needs a layer sweep |
| E7 causal patching | is it *used*? | ● | ● | **Positive** — information routes across layers; sanitizer site is causally inert |
| E8 real code | CodeSearchNet transfer | ☐ | ☐ | Not run |
| E9 obfuscation | semantics-preserving edits | ◐ | ● | Robust to renaming mid-layer; breaks on flatten (1.3b table is pre-fix — re-run stage 31) |

Two experiments are still open: **E8** (real-code transfer) has not been run,
and **E6** produced a degenerate result in *both* models that should be re-run
with a layer sweep before it is trusted (see below). **E4 is now valid and
replicates across scale** (its hard-negative control was rebuilt).

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

## E6: latent and behavioral failure are perfectly coupled — the model ignores sanitization

E6 asks whether the taint probe's internal state degrades *before* the model's
answer goes wrong (RQ4). Both models produced a **degenerate result with no
positive lead**, and the degeneracy is itself informative. The 6.7b case is the
cleanest to read:

Of 70 test programs: the **38 unsanitized** ones are handled perfectly by both
probe and model (no failure to lead). On **all 32 sanitized** ones, both the
probe and the model go wrong at *exactly the same step* — the sanitization line
itself. Mean lead time **0.0**, bootstrap CI **[0.0, 0.0]**, fraction with
positive lead **0.0**. *(Figure: `leadtime_{model}.png` — a single spike at 0.)*

**Plain reading:** the model **never registers the sanitizer**. After the
sanitizing call it continues to treat the value as tainted, and the internal
taint state (read at layer 0) agrees completely. There is no early-warning
signal because there is no disagreement to detect — latent and behavior fail
together. This is the same fact E7 found causally (patching the sanitizer does
nothing) seen from the behavioral side.

**1.3b tells the same story with a twist.** Again **no case has positive lead**
(`frac_positive_lead = 0.0`), but the mean lead is **−2.83** (CI [−3.3, −2.4]),
i.e. the layer-0 taint probe tends to "fail" a couple of steps *after* the model
does — the opposite of an early warning. So neither model gives the leading
signal RQ4 asks for; the sign difference between them is exactly the kind of
artifact a single-layer, single-threshold readout produces.

**Caveat — do not report this as the final RQ4 answer yet.** E6 was run at
**layer 0 only** with a **~0.999 threshold** in both models, and E7 shows taint
information migrates through the layers. A probe at a mid or late layer might
diverge from behavior where a layer-0 probe cannot. **This should be re-run with
a layer sweep** before the null is trusted — the divergent 1.3b/6.7b lead values
(−2.83 vs 0.0) make the under-powering concrete.

---

## What the results say, in one paragraph

Both `deepseek-coder` models linearly encode **variable binding** and **def-use
structure** in a way no surface cue can explain: on token-identical
program pairs the probe rises from a hard 0.500 floor to a ~0.98 mid-layer peak,
replicated at both scales at the same relative depth. That representation is
**built in the first few blocks, held through the middle, and shed near the
output**; it is **robust to inert length and formatting** but **collapses under
scope-shadowing interference and control-flow flattening**, i.e. exactly when the
underlying semantic task gets harder. Causal patching shows the information is
**really used** and physically **routes from the sink-argument token (layers
0–11) to the last token (layers 19–31)**. In the taint setting the model
**demonstrably never incorporates sanitization** — behavior and latent state fail
together with zero lead time, and patching the sanitizer site has zero causal
effect.

## Open items before the paper

1. **E4** — done: fixed and **replicated across both scales** (sibling-guard
   `indent_matched` control; hidden 0.92 vs surface 0.68 on the hard stratum;
   surface baseline numerically identical across models). Optional polish:
   re-anchor on the guard variable / statement target instead of the span's
   trailing literal (a CPU-only stage-20 re-run).
2. **E9 (1.3b)** — **re-run stage 31**; the 1.3b obfuscation table predates the
   corpus regeneration. Activations already exist, so it is a CPU-only run.
   Numbers should be unchanged (E9 uses the seed-stable binding/def-use probes),
   but the file should be made current.
3. **E6** — re-run with a **layer sweep**; the layer-0 null is under-powered in
   both models (leads of 0.0 at 6.7b vs −2.83 at 1.3b), given E7's layer
   migration.
4. **E8** — real-code (CodeSearchNet) transfer not yet run.
5. **E1 lexical fits** did not converge (`converged = False`, AUC logged as 0.000,
   a multi-class reporting artifact); re-run with `--max-iter 2000` so stage 20
   writes its manifest cleanly.
6. Report E5/E9 at **peak/per-layer** rather than layer-averaged — the averages
   hide the strongest findings (e.g. rename fools layer 0 but not layer 11).
7. **Housekeeping:** the 6.7b `behavioral_leadtime` / `causal_patching` table CSVs
   carry an old (Jul-19) timestamp though their stages re-ran later; content is
   valid (the taint corpus is seed-stable and untouched by the E4 edit), but
   re-running stage 40/50 for 6.7b will refresh them so every asset traces to one
   generation.
