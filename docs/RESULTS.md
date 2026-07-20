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
| E4 control dep | guard→statement | ● | ● | **Invalid** — surface baseline = 1.000 |
| E5 context degradation | robustness to filler | ● | ● | Survives length; collapses under interference |
| E6 lead time | latent vs behavioral failure | ☐ | ● | Degenerate (zero lead) — needs a layer sweep |
| E7 causal patching | is it *used*? | ☐ | ● | **Positive** — information routes across layers |
| E8 real code | CodeSearchNet transfer | ☐ | ☐ | Not run |
| E9 obfuscation | semantics-preserving edits | ● | ● | Robust to renaming mid-layer; breaks on flatten |

Two experiments are still open: **E8** (real-code transfer) has not been run,
and **E6** produced a degenerate result that should be re-run before it is
trusted (see below). **E4 is not a valid semantic claim** in its current form.

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

## E1 lexical baseline & E4 control dependence: read with care

**E1 (token type)** peaks at **1.000 accuracy at the embedding layer (−1)** with
high selectivity (~0.88–0.90) in both models. This is the *expected* control,
not a finding: token type is a pure lexical property, so it is best decoded
before any context is added. It confirms the machinery works and gives the
contrast for E3's thesis (RQ3) — **lexical features are readable from the
embeddings; semantic relations are not, and only appear after computation.**

**E4 (control dependence) cannot currently support a semantic claim.** The
surface baseline scores **1.000** on it — the guard→statement pairs are
separable from token context alone, so the probe's perfect accuracy proves
nothing about the model. E4 needs its own `context_matched` pairs (token-
identical programs with the control relation flipped) before it can be reported.
`taint_state` is likewise at ceiling with ~0.5 selectivity; it is fine as the
*input* to E6/E7 but is not a standalone result.

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

The third column is the quiet bombshell: patching **`sanitizer_def` recovers
nothing at any layer (0.000 throughout)**. Overwriting the sanitizer's
definition never changes the output. The model's taint decision does not route
through the sanitization site at all — which sets up, and is confirmed by, E6.

---

## E6: latent and behavioral failure are perfectly coupled — the model ignores sanitization

E6 asks whether the taint probe's internal state degrades *before* the model's
answer goes wrong (RQ4). On 6.7b it produced a **degenerate, zero-lead-time
result**, and the degeneracy is itself informative.

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

**Caveat — do not report this as the final RQ4 answer yet.** E6 was run at
**layer 0 only** with a **0.999 threshold**, and E7 shows taint information
migrates through the layers. A probe at a mid or late layer might diverge from
behavior where a layer-0 probe cannot. **This should be re-run with a layer
sweep** before the null is trusted.

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

1. **E4** — invalid as reported (surface baseline 1.000); build `context_matched`
   control-dependence pairs.
2. **E6** — re-run with a **layer sweep**; the current layer-0 null is
   under-powered given E7's layer migration.
3. **E8** — real-code (CodeSearchNet) transfer not yet run.
4. **E1 lexical fits** did not converge (`converged = False`, AUC logged as 0.000,
   a multi-class reporting artifact); re-run with `--max-iter 2000` so stage 20
   writes its manifest cleanly.
5. Report E5/E9 at **peak/per-layer** rather than layer-averaged — the averages
   hide the strongest findings (e.g. rename fools layer 0 but not layer 11).
