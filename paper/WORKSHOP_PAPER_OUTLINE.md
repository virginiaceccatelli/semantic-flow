# Six-page workshop paper outline

## Working title

**Semantic Flow Under Obfuscation: What Code Models Represent, What Survives, and What Becomes Sayable**

More security-forward alternative:

**Code Models Represent Security-Relevant Data Flow, but Control-Flow Flattening Breaks the Readout**

Prefer the first title if submitting to a code-model or interpretability venue. Prefer the second for a security or trustworthy-AI venue.

## The paper in one sentence

Code models make security-relevant source-to-sink flow linearly explicit beyond both local and whole-program lexical baselines, but this access depends on the source-level control-flow scaffold: control-flow flattening alone causes the full robustness failure across three models, while the surviving distinction is distributed across output coordinates rather than aligned with an explicit word such as *unsafe*.

## The three questions

The paper should answer only these questions:

1. **Presence:** Do code models represent security-relevant semantic flow beyond textual shortcuts?
2. **Robustness:** Which semantics-preserving transformation actually breaks access to it?
3. **Format:** Is the distinction expressed as a recognizable security concept in the model's own vocabulary space?

Everything in the main paper must answer one of these. If it does not, omit it or move it to the appendix.

## Strongest findings, in order

### Finding 1: security-relevant flow is constructed contextually

- E15 asks whether the value at a code-bearing sensitive argument is derived from untrusted input.
- On clean training programs:
  - local ±3-token surface reader: 0.488--0.491;
  - whole-program lexical reader: 0.464;
  - embedding layer: 0.482;
  - contextual hidden states: 1.000 / 1.000 / 0.997 across DeepSeek-Coder 1.3B, DeepSeek-Coder 6.7B, and StarCoder2-3B.
- On held-out clean programs the frozen hidden-state readout reaches 1.000 in all three models.
- The layerwise rise reproduces the binding/def--use profile: chance at input, rapid construction in the first quarter, ceiling near half depth.

**Claim:** contextual computation makes source-to-sensitive-sink flow linearly accessible; neither anchor identity nor whole-program lexical regularities explain it under the measured baselines.

**Do not claim:** the floor excludes a program analyzer that actually computes taint flow, or that high probe accuracy proves causal use.

### Finding 2: flattening alone accounts for the robustness failure

- Atomic opaque predicates and MBA encoding cost exactly 0.000 in all three models.
- Atomic renaming costs 0.014--0.118.
- Atomic control-flow flattening costs 0.312--0.340:
  - 1.3B: 1.000 → 0.688;
  - 6.7B: 1.000 → 0.667;
  - StarCoder2-3B: 1.000 → 0.660.
- The full four-transformation composition reaches 0.729 / 0.653 / 0.674.
- Cumulative-minus-atomic flattening interaction is +0.042 / -0.014 / +0.014, no larger than the independently measured draw-noise floor.

**Claim:** the failure is attributable to control-flow flattening, not generic code obfuscation or an interaction among transformations.

This is the paper's most important result. It is stronger than a cumulative ladder because it identifies the responsible structural change.

### Finding 3: residual accuracy is not retained semantic flow

- Under `flatten_only`, 46--56% of matched safe/unsafe pairs receive the same label.
- Under the full composition, model biases point in opposite directions:
  - 6.7B shifts toward unsafe;
  - StarCoder2 shifts toward safe.
- A constant class predictor scores 0.500 on the balanced dataset.

**Claim:** pooled post-flattening accuracy overstates retained information; matched-pair collapse and class bias show that much of the residual score is a prior rather than usable flow discrimination.

This is the operational consequence. Report pair collapse and unsafe/safe rates beside every headline robustness number.

### Finding 4: the boundary is general to program relations, with a security consequence

- E9 applies the same transformations to binding and def--use across all three models.
- At each model's best layer:
  - binding falls to 0.527--0.615 under flattening;
  - def--use falls to 0.402--0.545;
  - E15 source-to-sink flow falls to 0.660--0.688.
- The security readout is not uniquely fragile; it is at least as robust as the semantic primitives beneath it.

**Claim:** flattening disrupts frozen linear access to source-level program relations generally, including the relation required by a security audit.

This companion result prevents an inflated “security representations are uniquely fragile” claim and gives a deeper explanation: the failure tracks a change in structural presentation.

### Finding 5: the distinction is distributed, not named

There are three separate results; present them as a hierarchy rather than as three experiments.

1. **Explicit security words do not carry the unprompted distinction.** At the sink site, probability mass on tokens such as `unsafe`, `tainted`, or `vulnerable` does not consistently rise in unsafe programs. It is significantly inverted in 1.3B. Logit lens, J-lens, and R-lens agree, so the result is not specific to one readout.
2. **A distributed full-vocabulary direction does generalize.** A direction estimated on clean training-pair vocabulary differences is projected onto positively by all 72 held-out pairs in all three models from roughly 25% depth onward. The same-label projection is near chance, and the token-identical `last_token` site removes a direct anchor-token account.
3. **The direction has no lexical interpretation.** Its loadings are flat and spread over thousands of arbitrary fragments; it does not dominate the overall difference geometry relative to same-label concentration controls.

**Claim:** the source-to-sink distinction is linearly accessible and has a reproducible distributed projection into output coordinates, but it is not localized on an explicit security word in the unprompted state.

**Important qualification:** the full-vocabulary direction is label-defined on the training split and is supporting evidence about representational format, not an independently causal feature. Do not describe it as “the unsafe feature.”

## Results that should not carry the six-page paper

- **E15-D relevance redistribution:** one model only, small effect, sign test positive but mean permutation test null. Appendix or future work.
- **R-lens engineering and rule ablations:** useful instrument validation, not a code-semantics contribution. State that the genuine R-lens is validated on both DeepSeek models; put details in the appendix.
- **StarCoder2 R-lens:** not applicable because the homogenizing rules do not bind to its LayerNorm/non-gated-MLP architecture. Its logit- and J-lens runs are complete and sufficient for the three-model vocabulary comparison.
- **E13 causal binding interchange:** strong but a different paper arc. Mention once as evidence that the project distinguishes decodability from causal use; do not allocate a results section.
- **Context-length degradation, natural-code transfer, control dependence, archived intervention attempts:** omit from the main text.
- **Per-sink-family nulls and every layer table:** appendix only.

## Status of the lens runs

- DeepSeek-Coder 1.3B: logit lens, J-lens, R-lens, vocabulary contrast, positive control, and relevance redistribution complete.
- DeepSeek-Coder 6.7B: logit lens, J-lens, R-lens, vocabulary contrast, and positive control complete. R-lens relevance redistribution over AST roles has not run.
- StarCoder2-3B: logit lens, J-lens, vocabulary contrast, and positive control complete. A genuine R-lens/relevance run is not applicable to this architecture under the implemented rules.

The missing 6.7B relevance run does not block the proposed paper because relevance redistribution is excluded from the main argument.

---

# Page allocation

Six pages excluding references and appendix:

- Abstract: 0.25 page.
- 1. Introduction: 0.70 page.
- 2. Controlled benchmark and method: 1.05 pages.
- 3. Semantic flow is constructed with depth: 0.75 page.
- 4. Flattening breaks the representation: 1.45 pages.
- 5. Decodable does not mean lexicalized: 0.90 page.
- 6. Discussion, limitations, conclusion: 0.90 page.

Use three figures and two small tables. Do not add a fourth main figure.

---

# Abstract

Write one compact argument, not an experiment inventory.

1. **Problem.** Code-model representations may support program analysis on ordinary source while failing under semantics-preserving structural rewrites.
2. **Identification.** Introduce matched Python programs differing only in whether the sink argument is source-derived, with independent static/dynamic labels and local plus whole-program textual baselines.
3. **Presence result.** Hidden-state decoding reaches 0.997--1.000 across three models while surface and embedding controls remain at chance.
4. **Atomic robustness result.** Opaque predicates and MBA encoding independently cost nothing; renaming is modest; flattening alone produces the entire 0.31--0.34 collapse, with no detectable cumulative interaction.
5. **Failure interpretation.** Roughly half of flattened counterfactual pairs receive the same prediction and residual class biases reverse across models.
6. **Format result.** The distinction is not aligned with explicit security words, although a training-defined distributed full-vocabulary direction generalizes to every held-out pair.
7. **Conclusion.** Code models construct security-relevant semantic flow, but access to it depends on source-level control-flow presentation and is more distributed than a verbalized “unsafe” concept.

Do not mention E13, the R-lens rule ablation, relevance redistribution, or every sink family in the abstract.

---

# 1. Introduction

## Paragraph 1: the scientific question

- Program-analysis tasks require relations that are not local token properties: binding, def--use, and whether untrusted data reaches a sensitive call.
- Code models may contain these relations, or a fitted readout may exploit names and generator regularities retained in hidden states.
- Even a genuine clean-code relation may be represented in a form tied to source-level syntax rather than normalized program semantics.

## Paragraph 2: the robustness question

- Semantics-preserving obfuscations provide controlled interventions on presentation.
- Existing end-to-end studies ask whether model judgments change under obfuscation; this paper asks what happens to the internal semantic relation itself.
- The atomic/cumulative distinction is essential: a cumulative ladder can reveal failure but cannot identify which rewrite caused it.

## Paragraph 3: approach

- Construct matched safe/unsafe source-to-sink pairs.
- Train a frozen low-capacity readout only on clean programs.
- Compare it with local surface, whole-program lexical, and embedding controls.
- Apply the same four transformations both atomically and cumulatively.
- Use pair collapse, per-class rates, and companion binding/def--use runs to interpret the failure.
- Finally inspect whether the relation aligns with the model's own vocabulary directions.

## Paragraph 4: contributions

Use exactly three contributions:

1. A controlled benchmark showing contextual construction of source-to-sensitive-sink flow beyond two measured textual floors in three code models.
2. Atomic attribution showing control-flow flattening alone accounts for the entire frozen-readout collapse, whereas opaque predicates and MBA encoding cost exactly zero.
3. A representational-format result: the distinction has a reproducible distributed output-space direction but is not aligned with explicit security vocabulary.

End with the central interpretation:

> The models do not merely memorize dangerous API names, but neither do they compute a fully normalized analysis independent of how control flow is presented.

---

# 2. Controlled Benchmark and Method

## 2.1 What “semantic” means here

Give the binding minimal pair in one small inset:

```python
x = 3                  x = 3
def f():               def f():
    y = 7                  x = 7
    return x               return x
```

- The binding label flips while the use token, local windows, and distance remain fixed.
- Binding and def--use begin at an exact 0.500 declared floor and rise toward 0.99 in contextual layers.
- This is the controlled premise, not a separate results section: code models can construct relational program information beyond local form.

One paragraph is enough. Put complete E2/E3 curves in the appendix.

## 2.2 E15 source-to-sink benchmark

Define the target precisely:

> Is the value passed to this code-bearing sensitive argument derived from untrusted input?

Dataset:

```text
3 sink families × 4 flow structures × 20 bases × 2 labels = 480 programs
```

- Families: command execution, SQL execution, dynamic-code execution.
- Structures: direct, two-step assignment chain, branch merge, one helper boundary.
- Every base contains matched unsafe/safe members with the same source, trusted alternative, propagation, and sink; only the sink argument differs.
- Safe means the sink receives an independently trusted literal, not that a generic sanitizer makes arbitrary code execution safe.
- Train: 168 bases / 336 programs. Held out: 72 bases / 144 programs.

Show one compact matched example. The code example should make the only changed span visually obvious.

## 2.3 Ground truth and controls

Keep this dense:

- Labels must agree under restricted instrumented execution and an independent AST taint fixpoint.
- Dangerous APIs are recorder stubs; no sink executes.
- Pair members and all variants remain grouped by base.
- Frozen readout: fit once on clean train, never refit after transformation.
- Controls:
  - local ±3-token IDs;
  - whole-program token/character n-grams;
  - embedding layer;
  - grouped selectivity control.
- Cluster bootstrap over the 72 held-out bases.

Mention that every stage has hard integrity gates, but do not print the gate catalogue in the main paper.

## 2.4 Transformations

Use one table:

| Transformation | What changes | Atomic question |
|---|---|---|
| rename | identifier spelling | is the relation lexical? |
| opaque predicate | unreachable control/data decoys | do dead alternatives interfere? |
| MBA encode | expression form | is arithmetic surface form load-bearing? |
| flatten | structured control flow → dispatch loop | is source-level control structure load-bearing? |

- Apply each alone and in the fixed cumulative prefix.
- Validate every variant's transformation set and security label.
- The paper does not test all 15 transformation combinations; do not imply full factorial interaction coverage.

---

# 3. Semantic Flow Is Constructed With Depth

## Main result

Present one normalized-depth plot across all three models.

| Readout on clean training programs | 1.3B | 6.7B | StarCoder2-3B |
|---|---:|---:|---:|
| local surface | 0.491 | 0.491 | 0.488 |
| whole-program lexical | 0.464 | 0.464 | 0.464 |
| embedding | 0.482 | 0.482 | 0.482 |
| ~25% depth | 0.991 | 0.979 | 0.952 |
| ~50% depth | 1.000 | 1.000 | 0.997 |

Explain the curve in three sentences:

1. The target is not accessible to the measured textual readers or anchor token identity.
2. It becomes explicit rapidly after contextual computation begins.
3. The same shape appears across scale and model family and mirrors binding/def--use.

Then state the scope carefully:

> This establishes accessible information under named controls, not a claim that the model executes a sound taint analysis or causally uses the decoded direction.

## Figure 1

**Layerwise construction of semantic flow.** Three model curves over normalized depth, with horizontal bands/markers for local surface, whole-program lexical, and embedding controls. A small binding-pair inset can share the figure if legible.

---

# 4. Flattening Breaks the Representation

This is the paper's largest section.

## 4.1 Atomic attribution

Use the main table or bar plot:

| Held-out condition | 1.3B | 6.7B | StarCoder2-3B |
|---|---:|---:|---:|
| clean | 1.000 | 1.000 | 1.000 |
| rename only | 0.938 | 0.986 | 0.882 |
| opaque only | 1.000 | 1.000 | 1.000 |
| MBA only | 1.000 | 1.000 | 1.000 |
| flatten only | 0.688 | 0.667 | 0.660 |
| full cumulative | 0.729 | 0.653 | 0.674 |

Lead with the contrast, not the table mechanics:

- Two transformations that look adversarial are free in isolation.
- Renaming is mostly survivable, so the relation is not tied to original names.
- Flattening alone reproduces the full failure in all three models.
- The cumulative interaction is within independent-draw noise, so composition adds no detectable effect.

This licenses the sharp sentence:

> The readout is robust to substantial surface and expression changes but depends on the structured control-flow scaffold of source code.

## 4.2 Why 0.66 does not mean two-thirds of the relation survives

Report:

- unsafe/safe accuracy;
- predicted-unsafe rate;
- matched-pair same-label fraction.

Explain:

- Balanced pooled accuracy cannot distinguish retained relational information from a shifted class prior.
- At flattening, roughly half the matched pairs lose their distinction entirely.
- Opposite class biases across 6.7B and StarCoder2 show that similar aggregate accuracies arise through different failures.

This is methodologically important and easy to understand. Give one concrete example: the same safe/unsafe pair receives one label after flattening.

## 4.3 The failure follows structure, not security vocabulary

- Assignment chains are most fragile under renaming across all three models.
- No sink-family ordering reproduces.
- E9's binding and def--use readouts show the same flattening boundary.

Phrase the companion result positively:

> Source-to-sink flow is not a special brittle classifier. Its failure tracks the same structural boundary as the binding and def--use relations from which a flow analysis is built.

Do not include the entire E9 table. Use a three-row comparison at flattening or a small inset.

## Figure 2

**Atomic transformations identify the failure.** Grouped bars for clean and four atomic conditions across three models, plus a marker for the full cumulative condition. Add the draw-noise interval to make the null interaction visually explicit.

## Table 1

**When aggregate accuracy lies.** For `flatten_only`, show pooled accuracy, unsafe/safe accuracy, predicted-unsafe rate, and pair-collapse fraction across three models.

---

# 5. Decodable Does Not Mean Lexicalized

This section should be short and conceptual. Avoid turning it into a tour of lens implementations.

## 5.1 Ask a simple question

> Does the safe-to-unsafe change point toward the model's own words for the concept?

At the same sink-site states:

- Compare matched unsafe-minus-safe vocabulary scores.
- Use logit lens, J-lens, and R-lens where architecture-valid.
- Select candidate tokens on clean training pairs and freeze before held-out evaluation.
- Include same-label, permuted-orientation, embedding, and token-identical-site controls.

State instrument status in one sentence:

> All three readouts agree on the DeepSeek models; logit and validated J-lens results provide the three-model comparison, while a genuine R-lens is architecture-valid only for the two DeepSeek models.

## 5.2 The result is a dissociation

Present three levels:

| Question | Result |
|---|---|
| Is flow linearly decodable? | yes, near-perfectly |
| Does an explicit security word carry it unprompted? | no; sometimes inverted |
| Does a distributed output-space direction generalize? | yes, 72/72 held-out pairs in all three models |

Explain without mysticism:

- The relation is not a single “unsafe” vocabulary feature.
- A training-defined full-vocabulary difference direction nevertheless generalizes from about 25% depth onward.
- The direction's strongest token loadings are arbitrary fragments and it does not dominate same-label variation.
- Therefore the appropriate claim is distributed output alignment without lexical localization.

The most memorable sentence is:

> The models encode the distinction before they can name it, and they never name it with the obvious security word.

Immediately qualify “encode” as linear accessibility and “name” as alignment with explicit output tokens.

## 5.3 Connection to obfuscation

- The distributed projection weakens sharply under flattening in all three models.
- This gives convergent evidence with a different readout: both supervised accessibility and output-space organization depend on the same structural scaffold.
- Do not claim causal use; neither analysis intervenes on the security-flow state.

## Figure 3

**Accessible, distributed, not lexicalized.** A three-part compact figure:

1. hidden-state probe accuracy by depth;
2. held-out projection on the frozen full-vocabulary direction by depth;
3. explicit security-token contrast centered around zero or inverted.

Use one model for detailed depth curves and small replication markers for the other two. Do not plot every lens separately if their conclusions coincide.

---

# 6. Discussion, Limitations, and Conclusion

## 6.1 What the combined evidence says

Answer the three questions directly:

- **Presence:** yes, the security-relevant relation becomes linearly explicit beyond measured textual floors.
- **Robustness:** selective, not generic; names, dead branches, and MBA form are mostly cheap, while flattening is expensive.
- **Format:** distributed rather than aligned with an explicit security word.

Scientific interpretation:

> Code models form useful relational states over source-level program structure, but those states are not equivalent to a compiler-normalized semantic representation. They remain coupled to the control-flow presentation on which pretraining exposed the computation.

## 6.2 Why this matters

- A clean-code probe can substantially overstate robustness.
- “Semantics-preserving” is too coarse an evaluation category; transformations should be tested atomically.
- Pooled accuracy is insufficient for paired semantic properties.
- Program-analysis readouts should be evaluated across structural normal forms, especially flattened control flow.
- The representation being distributed and non-lexicalized explains why prompting for an explicit safety word is a different capability from internally tracking flow.

## 6.3 Limitations

Keep to six bullets:

1. Synthetic Python, three sink families, four flow structures; no natural malware or vulnerability corpus.
2. Safe means a trusted literal reaches the sensitive argument; the benchmark does not model sink-specific sanitization.
3. The whole-program lexical baseline excludes textual shortcuts, not an analyzer that computes flow.
4. Frozen linear failure may reflect a changed basis, not total absence of semantic information.
5. Full-vocabulary alignment is supervised by the training labels and observational; it is not a causal mechanism.
6. No causal intervention tests source-to-sink flow, and relevance redistribution currently has only a one-model preliminary result.

## Conclusion

End with three sentences:

1. Across three code models, source-to-sensitive-sink flow is constructed contextually and becomes almost perfectly accessible beyond local, global lexical, and embedding controls.
2. Atomic evaluation identifies control-flow flattening as the transformation that breaks this access, while opaque predicates and expression encoding leave it intact and composition adds no detectable interaction.
3. The distinction occupies a reproducible but distributed output-space direction rather than an explicit security word, revealing a representation that is semantic enough to track flow yet structurally tied to how source code presents the computation.

---

# Main-text assets

## Figure 1: construction with depth

- E15 accuracy over normalized depth for three models.
- Local surface, whole-program lexical, and embedding controls.
- Small binding minimal-pair inset if space permits.

## Figure 2: atomic attribution

- Clean plus four atomic transformations across three models.
- Full cumulative result as an overlaid marker.
- Draw-noise band for interpreting interaction.

## Figure 3: representational format

- Probe, distributed full-vocabulary projection, and explicit security-token contrast.
- Show the onset difference and the flattening degradation without plotting every implementation diagnostic.

## Table 1: benchmark and controls

- Dataset crossing, pair construction, train/held-out counts, label readings, and four readout arms.

## Table 2: post-flattening failure anatomy

- Pooled accuracy, unsafe/safe rates, predicted-unsafe rate, and pair collapse.

---

# Appendix only

- Full per-layer, per-condition, per-family, and per-structure E15 tables.
- All bootstrap intervals and draw-noise calculations.
- Complete E2/E3/E9 curves.
- J-lens and R-lens validation, conservation, architecture applicability, and candidate-pool construction.
- E15-C token tables and positive-control prompts.
- Full-vocabulary loadings and same-label concentration tests.
- Preliminary 1.3B relevance redistribution.
- E13 causal binding result, if desired as separate supporting context.
- Gate definitions, provenance hashes, manifests, and failure/refusal behavior.

## Final editorial rule

The paper should never make readers remember experiment IDs. In prose, call them:

- controlled binding/def--use foundation;
- source-to-sink benchmark;
- atomic obfuscation audit;
- vocabulary-format analysis.

The clean narrative is **relation → structural boundary → representational format**. Do not interrupt it with experiment chronology.
