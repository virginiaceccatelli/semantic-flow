# Workshop paper outline: semantic representation along three dimensions

## Working title

**When Is a Code Representation Semantic? Representation, Robustness, and Causal Use in Code Language Models**

Alternative, slightly more result-led title:

**Code Models Build Scope-Sensitive Representations, but Their Causal Use Remains Open**

The first title is preferable for an Interpretability as a Science workshop. It makes the paper's organizing framework visible, while the second foregrounds the asymmetry in the evidence.

## One-sentence paper claim

- DeepSeek-Coder internally computes scope-sensitive binding and def--use relations that cannot be recovered from the controlled surface form, and these relations are robust to distance and renaming but fragile under genuine scope and control-flow interference; whether the downstream model computation causally uses the binding relation itself remains open under the strongest available intervention.

## Claim discipline

- Use **representation**, **robustness**, and **causal use** as three increasingly strong claims requiring different evidence.
- State the evidential status immediately and repeat it in the conclusion:
  - representation: established;
  - robustness: established within the tested transformations;
  - causal use: not established.
- Do not use “understanding” except when explaining what is not claimed.
- Do not describe E11 as a positive causal result. Its preregistered verdict is NO-GO and its use-position null was retracted after the dose-response control.
- Describe E13 as the strongest causal design and report only gates that are valid. As of the repository's 2026-08-12 record, H0--H3 pass on the 6.7B model and H4--H5 require a corrected rerun. The result is therefore a validated, testable causal question rather than a positive causal finding.
- Where the repository disagrees with itself, follow the newer `docs/RESULTS.md` and the three supplied notes: `results/STATUS.yaml` still contains the older “implemented, not run” description of E13.

## Five-page allocation

- Abstract: about 180--220 words; normally outside or only minimally affecting the five-page budget, depending on the final workshop template.
- 1. Introduction: 0.65 page.
- 2. Framework and experimental setup: 0.75 page.
- 3. Representation: 1.10 pages.
- 4. Robustness: 0.95 page.
- 5. Causal use: 1.05 pages.
- 6. Discussion, limitations, and conclusion: 0.50 page.
- References: follow the workshop rule; if references count toward the limit, compress Section 2 and move all secondary results to the appendix.
- Main-text visual budget: two figures and one compact status/results table. Do not spend main-text space on separate figures for control dependence, real-code transfer, raw activation patching, J-lens validation, or retired experiments.

---

# Abstract

Write this as one compact argument, not as a catalogue of experiments.

- Open with the identification problem:
  - code-model behavior and probe accuracy can be explained by identifier identity, token distance, indentation, and generator regularities;
  - consequently, decodability alone does not show that a model computed a program relation.
- State the paper's three-dimensional framework in one sentence:
  - ask whether a relation is represented beyond the surface form, whether that representation survives meaning-preserving changes, and whether the model's downstream computation causally uses it.
- Introduce the central construction:
  - paired Python programs are token-identical except for one character that changes which definition a use resolves to;
  - the probing sites, their local windows, and their distance are unchanged while the label flips;
  - under the measured surface feature family, the baseline and the embedding-layer probe are exactly 0.500 by construction.
- State the representation result with the most useful numbers:
  - in DeepSeek-Coder 1.3B and 6.7B, binding rises from 0.500 at the input to 0.984 in middle layers and declines to 0.930/0.914 at the final layer;
  - def--use has the same layerwise profile and remains approximately 0.96--0.99 for pairs 50--200 tokens apart.
- State the robustness result as a contrast rather than a list:
  - at 500 inserted tokens, inert prose retains 0.921 binding accuracy, whereas scope-shadowing context reduces it to 0.570 and reaches chance at 1,000 tokens;
  - consistent renaming misleads early layers but preserves 0.85--0.90 middle-layer decoding; control-flow flattening reduces the best layer to about 0.750.
- State the causal result conservatively:
  - a factorial binding-interchange experiment removes token transport, arithmetic, and fixed answer-direction explanations;
  - ground truth, behavioral competence, use-site decoding, and whole-state interchange pass (H0--H3), but the low-rank intervention and its held-out falsification are not yet valid (H4--H5);
  - therefore causal use remains open.
- Close with the scientific conclusion:
  - the results establish a scope-sensitive but structurally constrained internal representation;
  - they also show why causal conclusions require matched positive controls, matched intervention magnitude, and explicit alternative hypotheses.
- Do not mention every failed experiment in the abstract. “Causal use remains open under the strongest controlled test” is enough.

---

# 1. Introduction

## Paragraph 1: the scientific problem

- Begin from the ambiguity of apparent code understanding:
  - equal names usually denote the same variable;
  - definitions and uses tend to occur near each other;
  - indentation often exposes control structure;
  - a representation containing the source text will therefore support high probe accuracy even if the model never computes the intended semantic relation.
- Define the target at the right strength:
  - the paper does not ask whether a model “understands code” in a general or human-like sense;
  - it asks whether hidden states contain particular program relations beyond controlled surface cues, how stable those relations are, and whether downstream computation reads them.
- Briefly connect to prior probing work on syntax, AST structure, identifiers, data flow, and program state: cite Wan et al. (2022), Hernández López et al. (2022), Troshin and Chirkova (2022), Ma et al. (2024), GraphCodeBERT for data-flow-aware pretraining (Guo et al., 2021), and Jin and Rinard (2024).
- State the gap precisely: these works motivate recoverability, but recoverability can reflect input information or probe capacity and does not establish causal use; cite Hewitt and Liang (2019) for control tasks.

## Paragraph 2: the construction that identifies representation

- Show the minimal pair in a two-column code box:

  ```python
  x = 3                  x = 3
  def f():               def f():
      y = 7                  x = 7
      return x               return x
  # returns 3             # returns 7
  ```

- Explain the example slowly enough for a non-program-analysis reader:
  - Python treats a name assigned inside a function as local to that function;
  - changing `y` to `x` therefore changes the definition selected by the identical `x` in `return x`;
  - the use token stays fixed, its position stays fixed, and the local windows around the compared anchors remain fixed.
- Phrase the baseline carefully:
  - “The measured surface baseline, which uses the ±3 token windows around both anchors and their bucketed distance, scores exactly 0.500 on these paired examples.”
  - Avoid the stronger sentence “no feature of the text can separate the labels.” The repository explicitly notes that a cross-position string-equality feature lies outside the current baseline and remains an open control.
- Explain why the embedding layer matters:
  - it is a model-derived but context-free lexical representation;
  - its exact 0.500 score shows that binding information appears only after contextual computation.

## Paragraph 3: three dimensions of evidence

- Present the framework in a single compact sequence:
  1. **Representation:** can a low-capacity readout recover the relation above the controlled surface floor?
  2. **Robustness:** does a frozen readout survive changes in form when meaning is preserved, and fail selectively when the relational problem becomes harder?
  3. **Causal use:** does an intervention on the proposed state change behavior according to the counterfactual semantic relation rather than merely moving a token or answer direction?
- Make clear that these dimensions are not interchangeable:
  - robustness strengthens the representational interpretation but does not imply downstream use;
  - an answer-changing intervention is not sufficient unless it isolates the semantic variable.
- Cite causal abstraction/interchange work here: Geiger et al. (2023), Wu et al. (2023; Boundless DAS), Huang et al. (2024; RAVEL), Feng and Steinhardt (2024), and Wu, Geiger, and Millière (2025).

## Paragraph 4: contributions and headline findings

- Use three contributions matching the paper structure:
  - **Representation:** binding and def--use are built rapidly with depth, peak in middle layers, and are partly shed near the output, replicated across 1.3B and 6.7B.
  - **Robustness:** distance and spelling are comparatively cheap, while scope interference and control-flow flattening are expensive; the representation is more abstract than lexical identity but not invariant to arbitrary semantics-preserving restructuring.
  - **Causal use and methodology:** the strongest binding-interchange test passes all prerequisites but not yet its decisive low-rank and held-out gates, leaving causal use open; earlier interventions reveal general failure modes involving surface transport, unmatched positive controls, and intervention dose.
- End with the paper's central interpretation:
  - the evidence supports a computed, scope-sensitive, structurally constrained relation;
  - it does not support the broader claim that the model forms a fully normalized symbolic program representation.

---

# 2. Framework and Experimental Setup

This section must be compact. Put implementation details in Appendix A, but retain every detail needed to interpret the three claims.

## 2.1 Operational definitions

- Introduce common notation:
  - program (x);
  - semantic relation (s(x)), chiefly binding and def--use;
  - hidden state (h_l(x)) at layer (l);
  - decoder (g).
- Define representation operationally:
  - (g(h_l(x)) \approx s(x)), evaluated against a surface-only baseline on the same controlled stratum;
  - say explicitly that this is a claim about accessible information, not human-like understanding or causal use.
- Define robustness using a transformation (T):
  - for meaning-preserving (T), (s(T(x)) = s(x));
  - freeze (g) after training on base programs and test whether (g(h_l(T(x)))) still recovers the relation;
  - this isolates changes in representation from probe refitting.
- Define causal use:
  - an intervention that installs semantic state (s') should make output behavior match the corresponding counterfactual, not merely differ from the original output;
  - cite interchange intervention and causal abstraction work.
- Add one sentence on the intended hierarchy:
  - representation is necessary but insufficient for robustness, and both are insufficient for causal use.

## 2.2 Models and data

- Models:
  - DeepSeek-Coder 6.7B base as the main model;
  - DeepSeek-Coder 1.3B base as a cross-scale replication;
  - base models are used to study representations learned through code pretraining without instruction-tuning behavior as an additional variable.
- Synthetic datasets:
  - canonical core generation uses 200 binding programs, 200 taint programs, and 100 shadowing programs;
  - context study uses 40 base programs, five filler types, and 0/50/100/200/500/1,000 inserted-token targets;
  - obfuscation uses 40 bases and a cumulative five-level ladder;
  - E13 uses 400 bases, four programs per base crossing binding structure with value assignment.
- Natural-code transfer:
  - fixed sample of 200 AST-parseable Python functions from the CodeSearchNet validation split (Husain et al., 2019);
  - present this only as transfer of the decoder/layer signature, not transfer of the exact 0.500 identification.
- Ground truth:
  - generators record exact program structure;
  - def--use extraction is independently cross-checked against Beniget, using set inclusion where branches admit multiple reaching definitions and equality on straight-line code;
  - obfuscated programs are executed and required to be observationally equivalent to their source;
  - E13 execution labels are independently checked by a scope-aware AST interpreter.
- Token alignment:
  - AST spans are mapped to exact tokenizer positions and verified against source text;
  - mention the exact code round-trip tokenizer guard only in the appendix, unless reproducibility is a workshop emphasis.

## 2.3 Readouts, controls, and uncertainty

- Pairwise probe representation:
  - for identifier positions (i,j), use ([h_i;h_j;h_i-h_j;|h_i-h_j|]);
  - linear classifier, (C=0.1), maximum 20,000 rows per task/layer;
  - five-fold `StratifiedGroupKFold`, with all rows from a source program kept in the same fold.
- Explain why linear:
  - the intended claim is that a relation is explicitly accessible in the state;
  - higher-capacity probes would make it harder to distinguish model structure from structure learned by the probe.
- Selectivity control:
  - retrain the identical probe after shuffling labels within each program;
  - report target accuracy minus shuffled-label accuracy;
  - preserve program-level grouping and label balance.
- Surface baseline:
  - identical training/evaluation protocol but no hidden states;
  - features are ±3-token ID windows around both anchors and bucketed pair distance;
  - report results per negative stratum rather than pooling easy and hard negatives.
- Negative strata:
  - `diff_name`, `distance_matched`, `same_name_diff_binding`, and decisive `context_matched`;
  - both members of a context-matched counterfactual remain in one CV group.
- Metrics:
  - accuracy for balanced controlled strata;
  - AUC for natural-code transfer, where identifiers are informative and a thresholded aggregate can obscure ranking quality;
  - cluster-bootstrap confidence intervals over base programs for paired intervention effects.
- Include a compact table with one row per dimension:

  | Dimension | Main test | Required control | Status |
  |---|---|---|---|
  | Representation | context-matched linear decoding | surface and embedding floor = 0.500 | established |
  | Robustness | frozen decoder under perturbation | matched lengths; execution-verified transforms | established within tested shifts |
  | Causal use | crossed-arm binding interchange | whole-state ceiling, answer-direction falsification, matched random controls | open; H0--H3 pass |

---

# 3. Representation: Is the Relation Present Beyond Surface Form?

## 3.1 Variable binding is built with depth

- Start with the question in plain language:
  - does an identifier use encode which definition it resolves to, rather than only how it is spelled?
- Make `context_matched` the only headline stratum:
  - surface and embedding accuracy are exactly 0.500 for both model scales;
  - first transformer block: 0.570 for 1.3B and 0.531 for 6.7B;
  - layer 3: 0.961 and 0.914;
  - peak: 0.984 at L7 for 1.3B and 0.984 across L11--15 for 6.7B;
  - final layer: 0.930 at L23 and 0.914 at L31.
- Interpret the profile in three stages:
  - **absent at input:** controlled binding cannot be read from token identity alone;
  - **rapidly constructed:** most of the recoverable relation appears in the first few transformer blocks;
  - **partly shed near output:** the later state is reorganized toward next-token prediction, so representation need not increase monotonically with depth.
- Be careful with “absent”:
  - write “not linearly recoverable above the measured floor at the input,” rather than metaphysically absent from every possible representation.
- Cross-scale interpretation:
  - emphasize replicated curve shape and relative depth rather than claiming a scaling law from two models;
  - the larger model holds the peak over more layers, but this is descriptive.
- Recommended Figure 1:
  - two panels, 1.3B and 6.7B;
  - x-axis transformer depth normalized to [0,1], with actual layer numbers in secondary labels;
  - y-axis context-matched binding accuracy;
  - horizontal line at 0.500;
  - optionally add def--use as a thinner line if legible;
  - do not plot easy negative strata in the main figure. Put them in Appendix B.
- Caption should carry the identification claim:
  - the controlled relation begins at the surface/embedding floor, rises sharply after contextual computation, and peaks at the same relative depth across scales.

## 3.2 Def--use extends the result from reference to data flow

- Define a def--use edge for a general ML reader:
  - a directed relation connecting a definition to a later occurrence that reads the value it defines.
- Design:
  - use the same pairwise feature construction;
  - distance-match negative pairs so proximity cannot solve the task;
  - bucket results by token distance.
- Results:
  - peak accuracy is approximately 0.99 in middle layers;
  - the layerwise trajectory closely matches binding;
  - the hardest 50--200-token bucket still reaches about 0.96--0.99.
- Interpretation:
  - the shared profile suggests that the model constructs related reference/data-flow structure rather than only retaining lexical adjacency;
  - long-range accuracy motivates the later robustness result: raw distance alone is not the dominant limit.
- Do not give this experiment equal visual space to binding. One paragraph plus a line/table entry is enough in five pages.

## 3.3 Contrast and external relevance

- Control dependence as a negative contrast:
  - hidden-state decoding is excellent, but a surface-only baseline already reaches about 0.927 accuracy and 0.990 AUC;
  - therefore high decodability is not automatically a semantic result;
  - the experiment demonstrates that the paper's criterion can demote a superficially impressive result.
- Natural code as a transfer analysis:
  - on 200 CodeSearchNet Python functions, 6.7B peak AUC is 0.978 for binding and 0.979 for def--use, compared with surface AUC 0.673 and 0.590;
  - 1.3B reaches approximately 0.980/0.975 peak AUC;
  - the early-middle peak and late decline resemble the synthetic profile.
- Limitation:
  - natural identifiers carry real information, with embedding AUC already about 0.96;
  - the result rules against a pure synthetic-template explanation, but does not transfer the clean semantic isolation.
- Move the full control-dependence and CodeSearchNet tables to Appendix C.

## Section 3 takeaway sentence

- “The controlled layerwise profile is evidence that the models compute scope-sensitive reference and data-flow relations, not merely that a classifier can recover labels correlated with the source text.”

---

# 4. Robustness: What Is the Representation Made Of?

## Opening logic

- A decodable relation may still be tied to identifier strings, formatting, or generator-specific structure.
- Robustness is therefore not an additional benchmark score; it identifies which changes preserve the same readout and which changes destroy it.
- State the frozen-probe rule again in one clause: train on base programs once, never refit on transformed inputs.

## 4.1 Context degradation separates distance from interference

- Explain the controlled comparison:
  - insert filler between the tracked definition and use;
  - measure filler size in actual tokenizer tokens;
  - use the same base programs across conditions.
- Explain the filler ladder by the semantic problem it creates:
  - `comment_prose`: inert text, mostly distance;
  - `dead_code`: unreachable code;
  - `lexical_decoy`: fresh but similar-looking names;
  - `competing_update`: rebindings of other variables;
  - `scope_shadow`: reuse of the tracked names inside a nested scope.
- Main 6.7B results at 500 tokens:
  - comment prose 0.921;
  - dead code 0.794;
  - lexical decoy 0.795;
  - competing update 0.859;
  - scope shadow 0.570.
- At 1,000 tokens:
  - scope shadow reaches approximately 0.498, while every other filler remains above 0.70;
  - do not repeat every numeric value in the prose; let Figure 2 show the trajectories.
- Layerwise result:
  - under shadowing, block 0 is relatively stable while the middle layers that normally build binding collapse most strongly;
  - this localizes the degradation to the contextual computation rather than the lexical lookup.
- Interpretation:
  - “distance is cheap; interference is expensive” should be the subsection's explicit conclusion;
  - avoid claiming comments cost literally nothing, because performance does decline from the unperturbed condition.

## 4.2 Obfuscation separates spelling from structural scaffold

- Design:
  - cumulative ladder: normalization, consistent local renaming, opaque predicates, mixed boolean-arithmetic encoding, control-flow flattening;
  - every variant is execution-verified against its source;
  - all levels for a base are retained or dropped together.
- Report best-layer binding accuracy, because layer-averaged values hide the key result:
  - normalized approximately 1.000;
  - rename 0.897 at L11;
  - opaque predicates 0.857;
  - encoded arithmetic 0.846;
  - flattened control flow 0.750.
- Layerwise renaming result:
  - embeddings/block 0 fall below chance, about 0.29--0.33;
  - middle layers 7--15 remain around 0.85--0.90;
  - early states remain lexical enough that consistent renaming systematically misleads the frozen decoder, while middle states support a more abstract relation.
- Interpret flattening as a boundary, not a contradiction:
  - the representation is invariant to some surface changes but not to arbitrary semantics-preserving compilation;
  - it appears tied to the structured scope/control scaffold in which source code presents the computation.
- Mention that def--use shows the same qualitative pattern; put its full table in the appendix.

## Recommended Figure 2

- Two panels sharing a clear “frozen probe” label:
  - left: binding accuracy against inserted token count for comment prose and scope shadow; optionally gray lines for the other fillers;
  - right: best-layer binding accuracy across the cumulative obfuscation ladder, with a small inset or callout showing early-layer versus middle-layer behavior after renaming.
- The caption should synthesize rather than restate:
  - inert distance and lexical rewriting are tolerated most strongly where the relation is built, while competing scope and flattened control structure define the main failure surface.

## Section 4 takeaway sentence

- “The representation is neither a lexical lookup nor a fully normalized symbolic analysis: it abstracts away from names and distance, but remains dependent on the program's presented scope and control structure.”

---

# 5. Causal Use: Does the Model Read the Relation?

This section should be framed around identification, not around a chronological report of every experiment.

## 5.1 Why decoding and answer-changing interventions are insufficient

- State the causal target:
  - after editing hidden state to install an alternative binding (s'), behavior should follow the value selected by (s').
- Distinguish this from generic answer movement:
  - changing a logit does not show that the binding relation was transported;
  - whole-state patching can carry token identity and many correlated features;
  - a small low-rank null is uninterpretable without showing that an edit of the same magnitude can affect the same site.
- Summarize three failure modes learned from earlier work, without giving each a full subsection:
  - **surface transport (E7):** early patching at the sink argument also restores the only changed input token;
  - **dose ambiguity (E11):** at the use site, response efficiency increases about 18-fold over the dose sweep, and a known-correct edit of the same small magnitude is also null;
  - **capability bottleneck (E12):** the 1.3B model fails the chained-arithmetic behavioral gate (about 0.418 balanced accuracy), so the experiment cannot answer a question about semantic state.
- If space permits, mention the general rule derived from these failures:
  - positive controls must match the test in kind, site, and scale.
- Cite activation-patching best practices (Zhang and Nanda, 2023), DAS/interchange work (Geiger et al., 2023), and Nikankin et al. (2024) when discussing arithmetic bottlenecks.

## 5.2 Binding interchange provides a direct falsification test

- Present the 2×2 design clearly:
  - cross binding structure (outer/source versus inner/target definition) with value assignment (`ab` versus `ba`);
  - four programs per base;
  - within each arm, programs differ by one upstream character but have an identical use token and equal token length;
  - the mutation is at least four tokens before the use;
  - no arithmetic occurs: the function returns the selected variable.
- Show the two arms in a compact schematic:

  | Arm | Host/source answer | Donor/target answer | Required movement |
  |---|---:|---:|---|
  | `ab` | a | b | a → b |
  | `ba` | b | a | b → a |

- Explain the identification in words:
  - fit the low-rank alignment on arm `ab` and evaluate it without refitting on held-out arm `ba`;
  - a representation of which definition is in scope should transfer and reverse its token-level effect;
  - a fixed answer or token-`b` direction can succeed on `ab` but should fail on `ba`.
- Relate this directly to prior work:
  - learned low-rank interchange is established by DAS/Boundless DAS and used in RAVEL and symbolic-program binding research;
  - the contribution here is the value-assignment factorial plus a construction-pinned surface floor in a pretrained code model, not a new intervention algorithm.

## 5.3 Gates and current evidence

- Give the gate chain in one compact table. This is the main causal-use result table:

  | Gate | Question | Current 6.7B result | Reading |
  |---|---|---|---|
  | H0 | Do execution and an independent scope interpreter agree? | pass, 400/400 bases; invariants 1.000 | data valid |
  | H1 | Does the model resolve the binding behaviorally? | pass, 1.000 overall and in weakest cell | task is within capability |
  | H2 | Is binding decodable at the use over the surface floor? | pass, 1.000 vs 0.500 | intervention site contains the relation |
  | H3 | Can whole-state interchange move behavior in both arms? | pass; logit movement +4.781/+4.799, flip rate 0.857 | both arms are causally testable |
  | H4 | Does low-rank interchange beat matched controls on training arm? | not yet valid | no claim |
  | H5 | Does it transfer to held-out arm while answer direction fails? | not yet valid | causal use remains open |

- Explain why H3 is unusually important:
  - a held-out null is meaningful only if whole-state replacement shows that the site and both arms can affect behavior;
  - structural-zero controls are exactly zero where no movement should occur, supporting hook and anchor correctness.
- Explain why the first H4/H5 run cannot be interpreted:
  - the learned rank-1 edit exceeded the whole-state ceiling and moved about 48% of the hidden-state norm;
  - the answer-direction comparison moved only about 1% and was therefore not norm-matched;
  - this recreated the earlier dose problem inside the decisive control;
  - the control has been corrected but the valid rerun is not in the repository record used for this outline.
- State the conclusion without softening or dramatizing it:
  - prerequisites for a decisive test pass;
  - the decisive semantic-versus-answer comparison does not yet have a valid result;
  - therefore the paper establishes representation and robustness, not causal use.

## 5.4 What each eventual result would mean

- If H4 and H5 pass:
  - conclude that a low-rank use-site subspace transports which definition is in scope across crossed value assignments;
  - still avoid “the model understands binding”; constrain the claim to this task, site, intervention family, and models.
- If H4 passes and H5 fails while the answer-direction control fails on `ba` as designed:
  - conclude that the learned training-arm subspace behaves like an answer direction rather than an abstract binding variable.
- If the answer-direction control also transfers to `ba`:
  - the discriminator is broken and no semantic conclusion is licensed.
- If H3 later fails under a revised run:
  - the held-out arm is not causally testable, so H5 cannot be interpreted.
- Put this branching interpretation in Appendix D if the main text is tight.

## Section 5 takeaway sentence

- “The use site contains the controlled binding relation and can causally affect the answer under whole-state replacement, but the evidence does not yet show that a low-rank binding representation, rather than an answer-aligned feature, is what downstream computation reads.”

---

# 6. Discussion, Limitations, and Conclusion

## Paragraph 1: answer the three questions directly

- Representation:
  - yes, under the paper's operational definition and measured surface controls;
  - binding and def--use emerge rapidly after the embedding layer and share a middle-layer peak.
- Robustness:
  - qualified yes;
  - the same frozen readout survives inert distance and renaming most strongly in middle layers, but degrades under shadowing and flattening.
- Causal use:
  - unresolved;
  - H0--H3 establish a valid site and testable behavior, while H4--H5 remain pending a valid matched-control run.

## Paragraph 2: scientific interpretation

- The joint evidence suggests a sequence of representational formats:
  - lexical at the input/earliest layer;
  - relational in early-middle and middle layers;
  - increasingly output-oriented near the final layers.
- Phrase this as an interpretation consistent with the curves, not a demonstrated mechanistic decomposition.
- The robustness boundary suggests the relational format is tied to source-level scope and control scaffolding, not a compiler-like canonical semantics.
- The control-dependence contrast supports a graded criterion:
  - not every program relation that is decodable should receive the same semantic interpretation.

## Paragraph 3: limitations

- Models:
  - two sizes from one model family; replication across architecture and training corpus remains necessary.
- Language/task scope:
  - Python only, short synthetic programs, mostly lexical scoping and def--use;
  - results do not establish general execution, alias analysis, heap reasoning, exceptions, or interprocedural semantics.
- Surface floor:
  - exact for the specified local-window-plus-distance baseline on constructed pairs, not for every computable feature of the complete source;
  - add the cross-position name-equality baseline before submission if possible.
- Natural code:
  - transfer is supportive but does not preserve the controlled 0.500 isolation;
  - static ground truth shares parts of the analysis pipeline.
- Robustness:
  - frozen-probe failure can mean that the representation changed basis, not necessarily that semantic information vanished;
  - execution equivalence is tested on the generator's observations, not proved for all inputs.
- Causality:
  - the strongest causal claim is open;
  - whole-state interchange establishes causal potency but is not specific to binding;
  - a future positive low-rank result would remain local to the selected layer/site and task distribution.

## Final paragraph

- End with the methodological point rather than a generic call for more work:
  - interpretability claims become scientific when each stronger word is attached to a stronger falsification;
  - a controlled floor supports representation, frozen transformations reveal its basis, and crossed interventions are needed for causal use;
  - on this standard, code models clearly compute scope-sensitive relations, their invariances are limited and measurable, and their causal use is the remaining question.

---

# Main-text visual and table plan

## Figure 1: construction and layerwise representation

- Left inset: the one-character binding flip with the use token highlighted.
- Main panels: context-matched binding accuracy across normalized depth for 1.3B and 6.7B.
- Horizontal surface/embedding floor at 0.500.
- Optional faint def--use curve if it remains legible.
- The figure should occupy no more than about 0.45 page.

## Figure 2: robustness failure surface

- Left: context length curves, emphasizing comment prose versus scope shadow.
- Right: cumulative obfuscation best-layer accuracy, with an early-versus-middle renaming annotation.
- Use consistent colors for “surface change” and “semantic/structural interference.”
- Avoid plotting all filler types with equal visual weight; gray supporting lines are sufficient.

## Table 1: three-dimensional evidence, with causal gates

- Prefer one combined compact table:
  - rows for binding, def--use, context, obfuscation, and E13 H0--H5;
  - columns for test, decisive control, result, and claim status.
- If this is too dense, keep the three-row framework table in Section 2 and use the six-row H0--H5 table in Section 5.

---

# Appendix plan (unlimited length)

## Appendix A: full methods and reproducibility

- Program-generation grammar and all template families.
- Exact model identifiers, layer maps, dtype, maximum lengths, random seed 42, hardware, and software versions.
- Tokenizer integrity guard and the exact code round-trip test.
- AST-span-to-token alignment algorithm and verification failures.
- Def--use extractor and Beniget cross-check, including the self-referential-update bug that the comparison caught.
- Full pairwise feature equation, classifier hyperparameters, group-level sampling, five-fold splits, and shuffled-label construction.
- Surface feature encoding and distance buckets.
- Cluster bootstrap procedure, resampling unit, number of resamples, and paired contrasts.
- Manifest/reproducibility workflow and tests.

## Appendix B: full representation results

- All binding strata by layer and model.
- Surface, embedding, hidden-state, shuffled-label, selectivity, accuracy, and AUC tables.
- Def--use distance buckets with counts.
- Lexical token-type sanity check: embedding accuracy 1.000, clearly labelled as validation rather than a scientific finding.
- Exact group/sample counts per task, layer, and stratum.

## Appendix C: secondary and transfer analyses

- Control dependence:
  - surface accuracy 0.927/AUC 0.990;
  - hidden-state performance and positive/hard-negative recall;
  - explain why it is a contrast rather than a headline semantic result.
- CodeSearchNet:
  - all 1.3B and 6.7B binding/def--use results;
  - embedding and surface baselines;
  - stratum sizes once added to the output.
- Complete context degradation curves for both tasks, all filler types, both models, and layerwise panels.
- Complete obfuscation ladder for binding and def--use, with execution-equivalence yield.

## Appendix D: causal experiments and audit trail

- E13 factorial generation, invariants, independent interpreter, metric, ranks, alignment training split, and every control.
- H0--H3 full tables and bootstrap intervals.
- First invalid H4/H5 run shown only as a diagnostic:
  - rank-1 result exceeding ceiling;
  - 48% state-norm edit;
  - unmatched 1% answer-direction control;
  - why no causal verdict follows.
- Predefine how the corrected H4/H5 rerun will update the main paper under each outcome.
- E7 activation-patching migration as descriptive supporting evidence, with the surface-transport limitation.
- E11 J-space NO-GO, readout-position measurement, use-site 18× convex dose response, missing `probe_basis` control, and retracted null.
- E12 arithmetic behavioral-gate failure and proximity simulation.
- J-lens implementation validation as instrument validation only.
- Retired lead-time and control-dependence-lens analyses, with the exact reason each claim was withdrawn.

## Appendix E: scope of claims and robustness checks requested before submission

- Add a cross-position string-equality surface baseline. This is the highest-priority representational control because the existing ±3-token baseline cannot express name equality between separated anchors.
- Report confidence intervals and sample counts for every headline number, particularly the context-matched stratum.
- Report E5/E9 at preregistered or calibration-selected layers, not only post hoc best layers; retain best-layer values as descriptive summaries.
- Add architecture-family replication if feasible; otherwise state clearly that scale replication is within DeepSeek-Coder.
- Add context-matched mutations to natural CodeSearchNet functions if time permits; otherwise keep the transfer claim narrow.
- Synchronize `results/STATUS.yaml`, `docs/RESULTS.md`, and the final manuscript before release.

---

# Citation placement and bibliography additions

The existing bibliography already supports the initial code-representation context:

- Wan et al. (2022): structural analysis of pretrained code models.
- Hernández López et al. (2022): AST-Probe.
- Troshin and Chirkova (2022): probing pretrained source-code models.
- Ma et al. (2024): syntax and semantics capacities of code pretrained models.
- Jin and Rinard (2024): emergent program-semantics representations.
- Guo et al. (2021): data-flow-aware code pretraining.
- Hewitt and Liang (2019): probe control tasks.
- Zhang and Nanda (2023): activation-patching methodology.
- Husain et al. (2019): CodeSearchNet.

Add full BibTeX records for the papers already cited in the supplied notes and E13 design:

- Geiger et al. (2023), *Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations* — introduce causal abstraction and distributed alignment/interchange.
- Wu et al. (2023), *Interpretability at Scale: Identifying Causal Mechanisms in Alpaca* — Boundless DAS and scalable learned causal alignments.
- Feng and Steinhardt (2024), *How Do Language Models Bind Entities in Context?* — causal intervention on entity binding; contrast attribute retrieval with source-code scope resolution.
- Huang et al. (2024), *RAVEL: Evaluating Interpretability Methods on Disentangling Language Model Representations* — isolation and disentanglement controls for representation interventions.
- Wu, Geiger, and Millière (2025), *How Do Transformers Learn Variable Binding in Symbolic Programs?* — closest conceptual comparison; distinguish a small from-scratch symbolic model from pretrained code models and emphasize the present value-assignment factorial.
- Nikankin et al. (2024), *Arithmetic Without Algorithms: Language Models Solve Math With a Bag of Heuristics* — use only when explaining why E12's chained arithmetic is a confounded capability requirement.

Do not cite Gurnee et al. (2026) as support for the central representation or causal claims. If retained, use it only in the appendix to motivate the J-lens instrument, since the surviving main-paper argument does not depend on verbalizability or a global workspace.

---

# Material to omit from the five-page main paper

- E1 lexical token classification beyond one sentence as a pipeline sanity check.
- Taint-state probing and behavioral lead time.
- J-lens taint and J-lens control-dependence tracks.
- Detailed activation-patching heatmaps.
- Full E11 operation-family tables and lens comparisons.
- Store-transition family diagnostics from E12.
- Every stage number, script name, manifest path, and run command.
- Aggregate static-probe accuracy that pools easy negative strata.
- Separate prose discussions of normalization, opaque predicates, and arithmetic rewriting; group them as intermediate obfuscations and emphasize the renaming/flattening contrast.
- Broad claims about model understanding, semantic equivalence under arbitrary compilation, or causal use of binding.
- Historical narrative about all failed experiments in the main text. Retain the three general lessons and move the complete audit trail to the appendix.

---

# Suggested final prose rhythm

- Preserve the supplied notes' pattern of short declarative claims followed by a careful qualification: “Distance is cheap; interference is not.” Then explain exactly what the comparison holds fixed.
- Use paired contrasts frequently: surface versus semantic, distance versus interference, representation versus use, answer movement versus binding transport.
- Prefer medium-length sentences with one main logical turn. Use a longer sentence only when its clauses encode an explicit experimental contrast.
- Define a technical term immediately before using it. For example: “A def--use edge links a definition to a later occurrence that reads its value.”
- Let numbers serve an argument. Give the endpoints and decisive contrast in the main text; move exhaustive layer and condition values to tables.
- State limitations in the sentence where a result is interpreted, not only in a final limitations section.
- Use “supports,” “is consistent with,” and “does not establish” to match the evidential strength. Reserve “shows” for direct measurements and passed controls.
