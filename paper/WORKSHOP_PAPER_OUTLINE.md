# Workshop paper outline: security-relevant semantic representations under obfuscation

## Working title

**Do Code Models See Through Obfuscation? Auditing Security-Relevant Semantic Representations**

Alternative: **Code Models Track Security-Relevant Data Flow, but Control-Flow Flattening Breaks the Readout**

Use the first title for AI4GOOD. It makes the security question explicit while retaining semantic representation as the scientific premise.

## One-sentence paper claim

- Across DeepSeek-Coder 1.3B and 6.7B and StarCoder2-3B, middle-layer states distinguish whether untrusted input reaches a code-bearing sensitive argument at 1.000 on held-out programs, against chance-level local-surface, whole-program-lexical and embedding controls — and **control-flow flattening applied alone destroys that readout (0.660--0.688) while opaque predicates and mixed-boolean-arithmetic encoding applied alone cost exactly nothing**, with the composition of all four adding no measurable interaction.

## The argument, in five steps, ranked by what the data carries

1. **Robustness, decomposed — the headline.** Atomic conditions isolate each transformation. Flattening alone reproduces the entire collapse; the interaction between transformations is inside the measured draw-noise floor in all three models. This is the paper's strongest and newest result (§5.2--5.3).
2. **Operational safety measurement.** Pooled accuracy conceals the failure that matters: half the matched pairs collapse to one label, class biases run in opposite directions across models, and the dangerous errors appear under *renaming alone* (§5.4--5.5).
3. **Representation.** The property is decodable at ceiling over **two** measured floors, one of which reads the entire program text (§5.1).
4. **Verbalisation — a clean null.** The same states, mapped into each model's own output vocabulary through three agreeing lenses, carry no security concept; 1.3B's contrast is significantly inverted. Decodable is not verbalised (§5.8).
5. **Causal use — supporting only.** E13 establishes causal transport for *binding*, at one site, layer, model and construction. Nothing in E15 is causal (§6.2).

Framing: a **semantic-representation and robustness** paper motivated by trustworthy program analysis in security. Not malware classification, not end-to-end vulnerability detection, not a new causal-intervention project.

## Paper position

- This is a paper about **auditing the robustness of AI code analysis**, not malware detection or interpretability for its own sake.
- The deployment-motivated question is whether a defender can recover a security-relevant program property when an adversarial author changes surface form while preserving behavior.
- Binding and def--use provide the controlled foundation: the models compute program relations beyond the declared surface reader.
- E15 is the application-facing contribution: it tests untrusted source-to-sensitive-sink flow under obfuscation, decomposes the failure into independent and compositional parts, and asks whether the difference is expressed in the model's own vocabulary basis.
- E13 becomes supporting mechanistic evidence. It should no longer be the narrative climax.

## Claim discipline

- Keep these claims distinct:
  - controlled binding/def--use representation: established for the specified probe and surface controls;
  - clean security-flow auditability: established on the synthetic E15 benchmark;
  - frozen-readout robustness: established within the tested transformations;
  - causal use: established locally for binding by E13, not for source-to-sink flow;
  - deployment-ready security auditing: not established.
- Do not call E15 a malware detector. Its programs instantiate primitives relevant to command injection, SQL injection, and dynamic-code injection.
- Do not equate taint flow with vulnerability generally. E15 asks whether untrusted data reaches a code-bearing sensitive argument.
- **The `flatten_only` arm supports it**, in all three models, so "control-flow flattening breaks the readout" may now be written without hedging. Keep the vocabulary precise: an **independent transformation effect** is what an atomic condition's `delta_clean` shows; a **marginal effect** is `delta_previous` along the cumulative chain; an **interaction** is `delta_atomic`, cumulative minus its atomic counterpart. Anything not supported by an atomic row is written as a cumulative effect.
- Do not read the `rename` row of the interaction table as an interaction: `rename_only` and `rename_cumulative` are the same transformation under independent draws, so that row is the measured draw-noise floor for the column.
- The vocabulary experiment returned a **null**, and 1.3B's contrast is significantly *inverted*. Report the sign. Never write "the model represents unsafe". The null is compatible with the probe succeeding — that dissociation is the point of including it.
- Do not claim the whole-program lexical baseline pins the floor against every predictor. It bounds generator-level *textual* shortcuts; a reader that performs the taint analysis is still out of scope.
- Do not claim security representations are *specifically* fragile. The completed E9 companion shows binding and def--use break the same way, and the security readout is at least as robust as either.
- Do not claim security representations are *specifically* more fragile until the companion E9 comparison is complete on StarCoder2.
- Do not infer retained flow from level-4 accuracy alone. Pair collapse and opposite class biases show that much of the residual accuracy is a model prior.
- E13 is positive: all H0--H5 pass on DeepSeek-Coder 6.7B. Constrain it to one construction, use site, layer, model, and intervention family.
- Avoid “understanding” except when saying what is not claimed.

## Target length

Target a focused 7--8 page AI4GOOD workshop paper, excluding references and appendices:

- Abstract: 180--220 words.
- Introduction and threat model: 0.9 page.
- Related work: 0.6 page.
- Controlled foundations and audit methodology: 1.2 pages.
- Security source-to-sink benchmark: 1.1 pages.
- Obfuscation results (atomic, cumulative, interactions, per-class): 2.0 pages.
- Baselines and the vocabulary-space contrast: 0.5 page.
- Mechanistic evidence and implications: 0.8 page.
- Limitations and conclusion: 0.7 page.

---

# Abstract

Write the abstract as a security-auditing argument:

- AI-assisted code auditors may perform well on ordinary source while failing when an adversarial author preserves behavior but conceals data and control flow.
- Accuracy on clean code or probe decodability does not show that the relevant flow exists beyond identifiers and local syntax, nor that a frozen audit survives obfuscation.
- Give the controlled foundation in one sentence: token-matched binding pairs pin the declared surface and embedding readers to 0.500 while middle-layer decoding approaches 0.99.
- Introduce E15 precisely: 480 clean Python programs; three sensitive-sink families; four flow structures; matched safe/unsafe programs differing only at the sink argument; 336 clean training programs and 144 held-out programs per condition; independent dynamic and static labels; frozen evaluation through an execution-verified ladder.
- Report clean local surface 0.488--0.491, whole-program lexical 0.464, embedding 0.482, and middle-layer decoding 0.997--1.000 across three models.
- **Lead the results sentence with the atomic decomposition**: opaque predicates and MBA encoding cost exactly nothing alone; renaming costs 0.014--0.118; control-flow flattening alone costs 0.312--0.340, with the composition of all four adding no interaction beyond draw noise.
- State the operational asymmetry: StarCoder2's `rename_only` accuracy is 0.882 overall but 0.764 on unsafe programs, and about half of matched pairs collapse to one label under flattening.
- One clause on the vocabulary experiment, and only its outcome: the distinction is *not* expressed in the models' own output-aligned coordinates. Do not spend abstract space on the method.
- Conclude that security readouts require semantics-preserving adversarial tests, per-class rates, and matched-pair diagnostics rather than pooled clean accuracy.
- Omit E13 from the abstract unless the security result is already clear and space remains.

---

# 1. Introduction and Threat Model

## 1.1 Practical problem

- Code models are plausible components of vulnerability triage, code review, and security analysis.
- A security auditor faces a distribution shift with an adversary behind it: the author can rename variables, insert opaque branches, encode expressions, or flatten control flow while preserving behavior.
- The central question is not only whether a model classifies clean code, but whether its internal representation of security-relevant flow remains auditable under concealment.

## 1.2 Why ordinary evaluation is insufficient

- Sensitive APIs and names such as `raw`, `safe`, or `command` can make clean benchmarks solvable lexically.
- A detector refitted on every transformation can learn new shortcuts and conceal a representational failure.
- Pooled accuracy can hide the dangerous failure mode: systematically predicting vulnerable programs as safe.
- The paper therefore combines matched counterfactual programs, surface and embedding controls, frozen readouts, execution-verified transformations, per-class rates, and matched-pair collapse.

## 1.3 Threat model

- **Defender:** an auditor using a fixed code model and frozen linear readout.
- **Adversary:** controls program surface form and applies the tested semantics-preserving transformations; does not change weights or the security label.
- **Protected property:** whether the value at a code-bearing sensitive argument derives from untrusted input.
- **Attack success:** reduce the readout's ability to distinguish matched safe and unsafe programs, especially through false negatives.
- **Out of scope:** executable malware, reflection, dynamic loading, heap aliasing, concurrency, and adaptive white-box attacks.

## 1.4 Contributions

1. **Controlled semantic foundation.** Binding and def--use become decodable only after contextual computation and survive lexical changes better than structural interference.
2. **Security-flow benchmark.** A gated, 480-program benchmark crossing three sink families with four flow structures, matched pairs, and two independent ground-truth readings.
3. **Obfuscation audit, decomposed.** Frozen-readout evaluation across three models, with each transformation applied both alone and in composition, isolates control-flow flattening as the single cause of the collapse — while opaque predicates and expression encoding cost nothing and the interaction stays inside a measured draw-noise floor — plus class- and structure-specific failures hidden by pooled accuracy.
4. **A representational-format probe.** An observational vocabulary-space contrast asks whether the safe/unsafe difference is expressed in the model's own output-aligned coordinates, with train-only token discovery, permutation and mismatched-pair controls, and a reportable null.
5. **Evidence standard.** Security readouts should be evaluated with explicit surface floors (local *and* whole-program), held-out obfuscations applied both atomically and cumulatively, pair-collapse diagnostics, and per-class errors.

---

# 2. Related Work

Organize by the security problem rather than experiment chronology.

## 2.1 Learned program representations

- Cover probing of syntax, identifiers, namespaces, data flow, control flow, and latent execution state.
- Explain the identification gap: hidden states retain source text, so recoverability need not mean contextual computation formed the relation.
- Position the context-matched binding construction and surface controls as the answer to this gap.

## 2.2 ML for vulnerability and taint analysis

- Discuss learned vulnerability detection and source-to-sink/data-flow analysis.
- Distinguish E15 from end-to-end vulnerability classification: it isolates one security-relevant semantic bit under controlled transformations.
- Note that the safe member uses an independently trusted literal, not a generic “sanitizer,” because mitigation is sink-specific.

## 2.3 Adversarial code transformations

- Relate renaming, opaque predicates, mixed boolean-arithmetic encoding, and control-flow flattening to evasion and robustness evaluation.
- Emphasize independent label checking and that dangerous APIs are replaced by recorders during execution.
- Position frozen transfer as the key distinction from retraining on obfuscated samples.

## 2.4 Causal analysis

- Briefly cite causal abstraction, activation patching, and DAS/interchange.
- E13 supports the premise that one controlled code relation is causally transportable; E15 itself remains observational.

---

# 3. Controlled Foundations and Audit Methodology

## 3.1 Operational definitions

Let `x` be a program, `s(x)` a semantic or security property, `h_l(x)` its layer-`l` state, `g` a linear readout, and `T` a transformation.

- **Representation:** `g(h_l(x))` recovers `s(x)` above the declared surface reader on a controlled distribution.
- **Frozen auditability:** fit `g` on clean programs and evaluate the same `g` on held-out `T(x)` without refitting.
- **Robustness cost:** the paired performance drop from clean held-out programs to their transformed variants.
- **Causal use:** an intervention installing an alternative semantic state changes output according to the semantic counterfactual. E13 tests this for binding, not E15 flow.

## 3.2 Binding and def--use foundation

Show the one-character binding pair:

```python
x = 3                  x = 3
def f():               def f():
    y = 7                  x = 7
    return x               return x
# outer definition      # inner definition
```

- The use token, anchor windows, and separation remain fixed while its resolved definition changes.
- The declared surface reader and context-free embedding row are exactly 0.500.
- In the attached paper's runs, binding peaks at 0.981 for DeepSeek 1.3B and 0.991 for 6.7B; def--use follows the same profile and remains strong at 50--200 token distances.
- Use this to establish that contextual states can contain program relations beyond local lexical cues. Do not make it the largest result section.

## 3.3 Common controls

- Linear readouts and grouped splits by source/base program.
- Surface-only local token windows and the context-free embedding layer.
- Shuffled-label control, with the E15 caveat that within-pair shuffling can only swap two opposite labels and is not an ideal 0.500 null.
- Cluster bootstrap over base programs.
- Per-family, per-structure, per-class, and per-layer reporting.

## Table 1: evidence ladder

| Claim | Main test | Load-bearing control | Status |
|---|---|---|---|
| Binding/def--use representation | context-matched decoding | surface and embedding = 0.500 | established |
| Clean source-to-sink auditability | E15 clean held-out | surface/embedding near 0.500; grouped bases | established synthetically |
| Obfuscation robustness | frozen E15 readout | label-preserving held-out transforms | established for tested ladder |
| Independent transformation effect | E15 atomic arms | `normalize` reference row; per-variant AST verification | **established** — flattening alone, three models |
| Compositional (interaction) effect | atomic vs. cumulative difference | rename row as measured draw-noise floor | **established as null** — no interaction above noise |
| Generator-level textual shortcut ruled out | whole-program lexical arm | frozen transfer, vectorizer fitted on training text only | **established** — 0.465--0.535 in every condition |
| Vocabulary-space expression of the property | E15-C contrast | permutation, mismatched pairs, embedding/token-identity, role strata, random and Gram-matched lenses | **null**, three models; 1.3B inverted |
| Failure is not security-specific | E9 companion, three models | same transformations on binding and def--use | **established** — same boundary, security readout no more fragile |
| Binding causal use | crossed E13 interchange | reversed value arm and matched controls | established locally |
| Security-flow causal use | no intervention performed | -- | open |

---

# 4. E15: Security-Relevant Source-to-Sink Audit

## 4.1 Security property and sink families

Ask one question per program:

> Is the value passed to this code-bearing, security-sensitive argument derived from untrusted input?

| Family | Sources | Sensitive argument |
|---|---|---|
| command execution | request arguments, `sys.argv[1]` | `os.system(x)`, `subprocess.call(x, shell=True)` |
| SQL execution | request query/form input | SQL-text argument of `cursor.execute(x)` |
| dynamic execution | request form input, `input()` | `eval(x)`, `exec(x)` |

- The safe member passes an independently trusted literal; the unsafe member passes the source-derived value.
- Both members contain the same source, trusted chain, propagation code, and sink and differ only at the sink-argument span.
- This isolates source-to-sensitive-argument flow; it is not a taxonomy of vulnerabilities or mitigations.

## 4.2 Factorial benchmark

```text
3 sink families x 4 flow structures x 20 base seeds x 2 labels
= 480 clean programs
```

- `direct`: source/trusted value directly reaches the sink.
- `assign_chain`: two aliasing assignments.
- `branch_merge`: two definitions meet at a join.
- `helper`: one parameter-to-return helper boundary.
- Fourteen bases per family/structure form the training set: 168 bases and 336 programs.
- Six per family/structure are held out: 72 bases and 144 programs.
- Pair members and every transformed variant inherit the base split; only held-out programs are obfuscated.

## 4.3 Ground truth and gates

- **Dynamic reading:** restricted execution with provenance-carrying values and recorder stubs replacing sensitive APIs; no dangerous call executes.
- **Static reading:** AST taint fixpoint with one-level helper summaries.
- Both readings must agree with each other and the intended label.
- Six hard gates. **S0--S3** validate corpus size and balance, pair confinement, parsing, alignment, independent labels, split integrity, per-condition counts and transformation isolation, matched transformation draws, transformed-label preservation, activation completeness, probe provenance, the four required arms, expected result cells, and both classes plus per-class and pair metrics in every cell. **J0--J1** validate the lens instrumentation and the vocabulary contrast: unchanged forward logits, all lenses present and matching the model and the frozen vocabulary, finite scores, one matched member per polarity, one recorded orientation, train-only token discovery frozen before held-out scoring, and the controls having run.
- Every gate is **mechanical**. None requires a positive security-token result, and J0/J1 must pass when the vocabulary result is null. Lens fidelity is a diagnostic that warns per (layer, lens) and never blocks, so the report can distinguish *mechanically invalid*, *mechanically valid with weak lens fidelity*, *valid null*, and *positive above controls*.
- Failures exit nonzero with the gate name, expected/observed values, offending IDs, and the exact rerun command; invalid rows are never silently dropped.

## 4.4 Obfuscation protocol: atomic and cumulative

Reuse E9's four rewrites unchanged — consistent local renaming, opaque dead branches, mixed boolean-arithmetic encoding, control-flow flattening into a dispatch loop — and apply them **two ways**:

| kind | conditions | what it measures |
|---|---|---|
| baseline | `normalize` (AST round-trip) | unparse artifacts, so they are never confounded with a transformation |
| **atomic** | `rename_only`, `opaque_only`, `encode_only`, `flatten_only` | the **independent** effect of each transformation |
| **cumulative** | `rename_cumulative` → `rename_opaque` → `rename_opaque_encode` → `rename_opaque_encode_flatten` | **adversarial composition**, and the marginal cost of each added step |

`144 held-out programs × 9 transformed conditions = 1296 variants` per model.

- No new obfuscation algorithm, and no arbitrary pairwise or higher-order combinations: 8 arms, not the full 15-arm lattice. An interaction between, say, opaque predicates and flattening *without* renaming is not measurable and must not be implied.
- Both pair members use the same transformation draw and are revalidated at every condition.
- Each variant's transformations are **verified from its own AST** and must equal exactly what its condition declares; a draw that under- or over-delivers is redrawn (recorded), and a base that never satisfies its condition fails the gate rather than being dropped.
- Fit on clean training programs once; freeze before evaluating clean and all transformed held-out programs.

## 4.5 Baselines: four arms, all frozen, all transferred

| arm | features | bounds |
|---|---|---|
| `local_surface` | ±3 token ids at the anchor | "the identifier at the anchor gives it away" |
| `whole_program_lexical` | token uni/bigrams + char 3--5-grams over the **entire** program | "the generator left a textual shortcut somewhere in the file" |
| `embedding` | layer −1 state | token identity before any computation |
| `hidden_state` | probed layers ≥ 0 | the claim |

The whole-program arm is CPU-only, fitted on clean training programs alone with the vectorizer refitted inside every CV fold, and frozen through all conditions. It is given no AST, graph or taint features on purpose: it bounds the textual shortcut, it is not a competing program analysis, and a chance score there still does not rule out a predictor that runs the taint analysis itself.

---

# 5. Results: What Survives Obfuscation?

## 5.1 Clean flow becomes explicit with depth

At the sink-argument site:

| Measurement | DeepSeek 1.3B | DeepSeek 6.7B | StarCoder2-3B |
|---|---:|---:|---:|
| local surface baseline | 0.491 | 0.491 | 0.488 |
| **whole-program lexical baseline** | **0.464** | **0.464** | **0.464** |
| embedding layer | 0.482 | 0.482 | 0.482 |
| near 10% depth | 0.777 | 0.758 | 0.896 |
| near 25% depth | 0.991 | 0.979 | 0.952 |
| near 48% depth | 1.000 | 1.000 | 0.997 |

- The property is not recoverable from the measured local surface or anchor identity but becomes linearly explicit in contextual states.
- It replicates across scale and across two model families/pretraining corpora.
- Do not call the embedding row three independent results: the fixed identifier pool induces the same token partition under both tokenizers.
- Unlike E2, E15's floor is not exact against every possible whole-program reader. E15 measures transfer of the declared readout.

## 5.2 Independent transformation effects (atomic arms)

Report each transformation applied **alone**, at the layer nearest 48% depth, with cluster-bootstrap intervals, against the `normalize` reference row:

| Condition | DeepSeek 1.3B | DeepSeek 6.7B | StarCoder2-3B |
|---|---:|---:|---:|
| clean held-out | 1.000 | 1.000 | 1.000 |
| normalize | 1.000 | 1.000 | 1.000 |
| rename_only | 0.938 [0.889, 0.972] | 0.986 [0.965, 1.000] | 0.882 [0.833, 0.931] |
| opaque_only | **1.000** | **1.000** | **1.000** |
| encode_only | **1.000** | **1.000** | **1.000** |
| **flatten_only** | **0.688** [0.618, 0.750] | **0.667** [0.597, 0.729] | **0.660** [0.583, 0.736] |

- This is the table that licenses "transformation X costs Y **on its own**", and it settles the attribution the earlier draft had to hedge.
- **Two transformations are exactly free.** Opaque dead branches and MBA encoding score 1.000 in every model. Say this plainly: an adversary's surface noise buys nothing.
- **Flattening alone costs 0.312 / 0.333 / 0.340** — the whole collapse, from one transformation, replicated three times.
- Renaming costs 0.014--0.118, and its damage is one-sided (§5.5).

## 5.3 Adversarial composition and interactions (cumulative ladder)

Report the layer nearest 48% depth: 1.3B L11, 6.7B L15, StarCoder2 L15.

| Condition | DeepSeek 1.3B | DeepSeek 6.7B | StarCoder2-3B |
|---|---:|---:|---:|
| clean held-out | 1.000 | 1.000 | 1.000 |
| rename_cumulative | 0.958 | 0.951 | 0.910 |
| + opaque predicates | 0.944 | 0.965 | 0.931 |
| + MBA encoding | 0.951 | 0.965 | 0.938 |
| + control-flow flattening | 0.729 | 0.653 | 0.674 |

**The interaction is null.** Cumulative minus its atomic counterpart, for flattening:

| | DeepSeek 1.3B | DeepSeek 6.7B | StarCoder2-3B |
|---|---:|---:|---:|
| interaction (cumulative − atomic) | +0.042 | −0.014 | +0.014 |
| measured draw-noise floor | 0.021 | 0.035 | 0.028 |

The floor is `rename_only` vs `rename_cumulative` — the identical transformation under independent draws — so it is measured, not assumed. No model shows an interaction distinguishable from it.

- Include cluster-bootstrap intervals in the paper table or figure.
- Both lexical arms remain near chance in every condition (the local window measured 0.444--0.521 in the cumulative-only runs).
- Report `delta_previous` beside accuracy: it is the **marginal** cost of the step each condition adds, and the only column that licenses "adding X costs Y".
- Report `delta_atomic` — cumulative minus its atomic counterpart — as the **interaction**, with the `rename` row flagged as the draw-noise floor for the column (identical transformations, independent draws).
- The attribution test passes: `flatten_only` shows the loss alone, so the paper may say **"control-flow flattening breaks the readout"** without hedging.
- State the negative result explicitly — **composition adds nothing**. That is a claim about the threat model: an adversary gains nothing by stacking transformations.
- Do not rank models. The ordering is unstable.

## 5.4 Accuracy understates the cumulative collapse

At `flatten_only`:

| Diagnostic | DeepSeek 1.3B | DeepSeek 6.7B | StarCoder2-3B |
|---|---:|---:|---:|
| accuracy | 0.688 | 0.667 | 0.660 |
| unsafe / safe accuracy | 0.625 / 0.750 | **0.833 / 0.500** | 0.667 / 0.653 |
| matched pairs given same label | 0.514 | 0.556 | 0.458 |

Under the full cumulative condition:

| Diagnostic | DeepSeek 1.3B | DeepSeek 6.7B | StarCoder2-3B |
|---|---:|---:|---:|
| accuracy | 0.729 | 0.653 | 0.674 |
| unsafe / safe accuracy | 0.667 / 0.792 | **0.861 / 0.444** | **0.569 / 0.778** |
| matched pairs given same label | 0.431 | 0.583 | 0.486 |

- Similar pooled accuracies arise through opposite biases: 6.7B collapses toward “unsafe,” StarCoder2 toward “safe.” A constant predictor of either class scores exactly 0.500 on this balanced set.
- The parsimonious reading is that the frozen flow distinction largely breaks and residual accuracy reflects model-specific priors.
- Pair agreement and per-class error are therefore load-bearing security metrics.

## 5.5 Dangerous errors appear before flattening

- Under `rename_only`, StarCoder2 scores 0.882 pooled but **0.764 on unsafe versus 1.000 on safe** — the entire loss is false negatives, with control flow untouched.
- In its assignment chains it drops to 0.639 while `branch_merge` stays at 1.000.
- DeepSeek 6.7B loses almost nothing (0.986) and 1.3B loses symmetrically (0.917/0.958). Aggregate robustness cannot characterize audit risk.

## 5.6 Structure, not API spelling, predicts fragility

Report rename/flatten accuracy:

`rename_only` / `flatten_only`:

| Structure | DeepSeek 1.3B | DeepSeek 6.7B | StarCoder2-3B |
|---|---:|---:|---:|
| direct | 1.000 / 0.722 | 1.000 / 0.639 | 0.972 / 0.639 |
| branch merge | 1.000 / 0.694 | 1.000 / 0.833 | 1.000 / 0.806 |
| helper | 0.972 / 0.639 | 0.972 / 0.667 | 0.917 / 0.556 |
| assignment chain | **0.778** / 0.694 | 0.972 / **0.528** | **0.639** / 0.639 |

- Assignment chains are most fragile across all three models, with helper boundaries next.
- Branch merges being more robust than two-step alias chains argues against a simple “longer path is harder” explanation.
- Sink-family ordering does not replicate, supporting flow rather than dangerous-API memorization.
- Treat the cause of assignment-chain fragility as an open mechanistic question.

## 5.7 The whole-program lexical baseline

Report the four arms side by side, condition by condition: `hidden_state` at the reported layer, `embedding`, `local_surface`, `whole_program_lexical`.

- The purpose is to bound **generator-level textual shortcuts**, which the ±3-token window structurally cannot see.
- **Result: near chance everywhere** — 0.464 on clean-training CV and 0.465--0.535 across all ten held-out conditions in all three models, against 1.000 for the hidden state. The hidden-state result is not something a wider textual window would have recovered, and the clean-code number is not a corpus artifact.
- Either way, state the boundary: this does not bound a predictor that performs the taint analysis itself, and no baseline in this paper does.

## 5.8 The difference is not in the model's own vocabulary (E15-C)

An observational readout experiment on the same states, answering a different question from every table above: **is the safe→unsafe difference expressed in output-aligned coordinates?** The answer is no, in all three models.

| clean held-out, R-lens, matched depth | DeepSeek 1.3B | DeepSeek 6.7B | StarCoder2-3B |
|---|---:|---:|---:|
| concept token surviving the tokenizer | `" vulnerable"` | `" vulnerable"` | `" unsafe"` |
| held-out sign consistency | **0.153** | 0.403 | 0.694 |
| permutation p | 0.000 | 0.004 | 0.008 |
| verdict | **inverted** | stable non-security | stable non-security |

- Method in one paragraph: three readouts (logit lens, J-lens, R-lens) with **R-lens declared primary in code before any result**; orientation fixed once as `score_unsafe − score_safe`; token discovery on clean *training* pairs only, frozen to disk before any held-out pair is scored; a small security lexicon fixed in advance and validated per model, with omissions recorded and nothing substituted.
- **Why this is a real null.** (i) The three lenses agree — pairwise cosine of their mean vocabulary-difference vectors is 0.75--0.97 — so it is not an artifact of the primary-lens choice. (ii) It is not token identity: the embedding-layer contrast is null (p = 0.71--0.81) and 75% of pairs share the same anchor token at the sink argument. (iii) Something *does* replicate — frozen discovered tokens reappear in the held-out top-k at 0.875 / 0.750 / 0.875 against 0.000--0.031 for random control tokens — but those tokens are semantically arbitrary (`" ?"`, `" liber"`, `"OrNull"`). (iv) Under flattening the vocabulary contrast degrades alongside the probe, so trained and output-aligned auditability are lost together.
- **The claim this licenses:** linear decodability and expression in a model's own output vocabulary are **different properties**, and E15 exhibits the first without the second. Do not write "the model represents unsafe" anywhere.
- Report the inverted 1.3B sign explicitly rather than burying it; a two-sided test would have mislabelled it as a positive result.
- Diagnostic caveat to state once: R-lens relevance conservation is 1.0001 / 0.9993 on the deepseek models but **0.154 on StarCoder2**, so that model's lens numbers carry a fidelity caveat. This warns, it does not invalidate — the gates are mechanical and both passed.

## 5.9 The boundary is general, not security-specific

The companion E9 run is complete on all three models, which retires a caveat the earlier draft had to carry.

| best layer per task | rename | flatten |
|---|---:|---:|
| binding | 0.708 -- 0.883 | 0.527 -- 0.615 |
| def--use | 0.689 -- 0.864 | 0.402 -- 0.545 |
| **E15 security flow** | **0.882 -- 0.986** | **0.660 -- 0.688** |

- The same transformations break binding and def--use the same way, and the security readout is *at least as robust* as the primitives it rests on.
- Supported claim: **structural obfuscation breaks frozen linear readouts of program relations, security ones included.** Not: "security representations are specifically fragile."

## Section 5 takeaway

> Clean-code success substantially overstates the reliability of a frozen security-flow readout. Two of four semantics-preserving transformations cost nothing at all; renaming already produces one-sided false negatives; and **control-flow flattening alone destroys the matched distinction in all three models**, with composition adding no measurable interaction. What survives is each model's class prior, not flow information — and the distinction was never expressed in the models' own output vocabulary to begin with.

---

# 6. Mechanistic Evidence and Security Implications

## 6.1 Why the audit is plausible

- Binding and def--use are prerequisites for following source-to-sink flow.
- Their controlled profiles show contextual computation forming relational information rather than preserving only identifiers.
- E9 supplies the companion result that binding/def--use also survive renaming better than flattened control structure in DeepSeek models.
- Until StarCoder2 E9 is complete, do not claim a security-specific weakness rather than a general limitation of frozen linear semantic readouts.

## 6.2 Causal evidence is supporting, not the E15 claim

Summarize E13 in one paragraph or appendix table:

- Cross binding structure with value assignment, fitting a rank-1 interchange on one arm and testing on an arm requiring the opposite answer-token movement.
- All H0--H5 pass on DeepSeek-Coder 6.7B.
- The learned edit transports the installed binding on 100% of held-out rows in both arms.
- A mean-difference baseline transports at about 76% but at higher intervention dose.
- Interpretation: binding is causally potent at one controlled site, layer, model, and construction.
- Boundary: E13 does not establish causal use of E15's source-to-sink property.

## 6.3 Implications for trustworthy code auditing

- Report false-negative robustness, not only balanced accuracy.
- Preserve matched safe/unsafe pairs and measure when both receive the same label.
- Test frozen transfer; retraining on each obfuscation answers a different question.
- Stratify by flow structure because API-level aggregates hide the boundary.
- Treat an internal readout as an audit instrument whose validity can shift under adversarial transformations, not as a stable semantic oracle.

---

# 7. Limitations and Conclusion

## Limitations

- **Synthetic scope:** Python only, three sink families, four structures, short generated programs, and no real malware or naturalistic vulnerability corpus.
- **Narrow policy:** the label is untrusted flow to a code-bearing argument; vulnerability also depends on path feasibility, environment, sink semantics, and mitigations outside this benchmark.
- **Declared surface floor:** the local-window and whole-program-lexical baselines are near chance but are not a construction-exact lower bound against whole-program *analysis*. A reader that performs the taint analysis itself would score 1.0.
- **Eight arms, not the lattice:** four atomic and four cumulative conditions, not all 15 combinations. An interaction between two transformations *without* renaming is not measurable and must not be implied.
- **Probe dependence:** "flattening breaks the readout" is a claim about a frozen linear readout at one position. A failing probe does not prove the model lost the information — though the parallel failure of the output-aligned readout (§5.8) is consistent with real loss.
- **Vocabulary alignment is not causal use, and the null is bounded.** E15-C is observational. The null does not show the information is absent — only that it is not expressed in the candidate output-aligned coordinates this design can search. R-lens relevance conservation is 0.154 on StarCoder2 (against ~1.000 on both deepseek models), so that model's lens numbers carry a fidelity caveat.
- **The vocabulary search is restricted.** A J/R-lens vector costs one vector-Jacobian product per candidate token, so the candidate pool is selected by a full-vocabulary logit-lens pass; a direction only the J-lens or R-lens would surface, on a token outside that pool, cannot be discovered.
- **Security lexicon coverage is tokenizer-dependent.** Which security words are single vocabulary tokens differs by model, so the concept-token sets are not identical across models and the cross-model comparison is correspondingly weaker there.
- **Probe dependence:** frozen linear failure can mean a changed representation basis rather than loss of all flow information.
- **Ground truth:** the static checker is valid for this generator, not a general taint analyzer; results are accepted only when instrumented execution agrees.
- **Layer matching:** compare matched relative depth; StarCoder2's closest point is 52% rather than 48%.
- **Observational security result:** no intervention establishes causal use of source-to-sink state.
- **Fixed threat model:** transformations are not adaptively optimized against the model or readout.

## Conclusion structure

1. Code models construct scope-sensitive binding, def--use, and source-to-sink information beyond both measured surface floors, local and whole-program.
2. This supports an accurate clean-code security-flow readout in three models, but frozen reliability depends strongly on presentation.
3. Opaque predicates and expression encoding are tolerated completely; renaming is tolerated in aggregate yet already produces one-sided false negatives for particular structures and models.
4. **Control-flow flattening alone erases the matched flow distinction**, with residual pooled accuracy reflecting opposing class priors — and composing it with the other transformations adds nothing measurable.
5. Whether that information is expressed in the model's own vocabulary basis is a separate question from whether a probe can decode it, and this paper reports the answer it measured — including if that answer is null.
6. Trustworthy evaluation of AI code auditors must treat semantics-preserving obfuscation applied both atomically and cumulatively, structural strata, pair collapse, and false-negative rates as first-class validity tests.

End on the security lesson:

> A code model can contain a highly decodable semantic relation on clean source and still provide an unreliable audit surface under adversarially equivalent code. Security evaluation must test the invariance of the representation it intends to trust.

---

# Main-text visual and table plan

## Figure 1: from controlled semantics to security flow

- Left: the one-character binding pair and exact 0.500 floor.
- Right: clean E15 sink-flow accuracy across normalized depth for all three models, with surface/embedding controls.
- Purpose: show the audit builds on a controlled semantic phenomenon rather than API-token recognition.

## Figure 2: frozen audit under obfuscation

- Accuracy across clean, normalize, rename, opaque, encode, and flatten at matched relative depth for all three models.
- Include the surface arm as a thin chance reference and mark that the ladder is cumulative.

## Figure 3: pooled accuracy versus operational failure

- Unsafe and safe accuracy under rename and flatten.
- Fraction of matched pairs receiving the same prediction.
- Highlight StarCoder2 assignment-chain false negatives under renaming.

## Table 1: benchmark and validity controls

- Sink families, flow structures, train/held-out counts, independent ground truth, pair invariant, frozen transfer, and gates S0--S3.

## Table 2: structure-specific robustness

- Use the rename/flatten table from Section 5.6.
- Keep sink-family results in the appendix because the cross-model null is the relevant finding.

## Table 3: atomic versus cumulative

- One row per (atomic, cumulative) pair: atomic accuracy, cumulative accuracy, the interaction, and the marginal step cost.
- Flag the `rename` row as the draw-noise floor.
- This is the table the "flattening" sentence in the abstract has to be checked against.

## Table 4: the four arms

- `hidden_state`, `embedding`, `local_surface`, `whole_program_lexical`, per condition.

## Figure 4: vocabulary-space contrast

- Held-out sign consistency by layer for the three lenses, with the permutation and mismatched-pair control bands drawn behind them.
- Inset or companion panel: cosine between each condition's mean vocabulary-difference vector and the clean held-out one, so the atomic/cumulative story and the vocabulary story are read on the same axis.
- If the result is null, this figure still belongs in the paper — it is what makes "no stable vocabulary-aligned security concept was found" a measurement rather than an absence.

## Appendix plan

- Full per-layer and per-model E15 tables with bootstrap intervals.
- Per-family results and complete per-class rates.
- Gate definitions, failures, and provenance digests.
- Transformation examples and label-preservation checks.
- Full binding/def--use/context robustness results from the attached paper.
- Compact E13 factorial and gate table.
- Archived intervention failures only if needed to motivate control design.
