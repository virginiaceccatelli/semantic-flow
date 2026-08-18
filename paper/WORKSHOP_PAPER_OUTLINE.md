# Workshop paper outline: security-relevant semantic representations under obfuscation

## Working title

**Do Code Models See Through Obfuscation? Auditing Security-Relevant Semantic Representations**

Alternative: **Code Models Track Security-Relevant Data Flow, but Control-Flow Flattening Breaks the Readout**

Use the first title for AI4GOOD. It makes the security question explicit while retaining semantic representation as the scientific premise.

## One-sentence paper claim

- Across DeepSeek-Coder 1.3B and 6.7B and StarCoder2-3B, middle-layer states distinguish whether untrusted input reaches a code-bearing sensitive argument despite chance-level local-surface, whole-program-lexical, and embedding controls. A frozen readout remains at 0.868--0.979 through renaming, opaque predicates, and mixed-boolean-arithmetic encoding, but falls to 0.562--0.632 after cumulative control-flow flattening, often by collapsing toward model-specific class priors rather than retaining usable flow information.

## The argument, in five steps

The paper makes one direct argument, and every section serves one step of it:

1. **Controlled binding and def--use experiments show that code models construct semantic program relations beyond the declared surface reader.** (§3, from the existing E2/E3 results.)
2. **E15 shows those representations support a security-relevant source-to-sensitive-sink readout.** (§4--5.1.)
3. **Atomic and cumulative obfuscation reveal which transformations independently disrupt that readout and which failures arise through composition.** (§5.2--5.3. Atomic conditions give independent effects; cumulative conditions measure adversarial composition; their difference is the interaction. Flattening is named as a cause only where `flatten_only` supports it.)
4. **A contrastive vocabulary-space experiment tests whether the safe-to-unsafe difference is expressed in the model's own output-aligned coordinates as a recognizable security-relevant concept.** (§5.8. Observational; a null is a reportable result.)
5. **Pooled accuracy can conceal dangerous false negatives and matched-pair collapse.** (§5.4--5.5, and the evidence-standard contribution.)

This remains a **semantic-representation and robustness** paper motivated by trustworthy program analysis in security. It is not malware classification, not end-to-end vulnerability detection, and not a new causal-intervention project.

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
- Do not say flattening alone causes the failure **unless the `flatten_only` arm supports it**. The three-model table is cumulative-only: levels 1--3 cost at most 0.13, and adding flattening causes the large marginal drop. With the atomic arms in hand, the vocabulary is fixed: an **independent transformation effect** is what an atomic condition's `delta_clean` shows; a **marginal effect** is `delta_previous` along the cumulative chain; an **interaction** is `delta_atomic`, cumulative minus its atomic counterpart. Anything not supported by an atomic row is written as a cumulative effect.
- Do not read the `rename` row of the interaction table as an interaction: `rename_only` and `rename_cumulative` are the same transformation under independent draws, so that row is the measured draw-noise floor for the column.
- Do not describe the vocabulary-space experiment as causal, and do not summarise it as "the model represents unsafe" because a security token appeared in a top-k list. A semantic reading needs train-only discovery, held-out replication, a consistent orientation, an effect above the permutation *and* mismatched-pair controls, stability across identifier roles, and evidence not reducible to the differing sink-argument token. A **null there is a result** and is compatible with the probe succeeding.
- Do not claim the whole-program lexical baseline pins the floor against every predictor. It bounds generator-level *textual* shortcuts; a reader that performs the taint analysis is still out of scope.
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
- Report clean local surface 0.488--0.491, whole-program lexical (near chance), embedding 0.482, and middle-layer decoding 0.997--1.000 across three models.
- Report the atomic arms (each transformation alone) and the cumulative ladder separately, and say which of the two the "flattening" sentence rests on.
- Report levels 1--3 at 0.868--0.979 and cumulative flattening at 0.562--0.632.
- State the operational asymmetry: StarCoder2's renaming accuracy is 0.868 overall but 0.750 on unsafe programs and 0.222 on unsafe assignment chains.
- One clause, at most, on the vocabulary-space experiment, and only its outcome — whether the safe/unsafe difference is expressed in the model's own output-aligned coordinates, or is not. Do not spend abstract space on the method.
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
3. **Obfuscation audit, decomposed.** Frozen-readout evaluation across three models reveals a replicated boundary between levels 1--3 and cumulative flattening, plus class- and structure-specific failures hidden by pooled accuracy — and the atomic arms separate what each transformation does alone from what composition adds.
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
| Independent transformation effect | E15 atomic arms | `normalize` reference row; per-variant AST verification | measurable by design; run pending |
| Compositional (interaction) effect | atomic vs. cumulative difference | rename row as draw-noise floor | measurable by design; run pending |
| Generator-level textual shortcut ruled out | whole-program lexical arm | frozen transfer, vectorizer fitted on training text only | measurable by design; run pending |
| Vocabulary-space expression of the property | E15-C contrast | permutation, mismatched pairs, embedding/token-identity, role strata, random and Gram-matched lenses | observational; outcome open, null reportable |
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
| whole-program lexical baseline | | | |
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
| normalize | | | |
| rename_only | | | |
| opaque_only | | | |
| encode_only | | | |
| flatten_only | | | |

- This is the table that licenses "transformation X costs Y **on its own**".
- Whether flattening may be named as the cause of the collapse is decided **here**, by the `flatten_only` row, and nowhere else.
- If `flatten_only` is survivable and the cumulative flatten condition is not, the finding is an **interaction**, and §5.3 is where it belongs.

## 5.3 Adversarial composition and interactions (cumulative ladder)

Report the layer nearest 48% depth: 1.3B L11, 6.7B L15, StarCoder2 L15.

| Condition | DeepSeek 1.3B | DeepSeek 6.7B | StarCoder2-3B |
|---|---:|---:|---:|
| clean held-out | 1.000 | 1.000 | 1.000 |
| rename | 0.931 | 0.965 | 0.868 |
| opaque predicates | 0.951 | 0.979 | 0.938 |
| MBA encoding | 0.938 | 0.958 | 0.924 |
| cumulative flattening | 0.632 | 0.562 | 0.569 |

- Include cluster-bootstrap intervals in the paper table or figure.
- Both lexical arms remain near chance in every condition (the local window measured 0.444--0.521 in the cumulative-only runs).
- Report `delta_previous` beside accuracy: it is the **marginal** cost of the step each condition adds, and the only column that licenses "adding X costs Y".
- Report `delta_atomic` — cumulative minus its atomic counterpart — as the **interaction**, with the `rename` row flagged as the draw-noise floor for the column (identical transformations, independent draws).
- Describe a replicated boundary: levels 1--3 are mostly survivable at the sink site, while adding flattened dispatch causes roughly 0.30--0.40 additional loss. Attribute that loss to flattening only if `flatten_only` shows it alone; otherwise call it a cumulative effect and say so in the sentence, not in a footnote.
- Do not rank models. The ordering is unstable.

## 5.4 Accuracy understates the cumulative collapse

| Diagnostic | DeepSeek 1.3B | DeepSeek 6.7B | StarCoder2-3B |
|---|---:|---:|---:|
| accuracy | 0.632 | 0.562 | 0.569 |
| unsafe / safe accuracy | 0.583 / 0.681 | 0.792 / 0.333 | 0.417 / 0.722 |
| predicted unsafe | 0.451 | 0.729 | 0.347 |
| matched pairs given same label | 0.514 | 0.653 | 0.444 |

- Similar pooled accuracies arise through opposite biases: 6.7B collapses toward “unsafe,” StarCoder2 toward “safe,” while 1.3B loses much of the pair distinction without the same bias.
- The parsimonious reading is that the frozen flow distinction largely breaks and residual accuracy reflects model-specific priors.
- Pair agreement and per-class error are therefore load-bearing security metrics.

## 5.5 Dangerous errors appear before flattening

- Under renaming, StarCoder2 scores 0.868 pooled but 0.750 on unsafe versus 0.986 on safe programs.
- For its assignment chains, unsafe accuracy is 0.222 and safe accuracy 0.944: the readout misses 78% of vulnerable members after only consistent renaming.
- DeepSeek 1.3B's similarly sized pooled loss is symmetric at 0.931/0.931. Aggregate robustness cannot characterize audit risk.

## 5.6 Structure, not API spelling, predicts fragility

Report rename/flatten accuracy:

| Structure | DeepSeek 1.3B | DeepSeek 6.7B | StarCoder2-3B |
|---|---:|---:|---:|
| direct | 1.000 / 0.611 | 1.000 / 0.611 | 1.000 / 0.806 |
| branch merge | 1.000 / 0.806 | 1.000 / 0.750 | 0.972 / 0.694 |
| helper | 0.917 / 0.556 | 1.000 / 0.528 | 0.917 / 0.444 |
| assignment chain | 0.806 / 0.556 | 0.861 / 0.361 | 0.583 / 0.333 |

- Assignment chains are most fragile across all three models, with helper boundaries next.
- Branch merges being more robust than two-step alias chains argues against a simple “longer path is harder” explanation.
- Sink-family ordering does not replicate, supporting flow rather than dangerous-API memorization.
- Treat the cause of assignment-chain fragility as an open mechanistic question.

## 5.7 The whole-program lexical baseline

Report the four arms side by side, condition by condition: `hidden_state` at the reported layer, `embedding`, `local_surface`, `whole_program_lexical`.

- The purpose is to bound **generator-level textual shortcuts**, which the ±3-token window structurally cannot see.
- Near-chance across all conditions ⇒ the hidden-state result is not something a wider textual window would have recovered, and the clean-code number is not a corpus artifact.
- Above chance in any condition ⇒ attach the caveat to every number in that condition, in the text, not in an appendix.
- Either way, state the boundary: this does not bound a predictor that performs the taint analysis itself, and no baseline in this paper does.

## 5.8 Is the difference in the model's own vocabulary? (E15-C)

An observational readout experiment on the same states, reported as its own subsection because it answers a different question from every table above: **is the safe→unsafe difference expressed in output-aligned coordinates?**

- Three readouts — logit lens, J-lens, R-lens — with **R-lens primary, declared before any result** (the target includes early and middle layers, where E14 measured the J-lens's raw-autograd backward to be least faithful).
- Orientation fixed once: `delta(pair, token) = score_unsafe − score_safe`.
- Token discovery on **clean training pairs only**, frozen to disk before held-out scoring; a small security lexicon fixed in advance and validated per model (omissions recorded, nothing substituted — the tokenizers disagree about which security words are single tokens).
- Report: held-out semantic mass and sign consistency; the permutation and mismatched-pair controls; top-k enrichment of the frozen set against random control tokens; the cosine between each obfuscation condition's mean vocabulary-difference vector and the clean one; and agreement among the three lenses by layer.
- **Interpretations this can support**: explicit security vocabulary; stable non-security vocabulary (output-aligned flow information without verbalisation); probe succeeds while all lenses fail (linearly decodable without demonstrated vocabulary alignment); R-lens succeeds where J-lens fails (the intermediate-layer limitation); all readouts collapse under structural obfuscation (loss of both trained and output-aligned auditability).
- **A null is a result**, and it is the one to write plainly if the checklist does not hold. Vocabulary alignment is not causal use in any case.

## Section 5 takeaway

> Clean-code success substantially overstates the reliability of a frozen security-flow readout: surface transformations can preserve average accuracy while selectively hiding vulnerable programs, and cumulative control-flow flattening largely destroys the matched distinction across three models.

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
- **Cumulative vs. atomic:** the cumulative ladder adds flattening after prior transformations. The atomic arms are what make an attribution possible; where they do not support one, the paper says "cumulative effect".
- **Vocabulary alignment is not causal use.** E15-C is observational: a vocabulary direction that separates the two members does not show the model uses it, and a null there does not show the information is absent — only that it is not expressed in the candidate output-aligned coordinates this design can search.
- **The vocabulary search is restricted.** A J/R-lens vector costs one vector-Jacobian product per candidate token, so the candidate pool is selected by a full-vocabulary logit-lens pass; a direction only the J-lens or R-lens would surface, on a token outside that pool, cannot be discovered.
- **Security lexicon coverage is tokenizer-dependent.** Which security words are single vocabulary tokens differs by model, so the concept-token sets are not identical across models and the cross-model comparison is correspondingly weaker there.
- **Probe dependence:** frozen linear failure can mean a changed representation basis rather than loss of all flow information.
- **Ground truth:** the static checker is valid for this generator, not a general taint analyzer; results are accepted only when instrumented execution agrees.
- **Layer matching:** compare matched relative depth; StarCoder2's closest point is 52% rather than 48%.
- **Observational security result:** no intervention establishes causal use of source-to-sink state.
- **Fixed threat model:** transformations are not adaptively optimized against the model or readout.

## Conclusion structure

1. Code models construct scope-sensitive binding, def--use, and source-to-sink information beyond the measured local surface cues.
2. This supports an accurate clean-code security-flow readout in three models, but frozen reliability depends strongly on presentation.
3. Renaming, opaque predicates, and expression encoding are mostly tolerated in aggregate, yet can already produce dangerous false negatives for particular structures and models.
4. Adding control-flow flattening largely erases the matched flow distinction, with residual pooled accuracy reflecting opposing class priors.
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
