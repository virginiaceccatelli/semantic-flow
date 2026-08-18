# Workshop paper outline: security-relevant semantic representations under obfuscation

## Working title

**Do Code Models See Through Obfuscation? Auditing Security-Relevant Semantic Representations**

Alternative: **Code Models Track Security-Relevant Data Flow, but Control-Flow Flattening Breaks the Readout**

Use the first title for AI4GOOD. It makes the security question explicit while retaining semantic representation as the scientific premise.

## One-sentence paper claim

- Across DeepSeek-Coder 1.3B and 6.7B and StarCoder2-3B, middle-layer states distinguish whether untrusted input reaches a code-bearing sensitive argument despite chance-level local surface and embedding controls. A frozen readout remains at 0.868--0.979 through renaming, opaque predicates, and mixed-boolean-arithmetic encoding, but falls to 0.562--0.632 after cumulative control-flow flattening, often by collapsing toward model-specific class priors rather than retaining usable flow information.

## Paper position

- This is a paper about **auditing the robustness of AI code analysis**, not malware detection or interpretability for its own sake.
- The deployment-motivated question is whether a defender can recover a security-relevant program property when an adversarial author changes surface form while preserving behavior.
- Binding and def--use provide the controlled foundation: the models compute program relations beyond the declared surface reader.
- E15 is the application-facing contribution: it tests untrusted source-to-sensitive-sink flow under obfuscation.
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
- Do not say flattening alone causes the failure. Level 4 is cumulative: levels 1--3 cost at most 0.13, and adding flattening causes the large marginal drop. A flatten-only arm remains the clean attribution test.
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
- Obfuscation results: 2.0 pages.
- Mechanistic evidence and implications: 0.8 page.
- Limitations and conclusion: 0.7 page.

---

# Abstract

Write the abstract as a security-auditing argument:

- AI-assisted code auditors may perform well on ordinary source while failing when an adversarial author preserves behavior but conceals data and control flow.
- Accuracy on clean code or probe decodability does not show that the relevant flow exists beyond identifiers and local syntax, nor that a frozen audit survives obfuscation.
- Give the controlled foundation in one sentence: token-matched binding pairs pin the declared surface and embedding readers to 0.500 while middle-layer decoding approaches 0.99.
- Introduce E15 precisely: 480 clean Python programs; three sensitive-sink families; four flow structures; matched safe/unsafe programs differing only at the sink argument; 336 clean training programs and 144 held-out programs per condition; independent dynamic and static labels; frozen evaluation through an execution-verified ladder.
- Report clean surface 0.488--0.491, embedding 0.482, and middle-layer decoding 0.997--1.000 across three models.
- Report levels 1--3 at 0.868--0.979 and cumulative flattening at 0.562--0.632.
- State the operational asymmetry: StarCoder2's renaming accuracy is 0.868 overall but 0.750 on unsafe programs and 0.222 on unsafe assignment chains.
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
3. **Obfuscation audit.** Frozen-readout evaluation across three models reveals a replicated boundary between levels 1--3 and cumulative flattening, plus class- and structure-specific failures hidden by pooled accuracy.
4. **Evidence standard.** Security readouts should be evaluated with explicit surface floors, held-out obfuscations, pair-collapse diagnostics, and per-class errors.

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
- Four hard gates validate corpus size and balance, pair confinement, parsing, alignment, independent labels, split integrity, transformed-label preservation, activation completeness, probe provenance, required controls, expected result cells, and both classes in every cell.
- Failures exit nonzero with expected/observed values and offending IDs; invalid rows are never silently dropped.

## 4.4 Obfuscation protocol

Reuse E9's cumulative ladder unchanged:

0. AST normalization;
1. consistent local renaming;
2. opaque dead branches;
3. mixed boolean-arithmetic encoding;
4. control-flow flattening into a dispatch loop.

- Both pair members use the same transformation draw and are revalidated at every level.
- Fit on clean training programs once; freeze before evaluating clean and all transformed held-out programs.

---

# 5. Results: What Survives Obfuscation?

## 5.1 Clean flow becomes explicit with depth

At the sink-argument site:

| Measurement | DeepSeek 1.3B | DeepSeek 6.7B | StarCoder2-3B |
|---|---:|---:|---:|
| surface baseline | 0.491 | 0.491 | 0.488 |
| embedding layer | 0.482 | 0.482 | 0.482 |
| near 10% depth | 0.777 | 0.758 | 0.896 |
| near 25% depth | 0.991 | 0.979 | 0.952 |
| near 48% depth | 1.000 | 1.000 | 0.997 |

- The property is not recoverable from the measured local surface or anchor identity but becomes linearly explicit in contextual states.
- It replicates across scale and across two model families/pretraining corpora.
- Do not call the embedding row three independent results: the fixed identifier pool induces the same token partition under both tokenizers.
- Unlike E2, E15's floor is not exact against every possible whole-program reader. E15 measures transfer of the declared readout.

## 5.2 Levels 1--3 are largely survivable; cumulative flattening is not

Report the layer nearest 48% depth: 1.3B L11, 6.7B L15, StarCoder2 L15.

| Condition | DeepSeek 1.3B | DeepSeek 6.7B | StarCoder2-3B |
|---|---:|---:|---:|
| clean held-out | 1.000 | 1.000 | 1.000 |
| rename | 0.931 | 0.965 | 0.868 |
| opaque predicates | 0.951 | 0.979 | 0.938 |
| MBA encoding | 0.938 | 0.958 | 0.924 |
| cumulative flattening | 0.632 | 0.562 | 0.569 |

- Include cluster-bootstrap intervals in the paper table or figure.
- The surface arm remains between 0.444 and 0.521 in every condition.
- Describe a replicated boundary: levels 1--3 are mostly survivable at the sink site, while adding flattened dispatch causes roughly 0.30--0.40 additional loss.
- Do not rank models. The ordering is unstable.

## 5.3 Accuracy understates the level-4 collapse

| Diagnostic | DeepSeek 1.3B | DeepSeek 6.7B | StarCoder2-3B |
|---|---:|---:|---:|
| accuracy | 0.632 | 0.562 | 0.569 |
| unsafe / safe accuracy | 0.583 / 0.681 | 0.792 / 0.333 | 0.417 / 0.722 |
| predicted unsafe | 0.451 | 0.729 | 0.347 |
| matched pairs given same label | 0.514 | 0.653 | 0.444 |

- Similar pooled accuracies arise through opposite biases: 6.7B collapses toward “unsafe,” StarCoder2 toward “safe,” while 1.3B loses much of the pair distinction without the same bias.
- The parsimonious reading is that the frozen flow distinction largely breaks and residual accuracy reflects model-specific priors.
- Pair agreement and per-class error are therefore load-bearing security metrics.

## 5.4 Dangerous errors appear before flattening

- Under renaming, StarCoder2 scores 0.868 pooled but 0.750 on unsafe versus 0.986 on safe programs.
- For its assignment chains, unsafe accuracy is 0.222 and safe accuracy 0.944: the readout misses 78% of vulnerable members after only consistent renaming.
- DeepSeek 1.3B's similarly sized pooled loss is symmetric at 0.931/0.931. Aggregate robustness cannot characterize audit risk.

## 5.5 Structure, not API spelling, predicts fragility

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
- **Declared surface floor:** the local-window baseline is near chance but is not a construction-exact lower bound against whole-program analysis.
- **Cumulative ladder:** level 4 adds flattening after prior transformations; run the supported flatten-only arm before cleanly attributing the effect to flattening.
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
5. Trustworthy evaluation of AI code auditors must treat semantics-preserving obfuscation, structural strata, pair collapse, and false-negative rates as first-class validity tests.

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

- Use the rename/flatten table from Section 5.5.
- Keep sink-family results in the appendix because the cross-model null is the relevant finding.

## Appendix plan

- Full per-layer and per-model E15 tables with bootstrap intervals.
- Per-family results and complete per-class rates.
- Gate definitions, failures, and provenance digests.
- Transformation examples and label-preservation checks.
- Full binding/def--use/context robustness results from the attached paper.
- Compact E13 factorial and gate table.
- Archived intervention failures only if needed to motivate control design.
