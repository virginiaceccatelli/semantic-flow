# Testing semantic decodability and verifier-guided fine-tuning

Prepared 2026-10-08 from the saved repository results. This is an experimental proposal, not a completed training run.

## Assessment

A reliable synthetic/real gap is a finding about the generalization of a specified readout. It does not establish that semantic information is absent or impossible to recover. Failure of a synthetic-trained probe on real programs is different from failure of a probe trained on real programs. Even the latter is conditional on labels, samples, read positions, features and probe capacity. Noise, representational shifts, shortcuts, missing context and label errors remain competing explanations.

The available stage-20 results are separately fitted within-dataset grouped CV, not frozen synthetic-to-real transfer. Binding and def-use remain highly predictable overall on CodeSearchNet Python; the same-name/different-binding negative subset is much harder. These results support a hard-case generalization problem but not a blanket real-code decoding failure. Real taint and real context-matched binding scores are absent. Shadowing is not a separately measured balanced task here.

The raw stage-20 tables are `results/tables/static_probes_{model}_{dataset}.csv`. Their per-layer maxima are descriptive, not validation-selected test estimates, and stratum denominators, per-program predictions and uncertainty cannot be reconstructed from them. Whole-task group counts must not be used as hard-stratum sample counts. Stage 280 ([definition survival](DEF_SURVIVAL.md)) is the controlled synthetic-versus-real transfer experiment that replaces this comparison.

The same-name/different-binding subset contains negatives only: its score is specificity, not balanced binary accuracy. A score near 0.5 does not imply chance. Real-code aggregate positive prevalence is about 21%, so majority accuracy is already about 79%. Embedding-only performance is also high on pooled tasks. Accuracy above the local lexical baseline does not alone isolate contextual semantic computation. Probe controls matter; see [Hewitt and Liang](https://aclanthology.org/D19-1275/).

## Establish what the labels mean before training

Keep three targets distinct:

1. Lexical binding: which scope/symbol an occurrence denotes. Use compiler-consistent symbol tables and stable scope IDs. Reassignments to one symbol are not automatically different lexical bindings.
2. Reaching definition: which assignment can supply a use, either as a static may-reach set or for a specified execution. State which semantics applies.
3. Taint: whether a specified source can influence a sink under a declared propagation/sanitizer policy. Keep static may-taint, must-taint, and concrete execution taint separate.

The legacy `src/graphs/dfg_extractor.py` selects the latest prior AST-visit definition from current/global scopes. That is not a general exact Python analyzer. Audit labels around branching, joins, loops, closures, globals/nonlocals and unbound locals before reusing them as rewards. This code inspection identifies a validity risk; it does not measure the error rate in the saved corpus.

Python's [symtable interface](https://docs.python.org/3/library/symtable.html) provides compiler scope classifications; source-occurrence resolution and reaching definitions still need implementation. [CodeQL's Python data-flow documentation](https://codeql.github.com/docs/codeql-language-guides/analyzing-data-flow-in-python/) describes modeled flow and taint analyses, not an exact oracle for unrestricted runtime behavior.

For the first training experiment, restrict to supported language constructs and libraries, record unknown/unsupported cases, and report coverage. Use exact scope resolution on that subset; use a CFG analysis with explicit joins for static reaching sets, or independently instrumented execution for concrete flow. Concrete traces certify those inputs only. Specify heap aliasing, library summaries, sanitizers and explicit versus implicit flow for taint. Cross-check the verifier against hand-reviewed cases and an independent implementation. Do not silently convert unsupported cases into negative labels.

Audit causal visibility too. A decoder state at a use cannot depend on later source tokens. Python's lexical classification can depend on a later assignment in the same function. For each target, either establish that its answer is determined by the visible prefix or read after the complete function at a query token. Report use-site and end-of-program measurements separately.

## Resolve the baseline first

Build one audited task definition across synthetic and natural programs, with matched candidate sets and token alignment. Use program/repository-disjoint train, validation and locked test splits; keep clones, renamings and mutations together, and separate synthetic template families. Split before augmentation. Reserve separate data pools for model fine-tuning, probe fitting and final evaluation.

Measure the full probe matrix: synthetic-to-synthetic, synthetic-to-real, real-to-real, and real-to-synthetic. The first domain is probe training, the second probe evaluation; the language model stays frozen. Fit linear probes first, then a small capacity-controlled nonlinear probe as a diagnostic, using identical budgets across domains. Include learning curves, train-majority, embedding-only, lexical/position/string-equality and shuffled-label controls. Select layer, regularization and thresholds on validation only. Report balanced accuracy, AUROC/AUPRC where appropriate, specificity and sensitivity on matched hard examples, group counts, and source-cluster bootstrap intervals.

The existing `docs/cruxeval/PROBES.md` describes A within-corpus and B transfer machinery, but does not supply a completed measured transfer result in the inspected artifacts. Do not substitute a lens null or a branch-outcome result for binding-probe transfer; they measure different properties. Audit benchmark provenance before labeling any additional benchmark human-written real code.

To test the noise hypothesis, vary irrelevant context length, competing scopes, def-use distance and control-flow complexity independently, while holding the queried semantic fact fixed. Compare synthetic, natural code and semantics-preserving simplified natural code. Freeze the probe for transfer tests and also retrain within each condition. Recovery after removing unrelated context is evidence for interference; recovery only after probe retraining suggests a readout shift. Neither alone proves information was erased. Verify that each transformation actually preserves the target and report label/support changes.

## Training implementation

Start with DeepSeek-Coder 1.3B as a pilot, then replicate on 6.7B and StarCoder2 if the pilot is positive. Preserve the original model/tokenizer revisions and exact extraction conventions. Record package versions, adapter configuration, precision and data manifests. Compare base and adapted models in the same numerical precision; if QLoRA is necessary, include a matching quantized-base control.

Generate compact structured answers from the verifier instead of unconstrained prose explanations. Each training record should contain `program_id`, repository/template group, source hash, task, query occurrence/candidate IDs, optional concrete input, verifier version, assumptions, status, expected answer and a bounded list of required intermediate facts. Only code, queries and necessary inputs enter the prompt. Oracle labels and expected steps stay in completion targets/reward metadata.

An intermediate-fact sequence could identify the relevant scope, enumerate candidate definitions, apply assignment/branch transfer rules, and return the resolved definition or taint state. At control-flow joins, allow sets rather than forcing an arbitrary single answer. IDs must be local opaque references rather than label-bearing identifiers; randomize candidate order. Require a fixed schema and required fields, reject duplicates or contradictions, and score missing fields as incorrect. Use equivalence-aware comparison when several valid proofs exist.

**Stage 1: SFT.** Train on verifier-produced answers and intermediate facts, with completion-only loss. Start with LoRA rank 16 or 32 across attention and MLP linear layers, BF16 where supported, learning rate around 1e-4 for adapters and one epoch; tune on validation. These are pilot starting values, not established optima. Choose sequence length from measured corpus lengths and reject truncations that remove required context. Begin with roughly 10,000 audited records and increase only if coverage or learning curves justify it. Use synthetic diversity plus independently split natural training programs. Maintain a synthetic-only arm if the claim concerns transfer without real-code supervision.

**Stage 2: verifier-reward RL.** Continue from the SFT checkpoint with GRPO and a deterministic reward callback. Sample 4–8 completions per query, start adapter learning rate around 5e-6, and explicitly set a modest KL coefficient such as 0.02 to a frozen SFT reference; tune learning rate/KL on validation. Track effective prompt and completion batch sizes. Cache oracle analysis by source hash and score generated structured facts cheaply. Do not train a learned reward model when a trusted deterministic verifier is available.

One proposed scalar reward for a fixed required set of K facts is:

`R = valid_schema * (0.7 * exact_final_answer + 0.3 * verified_required_facts / K)`.

The denominator is fixed by the task, not by how many facts the model chooses to emit. Invalid schema or contradictory/duplicate required entries receives zero. The final-answer-only ablation uses `R = valid_schema * exact_final_answer`. Define partial-credit semantics before training, and test attempted exploits (omitted steps, repeated easy facts, fabricated IDs, answer-only shortcuts). Normalize each task before combining tasks and log reward components separately.

Aggregating checked steps into a sequence reward is process-informed RL; standard scalar GRPO does not assign a distinct advantage to each intermediate step. For true step-local credit, implement an environment with step transitions and per-step rewards/returns, or a tested trainer modification. Start with the scalar version plus step-level SFT to keep the first experiment interpretable. Check the fraction of generation groups with zero reward variance: all-wrong or all-correct groups supply no relative reward signal. Adjust curriculum or improve SFT rather than assuming RL is learning from them.

TRL provides [SFTTrainer](https://huggingface.co/docs/trl/sft_trainer) and [GRPOTrainer](https://huggingface.co/docs/trl/grpo_trainer) with custom rewards. Its documented GRPO default KL coefficient is zero, so set it explicitly if desired. Pin the installed release and verify its configuration fields before writing the launcher. [Process-supervision research](https://openai.com/index/improving-mathematical-reasoning-with-process-supervision/) motivates checking intermediate outputs, but does not establish internal representational alignment for this code task.

## Checkpoints and necessary comparisons

Save the base, post-SFT, and RL checkpoints at 10%, 25%, 50%, 75% and 100% of the planned optimizer budget. Run intermediate diagnostics on validation; keep final test results out of checkpoint selection. Use at least three training seeds if feasible.

Minimum arms are base; answer-only SFT; structured-step SFT; structured-step SFT plus final-answer-reward RL; and structured-step SFT plus final-and-step-reward RL. Compare against additional SFT at a comparable compute budget, report both training tokens and rollout cost, and keep training sources matched. If resource constrained, run these on 1.3B before scaling. Mix ordinary code training only as a separately documented retention intervention, applied consistently across the compared arms.

At every checkpoint:

- Evaluate task behavior and verified step correctness on held-out natural programs, including same-name hard cases, with general code completion/execution retention tests.
- Extract activations from unchanged raw-code prompts before any generated explanation or label appears. Also measure a separate query-conditioned setting if useful. Never compare a base use-site state with an adapted answer-token state as if they were the same measurement.
- Refit probes on independent probe-training data at each checkpoint to measure current decodability. Separately reuse the original frozen probe to measure stability of the old readout. Include the synthetic-to-real matrix and embedding/surface controls at each checkpoint.
- Use source-cluster paired bootstrap comparisons on the same final examples. Report effect sizes and uncertainty, per-property coverage, and performance by length/scope/flow complexity. Select layers on validation or preregister a middle-layer band to limit post-hoc selection.
- If decodability and behavior improve, repeat a controlled binding interchange/ablation on audited held-out pairs with random, dose-matched and answer-direction controls. Extend the existing DAS design only where valid counterfactual donors exist.

Do not reward the evaluation probe's score or optimize a direct probe loss in the primary experiment. That would make increased decodability an explicitly trained objective and weaken the inference that verifier-guided task training improved it spontaneously. Such a loss can be a clearly labeled engineering ablation.

## Interpreting outcomes

Better fresh probes plus better held-out behavior supports improved semantic accessibility and capability. A failed old probe with successful fresh probes indicates representation drift. Better structured explanations alone establishes output compliance, not faithful internal reasoning. Better probes without better behavior establishes decodability, not causal use. Gains confined to prompted explanations establish task-conditioned accessibility, not automatic representation during ordinary code reading. RL must beat the SFT and compute controls before attributing the result specifically to RL.

Even successful fine-tuning cannot prove that the original failure was caused by noise: training may construct new features, change their organization, or improve the model's underlying computation. The controlled complexity experiment is the direct test of the proposed mechanism.
