# Security monitor: develop on generated functions, report on real code

This is a new extension of SemFlow, separate from the completed binding paper.
The defining evaluation is **held-out real Python code**. Generated functions
exist to fit and validate instruments, prompts and thresholds. Their results
are always labelled `SYNTHETIC DEVELOPMENT ONLY — NOT FINAL RESULTS`.
No real-code performance or successful repair is claimed by this implementation.

## The one task

For a designated variable argument at a sink, determine whether externally
supplied input reaches it. The model answers `1` for external input and `0`
for constants. In reports, `unsafe` is shorthand for this positive flow label,
**not proof of an exploitable vulnerability**. Start with direct assignments
and scope resolution. Sanitization correctness, arbitrary call graphs,
branch-dependent multiple reaching definitions, and whole repositories are
outside the first version.

The monitor asks whether internal evidence predicts an incorrect flow judgment.
It does not claim to expose the model's complete reasoning or intrinsic confidence.

## Implemented

* A tokenizer-independent JSONL format shared by generated and real functions.
  Each record contains source, entry point/external parameters, designated sink
  argument span, candidate assignment spans, and independently verified labels.
  Assignment states are read after the complete RHS, not before its value is visible.
* Sixteen controlled Python scaffolds. Each base crosses binding (outer/inner)
  with external-input location (outer/inner), neutral/reassuring/suspicious names,
  and clean/unused-scope variants. The generator verifies all 24 variants using
  an identity marker and an inert sink. Only internally generated code executes.
* Four disjoint scaffold splits: `probe_train`, `monitor_train`, `calibration`,
  `synthetic_test`. All variants of a source group stay together. These small
  scaffold changes establish instrument transfer; they do not establish real-world generality.
* One frozen pairwise linear probe per selected layer: `[use, definition,
  use-definition, abs(use-definition)]`. Train on clean `probe_train` functions.
  At inference, rank any supplied set of at least two candidate assignments.
  Top margin, score, normalized entropy and layer disagreement become features.
  Softmax over candidate scores is a support summary, not a calibrated binding probability.
* The **published J-lens** from `src/workspace_lens/`, reused without refitting.
  It reads the same use state as the probe and records full-vocabulary ranks for
  danger (`unsafe/dangerous/vulnerable`), safety (`safe/secure/protected`),
  uncertainty (`unsure/uncertain/unknown`), certainty (`sure/certain/known`),
  and generic code words (`string/number/index`). Bare/space-prefixed and initial
  capital spellings count only when they tokenize as one token. Missing words
  are recorded, never truncated. A missing entire family aborts extraction.
  Features are mean normalized log ranks across available words in each family.
  Generic words are a fixed diagnostic family, **not a frequency-matched causal control**.
* Five feature comparisons, each with the same fitted error-predictor procedure:
  output only; output + probe; output + J-lens; output + both; output + ordinary
  logit-lens vocabulary features. The last comparison checks whether Jacobian
  transport adds value. No feature uses the true reaching definition or flow label.
* Logistic error predictors fit only on `monitor_train`. Separate sigmoid
  calibration and a review threshold use `calibration`. No class weighting of
  error predictors alters the observed error base rate. Default review budget: 10%.
* Frozen evaluation on `real_test`, or predictions without labels on `inference`.
  Artifacts pin the model, prompt, layers, vocabulary and lens identity. Changing
  those requires a new development run; evaluation never silently refits.
* Inspection JSONL with source spans, per-layer selected definitions, candidate
  scores, J/logit top words and ranks, final answer, estimated error risk and flag.
  This can feed a future UI without coupling the science to a web application.

## Evaluation and claims

The primary operational quantity is recall of **incorrect 0/constant judgments**
when reviewing the highest-risk 10% of examples. Selection uses scores only.
Also report total error recall, accepted error rate, errors among accepted
constant judgments, false alarms, AUROC, Brier score and per-layer binding accuracy.
Always show both the ranking budget and the fixed calibration threshold: under
real distribution shift, the latter's review rate may exceed 10%.

Paired 95% bootstrap intervals compare combined features against output, probe,
and J-lens by resampling whole source groups. Set group IDs to a coarser unit
(for example a repository) if that is the intended independent sampling unit.
All variants of one original function must at minimum share one group.
Zero eligible errors yield undefined recall, not successful prediction.

The output decision is forced-choice between exact single-token continuations
for `0` and `1`, not unrestricted generation. Trailing/leading whitespace is
resolved using tokenizer structure alone, then both prefix continuations are
verified. Answer token mass and full-vocabulary
argmax agreement are reported so formatting failure is visible.
The fit refuses if clean behavioral balanced accuracy is below 0.75 or either
error-predictor/calibration split has fewer than 10 correct or 10 incorrect
decisions. It writes `fit_diagnostics.json` first. A failure means the next step
is development of the task/prompt or more development examples, not lowering a
gate to produce an apparent result. Any revised prompt is a new version before
real-test inspection.

Danger vocabulary and uncertainty vocabulary remain separate. A highly ranked
`unsafe` may reflect a confident danger judgment, while `unsure` may reflect
uncertainty about either class. Improvement in error prediction is evidence for
useful monitoring, not on its own proof of semantic reliance. Controlled naming
contrasts and later interventions supply that interpretation.

## Cluster: run the development stage

Use the existing cluster `uq` environment and checkout, as in `jobs/common.csh`.
There are no additional Python dependencies beyond the repository and the
already-vendored published `jlens` package. If needed, install that package once:

```bash
/scratch_NOT_BACKED_UP/NOT_BACKED_UP/vceccate/envs/uq/bin/python -m pip install --no-deps -e third_party/jacobian-lens
```

The job reuses `results/workspace_lens/deepseek-coder-6.7b/j-lens/lens.pt`, its
metadata, and the passing stage-202 gate. It does **not** rerun the expensive
J-lens fit. If artifacts are missing, use the existing `make lens-check`,
`make lens-fit-dry`, `make lens-fit`, and `make lens-validate` targets for that
model; inspect the cost estimate first. This is the existing E19 prerequisite,
not an additional monitor experiment.

Run from the cluster checkout on an available GPU:

```bash
screen -L -Logfile security-monitor-v1.log -dmS security-monitor-v1 \
  env MODEL=deepseek-coder-6.7b MONITOR_RUN=v1 MONITOR_BASES=8 \
  csh jobs/security_monitor.csh
```

Defaults: 3,072 functions (768 per split), layers 6/11/20, bfloat16,
2,048-token maximum, one forward per function, one model resident. A 24 GB
GPU is a reasonable starting allocation for 6.7B; actual memory depends on
sequence lengths and model-loading configuration. CPU fitting holds the small
anchor cache in RAM. No GPU-hour estimate has been measured for this new stage.

The job stops on failures. Extraction resumes per function with atomic cache
writes. Fitted tools and reports are never overwritten. Use a new `MONITOR_RUN`
for a changed design; when resuming, `MONITOR_BASES` does not regenerate an
existing dataset. Review the validated split counts in the log.

Outputs:

```text
data/security_monitor/dev-v1.jsonl
results/security_monitor/deepseek-coder-6.7b/v1/
  features/                    # resumable .npz files + manifest + completion marker
  frozen/fit_diagnostics.json   # capability / error-count checks, even after a failed fit
  frozen/monitor.pkl            # trusted local sklearn models; load only your own artifacts
  frozen/monitor.json           # readable configuration, training provenance, thresholds
  synthetic_test/report.md      # development only
  synthetic_test/metrics.json
  synthetic_test/predictions.jsonl
  synthetic_test/inspection.jsonl
  synthetic_test/paired_changes.json
```

## Real-code handoff: same extractor, frozen tools

Choose real self-contained functions where the designated argument resolves to
one identifiable assignment. Preserve actual source, names and comments.
Include supporting context above the use as needed within the fixed token
budget. Do not remove difficult examples without recording exclusions.
The importer does not automatically infer security truth from function names,
a vulnerability dataset label, or arbitrary repository code execution.

Prepare a JSON list of reviewed annotations. `source_file` is relative to this
JSON file. Line numbers refer to the exact file, not a reformatted function.
`candidate_lines` identify simple assignments, whose full spans are found by AST.
`sink_line` must contain exactly one call, and its selected positional argument
must be a variable name. No side-effecting or nested call on that line is supported
in this first importer. Direct JSONL supports the same source-span contract.

```json
[
  {
    "id": "projectA-functionX-original",
    "group_id": "projectA-functionX",
    "source_file": "functionX.py",
    "sink_line": 12,
    "argument_index": 0,
    "candidate_lines": [3, 7],
    "reaching_definition_line": 3,
    "unsafe": true,
    "query": {
      "entrypoint": "process",
      "external_parameters": ["user_input"],
      "sink": "subprocess.run (its first positional argument)"
    },
    "variant": "original",
    "provenance": {
      "repository": "repository URL or stable identifier",
      "revision": "exact commit",
      "path": "original/repository/path.py",
      "original_sha256": "SHA256 of the original extracted source",
      "verification": "Reviewer and evidence establishing the reaching definition and external-input flow"
    }
  }
]
```

This is a schema illustration, not a real example or a runnable file. Keep
originals and all verified transformations under the same group. Retain the
original hash when annotating variants. Real-test labels require independent
verification; structural schema validation does not prove their correctness.
For a meaning-preserving variant, optionally set `reference_id` to the original
example's ID in the same dataset. This enables paired answer-flip rates and
changes in predicted risk. The validator requires the same group, split and flow
label; behavioral equivalence still requires independent verification.
For unlabeled application use `"split": "inference"` and omit the two labels.

With the cluster environment activated:

```bash
python scripts/210_security_monitor.py import-real \
  --spec data/security_monitor/real/annotations.json \
  --output data/security_monitor/real-test-v1.jsonl

python scripts/210_security_monitor.py extract \
  --dataset data/security_monitor/real-test-v1.jsonl \
  --model deepseek-coder-6.7b \
  --lens-dir results/workspace_lens/deepseek-coder-6.7b \
  --layers 6,11,20 --max-tokens 2048 \
  --output results/security_monitor/deepseek-coder-6.7b/real-v1/features

python scripts/210_security_monitor.py evaluate \
  --dataset data/security_monitor/real-test-v1.jsonl \
  --features results/security_monitor/deepseek-coder-6.7b/real-v1/features \
  --bundle results/security_monitor/deepseek-coder-6.7b/v1/frozen/monitor.pkl \
  --split real_test \
  --output results/security_monitor/deepseek-coder-6.7b/real-v1/evaluation
```

No synthetic labels, clean donor, or known correct binding is needed to compute
monitor predictions. Candidate definitions and the query annotation are still
required: automatic project-wide extraction is not implemented. Synthetic-to-real
calibration may fail; the frozen real-test Brier score and review rate measure that.
Do not tune on this real test set. A future real development set would require
an explicitly separate protocol and new unseen final evaluation.

## Pending, deliberately

1. **Curated real examples and final results.** The importer/evaluator are ready;
   a target codebase and independently reviewed annotations are still needed.
2. **DAS repair.** Existing `src/models/das.py` remains available, but no new
   repair fit is launched. First identify repeatable behavioral failures in
   development. Then learn a component on separate development counterfactuals
   and test frozen clean-donor repair on real pairs, including crossed assignments
   and matched interventions. The new cache already preserves use states and
   positions for this stage. A successful donor interchange does not by itself
   establish a unique semantic direction or a donor-free repair policy.

## Local verification

```bash
python -m pytest -q tests/test_security_monitor.py
python scripts/210_security_monitor.py --help
```

CPU tests cover executed generation, split leakage, Unicode/source alignment,
label-free prompts/inference, actual tiny-model J-lens extraction, calibration,
frozen transfer, cache identity, and reporting. They establish implementation
correctness, not effectiveness on DeepSeek or real security decisions.
