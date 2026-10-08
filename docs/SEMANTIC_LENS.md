# A semantic property lens for real programs

Stage 270 is independent of BranchExec and U-Space/J-lens. It learns a small
supervised classifier for **the outcome of a particular `if` condition on a
particular dataset-supplied input**. Ground truth comes from execution; no words such
as “semantic” or “branch” are used as latent-space anchors. The readout describes
representation under explicit questioning, not spontaneous execution while
reading, causal use, or a percentage of “real reasoning.”

## Data contract and provenance

Use the existing ExecSem raw JSONL interface: each problem has `task_id`,
`correct_submissions` and `incorrect_submissions` (each with `language: "py3"`
and `code`), and `test_cases_preview` (each with string `input`). Optional `id`
fields preserve submission/test identifiers; deterministic identifiers are used
when absent. Outputs, descriptions, and correctness verdicts are never rendered
in prompts. All submissions and all unique available preview inputs are used;
there is no correctness-based sampling, integer-output restriction, input
mutation, new test generation, or synthetic direction fitting. Dataset-supplied
inputs may themselves have been generated upstream.

“Preview” is the available source field, not a claim of complete judge-test
coverage or human authorship. Real, unchanged contestant programs are required;
generated inputs are allowed. The audit records `input_origin` as
`human_authored`, `generated`, or `mixed`, with supporting evidence. Unknown
origins are rejected. Legacy audits with `original_human_authored_tests: true`
remain supported; conflicting old/new assertions are rejected.

For the supplied `exec-sem/codecontests-plus-execsem` slice, use the new `audit`
command below. It writes the file hashes and documents the provenance already
reviewed for this study:

- The slice card says programs were selected from CodeContests+ `ccplus_1x`.
- The [CodeContests+ paper, section 4](https://aclanthology.org/2025.findings-emnlp.299.pdf)
  describes the underlying authentic contestant submission records.
- The [upstream dataset card](https://huggingface.co/datasets/ByteDance-Seed/Code-Contests-Plus)
  describes `1x` inputs as pre-generated using the Generator–Validator system.

The helper is specific to this reviewed slice: it requires its README, checks
its identifying markers, hashes it, and stores the references and review scope.
This is a documentation-based provenance record, not independent authentication
of each submission's author. It does not assert that inputs were human-authored.
An identical rerun leaves the audit unchanged; conflicting existing audits
require a new destination path. For another dataset, supply its own audit:

```json
{
  "schema_version": 2,
  "reviewer": "Reviewer of source documentation",
  "reviewed_at": "YYYY-MM-DD",
  "sources": {
    "train": {
      "sha256": "SHA256 of the raw training records file",
      "source_reference": "Versioned dataset record or publication URL",
      "human_written_programs": true,
      "input_origin": "generated",
      "program_origin_evidence": "Evidence for real contestant code, with reference",
      "test_origin_evidence": "Evidence identifying the input generation procedure"
    },
    "test": {
      "sha256": "SHA256 of the raw held-out records file",
      "source_reference": "Versioned dataset record or publication URL",
      "human_written_programs": true,
      "input_origin": "generated",
      "program_origin_evidence": "Evidence for real contestant code, with reference",
      "test_origin_evidence": "Evidence identifying the input generation procedure"
    }
  }
}
```

Input origin is retained in cases and coverage reports. The current pipeline
uses supplied previews unchanged; if more inputs are added in a later study,
select them by execution coverage independently of probe performance and record
the new generation procedure. Do not conflate generated programs with generated
inputs. Unit-test fixtures remain implementation checks, never research evidence.

The FPS-chrF slice is selected for lexical diversity, not a random sample of
programs. First run this pilot; a later replication should use non-overlapping,
randomly sampled upstream problems. This is a limitation to report, not a reason
to replace the current inputs.

Pass the original ExecSem `rows.jsonl` as `--splits` to preserve an existing
partition. It must cover every source problem; a partial pilot map fails closed.
If there is no prior partition, omit it: source train problems are split 80/20
with seed 0, and source test problems remain test. All submissions of a problem
stay together. Every member of any cross-split whitespace-normalized code
collision is removed. This does not detect all near-clones or pretraining
contamination; conclusions must retain that limitation.

## Execution and coverage

Each program/input runs twice unchanged and twice with AST instrumentation in
the existing container-only ExecSem sandbox. Runtime/image identity is recorded.
The tracer truth-tests each original condition once, preserving short-circuit
semantics, and records a separate bounded trace. Counts saturate at 2 because
only exactly-once conditions are eligible. Module docstrings and future imports
are preserved; the model always receives the unchanged original source.

Keep only successful, repeat-stable runs whose instrumented and original stdout
match. Unreached conditions, repeated evaluations, failed runs, missing traces,
changed outputs, overlarge outputs and unstable executions are not false labels.
A branch that selects the false arm without an `else` still has outcome false.
This is about condition outcome after it is reached, not general reachability.
Instrumentation is observational tooling, not a formal equivalence proof;
introspection-sensitive programs and hostile self-modification are limitations.
Two repeats also do not prove determinism. The sandbox retains its existing
stdout limit (4096 bytes) and resource restrictions; exclusions are reported.

`trace/coverage.json` is the first empirical deliverable. It includes retained
counts, exclusions, branches/problems changing outcome across supplied inputs, and the number of
implicit opposite-outcome pairs. Continue only if every split contains a
branch changing outcome across supplied inputs. This is a feasibility gate, not a statistical power
guarantee. This run never generates extra inputs silently when coverage fails.

## Runbook

Use the repository's configured Python environment with its normal model/test
dependencies. Run tracing on an approved Docker/Podman/Singularity host with the
image already installed; there is no host-execution fallback. Example paths
below refer to user-provided files, not files bundled with the repository.

```bash
python scripts/270_semantic_lens.py audit --train data/execsem/records/train.jsonl --test data/execsem/records/val.jsonl --dataset-card data/execsem/README.md --audit data/execsem/provenance.json
python scripts/270_semantic_lens.py prepare --train data/execsem/records/train.jsonl --test data/execsem/records/val.jsonl --audit data/execsem/provenance.json
python scripts/270_semantic_lens.py trace --runtime docker --image python:3.11-slim
python scripts/270_semantic_lens.py extract --model deepseek-coder-1.3b --device cuda --max-tokens 4096
python scripts/270_semantic_lens.py fit --shuffled-repeats 5
python scripts/270_semantic_lens.py evaluate --bootstrap 1000
python scripts/270_semantic_lens.py report
```

For Singularity use `--runtime singularity --image /absolute/path/python311.sif`.
The audit command writes `--audit`. Every experimental stage accepts `--out` (default `results/semantic_lens/deepseek-coder-1.3b`).
Keep that value identical across stages. Completed stages are manifest-verified
no-ops; changed configuration, implementation, source artifacts or dependencies
require a new output directory. Tracing and extraction checkpoint individual
cases, verify checkpoint digests on resume, and reject upstream mutations.
Extraction caches full/code-only/input-only states in one `.npz` per case.
Disk usage is about `cases × 3 × layers × hidden_dim × 4 bytes` before compression.
Dense probe fitting loads train and validation arrays for one condition at a time.

## Readout and controls

The full prompt is original source, dataset-supplied input, a line/column reference to
the queried `if`, and a fixed true/false question ending in `Answer:`. Columns
are zero-based UTF-8 byte offsets, matching Python AST coordinates. The readout
is the final prompt token at each zero-based transformer block output, before
final normalization. It has seen all input tokens and is question-conditioned.
No generated reasoning trace is required.

Extract code-only and input-only controls from the same model and population.
Drop a case from every condition when any prompt is too long; never truncate.
Recompute natural-pair coverage after filtering. Identical code-only prompts
must yield exactly identical states, providing a structural-zero check.
Generate up to 8 tokens greedily for the full prompt; parse only an initial
`true` or `false` word, preserving the raw response and all unparsed cases.
These are model answers, distinct from probe predictions.

Train weighted logistic regressions with C in {0.01, 0.1, 1, 10}, standardizing
using weighted train statistics only. Every problem receives equal total
training weight. Select the layer/C lexicographically by validation pair
ranking, AUROC, then balanced accuracy. Ties keep the earliest layer/smallest C.
Persist the raw-coordinate covector/intercept; threshold at score zero
(probability 0.5). No fitting or selection reads test activations or test labels.

Fit the same controls independently with the same selection protocol. Fit five
full-state shuffled-label repeats using seeds 0–4, with independently repeated
layer/C selection. Fit full-prompt and input-only lexical classifiers using a
union of token 1–2 grams and character 3–5 grams (25k features each, case-sensitive,
train-only vocabulary and IDF) and the same C grid and problem weights.

Primary pair ranking compares every true case against every false case at the
same program/site; ties score 0.5. Average within branch, then program, then
problem. Also report both-pair-outcomes-correct, problem-weighted balanced
accuracy and AUROC. Paired bootstrap resamples whole test problems, including
all their cases, with 1,000 replicates, seed 0. Report pointwise 95% intervals,
valid-replicate counts, and full-minus-baseline differences; these are not
simultaneous intervals. Fewer than two test problems yields no confidence
intervals. Model/probe error agreement includes unparsed model answers and
never gates the empirical population on correct model behavior.

## Reading the report

`report/report.html` is a standalone, escaped source view of **all** held-out
cases, not selected successes. Each branch has original source highlighting,
dataset-supplied inputs, observed outcome, probe prediction, signed score, probe
probability, and the raw model answer. Naturally contrasting tests get a
conditional explanation of agreement or disagreement. `report/report.md` gives
the metrics, uncertainty, selection and coverage; JSON/JSONL retain exact values.

A favorable result supports input-dependent control-flow representation beyond
the tested baselines. A failed readout does not establish lexical matching,
and a correct readout does not establish causal reliance. Report negative and
inconclusive results. Generalization is limited to the audited source domain,
eligible branches and available tests. Dynamic data-flow and causal interventions
are follow-up work, not implicit claims of this implementation.

Run the dedicated tests with:

```bash
python -m pytest tests/test_semantic_lens.py -q
```

Small test fixtures check implementation only and are never research evidence.
