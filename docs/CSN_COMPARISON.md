# Synthetic versus human-written CodeSearchNet

Stage 280 implements a new comparison. It does not reuse the approximate labels
or discard provenance as the old `data/real/csn_python_200.jsonl` importer did.
There is no language-model training in this experiment; only linear probes are
fitted. CruxEval is not used as the real-code population.

## Run on the cluster

Copy the updated checkout, including `src/csn_comparison.py`,
`scripts/280_csn_comparison.py`, and `tests/test_csn_comparison.py`. Commands below
are **bash**, run from the repository root in the existing Python 3.11 cluster
environment. Activate a GPU allocation only for extraction. No remote job has
been submitted by the assistant.

```bash
bash
.venv/bin/python -m pip install -r requirements-cruxeval.txt
.venv/bin/python -m pytest tests/test_csn_comparison.py -q

MODEL=deepseek-coder-1.3b
RUN="results/csn_comparison/${MODEL}/seed42"
```

Start with 1.3B. Repeat with `MODEL=deepseek-coder-6.7b` and
`MODEL=starcoder2-3b` in distinct run directories once the pipeline and coverage
are satisfactory. Each model needs its own tokenization and preparation.

### 1. CPU preparation, with network access to dataset and tokenizer

```bash
.venv/bin/python scripts/280_csn_comparison.py prepare \
  --model "$MODEL" \
  --real-programs 2000 --synthetic-pairs 1000 \
  --seed 42 --max-length 2048 --min-groups 5 \
  --output "$RUN/prepared"

cat "$RUN/prepared/report.md"
cat "$RUN/prepared/coverage.csv"
```

The importer resolves CodeSearchNet's Python training parquet revision to an
immutable commit, then shuffles the whole dataset's indices. It selects up to
2,000 **eligible hard-case functions**, so it may analyze substantially more
than 2,000 candidates. It preserves repository, source URL, dataset index,
reference graph and exclusions. It does not execute downloaded programs.
The dataset may be large on its first download; subsequent runs use the cache.

Inspect these CPU outputs before GPU work:

- `coverage.csv`: real/synthetic train/validation/test counts for each property;
- `audit.jsonl`: included/excluded functions, unsupported cases, unresolved
  external references, and legacy/reference disagreements;
- `programs.jsonl`: exact prompts, full reference graph and provenance for
  spot-checking labels;
- `report.md`: active/skipped properties and estimated uncompressed activation
  storage in bytes. Reserve space for this plus temporary float32 probe matrices.

Both classes and at least five repository/pair groups per partition are required
to activate a task. Five is a mechanical floor, not a power calculation. Aim
for at least 20 real test repositories per task. If coverage is inadequate,
increase `--real-programs` into a **new** preparation directory. Do not lower
coverage thresholds to obtain a desired result. Lexical binding may be skipped
because naturally occurring nested shadowing is scarce. Skipped means missing
evidence, never failed decoding. Def-use must be active before extraction.

Preparation refuses overwrite. To reproduce the exact dataset later, supply
`--revision` with the saved parquet commit. If the cluster cannot download data,
use `--input-jsonl /path/to/full-csn-export.jsonl`; each row must contain
`repository_name`, `func_code_url`, and `whole_func_string`. The legacy
200-function export is intentionally rejected.

### 2. GPU extraction

After reviewing coverage and the label audit:

```bash
.venv/bin/python scripts/280_csn_comparison.py extract \
  --prepared "$RUN/prepared" \
  --output "$RUN/activations" \
  --device cuda --dtype float16 --resume
```

This captures raw embedding output and every transformer block output, with an
independent final-normalization check. It extracts both domains once, rejects
truncation and nonfinite states, records the model revision, and resumes complete
programs only under the same configuration. Inputs and labels never appear as
an appended answer. Only code enters the model.

### 3. CPU probe fitting and evaluation

```bash
.venv/bin/python scripts/280_csn_comparison.py evaluate \
  --prepared "$RUN/prepared" \
  --store "$RUN/activations" \
  --output "$RUN/comparison" \
  --max-train-pairs 1000 --max-iter 20000 \
  --bootstrap 1000 --seed 42 --resume

cat "$RUN/comparison/report.md"
```

Optional `--scratch /existing/scratch/directory` places temporary feature
matrices on a suitable disk. The default solver is `lbfgs`, tested on this
pipeline including the exactly balanced synthetic embedding null; `saga` is
available explicitly but can fail to converge on that null. Fits must converge. A changed iteration budget,
solver or sampling budget requires a new evaluation directory, but can reuse
the completed preparation and activation artifacts. A failed result is not
certified by a passing `gates.json`.

Preparation, extraction and evaluation each have integrity-checked gates.
The prepared corpus never changes during extraction or probe fitting.

## What is measured

| Property | Positive label | Negative label |
|---|---|---|
| `defuse_edge` | candidate is the unique reference reaching definition | another earlier same-name definition |
| `lexical_binding` | candidate and use resolve to the same scope/name | an earlier same-name definition in a different scope |

Every real queried use contributes equal positive and same-name negative counts.
Different-name easy negatives are excluded. Def-use excludes ambiguous and
forward-reaching sites. Lexical binding requires an unambiguous scope/name and
also excludes sites with forward-reaching definitions. The reference is beniget;
the legacy visit-order analyzer does not define these labels. Class/global/nonlocal
constructs and direct `eval`/`exec` are excluded in this initial supported subset.
Unresolved external names are reported rather than assigned guessed labels.

Reassignment in one scope is not lexical shadowing. The new synthetic paired
construction flips whether an inner assignment shadows an outer parameter.
The tracked outer-definition/use pair, local token windows and distance remain
identical across opposite labels; the pair differs by one source character and
one token. Both tasks are certified against the same reference graph. This is
a new synthetic corpus; do not mix its numbers with old stage-20 maxima.

Real splits use a fixed repository hash; synthetic splits use a fixed pair-group
hash. The allocation is 60% training, 20% validation and 20% test. Complete
balanced units are sampled from training data to match training row counts
across the two domains. Real repository counts can still differ from synthetic
pair counts and are reported. Exact and conservatively alpha-normalized AST
duplicates are removed before fitting; this is not a guarantee against every
possible semantic clone or model-pretraining contamination.

All four arms run:

| Train probe on | Evaluate on |
|---|---|
| Synthetic training | Synthetic validation/test |
| Synthetic training | Real validation/test |
| Real training | The same real validation/test rows |
| Real training | The same synthetic validation/test rows |

The model is frozen. Standardization, surface vocabularies, probe fitting and
shuffling use training rows only. Each arm has a train-matched local surface
reader, shuffled-label probe, embedding control and train-majority classifier.
Probe regularization is fixed at the repository default `C=0.1`; there is no
test-driven hyperparameter search. Layers are selected using **source-domain**
validation balanced accuracy; ties choose the lowest transformer layer. The
same source-selected layer is then used for both test domains. Synthetic-to-real
transfer uses no real labels for fitting, scaling, threshold or layer selection.

## Read these outputs

- `selected_test.csv`: primary comparison, validation-selected layer per arm;
- `selected_shadow_strata.csv`: selected-layer results restricted to genuinely
  shadowed uses and other uses, with separate counts; tiny strata remain exploratory;
- `all_layers.csv`: validation/test curves and controls, including embedding −1;
- `paired_real_test.csv`: same-layer synthetic-trained minus real-trained
  accuracy, with paired repository-bootstrap intervals;
- `predictions/*.csv.gz`: exact row IDs, partitions, predictions and controls;
- `training_membership.csv`: which balanced rows actually trained each probe;
- `meta.json`, `gates.json`: settings, revisions, hashes and completion status.

Uncertainty resamples real repositories and synthetic pair instances, conditional
on fitted models. For seed sensitivity, repeat preparation and evaluation with
seeds 43 and 44 in new directories. Do not select the best seed. Changing the
model changes tokenization, so active populations may differ; check coverage
before interpreting cross-model differences.

The primary question is whether controlled synthetic decodability transfers to
supported hard cases from human code, and whether a real-trained probe recovers
more information on exactly those cases. This is a restricted static-name
comparison, not exact arbitrary Python execution analysis. The real data are
function snippets, so repository context is missing. Synthetic generalization
is across held-out instances of this construction, not unseen template families.

**Taint is not measured by this runner.** A trustworthy taint comparison requires
its own source/sink policy, library summaries and labels. This run also does not
test the proposed noise mechanism causally or establish model causal use.
