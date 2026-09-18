# CruxEval: preflight and probes

**Status:** stage 230 preflight and stages 231–234 A/B probes have run on the
cluster. Stages 235–238 implement the published full-Jacobian J-lens, matched
R-lens, full-vocabulary CruxEval output readout, and live-model causal erasure.
Stages 239–241 implement execution-grounded CruxEval value counterfactuals,
held-out low-rank DAS interchange with controls, and its report. No large
J/R-lens or DAS run has been performed locally. See [PROBES.md](PROBES.md) and
[MECHANISTIC.md](MECHANISTIC.md) for exact cluster commands. The optional
CruxEval obfuscation ladder remains unimplemented.

## What ran

`datasets.load_dataset("cruxeval-org/cruxeval")` returned the `test` split:
800 rows with `id`, `code`, `input`, and `output`. The pilot uses the first five
rows plus 45 rows sampled without replacement, seed 42. Tokenizer:
`deepseek-ai/deepseek-coder-6.7b-base`, loaded through SemFlow's guarded loader.
The input prompt is exactly `code + "\n\nf(" + input + ")"`; recorded outputs
are kept separately and never enter surface-reader features.

The [five-row review](../../results/cruxeval/deepseek-coder-6.7b/preflight_n50_seed42_iter20000/review.md)
shows complete code, inputs, outputs, every candidate use, its reaching
definitions/binding status, and the verified last covering token. The
[loader preview](../../results/cruxeval/deepseek-coder-6.7b/preflight_n50_seed42_iter20000/dataset_preview.json)
is the machine-readable counterpart.

| Population | Programs | Candidate uses | Def-use edges | Ambiguous uses | Shadowing uses |
|---|---:|---:|---:|---:|---:|
| First five | 5 | 25 | 25 | 0 | 0 |
| Pilot | 50 | 284 | 305 | 18 | 0 |

All 50 programs reproduced their recorded output on the supplied input in
fresh processes with a five-second timeout; all extracted definition/use
anchors passed. Of 328 AST Name loads, 44 have no source reaching definition
(including builtins); each is listed in the per-program coverage audit.

## Measured floors

The existing `run_surface_baseline` is used directly through a token-only store
adapter: ±3 token-ID windows at both endpoints, distance buckets of width 5,
standardized class-balanced logistic regression (`C=0.1`, `saga`, `tol=1e-3`),
five-fold `StratifiedGroupKFold`. Source hashes group exact duplicate code
together. All examples of a source remain in one fold; both real-label and
within-source shuffled-label folds are asserted disjoint and saved.

| Relation | Surface accuracy | Shuffled accuracy | Selectivity | Train-majority control |
|---|---:|---:|---:|---:|
| Def-use | 0.602310 | 0.593429 | 0.008881 | 0.738336 |
| Binding | Not run: no genuine shadowing in this sample | — | — | — |

These are arithmetic means of the five held-out fold accuracies, matching the
existing reader. There are 1,165 unique def-use pairs, 305 positive. The
majority control chooses its constant label from each training fold, then
scores that fold's test set. Its stronger accuracy matters: beating 0.602310
alone would not establish useful decoding. This small-sample floor and its
small selectivity authorize **no representational conclusion**.

The default 2,000-iteration fit failed the convergence assertion. The reported
run raises only the iteration budget to 20,000; all five real and shuffled
fits converged. The failed run's gate and manifest are retained. A high-floor
policy of 0.90 is declared for review, motivated by METHODS' demotion at 0.927;
METHODS does not specify 0.90 as a universal cutoff. This pilot is always
marked `pilot_only_no_representational_claim`, regardless of its accuracy.

Binding adequacy is declared as at least 50 uses resolving to shadowing
definitions across at least 20 source groups. This policy is provisional for
review, not a power calculation. Reassigning a name within one scope does not
count as lexical shadowing. The sample's zero is not a claim about all 800 rows.

## Graph and alignment contract

`src/data/cruxeval_graph.py` extends the existing `DataFlowGraph`, `VarEvent`,
and `DefUseEdge` representation using **beniget**, already used by SemFlow's
independent graph cross-check. The legacy extractor accepts arbitrary source
but selects a single definition by visit order; that is inadequate at real
branch joins and loop backedges. It remains unchanged and is audited against
the set-valued reference (zero rejected legacy edges in this pilot).

Labels describe static **may-reaching definitions**, not the executed path or
heap-object contents. Every source Name Load with a source definition is
enumerated; all possible edges are positive. A use has a single binding label
only when its reaching-definition set is a singleton. Binding pair records
exclude ambiguous uses. Pair construction retains all positive and same-name
negative pairs, caps different-name negatives at three per positive, and does
not duplicate distance-matched rows. This exact population is saved for reuse;
it is not the synthetic paired construction.

The adapter converts Python AST UTF-8 byte columns to character columns before
calling `TokenAligner`. It asserts exact identifier text, gap-free complete
coverage, and that the last covering token contains the identifier's end.
For split identifiers, the final token covers the end, while the union of
covering tokens covers the whole identifier. Function/import definitions use
their identifier spans rather than their whole AST statement. Full prompt
round-trip equality is also required. Truncation or coverage loss fails.

## Artifacts and gates

- `results/cruxeval/<model>/preflight_n50_seed42_iter20000/`: loader preview,
  complete tokenized programs/graphs, pair labels, source-group folds,
  per-program graph audit, review, and `gates.json`.
- [Tidy surface CSV](../../results/tables/cruxeval/surface_floor_deepseek-coder-6.7b_n50_seed42_iter20000.csv):
  aggregate plus stratum/distance rows; skipped binding is explicit, not zero.
- [Floor figure](../../results/figures/cruxeval/surface_floor_deepseek-coder-6.7b_n50_seed42_iter20000.png).
- `results/manifests/230_cruxeval_preflight_*.json`: args, git SHA, wall time,
  dependency versions, dataset fingerprint, tokenizer digest, selected row
  indices, and artifact digests.

Applicable preflight gates are assertions: dataset schema/count/IDs, source
parsing, input/output execution equality, full tokenizer round trip, complete
AST coverage, anchor correctness, source-disjoint folds, both classes per
fold, full held-out coverage, no unexpected row subsampling, and fit
convergence. A failed gate exits nonzero and records `status=failed`; valid
null/low results pass. No downstream authorization is emitted:
`review_confirmed=false`, `activation_extraction_allowed=false`.

## Cluster commands

For the independent Python 3.11 venv migration on `rad`, use
[CLUSTER_SETUP.md](CLUSTER_SETUP.md). The system Python 3.9 is too old.

Inside the repository's existing Python environment (CPU is sufficient):

```bash
python -m pip install -r requirements-cruxeval.txt
python -m pytest tests/test_cruxeval.py -q
python scripts/230_cruxeval_preflight.py \
  --model deepseek-coder-6.7b --sample-size 50 --seed 42 \
  --folds 5 --max-iter 20000 --out-root results/cruxeval_cluster
```

The separate output root avoids overwriting the reviewed local pilot. Both
dataset and tokenizer must be cached before using `HF_HUB_OFFLINE=1` on an
offline node. Only tokenizer assets are needed; no GPU/model weights. The
original surface reader materializes dense one-hot features, so larger samples
can need substantially more RAM. The stage refuses to overwrite an existing
run directory. Figures regenerate via `plot_floor(pd.read_csv(csv_path), path)`.

## Implementation after review

- Implemented stages 231–234 reuse forward hooks and `ActivationStore` for
  raw, unnormalized residuals at embedding output (`-1`) plus every block:
  33 read points for a 32-block model. Assert the layer grid and token/record
  identity with the approved preflight artifacts.
- The original request called synthetic transfer Design A; the latest request
  calls it **B**. Its implementation audits synthetic training provenance and the pinned 0.500 floor;
  reuse `load_frozen_probes` and `assemble_pair_features`. The 0.500 floor is
  a property of the synthetic training construction; it does not pin the
  real-code evaluation floor. **A** now means within-CruxEval training/testing;
  it reuses the saved source-group folds. Both implementations include controls,
  embedding comparison, per-split tidy outputs, and program-bootstrap intervals.
  Stage 231 remeasures the floor for the exact selected population.
  Frozen transfer alone does not supply RESULTS' open item 3, which explicitly
  calls for context-matched mutations of real code.
- Optional ladder: reuse atomic `ObfuscationLadder` steps, execute on the
  supplied input, drop every level of a failing base together, and re-extract
  each variant's graph. No ladder variants were generated here.
- Stages 235–238 reuse E19's stage-201 artifacts; they do not fit a second lens
  on CruxEval. They rerun the independent-corpus, matched-provenance,
  identity/head, forward-invariance, rule-binding, and nontrivial-J/R gates
  before full-vocabulary output-token ranks/pass@k at use, post-use, call, and
  teacher-forced answer positions. Causal erasure is reported separately with
  logit, off-target, stable-random, and exact edit-magnitude controls.
- Stages 239–241 derive paired programs rather than pairing unrelated CruxEval
  rows. A one-token literal mutation in a unique reaching definition must alter
  the instrumented runtime value and executed output while preserving the use
  anchor and token length. DAS and the answer-actuator control are fitted only
  on calibration source functions and frozen for two-direction evaluation on
  disjoint test functions. Completion is a mechanical pass, not a positive
  scientific verdict; the report compares against mean-difference, rank-random,
  magnitude-matched random, no-op, whole-state, and answer-only arms.
