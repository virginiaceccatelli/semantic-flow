# Methods

The methodological commitments behind every number in `results/tables/`.
This file is written to be lifted into the paper's Methods section.

## Probes

Logistic regression only (`C=0.1`, class-balanced), on standardized features.
Single-position tasks probe the raw hidden state; pairwise tasks probe
`[h_i; h_j; h_i−h_j; |h_i−h_j|]`. No MLP probes: with high-capacity probes,
"decodable" stops being evidence about the *representation* and starts being
evidence about the probe.

Fits use saga with `max_iter=2000, tol=1e-3`; convergence is recorded per fit
and surfaces in every results row (`converged`). Stage 20 fails its sanity
check if any reported fit did not converge.

## Ground truth and token alignment

Labels come from static analysis (`src/graphs/`): reaching-definition def-use
chains, AST-derived control dependence (guard nesting, join-point exact), and
generator-known taint state per line. Every source-level event is mapped to
token positions through its **AST span** and a **verified offset table**:
offsets are computed by incremental prefix decoding and checked to reproduce
the source exactly (`src/data/alignment.py`). The probing position for a
span is its last covering token (the first position whose state can integrate
the whole span under causal attention).

We do not match token strings: with subword vocabularies a variable name has
no reliable single token, and string matching silently mislabels shadowed
names — the exact phenomenon E2 measures.

### Tokenizer integrity

`AutoTokenizer` on transformers 5.x resolves deepseek-coder to a slow
sentencepiece path that mis-tokenizes code (`def func` → `['de','ff','unc']`,
whitespace dropped). `src/models/loader.py::load_tokenizer` therefore loads
via `PreTrainedTokenizerFast` and **rejects any tokenizer that fails an exact
code round-trip**. All results predating this guard are invalid.

## Cross-validation without leakage

Rows built from the same program share hidden-state vectors; random k-fold
therefore leaks train information into test folds. All CV is
`StratifiedGroupKFold` grouped by source example id, and dataset caps
(`max_samples=20000` per task × layer) drop whole groups, never rows.

## Selectivity control

For every probe we retrain the identical architecture on shuffled labels and
report `selectivity = accuracy − control_accuracy`. Labels are shuffled
*within* each source example (preserving each program's label marginals);
for example-level tasks where the label is constant within a program
(taint_state), a within-group shuffle would be a no-op, so the group→label
assignment is permuted *across* programs instead. Claims are made on
selectivity, not raw accuracy: a probe can score high accuracy from class
priors and per-program regularities alone.

## Negative-sampling strata

Pairwise tasks report held-out accuracy per negative stratum:
`same_name_diff_binding` (hard: defeats lexical shortcuts),
`diff_name` (capped 3× positives), `distance_matched` (defeats positional
shortcuts). An honest headline number is the hard-stratum accuracy, not the
pooled one.

## Frozen-probe evaluation (E5)

Degradation is measured by *evaluating* stage-20 probes on context variants,
never retraining them: retraining on each condition would measure the
condition's learnability, not the stability of the representation the probe
found. Ground truth is recomputed per variant so fillers that genuinely
change the program (competing updates) are scored against the new truth.

## Calibration and independence of signals (E6)

The taint-probe decision threshold is chosen on a held-out calibration split
(balanced-accuracy-maximizing cutoff) and fixed before touching test
examples. The latent signal (linear readout) and the behavioral signal
(model's own forced-choice log-probs) come from different mechanisms; the
former is never derived from the latter.

## Causal claims (E7)

Minimal pairs are verified token-length-matched with the difference confined
to the sink argument, so patched positions correspond one-to-one. Recovery is
reported per (layer, position); last-token patches are quarantined as the
trivial case. "Used" means recovery of the answer logit-diff > 0.5 at a
non-readout position while the frozen probe decodes the relation on both
sides.

## Reproducibility

Seed 42 everywhere by default (generator, CV splits, subsampling, bootstrap).
Every stage writes a manifest (`results/manifests/`) with git sha, arguments,
and wall time. Figures and tables are regenerable from the tidy CSVs alone
(stage 90), so the full chain data → figure is auditable.
