# ExecSem correctness probe + J-lens

Run from the repository root. DAS is intentionally deferred. These scripts use the schema in the supplied private dataset card, without executing submissions. They do not use the precomputed encoder embeddings: a causal-LM J-lens must use that same model's residual states.

## Setup and run

Use your existing GPU environment with the repository dependencies, plus the vendored lens:

```bash
pip install -e .
pip install -e third_party/jacobian-lens
```

Download only the two records files in an environment already authenticated for the private dataset (or use your existing local copy):

```bash
hf download exec-sem/codecontests-plus-execsem records/train.jsonl records/val.jsonl --repo-type dataset --local-dir data/execsem
```

Run each command separately. Every command prints completion only after writing its artifacts; an exception identifies the failing stage. Use a new `--out` directory for each language, model, seed, or token-budget experiment. Defaults below use Python, two submissions per class per problem, and DeepSeek-Coder-1.3B. This is a pilot of up to roughly 16,000 submissions, not a tiny smoke run. Extraction checkpoints every 50 examined programs in `extract_checkpoints/` and resumes automatically with the same command. It prints actual model parameter devices and requires single-GPU placement for `--device cuda` (no automatic CPU offloading). Progress distinguishes examined, retained and dropped programs. Extraction currently holds activations in host RAM (roughly 3 GB at that size); the compressed archive and metadata must remain together.

```bash
python scripts/220_execsem.py prepare --train data/execsem/records/train.jsonl --test data/execsem/records/val.jsonl
python scripts/220_execsem.py extract --model deepseek-coder-1.3b --device cuda --max-tokens 1024
python scripts/220_execsem.py probe
```

Outputs: `results/execsem/pilot/prepare.json`, `rows.jsonl`, `extracted.json`, `activations.npz`, `probe.json`, `probe.npz`. Check discarded prompt counts in `extracted.json` before interpreting results. Long statements plus long programs are excluded completely, which changes the population being evaluated. Increase the budget in a fresh run if necessary.

The supplied train problems are split 80/20 into train/validation. Supplied validation problems become untouched test problems. Duplicate whitespace-normalized programs are removed entirely; this catches exact duplicates, not all clones. No label, verdict, test results, or test-case preview enters the prompt. The full statement and submission precede a fixed `Assessment:` suffix. Residuals are taken at its last token, after each zero-based transformer block, before final normalization.

Probes standardize using train statistics, fit balanced logistic regression, select layer and C on validation AUROC, and evaluate the selected probe on test. The saved raw-coordinate covector points toward accepted submissions. `probe.json` includes the full validation sweep, test AUROC/balanced accuracy and a length/lines/word-count surface baseline. Test metrics must not be used to select later configurations. This first pilot does not supply confidence intervals or comprehensive lexical controls.

## Lens artifact

You can reuse the same-model independent-corpus J-lens already fitted by stage 201 if it covers the selected layer. The existing released recipe targets the penultimate block, so it cannot cover a probe selected at the final block. For coverage of every probe layer, fit a separate final-target sensitivity artifact. For DeepSeek-1.3B there are 24 blocks, so the target is 23:

```bash
python scripts/200_lens_corpus.py --model deepseek-coder-1.3b --n-prompts 25
python scripts/201_lens_fit.py --model deepseek-coder-1.3b --corpus data/lens_corpus/pile10k-n25.jsonl --kinds j-lens --target-layer 23 --output results/execsem/lens-final --dim-batch 4 --checkpoint-every 1 --dry-run
python scripts/201_lens_fit.py --model deepseek-coder-1.3b --corpus data/lens_corpus/pile10k-n25.jsonl --kinds j-lens --target-layer 23 --output results/execsem/lens-final --dim-batch 4 --checkpoint-every 1
```

Use the corpus path printed by stage 200 if different. Full Jacobian fitting is expensive; review the dry-run estimate first. Stage 201 resumes its own checkpoints. This lens uses independent corpus prompts, not test programs or correctness labels. The final target is an identity anchor and is equivalent to the ordinary final readout at that layer; it is not evidence of useful Jacobian transport by itself. For other models use their actual final block index.

```bash
python scripts/220_execsem.py inspect --lens results/execsem/lens-final/j-lens --split val
# Freeze your interpretation/categories before the next command:
python scripts/220_execsem.py inspect --lens results/execsem/lens-final/j-lens --split test
```

## Interpretation

Yes: use the best validation probe layer as the primary lens layer, with available adjacent layers as sensitivity checks. Never choose a layer because its test vocabulary looks attractive.

`probe_words.json` ranks unrestricted vocabulary tokens by cosine between the raw probe covector and `J.T @ (final_norm_gain * unembedding[token])`. Positive and negative top-20 lists describe geometric alignment to accepted/incorrect classification. This gain-only approximation follows existing repository direction analysis; it does not include the state-dependent normalization derivative, and is not a causal result.

`lens_val.json` / `lens_test.json` separately contain normalized J-lens and ordinary logit-lens activation readouts, per program and layer, with unrestricted top-20 tokens and predefined candidate ranks. The Markdown companions summarize token occurrence in these lists separately for AC/WA. A token can align with the probe direction without being a high-logit token in actual states, so examine both analyses.

The fixed vocabulary is in `configs/execsem_words.json`: correctness words, error mechanisms, generic code and unrelated controls. Both bare and space-prefixed single-token spellings are scored. Multi-token words are explicitly represented by null ranks when neither spelling is a single token; fragments are not substituted as evidence for the whole word.

Review the top lists for correctness meaning, algorithm language, generic code, prompt echo and uninterpretable fragments. Check whether candidate ranks differ between AC and WA within individual problems, whether effects also appear in the ordinary logit lens, and whether validation discoveries recur on held-out problems. The report supports manual review; it does not automatically certify words as representative. No empirical word assessment is available until a real run finishes.

Accepted/incorrect labels are proxies for behavior on upstream tests, not complete program semantics. Incorrect examples mix bug types; lexical shortcuts and pretraining contamination remain possible. A successful probe establishes decodability, and readable tokens suggest a vocabulary interpretation. Causal alignment requires the later DAS experiment with a clearly defined behavioral target and suitable same-problem counterfactual pairs.

## Within-problem AC versus WA (stage 221, CPU only)

After validation inspection, run this independent script; it needs only the saved
`lens_val.json`, not model weights or another extraction:

```bash
.venv/bin/python scripts/221_execsem_contrast.py --split val
cat results/execsem/pilot/contrast_val.md
```

Each problem must contain both accepted and incorrect submissions. Their scores
are averaged within class, then subtracted (AC minus WA); problem contrasts are
weighted equally. J-lens and logit-lens must contain identical submission IDs,
task IDs and labels. Missing-class problems are excluded and reported.

For every predefined supported word and each group's equally weighted word mean,
outputs include negative log rank, reciprocal rank and top-20 membership effects.
Positive effects mean higher ranks in AC. Thus positive correctness words are
expected to have positive effects, while negative correctness words are expected
to have negative effects. Missing single-token spellings are excluded, not given
an artificial worst rank. A malformed/missing rank for a supported word fails.

The J-minus-logit contrast subtracts the two within-problem effects before
bootstrapping, preserving their pairing. Confidence intervals resample whole
problems (2,000 replicates; `--bootstrap` and `--seed` configurable). They are
pointwise exploratory intervals, not multiple-testing-adjusted discoveries.
One-problem estimates have no interval. Validation layer 13 remains primary;
neighboring layers are sensitivity analyses.

Artifacts: `contrast_val.md` (group log-rank results),
`contrast_val_summary.csv` (all words/groups/metrics),
`contrast_val_pairs.csv` (individual problem effects),
`contrast_val_counts.csv` (retained class counts), and `contrast_val.json`
(all results, exclusions, unsupported words, source hash and bootstrap settings).
Use `--out` for another run directory. Only run `--split test` after freezing
validation hypotheses. Stage 221 does not implement random probe-direction controls.

## Pairwise ranking and differential vocabulary (stage 222)

Copy `scripts/222_execsem_ranking.py` alongside the current `220_execsem.py`.
The existing activations are reused. Start on CPU:

```bash
.venv/bin/python scripts/222_execsem_ranking.py fit
```

Fits antisymmetric AC-WA logistic ranking objectives, without an intercept, with
pair weights summing equally per problem. Feature scaling uses training pair
second moments only. Layer and C are selected by validation problem-averaged
ranking accuracy (ties score 0.5). `ranking_probe.json` includes the sweep,
original probe validation ranking performance and a length/lines/word-count
ranking baseline. The new ranking metric is not comparable to pooled AUROC.
`ranking_probe.npz` stores the classifier covector, mean training problem AC-WA
activation difference and training within-problem covariance-times-weight pattern.
No full lexical baseline is included yet. The length baseline cannot rule out
style or algorithm confounds.

Next, on GPU:

```bash
.venv/bin/python scripts/222_execsem_ranking.py vocabulary --split val --lens results/workspace_lens/deepseek-coder-1.3b/j-lens
cat results/execsem/pilot/ranking_vocab_val.md
```

Uses the ranking-selected layer (fails explicitly if the lens lacks it). For each
submission it unembeds the actual normalized transported state, removes its
vocabulary-wide mean logit, averages submissions within class, then subtracts WA
from AC. It saves full-vocabulary differences per problem for J-lens, logit-lens,
and their paired difference in `ranking_vocab_val.npz`. It discovers 20 positive
and 20 negative tokens per readout on validation and freezes IDs in
`ranking_vocab_discovery.json`. Bootstrap intervals and within-problem sign-flip
null comparisons are pointwise. Validation discovery intervals/p-values are
selection-biased and must not be treated as confirmatory significance. The null
assumes exchangeability of the two class means within a problem. No multiple
comparison correction is applied. This stage currently saves at completion,
not resumably; it performs only unembedding on saved states, not transformer
forwards. Avoid overwriting validation discovery after test inspection.

The JSON also contains qualitative readouts of mean-difference and covariance
patterns, distinct from actual-state contrasts. Normalizing a pattern and
unembedding it is a heuristic, not a measured perturbation response.

Algorithm panel: records per-problem/class J/logit ranks for supported spellings
of sorting, search, graph, recursion, counting and other terms, with a flag for
literal occurrence in any corresponding prompt. A shared algorithm can describe
both AC and WA, so these absolute ranks remain separate from correctness
contrasts. `merge` is not evidence for `merge sort`: unsupported multi-token names
have empty token-ID lists and are not scored. There are no algorithm labels in
the supplied schema. Evaluating recognition requires independent annotation of
the algorithm implemented by each submission, and code-only/statement controls
to distinguish computation from prompt echo. Flags alone do not resolve that.

After freezing validation choices, optional exploratory test commands:

```bash
.venv/bin/python scripts/222_execsem_ranking.py evaluate_test
.venv/bin/python scripts/222_execsem_ranking.py vocabulary --split test --lens results/workspace_lens/deepseek-coder-1.3b/j-lens
```

Test vocabulary uses the frozen validation token IDs. Since this test set has
already been inspected in earlier experiments, new results are exploratory;
strong confirmation needs fresh held-out problems. These stages establish
associations and ranking performance, not causal semantic alignment.
