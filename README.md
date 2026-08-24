# Semantic Flow

This repository tests whether code language models represent program semantics
separately from surface patterns in source text. The experiments focus on:

- which definition an identifier refers to (**binding**);
- which definition supplies a later use (**def–use**); and
- whether untrusted data reaches a security-sensitive function argument
  (**source-to-sink flow**).

The project also tests whether the model uses a binding representation in its
downstream computation. Most experiments are observational. The DAS interchange
experiment is causal at the tested model, layer, site, and program construction.

## Main results

- Binding is at chance at the input and model-free surface baseline
  (**0.500**), rises to approximately **0.984** in middle layers, and declines
  near the output. The pattern replicates at 1.3B and 6.7B at comparable relative
  depths. Def–use follows a similar pattern, with some decline as distance grows.
- Adding 1,000 tokens of irrelevant comments has little effect: binding remains
  at **0.921** in the reported comparison. Filler that reuses the tracked names
  reduces binding to chance. The main problem is semantic interference, not
  context length alone.
- Renaming all identifiers leaves middle-layer accuracy at **0.85–0.90**, while
  pushing the embedding-layer result below chance. Early layers depend strongly
  on identifier spelling, while much of the middle-layer representation does
  not.
- The source-to-sink readout reaches **1.000** on held-out programs over two
  measured chance baselines. Opaque branches and arithmetic rewriting have no
  measured cost when applied alone. Renaming costs **0.01–0.12**.
  Control-flow flattening alone costs **0.31–0.34**, within **0.03** of the cost
  of combining all four transformations. No additional interaction is
  distinguishable from measured draw noise.
- Across the full approximately 32,000-token vocabulary, **72/72** held-out
  safe/unsafe pairs point in the expected direction in every model. The largest
  token loadings are meaningless fragments rather than words such as `unsafe`.
  The distinction is aligned with the output vocabulary but distributed across
  many token dimensions.
- The absence of a clear security word is not caused by a blind readout. On a
  property that the models express through their own answer margin, the same
  readout scores **0.85–0.94**. At the same measurement cell, the security-word
  set scores **0.347 / 0.389** on two models, significantly in the opposite
  direction.
- For binding, a rank-1, magnitude-free DAS interchange changes which definition
  is in scope. It succeeds on **100%** of held-out rows in both arms of the 2×2
  design while moving **0.479** of `||h||`. The closed-form difference-in-means
  baseline succeeds on 76% while moving **0.711** of `||h||`; dose-matched
  random subspaces succeed on 1–2%.

For the complete experiments, controls, qualifications, and model-specific
results, see [docs/RESULTS.md](docs/RESULTS.md).

## Experimental design

### Controlled program pairs

The main construction uses program pairs with nearly identical text but
different semantics:

```python
x = 3                  x = 3
def f():               def f():
    y = 7                  x = 7
    return x               return x
#   returns 3              returns 7
```

The queried `x` appears at the same token position in both programs. Its local
context and distance from other tokens are unchanged. Only the binding changes.
A reader using only nearby tokens and distances therefore scores exactly
**0.500 by construction**. A hidden-state score above this floor cannot be
explained by those features.

This exact floor applies to the controlled binding construction. Other
experiments use their own measured baselines and controls, described in
[docs/METHODS.md](docs/METHODS.md).

### Ground truth

Labels come from program structure, not from the model or a human annotator.
Extractors under `src/graphs/` build these code-property-graph components:

| Component | Module | Used for |
|---|---|---|
| Abstract syntax tree (AST) | `ast_extractor.py` | exact source-span and token alignment |
| Control-flow graph (CFG) | `cfg_extractor.py` | statement flow, joins, and archived control-dependence measurements |
| Data-flow graph (DFG) | `dfg_extractor.py` | binding and def–use labels |
| Program-dependence graph (PDG) | `pdg_extractor.py` | source-to-sink taint paths |

The labels are cross-checked with `beniget`, instrumented execution, or an
independent scope-aware interpreter where appropriate.

Control dependence is not included as a main result. Although the model reaches
AUC **0.999**, a model-free reader using token windows and indentation already
reaches **0.927**. The result therefore does not isolate information computed by
the model. Details are in
[docs/ARCHIVE.md §4.3](docs/ARCHIVE.md#43-control-dependence).

### Four evidence levels

| Method | What it can establish | What it cannot establish |
|---|---|---|
| Linear probe with a controlled floor | a relation is linearly recoverable from a hidden state | that the model uses the relation |
| Frozen-probe transfer across behavior-preserving rewrites | whether the original readout survives a rewrite | that all information is gone when the probe fails |
| Output-basis and relevance readouts | whether the distinction is aligned with output tokens, represented by a word, or routed through particular input positions | causal use |
| Rank-1 DAS interchange | whether downstream computation reads the edited subspace at the tested site | generality beyond the tested construction, model, layer, and site |

The first three methods are observational. Only the DAS interchange edits the
model state and supports a causal conclusion.

## Experiment workflow

1. Generate small Python programs with exact binding, data-flow, or security
   labels.
2. Run a frozen model and save hidden states at verified token positions.
3. Fit low-capacity linear probes and evaluate model-free and shuffled-label
   controls.
4. Test the frozen probes on behavior-preserving rewrites.
5. Project states into vocabulary coordinates or attribute output scores to
   input positions.
6. For binding, intervene on a learned rank-1 subspace and measure the resulting
   answer change.

Synthetic programs are used because per-token ground truth must be exact. Linear
probes limit how much work the readout itself can perform. Hidden states are
saved once so repeated CPU analyses do not require another model run.

## Controls

| Control | Purpose |
|---|---|
| Grouped cross-validation by source program | prevents related rows from appearing in both training and test folds |
| Shuffled-label selectivity control | detects accuracy caused by class balance or program-specific regularities |
| Separate negative strata | prevents easy negatives from hiding failure on difficult cases |
| Local and whole-program surface baselines | measures how much the source text reveals without hidden states |
| Verified AST-span-to-token alignment | prevents incorrect labels for repeated or shadowed names |
| Independent ground-truth checks | detects extractor errors that could otherwise appear as model signal |
| Cluster bootstrap over programs | avoids treating correlated rows from one program as independent samples |
| Hard stage gates | stops dependent experiments when a required control or artifact is missing |

Tokenizer validation is also required. On Transformers 5.x, `AutoTokenizer` can
mis-tokenize DeepSeek Coder—for example, `def func` can become
`['de', 'ff', 'unc']`—without raising an error. The loader in
`src/models/loader.py` rejects tokenizers that fail an exact code round trip.

## Documentation

| File | Contents |
|---|---|
| [docs/METHODS.md](docs/METHODS.md) | complete methods, controls, metrics, and gates |
| [docs/RESULTS.md](docs/RESULTS.md) | completed results, measurements, interpretations, and limitations |
| [docs/PIPELINE.md](docs/PIPELINE.md) | setup, stage commands, prerequisites, and outputs |
| [docs/ARCHIVE.md](docs/ARCHIVE.md) | retired, superseded, or failed designs and why they were not used |

Machine-readable experiment status is stored in `results/STATUS.yaml`. Markdown
reports under `results/` are generated outputs, not primary documentation.

## Repository structure

```text
src/
  graphs/       AST, CFG, DFG, and PDG extraction
  data/         program generators, rewrites, alignment, execution, and reference labels
  models/       model loading, hooks, J-lens, R-lens rules, and DAS interchange
  probes/       linear probes, grouped cross-validation, controls, and datasets
  experiments/  experiment implementations
  analysis/     metrics, tables, figures, and cluster bootstrap
scripts/        numbered pipeline commands (00–131)
jobs/           GPU job scripts
configs/        model registry and experiment settings
results/        status, tables, figures, manifests, and generated reports
docs/           methods, results, pipeline, and archive
tests/          489 CPU-only tests
```

## Quickstart

```bash
conda create -n semflow python=3.11 -y && conda activate semflow
pip install -e ".[dev]"
make test                     # 489 CPU-only tests
make smoke                    # small end-to-end MPS run, approximately 15 minutes

# Foundation experiments
python scripts/00_generate_data.py --model deepseek-coder-1.3b
make extract probes context obfuscation assets MODEL=deepseek-coder-1.3b

# Security experiments: E15, E15-C, and E15-D
make sinkflow MODEL=deepseek-coder-1.3b

# Binding intervention: E13 DAS
make binding-pilot
```

See [docs/PIPELINE.md](docs/PIPELINE.md) before running the full model suite.

## Models

| Model | Use |
|---|---|
| `deepseek-coder-1.3b-base` | development, smoke tests, and pilot runs on Apple Silicon MPS |
| `deepseek-coder-6.7b-base` | main results; run in fp16 on one cluster GPU |
| `starcoder2-3b` | replication across a different corpus and architecture family |

The experiments use base models rather than instruction-tuned models because the
target is the representation learned during code pretraining, not chat behavior.

E15, E15-C, and E15-D stages 128–129 are complete for StarCoder2. The R-lens
analysis is not applicable to this architecture: its LayerNorm and non-gated MLP
do not match the homogenising rules, so stage 130 records no result instead of
reporting an invalid attribution.

## Contributions

1. Layer-by-layer measurements of binding, def–use, and taint-flow decodability
   against controlled or measured surface baselines.
2. An attributed failure boundary: the tested representations are comparatively
   robust to distance and identifier spelling, but fragile under scope
   interference and control-flow flattening.
3. Evidence that the safe/unsafe distinction is distributed across output
   coordinates rather than represented by a single security-related word.
4. A causal binding result using a rank-1, magnitude-free interchange with a
   held-out factorial arm that distinguishes binding transport from an answer
   direction.
5. An attribution-method result: on gated-MLP transformers, gate bilinearity—not
   normalization—is the main source of the LRP faithfulness gain. The effect is
   approximately **4.5×** and replicates across two models and two dtypes. For
   vocabulary projections in this project, the more expensive lenses do not
   improve the conclusions over a plain logit lens.
6. Experimental safeguards developed from four failed interventions, including
   construction-pinned floors, matched positive controls, magnitude-free edits,
   and hard prerequisite gates.
