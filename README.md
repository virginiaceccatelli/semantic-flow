# Semantic Flow

This repository studies one question:

> How is variable binding represented in code language models, how robust is that
> representation, and does the model causally use it to produce an answer?

The active evidence forms one sequence:

1. **Representation:** binding and definition-to-use relations become linearly
   recoverable in middle-layer hidden states.
2. **Robustness:** the frozen representation survives many surface changes but
   weakens under competing scopes and control-flow flattening.
3. **Causal use:** a rank-1 DAS interchange changes which definition the model
   behaves as though the variable refers to.
4. **Lenses:** the published **J-lens**, with the R-lens as a supporting
   replication, from
   [the 2026 global-workspace paper](https://transformer-circuits.pub/2026/workspace/index.html)
   and [the R-lens post](https://www.alignmentforum.org/posts/nv8oedrnLXKRzNEL9/),
   recover values at the answer position but essentially not at three earlier
   use-to-call positions; their transport has no consistent advantage over the
   logit lens on these models. See
   [docs/WORKSPACE_LENS.md](docs/WORKSPACE_LENS.md).

> **Naming.** Archived experiments used a corpus-averaged cotangent readout over a fixed candidate
> vocabulary — a different estimator, a different target layer, a different
> fitting corpus, and no normalization before the unembedding. They are now
> called the **cotangent lens** (`clens`) and the **conserving cotangent lens**
> (`clrp`) throughout the code and archived artifacts, so the two methods
> cannot be confused. `docs/WORKSPACE_LENS.md` §1 tabulates the differences.

The former security benchmark, output-vocabulary study, older standalone
cotangent-lens experiments, and the conserving-cotangent-lens taint-routing study
remain reproducible and are documented in [docs/ARCHIVE.md](docs/ARCHIVE.md),
including the former E16 and E18 binding results.

## Start here

For a first reading:

1. Read [docs/RESULTS.md](docs/RESULTS.md) for the evidence and conclusions.
2. Read [docs/METHODS.md](docs/METHODS.md) for the constructions, controls, and
   instrument mechanics.
3. Read the generated [DAS report](results/binding/deepseek-coder-6.7b/e13_report.md)
   for the causal binding tables.
4. Read [docs/WORKSPACE_LENS.md](docs/WORKSPACE_LENS.md) for the
   J-lens / R-lens experiment (E19): implementation, compatibility, deviations,
   controls, and the completed three-model result.
5. Use [docs/PIPELINE.md](docs/PIPELINE.md) to reproduce stages.
6. Use [docs/ARCHIVE.md](docs/ARCHIVE.md) for displaced tracks, failed designs,
   and the methodological history.

## The controlled binding construction

The core pair changes which definition the use of `x` resolves to:

```python
x = 3                  x = 3
def f():               def f():
    y = 7                  x = 7
    return x               return x
#   returns 3              returns 7
```

At the queried `x` in `return x`, the token identity, position, and bounded local
context are the same. The answer changes only because the inner definition comes
into scope. A bounded model-free surface reader is therefore at 0.500 by
construction. This is what lets a hidden-state result be interpreted as
contextual binding information rather than a local token shortcut.

## Main finding

### 1. Binding is represented

A linear probe reads which definition is in scope from hidden states at the
variable-use position. Accuracy begins at the 0.500 floor at the input, rises to
approximately 0.984 in middle layers, and declines toward the output.

The input-layer null matters: the relation is not directly available from the
queried token. Contextual processing makes it linearly recoverable.

This is observational evidence. A probe can detect information that the model
does not use.

### 2. The representation has a structural robustness boundary

The clean probe is frozen and evaluated on meaning-preserving variants. Long
irrelevant comments cause little damage, whereas reusing the tracked names in
competing scopes drives binding toward chance. Consistent identifier renaming
disrupts early lexical layers while leaving much of the middle-layer readout
intact. Opaque branches and equivalent arithmetic rewrites cause little
additional damage, while control-flow flattening produces the largest
reproducible collapse.

These results identify where the original linear representation transfers. A
failed frozen probe does not prove that every possible encoding has disappeared.

### 3. DAS shows causal use

Distributed Alignment Search learns a one-dimensional subspace at the unchanged
variable-use token. Given a host program and a donor program with the other
binding, DAS replaces only the host's component in that subspace with the
donor's component. The language model stays frozen.

The decisive design crosses binding with value assignment:

| arm | installing the other binding requires |
|---|---|
| fitted `ab` arm | answer `a → b` |
| held-out `ba` arm | answer `b → a` |

A fixed push toward answer token `b` should work in the first arm and fail or
reverse in the second. A binding component should follow whichever value the
installed definition supplies.

DAS produces the installed answer on 100% of held-out cases in both arms, in
DeepSeek-Coder 6.7B and StarCoder2 3B.

The decisive alternative is a separately trained rank-1 answer-token actuator
at the same site. It uses the same optimiser, steps, split, and per-row edit
norm as binding DAS, but never receives the donor binding state. It works on
the fitted `ab` arm and attenuates on crossed `ba`, whereas binding DAS remains
perfect. Dose-matched and rank-matched random subspaces test generic
disruption and provide a random rank-1 floor. A no-op detects hook or
measurement artifacts, a whole-state donor patch verifies that the intervention
site can affect the answer, and the mean donor−host direction tests the simplest
non-learned rank-1 alternative. The completed H0–H5 reports pass on both models.
No lens finds, initializes, or constrains the DAS direction. Earlier attempts to
use lens read directions as causal answer actuators are archived.

### 4. What is the causally used representation, and is it verbalized?

Having established with DAS that a binding component is present and causally
used, E19 asks what that representation looks like in output-vocabulary
coordinates: does the model verbalize the *language of binding* at the use
site? It applies the released full-vocabulary J-lens to
DeepSeek-Coder 1.3B/6.7B and StarCoder2-3B. All required implementation gates
pass, with R-lens used as a supporting replication.

The predeclared binding words are `local`, `global`, `inner`, `outer`, `scope`,
`scoped`, `shadow`, `shadowed`, `binding`, `bound`, `active`, `inactive`,
`definition`, `variable`, and `value`. Each concept includes every bare,
space-prefixed, and declared capitalization spelling that is a single token for
the model; split spellings are recorded as unavailable, never truncated. The
lens reads the unchanged use token, the following token, the call site, and the
answer position. Binding is crossed with value assignment (`ab`/`ba`), so a
binding word must move the same way when the concrete answer reverses. Matched
generic-code words, `earlier`/`later` and `kept`/`replaced` confound diagnostics,
and size/frequency-band-matched random concepts are evaluated in the same way.
Ranks, pass@k, crossed-arm score differences, and program-cluster bootstrap
intervals are reported for every lens, layer, read position, and concept.

As a secondary contrast, both lenses recover every value family at pass@10 =
1.000 when the value is about to be emitted, but the value is essentially absent
at the use token, following token, and call site, including for computed targets
that never occur in the prompt.

J-lens direction erasures use separate distractor controls, stable
random controls, exact edit-magnitude matching, and paired cluster-bootstrap
intervals. Strong effects occur beside the output head. Mid-network,
StarCoder2 is null; DeepSeek 6.7B has only a small L20 effect that does not beat
the logit direction; and on DeepSeek 1.3B the logit direction is stronger than
J-lens. The R-lens study closely replicates this pattern. It makes some local
improvements but does not recover a broad early-layer advantage. In the two
completed DeepSeek semantic panels, J-lens also surfaces controlled
binding-related vocabulary (`scope` or `global`) without surfacing the concrete
runtime value; R-lens supports that positive result as well.


## What the active evidence supports

The narrow conclusion is:

> In controlled programs, variable binding becomes linearly represented, remains
> stable under many surface changes but is fragile to structural interference,
> is causally read from a rank-1 component at the use site, but is not surfaced
> as a mid-network concrete-value token by the published J-lens. The R-lens
> independently supports this conclusion.

This conclusion is deliberately narrow. It does not establish general program
understanding, causal binding use at every layer or site, or transfer of the
controlled isolation to real code. It does not show that the DAS direction is
unique or recover a complete attention mechanism. A J-lens value null means
the published token-indexed linear readout does not surface the value; it does
not contradict the probe and DAS evidence that binding is represented and used.

## Repository map

```text
src/
  graphs/        AST, CFG, DFG, and program-structure extraction
  data/          controlled pair generation and verification
  models/        model loading, hooks, probes, and DAS
  experiments/   experiment implementations
  analysis/      metrics, tables, figures, and bootstrap utilities
scripts/         numbered pipeline commands
jobs/            GPU job scripts
configs/         model and experiment settings
results/         generated reports, tables, figures, and manifests
docs/            active methods/results, pipeline, and archive
tests/           CPU-only tests
```

## Quickstart

```bash
conda create -n semflow python=3.11 -y
conda activate semflow
pip install -e ".[dev]"

make test
make smoke

# Representation and robustness
python scripts/00_generate_data.py --model deepseek-coder-1.3b
make extract probes context obfuscation assets MODEL=deepseek-coder-1.3b

# Causal binding intervention
make binding-pilot

# J-lens with supporting R-lens replication (E19). Size the fit first — it is the one
# expensive stage in the repository.
make lens-fit-dry MODEL=deepseek-coder-6.7b
make lens         MODEL=deepseek-coder-1.3b LENS_HALVES=--halves
```

Read [docs/PIPELINE.md](docs/PIPELINE.md) before running the full model suite.

## Models in the active claim

| Model | Active role |
|---|---|
| `deepseek-coder-1.3b-base` | representation, robustness, J-lens, and R-lens replication |
| `deepseek-coder-6.7b-base` | representation, robustness, DAS, J-lens, and R-lens replication |
| `starcoder2-3b` | robustness, cross-architecture DAS, J-lens, and R-lens replication |

Base models are used because the target is the representation learned during
code pretraining rather than chat behavior.
