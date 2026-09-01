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
4. **Binding attribution:** on the same programs, a conserving cotangent lens moves
   answer relevance from the inactive definition toward the active one.
5. **Cotangent-lens lexical alignment:** at the unchanged, unprompted use-token
   state, several scope-related word contrasts track binding in both crossed
   value arms, independently of which literal is returned.
6. **Published J-lens and R-lens (E19):** the actual methods of
   [the 2026 global-workspace paper](https://transformer-circuits.pub/2026/workspace/index.html)
   and [the R-lens post](https://www.alignmentforum.org/posts/nv8oedrnLXKRzNEL9/),
   run through the released reference implementation on three code models. The
   gated result is negative: needed values surface at the answer position but
   essentially not at three earlier use-to-call positions, and Jacobian
   transport has no consistent advantage over the logit lens. See
   [docs/WORKSPACE_LENS.md](docs/WORKSPACE_LENS.md).

> **Naming.** Items 4 and 5 above are *not* the published J-lens and R-lens.
> They are a corpus-averaged cotangent readout over a fixed candidate
> vocabulary — a different estimator, a different target layer, a different
> fitting corpus, and no normalization before the unembedding. They are now
> called the **cotangent lens** (`clens`) and the **conserving cotangent lens**
> (`clrp`) throughout the code, the results and the tables, so the two methods
> cannot be confused. `docs/WORKSPACE_LENS.md` §1 tabulates the differences.

The former security benchmark, output-vocabulary study, older standalone
cotangent-lens experiments, and the conserving-cotangent-lens taint-routing study
remain reproducible and are documented in [docs/ARCHIVE.md](docs/ARCHIVE.md).
E18 is the active binding-specific cotangent-lens test.

## Start here

For a first reading:

1. Read [docs/RESULTS.md](docs/RESULTS.md) for the evidence and conclusions.
2. Read [docs/METHODS.md](docs/METHODS.md) for the constructions, controls, and
   instrument mechanics.
3. Read the generated [DAS report](results/binding/deepseek-coder-6.7b/e13_report.md)
   [conserving cotangent lens report](results/binding/deepseek-coder-6.7b/e16_report.md), and
   [cotangent lens verbalisation report](results/binding/deepseek-coder-6.7b/e18_report.md)
   for model-specific tables.
4. Read [docs/WORKSPACE_LENS.md](docs/WORKSPACE_LENS.md) for the published
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

The controls isolate distinct alternatives. A cotangent lens-derived answer direction,
scaled to the DAS edit norm, tests whether the learned intervention is merely a
fixed output-token push. Dose-matched and rank-matched random subspaces test
generic disruption and provide a random rank-1 floor. A no-op detects hook or
measurement artifacts, a whole-state donor patch verifies that the intervention
site can affect the answer, and the mean donor−host direction tests the simplest
non-learned rank-1 alternative. The cotangent lens does not find the DAS direction; it is
used only to construct the answer-direction control at the intervention layer.

### 4. The conserving cotangent lens attributes the answer to the active definition

The conserving cotangent lens changes only the backward attribution rules. It leaves the model's
forward activations, output scores, and emitted answer unchanged. For compatible
DeepSeek models, it divides the selected bound-value score among earlier input
positions while conserving that score.

On the binding pairs, exactly one of roughly 21 tokens changes: the inner
definition's name. The main statistic measures relevance movement between the
outer definition and the inner definition's unchanged value.

On DeepSeek-Coder 6.7B, the newly active inner value gains answer relevance and
the newly inactive outer definition loses it. The combined shift is about 13%
at the first measured layer and peaks near 22% in the middle, while the one
changed token carries only about 1.5% of the movement. Both crossed arms agree,
fixed-output-token conditions retain the effect, scoring the competing value
reverses it, and same-binding controls remain flat.

This is attribution, not causation. DAS supports the claim that the binding
component is used. The conserving cotangent lens supports the separate claim that the unedited
answer score is assigned to the semantically active definition.

The 1.3B conserving cotangent lens result is not interpreted because the model often assigns a zero
or negative score to the bound value in the shadowing condition, making
normalized relevance shares unstable.

### 5. The binding state aligns with some scope-related cotangent lens contrasts

E18 reads the same unchanged `x` in `return x`, with no appended question or
answer prompt. Nine predeclared single-token contrasts cover scope
(`local/global`, `inner/outer`, `inside/outside`, `nested/module`), position, and
action. Each contrast is read on the same 280 held-out bases in two crossed value
arms: the inner binding returns `b` in `ab` but `a` in `ba`. A word margin that
tracks the returned literal should therefore reverse across arms; a
binding-associated margin should keep the same orientation.

The strongest effects are striking: `nested/module` is 1.000/1.000 at L16 and
`local/global` is 0.996/1.000 at L20. The same orientation in `ab` and `ba` rules
out the simplest returned-literal explanation. The response is not uniform
across scope vocabulary—`inner/outer` is weak—and positional and action pairs
such as `later/earlier` and `replaced/kept` also respond strongly. The template
makes locality, order, proximity, and replacement coincide, so it cannot identify
which of these correlated properties drives the alignment.

The supported result is therefore **binding-associated lexical alignment**, not
clear internal verbalisation: several cotangent lens word margins follow the binding
independently of value identity, but the construction does not show that the
model specifically encodes the abstract concept of scope in those words.


## What the active evidence supports

The narrow conclusion is:

> In controlled programs, variable binding becomes linearly represented, remains
> stable under many surface changes but is fragile to structural interference,
> is causally read from a rank-1 component at the use site, is reflected in how
> the final answer is attributed to the active definition, and is associated
> with several scope-related lexical contrasts at that unprompted state.

This conclusion is deliberately narrow. It does not establish general program
understanding, causal binding use at every layer or site, or transfer of the
controlled isolation to real code. It does not show that the DAS direction is
unique, make conserving cotangent lens attribution causal, or recover a complete attention
mechanism. E18 establishes value-independent lexical alignment, not uniquely
scope-semantic or faithful internal verbalisation.

## Repository map

```text
src/
  graphs/        AST, CFG, DFG, and program-structure extraction
  data/          controlled pair generation and verification
  models/        model loading, hooks, probes, DAS, and conserving cotangent lens rules
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

# Binding conserving cotangent lens attribution
make binding-clrp MODEL=deepseek-coder-6.7b

# Unprompted cotangent lens verbalisation
make binding-lexlens MODEL=deepseek-coder-6.7b

# The PUBLISHED J-lens and R-lens (E19). Size the fit first — it is the one
# expensive stage in the repository.
make lens-fit-dry MODEL=deepseek-coder-6.7b
make lens         MODEL=deepseek-coder-1.3b LENS_HALVES=--halves
```

Read [docs/PIPELINE.md](docs/PIPELINE.md) before running the full model suite.

## Models in the active claim

| Model | Active role |
|---|---|
| `deepseek-coder-1.3b-base` | representation and robustness; binding conserving cotangent lens not interpretable |
| `deepseek-coder-6.7b-base` | representation, robustness, DAS, binding conserving cotangent lens, and cotangent lens verbalisation |
| `starcoder2-3b` | robustness and cross-architecture DAS replication; conserving cotangent lens rules not applicable |

Base models are used because the target is the representation learned during
code pretraining rather than chat behavior.
