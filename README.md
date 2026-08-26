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
4. **Binding attribution:** on the same programs, a conserving R-lens moves
   answer relevance from the inactive definition toward the active one.

The former security benchmark, output-vocabulary study, standalone J-lens
experiments, and R-lens taint-routing study remain reproducible and are documented
in [docs/ARCHIVE.md](docs/ARCHIVE.md). They are not part of the active claim.

## Start here

For a first reading:

1. Read [docs/RESULTS.md](docs/RESULTS.md) for the evidence and conclusions.
2. Read [docs/METHODS.md](docs/METHODS.md) for the constructions, controls, and
   instrument mechanics.
3. Read the generated [DAS report](results/binding/deepseek-coder-6.7b/e13_report.md)
   and [R-lens report](results/binding/deepseek-coder-6.7b/e16_report.md) for
   model-specific tables.
4. Use [docs/PIPELINE.md](docs/PIPELINE.md) to reproduce stages.
5. Use [docs/ARCHIVE.md](docs/ARCHIVE.md) for displaced tracks, failed designs,
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

At the queried `x` in `return x`:

- the token identity is the same;
- its position and local context are controlled;
- the answer changes only because the inner definition comes into scope.

A bounded model-free surface reader is therefore at 0.500 by construction. This
is what lets a hidden-state result be interpreted as contextual binding
information rather than a local token shortcut.

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

The clean probe is frozen and evaluated on meaning-preserving variants.

- Long irrelevant comments cause little damage.
- Reusing the tracked names in competing scopes drives binding toward chance.
- Consistent identifier renaming disrupts early lexical layers while much of the
  middle-layer readout survives.
- Opaque branches and equivalent arithmetic rewrites cause little additional
  damage.
- Control-flow flattening causes the largest reproducible collapse.

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

The controls test distinct alternatives:

- a J-lens-derived, dose-matched answer direction tests a fixed output-token
  push;
- a dose-matched random subspace tests generic disruption;
- a rank-matched random subspace provides a random rank-1 floor;
- a no-op detects hook or measurement artifacts;
- a whole-state donor patch verifies that the site can affect the answer;
- a mean donor−host direction tests the simplest non-learned rank-1 alternative.

The J-lens does not find the DAS direction. It is used only to construct the
strong answer-direction control at the intervention layer.

### 4. The R-lens attributes the answer to the active definition

The R-lens changes only the backward attribution rules. It leaves the model's
forward activations, output scores, and emitted answer unchanged. For compatible
DeepSeek models, it divides the selected bound-value score among earlier input
positions while conserving that score.

On the binding pairs, exactly one of roughly 21 tokens changes: the inner
definition's name. The main statistic measures relevance movement between the
outer definition and the inner definition's unchanged value.

On DeepSeek-Coder 6.7B:

- the newly active inner value gains answer relevance;
- the newly inactive outer definition loses relevance;
- the combined shift is about 13% at the first measured layer and peaks near 22%
  in the middle;
- the one changed token carries only about 1.5% of the movement;
- both crossed arms agree;
- fixed-output-token conditions retain the effect;
- scoring the competing value reverses it; and
- same-binding controls remain flat.

This is attribution, not causation. DAS supports the claim that the binding
component is used. The R-lens supports the separate claim that the unedited
answer score is assigned to the semantically active definition.

The 1.3B R-lens result is not interpreted because the model often assigns a zero
or negative score to the bound value in the shadowing condition, making
normalized relevance shares unstable.

## What the active evidence supports

The narrow conclusion is:

> In controlled programs, variable binding becomes linearly represented, remains
> stable under many surface changes but is fragile to structural interference,
> is causally read from a rank-1 component at the use site, and is reflected in
> how the final answer is attributed to the active definition.

It does not establish:

- general program understanding;
- causal binding use at every layer or site;
- transfer of the controlled isolation to real code;
- that the DAS direction is unique;
- that R-lens attribution is causal;
- a complete attention mechanism; or
- verbalisation of binding as a human-readable word.

A future binding-verbalisation study would be a new experiment, not a
reinterpretation of the R-lens attribution result.

## Repository map

```text
src/
  graphs/        AST, CFG, DFG, and program-structure extraction
  data/          controlled pair generation and verification
  models/        model loading, hooks, probes, DAS, and R-lens rules
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

# Binding R-lens attribution
make binding-rlens MODEL=deepseek-coder-6.7b
```

Read [docs/PIPELINE.md](docs/PIPELINE.md) before running the full model suite.

## Models in the active claim

| Model | Active role |
|---|---|
| `deepseek-coder-1.3b-base` | representation and robustness; binding R-lens attempted but not interpretable |
| `deepseek-coder-6.7b-base` | representation, robustness, DAS, and binding R-lens |
| `starcoder2-3b` | robustness and cross-architecture DAS replication; R-lens rules not applicable |

Base models are used because the target is the representation learned during
code pretraining rather than chat behavior.
