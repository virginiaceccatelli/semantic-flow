# Findings overview

## The main question

This repository asks whether code language models merely contain decodable traces
of variable binding or whether they use a binding representation when producing
an answer.

The active evidence follows one sequence:

```text
binding becomes represented
        ↓
the representation is tested under controlled rewrites
        ↓
DAS intervenes on a rank-1 binding component
        ↓
the answer follows the installed binding
        ↓
the R-lens attributes the unedited answer to the active definition
```

The security, output-vocabulary, standalone J-lens, and taint-routing tracks are
preserved in [ARCHIVE.md](ARCHIVE.md). They are not required for this argument.

## 1. Representation

The controlled programs change which definition a variable use resolves to while
holding the queried token, its position, and its bounded local context fixed. At
the input layer, binding accuracy is at the construction-pinned 0.500 floor.
Through contextual processing, a linear readout rises to approximately 0.984 in
middle layers.

This supports a representational claim: binding becomes linearly recoverable from
the model's contextual state. It does not yet show that downstream computation
uses the recovered information.

Definition-to-use edges show a similar profile and provide a closely related
structural replication.

## 2. Robustness

A probe trained on clean programs is frozen and evaluated after
meaning-preserving changes.

Long irrelevant context is comparatively cheap. Confusing the program with
competing uses of the tracked names is much more damaging. Consistent renaming
strongly disrupts the input and earliest layers, but much of the middle-layer
readout survives. Opaque predicates and equivalent arithmetic rewrites add
little damage. Control-flow flattening produces the largest reproducible
collapse.

The supported conclusion is about transfer of the original linear readout. When
it fails, the experiment does not prove that all possible binding information
has vanished.

## 3. Causal use with DAS

DAS is fitted at the unchanged variable-use token. It learns a rank-1 subspace
and replaces only the host program's component in that subspace with the donor
program's component.

The values assigned to the outer and inner definitions are crossed:

| arm | binding installation requires |
|---|---|
| fitted arm | answer `a → b` |
| held-out arm | answer `b → a` |

This crossing distinguishes binding from a fixed answer-token direction. A
direction meaning “increase token `b`” should not follow the reversed
requirement. A direction carrying “use the donor's definition” should.

DAS makes DeepSeek-Coder 6.7B and StarCoder2 3B emit the value selected by the
installed binding on 100% of held-out cases in both arms.

The comparison set separates several simpler explanations. A dose-matched answer
direction tests whether DAS merely pushes toward the token required during
fitting. Dose- and rank-matched random subspaces test generic disruption and set
random low-rank floors. A no-op detects intervention machinery artifacts, a full
donor-state patch verifies that the site can affect the answer, and a closed-form
mean donor−host direction tests the simplest non-learned rank-1 alternative.

The answer direction attenuates or reverses across the crossed arm, while DAS
does not. Random controls are weaker. The mean direction transports part of the
binding but requires a larger edit and remains less reliable.

This is the causal result: at the tested site and layer, downstream computation
uses a compact component whose effect follows which definition is in scope.

## 4. Binding attribution with the R-lens

The R-lens reads the same binding programs without changing the forward model. It
propagates the selected bound-value score backward and divides it among syntactic
roles while conserving the score.

Only the inner definition's name changes between the binding conditions. The
outer definition, inner value, use token, and other measured roles are
token-identical. The primary question is whether relevance moves between these
unchanged definitions.

On DeepSeek-Coder 6.7B, the newly active inner value gains relevance and the
newly inactive outer definition loses it on all 280 held-out bases. The combined
shift is approximately 13% at the first measured layer, peaks near 22% in the
middle, and declines toward the end. The changed name token carries only about
1.5% of the movement.

The result survives reversing the value assignment and scoring both programs at
the same fixed output token. Scoring the competing value reverses the relevance
shift as predicted, while controls that change values without changing the
binding remain flat. Together these comparisons separate binding-sensitive
attribution from a response to one answer token or to the changed input name.

The effect is best interpreted as a stable property of the template-level
binding contrast. A mismatched-base control reproduces it because all generated
bases share one template.

The 1.3B result is not interpreted: non-positive bound-value scores make its
normalized relevance shares unstable. StarCoder2 is outside the implemented
R-lens rules.

## Next experiment: internal semantic verbalisation

The next planned experiment will apply a frozen J-lens at the unchanged
variable-use position, without appending a question. It will test whether the
same intermediate states from which the binding probe succeeds align with a
small, predeclared semantic lexicon. This is an open question, not a current
result. The retired prompted study and its artifacts are recorded in
[ARCHIVE.md](ARCHIVE.md).

## The combined conclusion

The strongest supported statement is:

> In controlled programs, variable binding becomes linearly represented, remains
> stable under many surface changes but is fragile to structural interference,
> is causally read from a rank-1 component at the use site, is reflected in how
> the final answer is attributed to the active definition.

DAS and the R-lens are deliberately not merged into one claim. DAS edits the
model and establishes causal use. The R-lens edits nothing and establishes
attribution under a specified set of backward rules.

No experiment here establishes a complete mechanism. In particular, the project
does not yet show whether the probe-detectable binding state is expressible in
semantic vocabulary at the same intermediate position and layers.

## Where to read next

- [RESULTS.md](RESULTS.md): complete evidence, controls, numbers, and limits.
- [METHODS.md](METHODS.md): construction and instrument details.
- [DeepSeek DAS report](../results/binding/deepseek-coder-6.7b/e13_report.md).
- [StarCoder2 DAS report](../results/binding/starcoder2-3b/e13_report.md).
- [DeepSeek binding R-lens report](../results/binding/deepseek-coder-6.7b/e16_report.md).
- [ARCHIVE.md](ARCHIVE.md): displaced studies and failed designs.
