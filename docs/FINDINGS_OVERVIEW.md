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
the published J-lens tests whether needed values occupy a verbalizable workspace
        ↓
the concrete value does not surface before emission; binding vocabulary does in two DeepSeek panels
```

The earlier cotangent-lens and conserving-cotangent-lens tracks, including E16
and E18, are preserved in [ARCHIVE.md](ARCHIVE.md). They are not results from
the published J-lens or R-lens.

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

The decisive alternative is a separately trained, dose-matched rank-1 answer
actuator. It uses the same site, optimiser, steps and split, but never receives
the donor binding state. Dose- and rank-matched random subspaces test generic disruption and set
random low-rank floors. A no-op detects intervention machinery artifacts, a full
donor-state patch verifies that the site can affect the answer, and a closed-form
mean donor−host direction tests the simplest non-learned rank-1 alternative.

The answer actuator succeeds on its fitted arm and sharply attenuates on the
crossed arm, while DAS remains perfect. Random controls are weaker. The mean direction transports part of the
binding but requires a larger edit and remains less reliable.

This is the causal result: at the tested site and layer, downstream computation
uses a compact component whose effect follows which definition is in scope.

## 4. Lenses: J-lens, with R-lens supporting replication

E19 primarily uses Anthropic's released full-Jacobian J-lens, fitted on an
independent 100-prompt corpus for DeepSeek-Coder 1.3B/6.7B
and StarCoder2-3B. Required applicability, matched-pair, identity-anchor,
forward-invariance, and rule-binding gates pass.

Each value program is read at the use token, the following token, the call site,
and the answer position. All three lenses reach pass@10 = 1.000 at the answer
position. At the preceding three positions, the needed value is essentially
absent on all models, including arithmetic answers absent from the prompt.

Causal erasures use lens-specific distractor directions, stable random
directions, exactly magnitude-matched random displacements, and paired
cluster-bootstrap intervals. Large effects occur near the output. Mid-network,
StarCoder2 is null; DeepSeek 6.7B has only a small L20 effect that does not beat
the logit direction; and DeepSeek 1.3B's L20 logit direction is stronger than J.
The R-lens supporting study has modest local improvements over J but no
consistent advantage over the logit lens. A paper-minimal StarCoder2 fit
omitting the unpublished LayerNorm analogue leaves the conclusion unchanged.

A separate semantic-concept panel is positive on both completed DeepSeek
models. J-lens distinguishes the crossed binding arms with `scope` at layer 9
on 1.3B and `global` at layer 20 on 6.7B, while remaining stable across the two
value assignments. R-lens closely replicates this. Because the word differs by
model and logit-lens directions also carry some signal, the supported claim is
a binding-vocabulary-family signal, not a unique J-lens word code. StarCoder2
has no completed concept panel.

These models represent and causally use binding according to the probe and DAS
evidence, but the published J-space readout does not expose the needed value as a
mid-network verbalizable workspace representation.

## The combined conclusion

The strongest supported statement is:

> In controlled programs, variable binding becomes linearly represented, remains
> stable under many surface changes but is fragile to structural interference,
> is causally read from a rank-1 component at the use site, while the published
> J-lens does not surface the needed concrete values during use and its
> transport does not consistently improve on the logit lens, although it does
> surface controlled binding-related vocabulary in two DeepSeek models. R-lens
> independently supports this pattern.

The concrete-value J-lens null is specific to that published linear,
token-indexed readout. It does not negate the probe and DAS evidence that binding is
represented and used.

## Where to read next

- [RESULTS.md](RESULTS.md): complete evidence, controls, numbers, and limits.
- [METHODS.md](METHODS.md): construction and instrument details.
- [DeepSeek DAS report](../results/binding/deepseek-coder-6.7b/e13_report.md).
- [StarCoder2 DAS report](../results/binding/starcoder2-3b/e13_report.md).
- [J-lens technical report, with R-lens replication](WORKSPACE_LENS.md).
- [ARCHIVE.md](ARCHIVE.md): displaced studies and failed designs.
