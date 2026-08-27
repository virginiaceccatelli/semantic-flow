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
        ↓
the distinction becomes expressible in scope-related output words, late
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

## 5. Verbalisation

Everything above is read in the model's internal coordinates. A separate question
is whether the distinction surfaces in anything the model **emits**.

The same four programs get a question appended, rendered from the *outer* name
only so it is byte-identical in all four cells and the one-token counterfactual
survives:

```
z = 6                        z = 6
def f():                     def f():
    d = 3                        z = 3
    return z                     return z
# Question: does f return the z assigned inside f or outside f? Answer:
#                → outside                        → inside
```

Four question styles in two variants each, with chance pinned at 0.500 by the
factorial, and E13's own value question as a **positive control** built in from
the start — so that a null on the word styles can be told apart from a harness
that could not detect verbalisation if it were there.

**DeepSeek-Coder 6.7B can say it, in some words and not others.** Over 280
held-out programs, asked whether the returned variable is *local* or *global*, it
is right on 0.900, and above chance in both option orders (0.923 and 0.878). The
positive control returns 1.000, so the phrasings that fail are not the harness
failing. Two of the four do fail, each in a diagnosable way: the `inner`/`outer`
wording answers " inner" for every program, and the yes/no wording answers " yes"
to both polarities. And the wording chosen in advance as primary — "inside f or
outside f" — spans 0.502 to 0.980 across two orderings of the same question, so
phrasing dominates the result more than the model does.

**In vocabulary space the effect is large and late.** Mapping the answer-position
state into output coordinates, the inner-pole and outer-pole scope words almost
completely swap mass with the binding at layers 23–27 (+0.821 at layer 27), with
both crossed arms agreeing. Below layer 19 there is essentially nothing. A ranking
of the full 32k vocabulary on calibration bases only, given no lexicon, recovers a
coherent insideness cluster — ` Inside`, ` inside`, ` Within`, ` interior`,
` inner`, ` dentro` — and the scope word family carries three times what the
purely positional family does, and eighteen times a random floor. Since "the
nearest assignment wins" is a positional rule needing no scope concept, that
comparison favours the scope reading.

So the three results sit at three depths: causally used at layer 8, attributed at
layer 15, sayable at layers 23–27. The binding is represented and used well before
it becomes expressible.

**The 1.3B model teaches nothing here**, and the report says so rather than
reporting a null: every word style is at chance, but its positive control also
fails (0.811), so a null cannot be distinguished from an undetectable one.

Two limits carry over every reading. Answering a question about a program is not
introspection — the model can answer it by reading the text as any reader would.
And the attribution half of this experiment is unresolved: it was run on the one
wording the model answers with a constant, and its headline statistic proved
numerically ill-conditioned. One re-run resolves it.

## The combined conclusion

The strongest supported statement is:

> In controlled programs, variable binding becomes linearly represented, remains
> stable under many surface changes but is fragile to structural interference,
> is causally read from a rank-1 component at the use site, is reflected in how
> the final answer is attributed to the active definition, and becomes
> expressible in output-aligned scope vocabulary in the network's final quarter.

DAS and the R-lens are deliberately not merged into one claim. DAS edits the
model and establishes causal use. The R-lens edits nothing and establishes
attribution under a specified set of backward rules.

No experiment here establishes a complete mechanism. The verbalisation study adds
a third observational result rather than strengthening the causal one: it shows
the distinction is expressible in the model's own output vocabulary, not that the
model consults it when answering.

## Where to read next

- [RESULTS.md](RESULTS.md): complete evidence, controls, numbers, and limits.
- [METHODS.md](METHODS.md): construction and instrument details.
- [DeepSeek DAS report](../results/binding/deepseek-coder-6.7b/e13_report.md).
- [StarCoder2 DAS report](../results/binding/starcoder2-3b/e13_report.md).
- [DeepSeek binding R-lens report](../results/binding/deepseek-coder-6.7b/e16_report.md).
- [ARCHIVE.md](ARCHIVE.md): displaced studies and failed designs.
