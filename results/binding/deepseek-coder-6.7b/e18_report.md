# E18 — crossed-arm J-lens lexical alignment (deepseek-coder-6.7b)

**Result: binding-associated lexical alignment, not identified verbalisation.**
At the unchanged `x` in `return x`, several predeclared J-lens word margins move
with the binding on nearly every held-out program in both value arms. Since the
literal answer movement reverses between `ab` and `ba`, this cannot be explained
as a fixed preference for output token `a` or `b`. The template also changes
locality, order, distance, and replacement together, so the experiment cannot
identify the alignment uniquely as scope semantics or faithful verbalisation.

## What was measured

The frozen J-lens reads the original, unprompted variable-use state at layers 8,
12, 16, 20, and 24. No question, answer suffix, or generated explanation is
added. Nine predeclared single-token pairs cover scope, positional, and action
vocabulary. The reversal rate is the fraction of 280 held-out programs on which
a word margin moves in the predicted binding-relative direction.

All 1,600 exactness cells passed and all nine pairs survived tokenization. The
calibration-fitted binding probe reaches 1.000 on held-out states at every tested
layer, establishing that the read position contains binding information.

## The `ab`/`ba` control

| arm | outer value | inner value | activating inner binding changes answer |
|---|---|---|---|
| `ab` | `a` | `b` | `a → b` |
| `ba` | `b` | `a` | `b → a` |

A word margin that follows the binding in both arms is therefore not merely
tracking one returned identifier. This is the strongest claim the crossing
licenses.

## Main observations

| pair | family | L16 `ab` / `ba` | L20 `ab` / `ba` | L24 `ab` / `ba` |
|---|---|---:|---:|---:|
| `local/global` | scope | 0.964 / 0.968 | 0.996 / 1.000 | 0.986 / 0.993 |
| `nested/module` | scope | 1.000 / 1.000 | 0.964 / 0.982 | 0.993 / 0.996 |
| `later/earlier` | positional | 1.000 / 1.000 | 0.736 / 0.750 | 0.882 / 0.893 |
| `replaced/kept` | action | 0.925 / 0.946 | 0.982 / 0.982 | 0.814 / 0.793 |

`local/global` becomes strong in the middle layers and remains strong later.
`nested/module` is already saturated early. The directly phrased `inner/outer`
contrast is comparatively weak and inconsistent. Strong positional and action
pairs show that the family label “scope” is not uniquely selected by these data.

## Interpretation and limits

E18 shows that selected vocabulary directions are associated with the hidden
binding counterfactual independently of literal answer identity. It does not
show that the model explicitly represents the proposition “the inner/local
definition is in scope,” that the J-lens recovers the same feature as the probe
or DAS, or that a generated explanation would faithfully report the causal
state. Those stronger claims require program families that independently vary
scope, position, distance, and replacement.

Independent J-lens builds exceed 0.90 sign agreement only at L24, and the
held-out lexical validation contains 13 positions, so early-layer detail is less
secure than the late-layer association. Underlying values are in
[`lexlens_pair_directions.csv`](lexlens/lexlens_pair_directions.csv).
