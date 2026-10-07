# BranchExec: is the branch decided at the `if`, and does that decision drive the answer?

All rows are **real code** unless marked synthetic. Synthetic pairs only supplied the directions, the layer and the dose.

## Data

| source | programs | sites | members | pairs |
|---|---:|---:|---:|---:|
| synthetic | 1200 | 723 | 2025 | 1302 |
| cruxeval | 800 | 44 | 117 | 73 |
| mbpp | 972 | 296 | 816 | 520 |
| humaneval | 164 | 93 | 235 | 142 |

## 1. Is the branch outcome represented at the `if`? (readout)

Direction: synthetic difference-in-means, layer **8** chosen on synthetic validation (pair accuracy 0.826). Chance is exactly 0.5: the code is token-identical within a pair.

| read | order | pair accuracy at l* | 95% CI | real in-domain ceiling at l* |
|---|---|---:|---|---:|
| cond_last | first | 0.469 | [0.427, 0.510] | — |
| colon | first | 0.595 | [0.551, 0.638] | 0.682 |
| body_first | first | 0.637 | [0.594, 0.682] | — |
| def_colon | first | 0.525 | [0.481, 0.569] | — |
| answer | first | 0.556 | [0.515, 0.596] | 0.562 |
| cond_last | last | 0.500 | [0.500, 0.500] | — |
| colon | last | 0.500 | [0.500, 0.500] | 0.500 |
| answer | last | 0.512 | [0.463, 0.562] | 0.567 |

- Structural zero (input-last order, the `if` precedes the input): max |Δh| across pairs = 0; pair accuracy there is 0.5 by construction.
- Model-free input-token reader (grouped CV): pair accuracy 0.473.
- Frozen synthetic logistic probe at l*: 0.637 [0.593, 0.683].
- Shuffled-label synthetic direction at the colon: 0.437.

## 2. Behaviour (no intervention)

| split | source | n | acc, input first | acc, input last | acc on negated program | capable | answer = o_flip |
|---|---|---:|---:|---:|---:|---:|---:|
| real | cruxeval | 117 | 0.291 | 0.282 | 0.214 | 0.009 | 0.179 |
| real | humaneval | 235 | 0.285 | 0.438 | 0.289 | 0.009 | 0.268 |
| real | mbpp | 816 | 0.327 | 0.370 | 0.311 | 0.054 | 0.292 |
| syn_val | synthetic | 375 | 0.179 | 0.277 | 0.323 | 0.019 | 0.315 |

## 3. Is the decision used? (steering toward the other branch)

Blocks [6, 7, 8, 9, 10]; primary dose α = 0.02 chosen on synthetic validation (syn_val). Flip = greedy output is exactly `o_flip`, the output of the program with this `if` negated on the same input.

### Real, capable members (primary)

| condition | n | flip to o_flip | unsteered flip | 95% CI | keeps o | other | Δ log-odds(o_flip vs o) | 95% CI |
|---|---:|---:|---:|---|---:|---:|---:|---|
| semantic | 47 | 0.064 | 0.000 | [0.000, 0.162] | 0.936 | 0.000 | +0.02 | [+0.01, +0.03] |
| reverse | 47 | 0.021 | 0.000 | [0.000, 0.077] | 0.979 | 0.000 | -0.02 | [-0.03, -0.01] |
| random | 47 | 0.000 | 0.000 | [0.000, 0.000] | 1.000 | 0.000 | +0.00 | [-0.00, +0.01] |
| shuffled | 47 | 0.000 | 0.000 | [0.000, 0.000] | 1.000 | 0.000 | -0.01 | [-0.01, +0.00] |
| wrong_site | 47 | 0.000 | 0.000 | [0.000, 0.000] | 1.000 | 0.000 | -0.00 | [-0.01, +0.01] |
| answer_site | 47 | 0.064 | 0.000 | [0.000, 0.158] | 0.936 | 0.000 | +0.02 | [+0.00, +0.04] |
| actuator_cond | 47 | 0.000 | 0.000 | [0.000, 0.000] | 1.000 | 0.000 | +0.00 | [-0.01, +0.01] |
| actuator_answer | 47 | 0.447 | 0.000 | [0.265, 0.652] | 0.553 | 0.000 | +0.29 | [+0.23, +0.35] |

### Real, all members

| condition | n | flip to o_flip | unsteered flip | 95% CI | keeps o | other | Δ log-odds(o_flip vs o) | 95% CI |
|---|---:|---:|---:|---|---:|---:|---:|---|
| semantic | 1168 | 0.279 | 0.276 | [0.239, 0.319] | 0.312 | 0.409 | +0.01 | [+0.01, +0.01] |
| reverse | 1168 | 0.275 | 0.276 | [0.236, 0.313] | 0.316 | 0.409 | -0.01 | [-0.01, -0.01] |
| random | 1168 | 0.276 | 0.276 | [0.237, 0.315] | 0.317 | 0.408 | -0.00 | [-0.00, +0.00] |
| shuffled | 1168 | 0.275 | 0.276 | [0.236, 0.314] | 0.316 | 0.409 | -0.01 | [-0.01, -0.00] |
| wrong_site | 1168 | 0.275 | 0.276 | [0.236, 0.314] | 0.316 | 0.409 | -0.00 | [-0.00, -0.00] |
| answer_site | 1168 | 0.277 | 0.276 | [0.238, 0.315] | 0.316 | 0.408 | +0.01 | [-0.00, +0.01] |
| actuator_cond | 1164 | 0.277 | 0.277 | [0.238, 0.316] | 0.315 | 0.407 | -0.00 | [-0.01, -0.00] |
| actuator_answer | 1164 | 0.333 | 0.277 | [0.284, 0.383] | 0.262 | 0.405 | +0.17 | [+0.15, +0.20] |

### Synthetic validation (where the dose was chosen)

| condition | n | flip to o_flip | unsteered flip | 95% CI | keeps o | other | Δ log-odds(o_flip vs o) | 95% CI |
|---|---:|---:|---:|---|---:|---:|---:|---|
| semantic | 7 | 0.000 | 0.000 | [0.000, 0.000] | 1.000 | 0.000 | -0.02 | [-0.02, -0.01] |
| reverse | 7 | 0.000 | 0.000 | [0.000, 0.000] | 1.000 | 0.000 | -0.01 | [-0.03, +0.03] |
| random | 7 | 0.000 | 0.000 | [0.000, 0.000] | 1.000 | 0.000 | -0.01 | [-0.02, +0.00] |
| shuffled | 7 | 0.000 | 0.000 | [0.000, 0.000] | 1.000 | 0.000 | -0.00 | [-0.02, +0.03] |
| wrong_site | 7 | 0.000 | 0.000 | [0.000, 0.000] | 1.000 | 0.000 | -0.02 | [-0.03, -0.01] |
| answer_site | 7 | 0.000 | 0.000 | [0.000, 0.000] | 1.000 | 0.000 | -0.05 | [-0.08, -0.00] |
| actuator_cond | 7 | 0.000 | 0.000 | [0.000, 0.000] | 1.000 | 0.000 | -0.03 | [-0.05, -0.01] |
| actuator_answer | 7 | 0.000 | 0.000 | [0.000, 0.000] | 1.000 | 0.000 | +0.01 | [-0.00, +0.03] |

### Semantic condition by source (real, capable)

| source | n | flip rate |
|---|---:|---:|
| cruxeval | 1 | 0.000 |
| humaneval | 2 | 0.000 |
| mbpp | 44 | 0.068 |

Dose curves for every condition: `dose_curves.csv` and `branchexec.png`.

## 4. The model's own errors

Moved to stage 256 (`link/report.md`), which compares the readout with the model's answer within the same true branch and within pairs. The pooled comparison is confounded by the model's branch bias and is no longer reported here.

## Summary of the numbers

Readout of the branch outcome (real pairs, chance 0.5): 0.595 at the `if`, 0.556 at the answer. Mean log-odds shift toward the other branch's output at α = 0.02, all real members: semantic at the `if` +0.010, reverse -0.011, random -0.001, answer-token push at the `if` -0.004, semantic at the answer +0.005. Use of the outcome by the model's answers and repair of wrong answers are reported by stages 256–258.
