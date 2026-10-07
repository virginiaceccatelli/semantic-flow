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

Direction: synthetic difference-in-means, layer **16** chosen on synthetic validation (pair accuracy 0.863). Chance is exactly 0.5: the code is token-identical within a pair.

| read | order | pair accuracy at l* | 95% CI | real in-domain ceiling at l* |
|---|---|---:|---|---:|
| cond_last | first | 0.671 | [0.621, 0.720] | — |
| colon | first | 0.746 | [0.700, 0.791] | 0.796 |
| body_first | first | 0.712 | [0.671, 0.751] | — |
| def_colon | first | 0.487 | [0.445, 0.529] | — |
| answer | first | 0.582 | [0.534, 0.632] | 0.638 |
| cond_last | last | 0.500 | [0.500, 0.500] | — |
| colon | last | 0.500 | [0.500, 0.500] | 0.500 |
| answer | last | 0.581 | [0.525, 0.637] | 0.612 |

- Structural zero (input-last order, the `if` precedes the input): max |Δh| across pairs = 0; pair accuracy there is 0.5 by construction.
- Model-free input-token reader (grouped CV): pair accuracy 0.473.
- Frozen synthetic logistic probe at l*: 0.731 [0.684, 0.776].
- Shuffled-label synthetic direction at the colon: 0.482.

## 2. Behaviour (no intervention)

| split | source | n | acc, input first | acc, input last | acc on negated program | capable | answer = o_flip |
|---|---|---:|---:|---:|---:|---:|---:|
| real | cruxeval | 117 | 0.274 | 0.342 | 0.316 | 0.034 | 0.265 |
| real | humaneval | 235 | 0.315 | 0.562 | 0.289 | 0.017 | 0.277 |
| real | mbpp | 816 | 0.315 | 0.483 | 0.325 | 0.034 | 0.308 |
| syn_val | synthetic | 375 | 0.403 | 0.520 | 0.395 | 0.117 | 0.256 |

## 3. Is the decision used? (steering toward the other branch)

Blocks [14, 15, 16, 17, 18]; primary dose α = 0.2 chosen on synthetic validation (syn_val). Flip = greedy output is exactly `o_flip`, the output of the program with this `if` negated on the same input.

### Real, capable members (primary)

| condition | n | flip to o_flip | unsteered flip | 95% CI | keeps o | other | Δ log-odds(o_flip vs o) | 95% CI |
|---|---:|---:|---:|---|---:|---:|---:|---|
| semantic | 36 | 0.056 | 0.000 | [0.000, 0.143] | 0.944 | 0.000 | +0.20 | [+0.10, +0.30] |
| reverse | 36 | 0.000 | 0.000 | [0.000, 0.000] | 1.000 | 0.000 | -0.05 | [-0.09, -0.00] |
| random | 36 | 0.000 | 0.000 | [0.000, 0.000] | 1.000 | 0.000 | +0.01 | [-0.01, +0.02] |
| shuffled | 36 | 0.000 | 0.000 | [0.000, 0.000] | 1.000 | 0.000 | +0.05 | [+0.02, +0.10] |
| wrong_site | 36 | 0.000 | 0.000 | [0.000, 0.000] | 1.000 | 0.000 | +0.02 | [-0.00, +0.03] |
| answer_site | 36 | 0.250 | 0.000 | [0.108, 0.406] | 0.694 | 0.056 | +0.77 | [-0.08, +1.50] |
| actuator_cond | 36 | 0.000 | 0.000 | [0.000, 0.000] | 1.000 | 0.000 | +0.03 | [-0.08, +0.15] |
| actuator_answer | 36 | 0.806 | 0.000 | [0.625, 0.939] | 0.083 | 0.111 | +8.79 | [+7.29, +9.83] |

### Real, all members

| condition | n | flip to o_flip | unsteered flip | 95% CI | keeps o | other | Δ log-odds(o_flip vs o) | 95% CI |
|---|---:|---:|---:|---|---:|---:|---:|---|
| semantic | 1168 | 0.303 | 0.297 | [0.264, 0.343] | 0.313 | 0.384 | +0.07 | [+0.06, +0.09] |
| reverse | 1168 | 0.300 | 0.297 | [0.262, 0.339] | 0.317 | 0.384 | -0.07 | [-0.09, -0.05] |
| random | 1168 | 0.298 | 0.297 | [0.259, 0.338] | 0.312 | 0.390 | +0.00 | [-0.00, +0.00] |
| shuffled | 1168 | 0.301 | 0.297 | [0.262, 0.339] | 0.317 | 0.383 | +0.02 | [+0.01, +0.03] |
| wrong_site | 1168 | 0.301 | 0.297 | [0.262, 0.341] | 0.310 | 0.389 | +0.00 | [+0.00, +0.01] |
| answer_site | 1168 | 0.344 | 0.297 | [0.291, 0.399] | 0.248 | 0.408 | +0.50 | [+0.34, +0.66] |
| actuator_cond | 1164 | 0.296 | 0.298 | [0.258, 0.336] | 0.326 | 0.377 | -0.08 | [-0.10, -0.06] |
| actuator_answer | 1164 | 0.557 | 0.298 | [0.491, 0.623] | 0.088 | 0.356 | +4.29 | [+3.62, +4.95] |

### Synthetic validation (where the dose was chosen)

| condition | n | flip to o_flip | unsteered flip | 95% CI | keeps o | other | Δ log-odds(o_flip vs o) | 95% CI |
|---|---:|---:|---:|---|---:|---:|---:|---|
| semantic | 44 | 0.045 | 0.000 | [0.000, 0.121] | 0.955 | 0.000 | -0.03 | [-0.07, +0.01] |
| reverse | 44 | 0.023 | 0.000 | [0.000, 0.075] | 0.955 | 0.023 | +0.04 | [+0.00, +0.07] |
| random | 44 | 0.023 | 0.000 | [0.000, 0.079] | 0.977 | 0.000 | +0.01 | [-0.01, +0.02] |
| shuffled | 44 | 0.023 | 0.000 | [0.000, 0.075] | 0.955 | 0.023 | +0.04 | [+0.01, +0.07] |
| wrong_site | 44 | 0.000 | 0.000 | [0.000, 0.000] | 1.000 | 0.000 | +0.01 | [-0.00, +0.01] |
| answer_site | 44 | 0.045 | 0.000 | [0.000, 0.122] | 0.886 | 0.068 | +0.26 | [+0.09, +0.47] |
| actuator_cond | 44 | 0.023 | 0.000 | [0.000, 0.075] | 0.955 | 0.023 | -0.00 | [-0.04, +0.04] |
| actuator_answer | 44 | 0.068 | 0.000 | [0.000, 0.158] | 0.932 | 0.000 | +0.32 | [+0.09, +0.62] |

### Semantic condition by source (real, capable)

| source | n | flip rate |
|---|---:|---:|
| cruxeval | 4 | 0.000 |
| humaneval | 4 | 0.000 |
| mbpp | 28 | 0.071 |

Dose curves for every condition: `dose_curves.csv` and `branchexec.png`.

## 4. The model's own errors

Moved to stage 256 (`link/report.md`), which compares the readout with the model's answer within the same true branch and within pairs. The pooled comparison is confounded by the model's branch bias and is no longer reported here.

## Summary of the numbers

Readout of the branch outcome (real pairs, chance 0.5): 0.746 at the `if`, 0.582 at the answer. Mean log-odds shift toward the other branch's output at α = 0.2, all real members: semantic at the `if` +0.075, reverse -0.069, random +0.000, answer-token push at the `if` -0.078, semantic at the answer +0.503. Use of the outcome by the model's answers and repair of wrong answers are reported by stages 256–258.
