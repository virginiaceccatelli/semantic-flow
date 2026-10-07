# BranchExec: is the branch decided at the `if`, and does that decision drive the answer?

All rows are **real code** unless marked synthetic. Synthetic pairs only supplied the directions, the layer and the dose.

## Data

| source | programs | sites | members | pairs |
|---|---:|---:|---:|---:|
| synthetic | 1200 | 731 | 2061 | 1330 |
| cruxeval | 800 | 47 | 122 | 75 |
| mbpp | 972 | 299 | 820 | 521 |
| humaneval | 164 | 95 | 242 | 147 |

## 1. Is the branch outcome represented at the `if`? (readout)

Direction: synthetic difference-in-means, layer **13** chosen on synthetic validation (pair accuracy 0.848). Chance is exactly 0.5: the code is token-identical within a pair.

| read | order | pair accuracy at l* | 95% CI | real in-domain ceiling at l* |
|---|---|---:|---|---:|
| cond_last | first | 0.534 | [0.487, 0.579] | — |
| colon | first | 0.678 | [0.639, 0.721] | 0.673 |
| body_first | first | 0.629 | [0.581, 0.674] | — |
| def_colon | first | 0.561 | [0.518, 0.608] | — |
| answer | first | 0.556 | [0.511, 0.606] | 0.563 |
| cond_last | last | 0.500 | [0.500, 0.500] | — |
| colon | last | 0.500 | [0.500, 0.500] | 0.500 |
| answer | last | 0.483 | [0.433, 0.534] | 0.591 |

- Structural zero (input-last order, the `if` precedes the input): max |Δh| across pairs = 0; pair accuracy there is 0.5 by construction.
- Model-free input-token reader (grouped CV): pair accuracy 0.514.
- Frozen synthetic logistic probe at l*: 0.665 [0.625, 0.707].
- Shuffled-label synthetic direction at the colon: 0.568.

## 2. Behaviour (no intervention)

| split | source | n | acc, input first | acc, input last | acc on negated program | capable | answer = o_flip |
|---|---|---:|---:|---:|---:|---:|---:|
| real | cruxeval | 122 | 0.303 | 0.279 | 0.246 | 0.041 | 0.221 |
| real | humaneval | 242 | 0.285 | 0.459 | 0.281 | 0.004 | 0.273 |
| real | mbpp | 820 | 0.321 | 0.424 | 0.333 | 0.055 | 0.326 |
| syn_val | synthetic | 378 | 0.212 | 0.434 | 0.381 | 0.029 | 0.251 |

## 3. Is the decision used? (steering toward the other branch)

Blocks [11, 12, 13, 14, 15]; primary dose α = 0.2 chosen on synthetic validation (syn_val). Flip = greedy output is exactly `o_flip`, the output of the program with this `if` negated on the same input.

### Real, capable members (primary)

| condition | n | flip to o_flip | unsteered flip | 95% CI | keeps o | other | Δ log-odds(o_flip vs o) | 95% CI |
|---|---:|---:|---:|---|---:|---:|---:|---|
| semantic | 51 | 0.471 | 0.020 | [0.268, 0.698] | 0.529 | 0.000 | +0.58 | [+0.22, +0.95] |
| reverse | 51 | 0.078 | 0.020 | [0.017, 0.167] | 0.922 | 0.000 | -0.43 | [-0.73, -0.16] |
| random | 51 | 0.059 | 0.020 | [0.000, 0.137] | 0.941 | 0.000 | +0.01 | [-0.03, +0.04] |
| shuffled | 51 | 0.235 | 0.020 | [0.109, 0.386] | 0.765 | 0.000 | +0.19 | [+0.09, +0.30] |
| wrong_site | 51 | 0.294 | 0.020 | [0.164, 0.460] | 0.706 | -0.000 | +0.12 | [+0.07, +0.18] |
| answer_site | 51 | 0.451 | 0.020 | [0.288, 0.632] | 0.549 | 0.000 | +0.35 | [+0.19, +0.50] |
| actuator_cond | 51 | 0.059 | 0.020 | [0.000, 0.143] | 0.941 | 0.000 | -0.27 | [-0.67, -0.04] |
| actuator_answer | 51 | 0.941 | 0.020 | [0.863, 1.000] | 0.039 | 0.020 | +4.17 | [+2.87, +5.35] |

### Real, all members

| condition | n | flip to o_flip | unsteered flip | 95% CI | keeps o | other | Δ log-odds(o_flip vs o) | 95% CI |
|---|---:|---:|---:|---|---:|---:|---:|---|
| semantic | 1184 | 0.345 | 0.302 | [0.298, 0.394] | 0.265 | 0.389 | +0.25 | [+0.20, +0.31] |
| reverse | 1184 | 0.252 | 0.302 | [0.214, 0.291] | 0.355 | 0.394 | -0.21 | [-0.26, -0.17] |
| random | 1184 | 0.301 | 0.302 | [0.261, 0.341] | 0.309 | 0.390 | -0.01 | [-0.03, +0.01] |
| shuffled | 1184 | 0.324 | 0.302 | [0.282, 0.366] | 0.290 | 0.386 | +0.07 | [+0.05, +0.09] |
| wrong_site | 1184 | 0.322 | 0.302 | [0.278, 0.365] | 0.292 | 0.386 | +0.04 | [+0.02, +0.05] |
| answer_site | 1184 | 0.303 | 0.302 | [0.257, 0.349] | 0.290 | 0.407 | +0.04 | [-0.03, +0.10] |
| actuator_cond | 1180 | 0.281 | 0.302 | [0.242, 0.320] | 0.330 | 0.390 | -0.10 | [-0.17, -0.04] |
| actuator_answer | 1180 | 0.535 | 0.302 | [0.464, 0.607] | 0.090 | 0.375 | +2.86 | [+2.20, +3.60] |

### Synthetic validation (where the dose was chosen)

| condition | n | flip to o_flip | unsteered flip | 95% CI | keeps o | other | Δ log-odds(o_flip vs o) | 95% CI |
|---|---:|---:|---:|---|---:|---:|---:|---|
| semantic | 11 | 0.182 | 0.000 | [0.000, 0.500] | 0.545 | 0.273 | +0.47 | [+0.21, +0.82] |
| reverse | 11 | 0.091 | 0.000 | [0.000, 0.300] | 0.818 | 0.091 | -0.36 | [-0.72, -0.07] |
| random | 11 | 0.000 | 0.000 | [0.000, 0.000] | 1.000 | 0.000 | -0.04 | [-0.13, +0.03] |
| shuffled | 11 | 0.000 | 0.000 | [0.000, 0.000] | 0.909 | 0.091 | +0.21 | [+0.01, +0.47] |
| wrong_site | 11 | 0.000 | 0.000 | [0.000, 0.000] | 0.909 | 0.091 | +0.00 | [-0.09, +0.12] |
| answer_site | 11 | 0.000 | 0.000 | [0.000, 0.000] | 0.909 | 0.091 | +0.00 | [-0.13, +0.17] |
| actuator_cond | 11 | 0.000 | 0.000 | [0.000, 0.000] | 0.909 | 0.091 | +0.04 | [-0.13, +0.25] |
| actuator_answer | 11 | 0.273 | 0.000 | [0.000, 0.556] | 0.636 | 0.091 | +0.28 | [+0.07, +0.45] |

### Semantic condition by source (real, capable)

| source | n | flip rate |
|---|---:|---:|
| cruxeval | 5 | 0.600 |
| humaneval | 1 | 1.000 |
| mbpp | 45 | 0.444 |

Dose curves for every condition: `dose_curves.csv` and `branchexec.png`.

## 4. The model's own errors

Moved to stage 256 (`link/report.md`), which compares the readout with the model's answer within the same true branch and within pairs. The pooled comparison is confounded by the model's branch bias and is no longer reported here.

## Summary of the numbers

Readout of the branch outcome (real pairs, chance 0.5): 0.678 at the `if`, 0.556 at the answer. Mean log-odds shift toward the other branch's output at α = 0.2, all real members: semantic at the `if` +0.251, reverse -0.211, random -0.007, answer-token push at the `if` -0.098, semantic at the answer +0.038. Use of the outcome by the model's answers and repair of wrong answers are reported by stages 256–258.
