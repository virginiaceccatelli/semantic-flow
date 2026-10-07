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

Direction: synthetic difference-in-means, layer **21** chosen on synthetic validation (pair accuracy 0.898). Chance is exactly 0.5: the code is token-identical within a pair.

| read | order | pair accuracy at l* | 95% CI | real in-domain ceiling at l* |
|---|---|---:|---|---:|
| cond_last | first | 0.583 | [0.539, 0.628] | — |
| colon | first | 0.680 | [0.637, 0.726] | 0.651 |
| body_first | first | 0.585 | [0.542, 0.631] | — |
| def_colon | first | 0.514 | [0.467, 0.562] | — |
| answer | first | 0.485 | [0.438, 0.532] | 0.602 |
| cond_last | last | 0.500 | [0.500, 0.500] | — |
| colon | last | 0.500 | [0.500, 0.500] | 0.500 |
| answer | last | 0.592 | [0.542, 0.644] | 0.624 |

- Structural zero (input-last order, the `if` precedes the input): max |Δh| across pairs = 0; pair accuracy there is 0.5 by construction.
- Model-free input-token reader (grouped CV): pair accuracy 0.514.
- Frozen synthetic logistic probe at l*: 0.643 [0.601, 0.686].
- Shuffled-label synthetic direction at the colon: 0.590.

## 2. Behaviour (no intervention)

| split | source | n | acc, input first | acc, input last | acc on negated program | capable | answer = o_flip |
|---|---|---:|---:|---:|---:|---:|---:|
| real | cruxeval | 122 | 0.320 | 0.254 | 0.262 | 0.025 | 0.238 |
| real | humaneval | 242 | 0.302 | 0.496 | 0.289 | 0.004 | 0.306 |
| real | mbpp | 820 | 0.341 | 0.454 | 0.338 | 0.045 | 0.327 |
| syn_val | synthetic | 378 | 0.310 | 0.497 | 0.384 | 0.106 | 0.251 |

## 3. Is the decision used? (steering toward the other branch)

Blocks [19, 20, 21, 22, 23]; primary dose α = 0.1 chosen on synthetic validation (syn_val). Flip = greedy output is exactly `o_flip`, the output of the program with this `if` negated on the same input.

### Real, capable members (primary)

| condition | n | flip to o_flip | unsteered flip | 95% CI | keeps o | other | Δ log-odds(o_flip vs o) | 95% CI |
|---|---:|---:|---:|---|---:|---:|---:|---|
| semantic | 41 | 0.024 | 0.024 | [0.000, 0.075] | 0.976 | 0.000 | -0.04 | [-0.08, -0.00] |
| reverse | 41 | 0.073 | 0.024 | [0.000, 0.186] | 0.927 | 0.000 | +0.01 | [-0.03, +0.05] |
| random | 41 | 0.000 | 0.024 | [0.000, 0.000] | 1.000 | 0.000 | -0.02 | [-0.04, -0.00] |
| shuffled | 41 | 0.049 | 0.024 | [0.000, 0.150] | 0.951 | 0.000 | -0.01 | [-0.04, +0.01] |
| wrong_site | 41 | 0.000 | 0.024 | [0.000, 0.000] | 1.000 | 0.000 | -0.01 | [-0.03, +0.00] |
| answer_site | 41 | 0.098 | 0.024 | [0.023, 0.200] | 0.902 | 0.000 | -0.03 | [-0.11, +0.04] |
| actuator_cond | 41 | 0.000 | 0.024 | [0.000, 0.000] | 1.000 | 0.000 | -0.32 | [-0.51, -0.17] |
| actuator_answer | 41 | 0.829 | 0.024 | [0.667, 0.953] | 0.146 | 0.024 | +6.75 | [+5.04, +8.16] |

### Real, all members

| condition | n | flip to o_flip | unsteered flip | 95% CI | keeps o | other | Δ log-odds(o_flip vs o) | 95% CI |
|---|---:|---:|---:|---|---:|---:|---:|---|
| semantic | 1184 | 0.312 | 0.314 | [0.271, 0.353] | 0.332 | 0.356 | -0.01 | [-0.02, +0.00] |
| reverse | 1184 | 0.312 | 0.314 | [0.271, 0.354] | 0.335 | 0.353 | +0.01 | [+0.00, +0.02] |
| random | 1184 | 0.312 | 0.314 | [0.271, 0.352] | 0.332 | 0.356 | +0.00 | [-0.00, +0.01] |
| shuffled | 1184 | 0.312 | 0.314 | [0.271, 0.353] | 0.330 | 0.357 | +0.01 | [-0.00, +0.01] |
| wrong_site | 1184 | 0.312 | 0.314 | [0.271, 0.353] | 0.331 | 0.357 | -0.00 | [-0.00, +0.00] |
| answer_site | 1184 | 0.313 | 0.314 | [0.271, 0.355] | 0.328 | 0.359 | +0.02 | [-0.03, +0.06] |
| actuator_cond | 1180 | 0.240 | 0.314 | [0.201, 0.282] | 0.410 | 0.350 | -0.31 | [-0.38, -0.25] |
| actuator_answer | 1180 | 0.569 | 0.314 | [0.498, 0.640] | 0.098 | 0.333 | +4.22 | [+3.48, +4.96] |

### Synthetic validation (where the dose was chosen)

| condition | n | flip to o_flip | unsteered flip | 95% CI | keeps o | other | Δ log-odds(o_flip vs o) | 95% CI |
|---|---:|---:|---:|---|---:|---:|---:|---|
| semantic | 40 | 0.100 | 0.025 | [0.022, 0.206] | 0.900 | 0.000 | +0.01 | [-0.02, +0.04] |
| reverse | 40 | 0.050 | 0.025 | [0.000, 0.135] | 0.925 | 0.025 | +0.01 | [-0.03, +0.05] |
| random | 40 | 0.000 | 0.025 | [0.000, 0.000] | 1.000 | 0.000 | -0.01 | [-0.03, +0.02] |
| shuffled | 40 | 0.025 | 0.025 | [0.000, 0.086] | 0.975 | 0.000 | -0.01 | [-0.04, +0.02] |
| wrong_site | 40 | 0.025 | 0.025 | [0.000, 0.086] | 0.975 | 0.000 | +0.01 | [-0.02, +0.03] |
| answer_site | 40 | 0.000 | 0.025 | [0.000, 0.000] | 0.975 | 0.025 | -0.02 | [-0.08, +0.02] |
| actuator_cond | 40 | 0.000 | 0.025 | [0.000, 0.000] | 1.000 | 0.000 | -0.31 | [-0.51, -0.15] |
| actuator_answer | 40 | 0.200 | 0.025 | [0.071, 0.353] | 0.800 | 0.000 | +0.39 | [+0.08, +0.87] |

### Semantic condition by source (real, capable)

| source | n | flip rate |
|---|---:|---:|
| cruxeval | 3 | 0.000 |
| humaneval | 1 | 0.000 |
| mbpp | 37 | 0.027 |

Dose curves for every condition: `dose_curves.csv` and `branchexec.png`.

## 4. The model's own errors

Moved to stage 256 (`link/report.md`), which compares the readout with the model's answer within the same true branch and within pairs. The pooled comparison is confounded by the model's branch bias and is no longer reported here.

## Summary of the numbers

Readout of the branch outcome (real pairs, chance 0.5): 0.680 at the `if`, 0.485 at the answer. Mean log-odds shift toward the other branch's output at α = 0.1, all real members: semantic at the `if` -0.009, reverse +0.013, random +0.001, answer-token push at the `if` -0.313, semantic at the answer +0.017. Use of the outcome by the model's answers and repair of wrong answers are reported by stages 256–258.
