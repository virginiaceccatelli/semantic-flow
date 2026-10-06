# BranchExec: is the branch decided at the `if`, and does that decision drive the answer?

All rows are **real code** unless marked synthetic. Synthetic pairs only supplied the directions, the layer and the dose.

## Data

| source | programs | sites | members | pairs |
|---|---:|---:|---:|---:|
| synthetic | 150 | 92 | 258 | 166 |
| cruxeval | 60 | 0 | 0 | 0 |
| mbpp | 60 | 11 | 32 | 21 |
| humaneval | 60 | 9 | 25 | 16 |

## 1. Is the branch outcome represented at the `if`? (readout)

Direction: synthetic difference-in-means, layer **24** chosen on synthetic validation (pair accuracy 0.765). Chance is exactly 0.5: the code is token-identical within a pair.

| read | order | pair accuracy at l* | 95% CI | real in-domain ceiling at l* |
|---|---|---:|---|---:|
| cond_last | first | 0.622 | [0.500, 0.778] | — |
| colon | first | 0.649 | [0.500, 0.833] | 0.486 |
| body_first | first | 0.568 | [0.250, 0.790] | — |
| def_colon | first | 0.378 | [0.194, 0.550] | — |
| answer | first | 0.405 | [0.175, 0.643] | 0.405 |
| cond_last | last | 0.500 | [0.500, 0.500] | — |
| colon | last | 0.500 | [0.500, 0.500] | 0.500 |
| answer | last | 0.514 | [0.286, 0.750] | 0.378 |

- Structural zero (input-last order, the `if` precedes the input): max |Δh| across pairs = 0; pair accuracy there is 0.5 by construction.
- Model-free input-token reader (grouped CV): pair accuracy 0.284.
- Frozen synthetic logistic probe at l*: 0.811 [0.667, 0.970].
- Shuffled-label synthetic direction at the colon: 0.459.

## 2. Behaviour (no intervention)

| split | source | n | acc, input first | acc, input last | acc on negated program | capable | answer = o_flip |
|---|---|---:|---:|---:|---:|---:|---:|
| real | humaneval | 25 | 0.240 | 0.720 | 0.320 | 0.000 | 0.320 |
| real | mbpp | 32 | 0.344 | 0.562 | 0.188 | 0.031 | 0.250 |
| syn_val | synthetic | 53 | 0.415 | 0.453 | 0.434 | 0.113 | 0.264 |

## 3. Is the decision used? (steering toward the other branch)

Blocks [22, 23, 24, 25, 26]; primary dose α = 0.05 chosen on synthetic validation (syn_val). Flip = greedy output is exactly `o_flip`, the output of the program with this `if` negated on the same input.

### Real, capable members (primary)

| condition | n | flip to o_flip | 95% CI | keeps o | other | Δ log-odds(o_flip vs o) | 95% CI |
|---|---:|---:|---|---:|---:|---:|---|
| semantic | 1 | 0.000 | [0.000, 0.000] | 1.000 | 0.000 | -0.00 | [-0.00, -0.00] |
| reverse | 1 | 0.000 | [0.000, 0.000] | 1.000 | 0.000 | +0.00 | [+0.00, +0.00] |
| random | 1 | 0.000 | [0.000, 0.000] | 1.000 | 0.000 | +0.00 | [+0.00, +0.00] |
| shuffled | 1 | 0.000 | [0.000, 0.000] | 1.000 | 0.000 | +0.00 | [+0.00, +0.00] |
| wrong_site | 1 | 0.000 | [0.000, 0.000] | 1.000 | 0.000 | -0.00 | [-0.00, -0.00] |
| answer_site | 1 | 0.000 | [0.000, 0.000] | 1.000 | 0.000 | +0.17 | [+0.17, +0.17] |
| actuator_cond | 1 | 0.000 | [0.000, 0.000] | 1.000 | 0.000 | +0.02 | [+0.02, +0.02] |
| actuator_answer | 1 | 1.000 | [1.000, 1.000] | 0.000 | 0.000 | +5.20 | [+5.20, +5.20] |

### Real, all members

| condition | n | flip to o_flip | 95% CI | keeps o | other | Δ log-odds(o_flip vs o) | 95% CI |
|---|---:|---:|---|---:|---:|---:|---|
| semantic | 30 | 0.200 | [0.048, 0.438] | 0.367 | 0.433 | +0.01 | [+0.00, +0.01] |
| reverse | 30 | 0.200 | [0.048, 0.438] | 0.367 | 0.433 | -0.01 | [-0.01, -0.00] |
| random | 30 | 0.200 | [0.048, 0.438] | 0.367 | 0.433 | -0.00 | [-0.00, +0.00] |
| shuffled | 30 | 0.200 | [0.048, 0.438] | 0.367 | 0.433 | +0.00 | [-0.00, +0.00] |
| wrong_site | 30 | 0.200 | [0.048, 0.438] | 0.367 | 0.433 | -0.00 | [-0.00, +0.00] |
| answer_site | 30 | 0.200 | [0.048, 0.438] | 0.367 | 0.433 | -0.02 | [-0.11, +0.05] |
| actuator_cond | 27 | 0.222 | [0.056, 0.438] | 0.407 | 0.370 | -0.01 | [-0.02, +0.01] |
| actuator_answer | 27 | 0.667 | [0.206, 1.000] | 0.000 | 0.333 | +2.81 | [+0.85, +5.09] |

### Synthetic validation (where the dose was chosen)

| condition | n | flip to o_flip | 95% CI | keeps o | other | Δ log-odds(o_flip vs o) | 95% CI |
|---|---:|---:|---|---:|---:|---:|---|
| semantic | 6 | 0.000 | [0.000, 0.000] | 1.000 | 0.000 | +0.00 | [-0.01, +0.01] |
| reverse | 6 | 0.000 | [0.000, 0.000] | 1.000 | 0.000 | -0.00 | [-0.01, +0.01] |
| random | 6 | 0.000 | [0.000, 0.000] | 1.000 | 0.000 | +0.00 | [-0.01, +0.01] |
| shuffled | 6 | 0.000 | [0.000, 0.000] | 1.000 | 0.000 | -0.00 | [-0.01, +0.00] |
| wrong_site | 6 | 0.000 | [0.000, 0.000] | 1.000 | 0.000 | -0.00 | [-0.01, +0.00] |
| answer_site | 6 | 0.000 | [0.000, 0.000] | 1.000 | 0.000 | +0.02 | [-0.00, +0.04] |
| actuator_cond | 6 | 0.000 | [0.000, 0.000] | 1.000 | 0.000 | -0.01 | [-0.02, +0.00] |
| actuator_answer | 6 | 0.000 | [0.000, 0.000] | 1.000 | 0.000 | -0.02 | [-0.03, -0.01] |

### Semantic condition by source (real, capable)

| source | n | flip rate |
|---|---:|---:|
| mbpp | 1 | 0.000 |

Dose curves for every condition: `dose_curves.csv` and `branchexec.png`.

## 4. The model's own errors

Agreement = projection at l*/colon, oriented so that positive means the readout points to the branch that truly runs.

| unsteered answer | n | mean agreement | share reading the other branch |
|---|---:|---:|---:|
| o | 17 | -2.324 | 0.529 |
| o_flip | 16 | +1.330 | 0.688 |
| other | 24 | +4.039 | 0.333 |

AUROC (correct vs answered o_flip) of the agreement score: 0.526.

## Reading the pattern

Readout at the `if` 0.649, at the answer 0.405; flip rate semantic 0.000, random 0.000, answer-site 0.000, actuator at the `if` 0.000. Pattern: **represented, not shown to be used** at the tested sites and doses. Thresholds here (0.6 readout, +0.05 flip margin) only label the pattern; the tables carry the evidence.
