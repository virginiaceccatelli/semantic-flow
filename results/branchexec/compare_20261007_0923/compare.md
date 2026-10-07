# Representation versus utilisation of branch outcomes, across models

Real code only. Representation: frozen synthetic direction, pair accuracy at the `if` (chance 0.5). Utilisation: among answers that are one of the two branch outputs, the share following the true branch.

| model | real pairs | l* (depth) | readout at `if` [95% CI] | real ceiling | readout at answer (input first / last) |
|---|---:|---|---|---:|---|
| deepseek-coder-1.3b | 735 | 8 (0.38) | 0.595 [0.551, 0.638] | 0.682 | 0.556 / 0.512 |
| starcoder2-3b | 743 | 13 (0.47) | 0.678 [0.639, 0.721] | 0.673 | 0.556 / 0.483 |
| deepseek-coder-6.7b | 735 | 16 (0.53) | 0.746 [0.700, 0.791] | 0.796 | 0.582 / 0.581 |
| starcoder2-7b | 743 | 21 (0.69) | 0.680 [0.637, 0.726] | 0.651 | 0.485 / 0.592 |

| model | accuracy (input first / last) | follows true branch | `if`-body share | same branch for both inputs (bias-only expectation) | tracking pairs |
|---|---|---:|---:|---|---:|
| deepseek-coder-1.3b | 0.315 / 0.375 | 0.533 | 0.626 | 0.926 (0.532) | 0.055 |
| starcoder2-3b | 0.312 / 0.416 | 0.508 | 0.651 | 0.913 (0.545) | 0.066 |
| deepseek-coder-6.7b | 0.311 / 0.485 | 0.511 | 0.694 | 0.877 (0.576) | 0.110 |
| starcoder2-7b | 0.331 / 0.442 | 0.514 | 0.671 | 0.886 (0.559) | 0.096 |

| model | readout in tracking pairs | readout in same-branch pairs | within-branch AUROC |
|---|---:|---:|---:|
| deepseek-coder-1.3b | 0.591 | 0.598 | 0.565 |
| starcoder2-3b | 0.714 | 0.661 | 0.522 |
| deepseek-coder-6.7b | 0.791 | 0.755 | 0.707 |
| starcoder2-7b | 0.791 | 0.667 | 0.640 |

Repair of wrong-branch answers at α = 0.4 (share now exactly correct):

| model | n | at `if`: true / away / random | at answer: true / away / random | answer-token actuator |
|---|---:|---|---|---:|
| deepseek-coder-1.3b | 322 | 0.130 / 0.040 / 0.034 | 0.289 / 0.258 / 0.093 | 0.832 |
| starcoder2-3b | 358 | 0.215 / 0.006 / 0.028 | 0.162 / 0.179 / 0.089 | 0.722 |
| deepseek-coder-6.7b | 347 | 0.020 / 0.014 / 0.003 | 0.438 / 0.199 / 0.040 | 0.648 |
| starcoder2-7b | 371 | 0.022 / 0.027 / 0.013 | 0.043 / 0.059 / 0.065 | 0.680 |

Branch-following in the CruxEval order (input after the code; stage 259):

| model | accuracy | decided | follows true branch | `if`-body share | same branch for both inputs (bias-only expectation) | tracking pairs | same branch chosen in both orders |
|---|---:|---:|---:|---:|---|---:|---:|
| deepseek-coder-1.3b | 0.375 | 715 | 0.613 | 0.624 | 0.684 (0.531) | 0.293 | 0.613 |
| starcoder2-3b | 0.413 | 781 | 0.626 | 0.598 | 0.624 (0.519) | 0.337 | 0.666 |
| deepseek-coder-6.7b | 0.485 | 785 | 0.721 | 0.580 | 0.450 (0.513) | 0.532 | 0.713 |
| starcoder2-7b | 0.441 | 806 | 0.648 | 0.615 | 0.579 (0.527) | 0.390 | 0.741 |

Single-block repair (stage 260): best margin (toward − away) per site, with the relative depth of that block:

| model | `if` | first body token | answer | overall best |
|---|---|---|---|---|
| deepseek-coder-1.3b | +0.053 @ 0.29 | +0.006 @ 0.29 | +0.245 @ 0.71 | answer @ 0.71 |
| starcoder2-3b | +0.204 @ 0.37 | +0.034 @ 0.50 | +0.075 @ 0.23 | if @ 0.37 |
| deepseek-coder-6.7b | +0.063 @ 0.34 | +0.006 @ 0.34 | +0.294 @ 0.53 | answer @ 0.53 |
| starcoder2-7b | +0.124 @ 0.41 | +0.027 @ 0.34 | +0.070 @ 0.66 | if @ 0.41 |
