# Stage 257: repairing wrong-branch answers

371 real members whose unsteered answer is the wrong branch's output. Blocks [19, 20, 21, 22, 23]. Repair = greedy answer becomes exactly the true output (0 before steering).

## Repair rate by dose

| condition | α=0.05 | α=0.1 | α=0.2 | α=0.4 | α=0.8 |
|---|---:|---:|---:|---:|---:|
| actuator_answer | 0.637 | 0.675 | 0.680 | 0.680 | 0.672 |
| answer_away | 0.013 | 0.016 | 0.059 | 0.059 | 0.000 |
| answer_true | 0.013 | 0.038 | 0.089 | 0.043 | 0.008 |
| if_away | 0.013 | 0.013 | 0.011 | 0.027 | 0.030 |
| if_true | 0.013 | 0.024 | 0.024 | 0.022 | 0.022 |
| random_answer | 0.008 | 0.011 | 0.024 | 0.065 | 0.078 |
| random_if | 0.003 | 0.008 | 0.000 | 0.013 | 0.008 |
| shuffled_if | 0.011 | 0.008 | 0.008 | 0.019 | 0.022 |

## Log-odds shift toward the true output, by dose

| condition | α=0.05 | α=0.1 | α=0.2 | α=0.4 | α=0.8 |
|---|---:|---:|---:|---:|---:|
| actuator_answer | +2.70 | +5.86 | +13.28 | +37.95 | +58.20 |
| answer_away | -0.02 | -0.04 | +0.00 | +0.33 | +0.97 |
| answer_true | +0.02 | +0.02 | -0.01 | -0.11 | +0.05 |
| if_away | +0.00 | -0.00 | -0.01 | -0.01 | +0.01 |
| if_true | -0.01 | -0.02 | -0.03 | -0.06 | -0.08 |
| random_answer | +0.00 | +0.00 | +0.01 | +0.06 | +0.39 |
| random_if | -0.00 | -0.00 | -0.00 | +0.00 | -0.00 |
| shuffled_if | +0.00 | +0.00 | -0.00 | -0.01 | +0.00 |

Confidence intervals (program-clustered bootstrap): `repair_summary.csv`.
