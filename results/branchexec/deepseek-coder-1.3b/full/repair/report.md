# Stage 257: repairing wrong-branch answers

322 real members whose unsteered answer is the wrong branch's output. Blocks [6, 7, 8, 9, 10]. Repair = greedy answer becomes exactly the true output (0 before steering).

## Repair rate by dose

| condition | α=0.05 | α=0.1 | α=0.2 | α=0.4 | α=0.8 |
|---|---:|---:|---:|---:|---:|
| actuator_answer | 0.438 | 0.649 | 0.733 | 0.832 | 0.438 |
| answer_away | 0.031 | 0.050 | 0.106 | 0.258 | 0.006 |
| answer_true | 0.028 | 0.075 | 0.152 | 0.289 | 0.000 |
| if_away | 0.003 | 0.003 | 0.016 | 0.040 | 0.071 |
| if_true | 0.019 | 0.028 | 0.059 | 0.130 | 0.261 |
| random_answer | 0.016 | 0.031 | 0.047 | 0.093 | 0.186 |
| random_if | 0.009 | 0.012 | 0.022 | 0.034 | 0.068 |
| shuffled_if | 0.006 | 0.009 | 0.012 | 0.047 | 0.168 |

## Log-odds shift toward the true output, by dose

| condition | α=0.05 | α=0.1 | α=0.2 | α=0.4 | α=0.8 |
|---|---:|---:|---:|---:|---:|
| actuator_answer | +0.63 | +1.32 | +3.03 | +8.10 | +25.80 |
| answer_away | -0.02 | -0.03 | +0.00 | +0.38 | +0.34 |
| answer_true | +0.03 | +0.06 | +0.15 | +0.38 | +0.85 |
| if_away | -0.02 | -0.03 | -0.04 | -0.01 | -0.03 |
| if_true | +0.02 | +0.05 | +0.09 | +0.14 | +0.21 |
| random_answer | +0.01 | +0.02 | +0.03 | +0.08 | +0.42 |
| random_if | +0.00 | +0.00 | +0.00 | +0.01 | +0.05 |
| shuffled_if | -0.01 | -0.01 | -0.01 | +0.06 | +0.20 |

Confidence intervals (program-clustered bootstrap): `repair_summary.csv`.
