# Stage 257: repairing wrong-branch answers

347 real members whose unsteered answer is the wrong branch's output. Blocks [14, 15, 16, 17, 18]. Repair = greedy answer becomes exactly the true output (0 before steering).

## Repair rate by dose

| condition | α=0.05 | α=0.1 | α=0.2 | α=0.4 | α=0.8 |
|---|---:|---:|---:|---:|---:|
| actuator_answer | 0.533 | 0.622 | 0.663 | 0.648 | 0.110 |
| answer_away | 0.012 | 0.032 | 0.104 | 0.199 | 0.000 |
| answer_true | 0.032 | 0.086 | 0.343 | 0.438 | 0.000 |
| if_away | 0.000 | 0.003 | 0.009 | 0.014 | 0.032 |
| if_true | 0.000 | 0.000 | 0.006 | 0.020 | 0.052 |
| random_answer | 0.000 | 0.003 | 0.017 | 0.040 | 0.118 |
| random_if | 0.000 | 0.000 | 0.003 | 0.003 | 0.003 |
| shuffled_if | 0.000 | 0.000 | 0.003 | 0.003 | 0.014 |

## Log-odds shift toward the true output, by dose

| condition | α=0.05 | α=0.1 | α=0.2 | α=0.4 | α=0.8 |
|---|---:|---:|---:|---:|---:|
| actuator_answer | +1.60 | +3.34 | +6.30 | +10.90 | +37.73 |
| answer_away | -0.16 | -0.29 | -0.45 | -1.13 | -1.36 |
| answer_true | +0.19 | +0.39 | +0.73 | +0.82 | +1.60 |
| if_away | -0.02 | -0.04 | -0.05 | -0.01 | +0.06 |
| if_true | +0.03 | +0.06 | +0.12 | +0.20 | +0.31 |
| random_answer | -0.00 | -0.00 | +0.01 | +0.10 | +0.48 |
| random_if | +0.00 | +0.00 | +0.00 | +0.01 | +0.05 |
| shuffled_if | +0.00 | +0.00 | +0.01 | +0.09 | +0.05 |

Confidence intervals (program-clustered bootstrap): `repair_summary.csv`.
