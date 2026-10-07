# Stage 257: repairing wrong-branch answers

358 real members whose unsteered answer is the wrong branch's output. Blocks [11, 12, 13, 14, 15]. Repair = greedy answer becomes exactly the true output (0 before steering).

## Repair rate by dose

| condition | α=0.05 | α=0.1 | α=0.2 | α=0.4 | α=0.8 |
|---|---:|---:|---:|---:|---:|
| actuator_answer | 0.413 | 0.587 | 0.711 | 0.722 | 0.680 |
| answer_away | 0.020 | 0.039 | 0.084 | 0.179 | 0.045 |
| answer_true | 0.042 | 0.075 | 0.112 | 0.162 | 0.011 |
| if_away | 0.008 | 0.008 | 0.011 | 0.006 | 0.039 |
| if_true | 0.056 | 0.115 | 0.151 | 0.215 | 0.321 |
| random_answer | 0.034 | 0.045 | 0.067 | 0.089 | 0.106 |
| random_if | 0.014 | 0.014 | 0.020 | 0.028 | 0.050 |
| shuffled_if | 0.025 | 0.061 | 0.092 | 0.115 | 0.117 |

## Log-odds shift toward the true output, by dose

| condition | α=0.05 | α=0.1 | α=0.2 | α=0.4 | α=0.8 |
|---|---:|---:|---:|---:|---:|
| actuator_answer | +0.70 | +1.49 | +3.84 | +13.31 | +50.84 |
| answer_away | +0.01 | +0.03 | +0.04 | -0.26 | +0.66 |
| answer_true | -0.01 | -0.01 | +0.01 | +0.01 | +0.38 |
| if_away | -0.09 | -0.18 | -0.28 | -0.29 | -0.18 |
| if_true | +0.09 | +0.16 | +0.27 | +0.41 | +0.55 |
| random_answer | +0.00 | +0.01 | +0.00 | +0.04 | +0.34 |
| random_if | +0.00 | +0.01 | +0.02 | +0.03 | +0.08 |
| shuffled_if | +0.02 | +0.05 | +0.11 | +0.16 | +0.18 |

Confidence intervals (program-clustered bootstrap): `repair_summary.csv`.
