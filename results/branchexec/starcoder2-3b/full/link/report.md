# Stage 256: readout ↔ the model's own branch choice

Pair categories (real pairs, both members' unsteered answers): same_branch 387, undecided 319, tracks 28, inverted 9

`tracks` = right branch for both inputs; `same_branch` = one branch for both inputs; `inverted` = wrong branch for both. Pair accuracy has chance exactly 0.5 in every row.

| read (l*) | category | pairs | pair accuracy | 95% CI |
|---|---|---:|---:|---|
| colon | tracks | 28 | 0.714 | [0.538, 0.889] |
| colon | same_branch | 387 | 0.661 | [0.606, 0.719] |
| colon | inverted | 9 | 0.778 | [0.500, 1.000] |
| colon | undecided | 319 | 0.693 | [0.635, 0.752] |
| colon | all | 743 | 0.678 | [0.638, 0.723] |
| answer | tracks | 28 | 0.536 | [0.333, 0.714] |
| answer | same_branch | 387 | 0.556 | [0.488, 0.621] |
| answer | inverted | 9 | 0.778 | [0.500, 1.000] |
| answer | undecided | 319 | 0.552 | [0.484, 0.617] |
| answer | all | 743 | 0.556 | [0.512, 0.607] |

## Within true branch: does the projection predict the branch the model answers?

| true branch | members | share answering the taken branch | AUROC | 95% CI |
|---|---:|---:|---:|---|
| not_taken | 386 | 0.635 | 0.511 | [0.413, 0.623] |
| taken | 341 | 0.669 | 0.534 | [0.419, 0.651] |

Weighted mean within-branch AUROC: 0.522 (0.5 = no link).

Layer profiles per category: `link_pairs.csv`.
