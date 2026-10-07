# Stage 256: readout ↔ the model's own branch choice

Pair categories (real pairs, both members' unsteered answers): same_branch 373, undecided 332, tracks 22, inverted 8

`tracks` = right branch for both inputs; `same_branch` = one branch for both inputs; `inverted` = wrong branch for both. Pair accuracy has chance exactly 0.5 in every row.

| read (l*) | category | pairs | pair accuracy | 95% CI |
|---|---|---:|---:|---|
| colon | tracks | 22 | 0.591 | [0.400, 0.760] |
| colon | same_branch | 373 | 0.598 | [0.537, 0.656] |
| colon | inverted | 8 | 0.375 | [0.100, 0.714] |
| colon | undecided | 332 | 0.596 | [0.535, 0.659] |
| colon | all | 735 | 0.595 | [0.551, 0.639] |
| answer | tracks | 22 | 0.591 | [0.355, 0.824] |
| answer | same_branch | 373 | 0.563 | [0.500, 0.617] |
| answer | inverted | 8 | 0.125 | [0.000, 0.429] |
| answer | undecided | 332 | 0.557 | [0.498, 0.617] |
| answer | all | 735 | 0.556 | [0.518, 0.595] |

## Within true branch: does the projection predict the branch the model answers?

| true branch | members | share answering the taken branch | AUROC | 95% CI |
|---|---:|---:|---:|---|
| not_taken | 364 | 0.588 | 0.552 | [0.442, 0.653] |
| taken | 326 | 0.669 | 0.579 | [0.472, 0.694] |

Weighted mean within-branch AUROC: 0.565 (0.5 = no link).

Layer profiles per category: `link_pairs.csv`.
