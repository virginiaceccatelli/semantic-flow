# Stage 256: readout ↔ the model's own branch choice

Pair categories (real pairs, both members' unsteered answers): undecided 344, same_branch 343, tracks 43, inverted 5

`tracks` = right branch for both inputs; `same_branch` = one branch for both inputs; `inverted` = wrong branch for both. Pair accuracy has chance exactly 0.5 in every row.

| read (l*) | category | pairs | pair accuracy | 95% CI |
|---|---|---:|---:|---|
| colon | tracks | 43 | 0.791 | [0.643, 0.927] |
| colon | same_branch | 343 | 0.755 | [0.690, 0.813] |
| colon | inverted | 5 | 1.000 | [1.000, 1.000] |
| colon | undecided | 344 | 0.727 | [0.665, 0.791] |
| colon | all | 735 | 0.746 | [0.699, 0.791] |
| answer | tracks | 43 | 0.488 | [0.318, 0.651] |
| answer | same_branch | 343 | 0.557 | [0.488, 0.621] |
| answer | inverted | 5 | 0.400 | [0.000, 0.500] |
| answer | undecided | 344 | 0.622 | [0.548, 0.694] |
| answer | all | 735 | 0.582 | [0.535, 0.631] |

## Within true branch: does the projection predict the branch the model answers?

| true branch | members | share answering the taken branch | AUROC | 95% CI |
|---|---:|---:|---:|---|
| not_taken | 364 | 0.679 | 0.714 | [0.621, 0.795] |
| taken | 346 | 0.711 | 0.699 | [0.578, 0.779] |

Weighted mean within-branch AUROC: 0.707 (0.5 = no link).

Layer profiles per category: `link_pairs.csv`.
