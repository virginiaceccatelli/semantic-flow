# Stage 256: readout ↔ the model's own branch choice

Pair categories (real pairs, both members' unsteered answers): same_branch 396, undecided 296, tracks 43, inverted 8

`tracks` = right branch for both inputs; `same_branch` = one branch for both inputs; `inverted` = wrong branch for both. Pair accuracy has chance exactly 0.5 in every row.

| read (l*) | category | pairs | pair accuracy | 95% CI |
|---|---|---:|---:|---|
| colon | tracks | 43 | 0.791 | [0.622, 0.944] |
| colon | same_branch | 396 | 0.667 | [0.607, 0.729] |
| colon | inverted | 8 | 0.625 | [0.333, 0.875] |
| colon | undecided | 296 | 0.682 | [0.620, 0.744] |
| colon | all | 743 | 0.680 | [0.639, 0.725] |
| answer | tracks | 43 | 0.442 | [0.237, 0.652] |
| answer | same_branch | 396 | 0.437 | [0.375, 0.501] |
| answer | inverted | 8 | 0.500 | [0.111, 0.861] |
| answer | undecided | 296 | 0.554 | [0.479, 0.628] |
| answer | all | 743 | 0.485 | [0.442, 0.531] |

## Within true branch: does the projection predict the branch the model answers?

| true branch | members | share answering the taken branch | AUROC | 95% CI |
|---|---:|---:|---:|---|
| not_taken | 396 | 0.652 | 0.592 | [0.485, 0.692] |
| taken | 367 | 0.692 | 0.693 | [0.587, 0.782] |

Weighted mean within-branch AUROC: 0.640 (0.5 = no link).

Layer profiles per category: `link_pairs.csv`.
