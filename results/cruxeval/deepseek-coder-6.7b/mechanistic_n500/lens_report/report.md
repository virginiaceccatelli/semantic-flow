# CruxEval J/R-lens

Full-vocabulary output-token readout; multi-token outputs are teacher-forced token by token.

| lens       | read     |   layer |   pass@10 |   sequence_pass@10 |   median_rank |
|:-----------|:---------|--------:|----------:|-------------------:|--------------:|
| j-lens     | answer   |      30 |  0.976873 |         0.84       |             0 |
| logit-lens | answer   |      30 |  0.976873 |         0.84       |             0 |
| r-lens     | answer   |      30 |  0.976873 |         0.84       |             0 |
| r-lens     | post_use |       4 |  0.209075 |         0          |           111 |
| r-lens     | use      |       0 |  0.208698 |         0.00167168 |           163 |
| r-lens     | call     |       7 |  0.201434 |         0.004      |           135 |
| j-lens     | call     |       5 |  0.172988 |         0.012      |           105 |
| logit-lens | use      |      30 |  0.132808 |         0.00200602 |            74 |
| j-lens     | use      |      30 |  0.132808 |         0.00200602 |            74 |
| j-lens     | post_use |       5 |  0.129639 |         0.0117018  |           131 |
| logit-lens | post_use |      30 |  0.104594 |         0.0123704  |           121 |
| logit-lens | call     |      30 |  0.103608 |         0          |           105 |

J/R provenance, corpus independence, identity anchor, model-head equivalence, forward invariance, rule binding and nontrivial J/R difference passed before readout.
Lens ranks are observational. Only the separately reported live-model erasure rows are causal, and only relative to their matched controls.
A null before the answer position means these linear vocabulary coordinates do not surface the value there; it does not show the execution state is absent.
