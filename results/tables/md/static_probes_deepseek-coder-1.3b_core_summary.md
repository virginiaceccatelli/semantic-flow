# Static probes — deepseek-coder-1.3b_core

## What this experiment asks

This table tests whether a linear classifier can recover a program relation from frozen model hidden states. Read the model score against the measured surface baseline and shuffled-label control: a higher hidden-state score supports representation of the relation, but does not by itself show that the model uses it.

| task               |   peak_layer |   accuracy |   selectivity |   auc |   control_accuracy |   n_groups | converged   |
|:-------------------|-------------:|-----------:|--------------:|------:|-------------------:|-----------:|:------------|
| binding            |            0 |      0.978 |         0.402 | 0.997 |              0.575 |        423 | True        |
| control_dep        |           11 |      0.971 |         0.377 | 0.997 |              0.593 |        289 | True        |
| defuse_edge        |           11 |      0.989 |         0.422 | 0.999 |              0.567 |        607 | True        |
| lexical_token_type |           -1 |      1.000 |         0.862 | 0.000 |              0.138 |        347 | True        |
| taint_state        |            0 |      1.000 |         0.525 | 1.000 |              0.475 |        200 | True        |
