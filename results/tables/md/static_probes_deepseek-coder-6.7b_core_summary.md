# Static probes — deepseek-coder-6.7b_core

## What this experiment asks

This table tests whether a linear classifier can recover a program relation from frozen model hidden states. Read the model score against the measured surface baseline and shuffled-label control: a higher hidden-state score supports representation of the relation, but does not by itself show that the model uses it.

| task               |   peak_layer |   accuracy |   selectivity |   auc |   control_accuracy |   n_groups | converged   |
|:-------------------|-------------:|-----------:|--------------:|------:|-------------------:|-----------:|:------------|
| binding            |            0 |      0.975 |         0.391 | 0.998 |              0.585 |        423 | True        |
| control_dep        |           31 |      0.973 |         0.388 | 0.996 |              0.585 |        289 | True        |
| defuse_edge        |            3 |      0.990 |         0.409 | 0.999 |              0.581 |        607 | True        |
| lexical_token_type |            0 |      1.000 |         0.843 | 0.000 |              0.157 |        347 | True        |
| taint_state        |            0 |      1.000 |         0.522 | 1.000 |              0.477 |        200 | True        |
