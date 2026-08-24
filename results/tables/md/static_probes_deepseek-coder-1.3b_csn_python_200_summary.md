# Static probes — deepseek-coder-1.3b_csn_python_200

## What this experiment asks

This table tests whether a linear classifier can recover a program relation from frozen model hidden states. Read the model score against the measured surface baseline and shuffled-label control: a higher hidden-state score supports representation of the relation, but does not by itself show that the model uses it.

| task        |   peak_layer |   accuracy |   selectivity |   auc |   control_accuracy |   n_groups | converged   |
|:------------|-------------:|-----------:|--------------:|------:|-------------------:|-----------:|:------------|
| binding     |            3 |      0.914 |         0.329 | 0.980 |              0.585 |        182 | True        |
| defuse_edge |            3 |      0.912 |         0.318 | 0.975 |              0.594 |        199 | True        |
