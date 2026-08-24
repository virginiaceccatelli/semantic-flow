# Static probes — deepseek-coder-6.7b_csn_python_200

## What this experiment asks

This table tests whether a linear classifier can recover a program relation from frozen model hidden states. Read the model score against the measured surface baseline and shuffled-label control: a higher hidden-state score supports representation of the relation, but does not by itself show that the model uses it.

| task        |   peak_layer |   accuracy |   selectivity |   auc |   control_accuracy |   n_groups | converged   |
|:------------|-------------:|-----------:|--------------:|------:|-------------------:|-----------:|:------------|
| binding     |            7 |      0.902 |         0.308 | 0.978 |              0.594 |        182 | True        |
| defuse_edge |            3 |      0.911 |         0.308 | 0.979 |              0.603 |        199 | True        |
