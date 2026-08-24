# Static probes — starcoder2-3b_csn_python_200

## What this experiment asks

This table tests whether a linear classifier can recover a program relation from frozen model hidden states. Read the model score against the measured surface baseline and shuffled-label control: a higher hidden-state score supports representation of the relation, but does not by itself show that the model uses it.

| task               |   peak_layer |   accuracy |   selectivity |   auc |   control_accuracy |   n_groups | converged   |
|:-------------------|-------------:|-----------:|--------------:|------:|-------------------:|-----------:|:------------|
| binding            |            3 |      0.913 |         0.331 | 0.983 |              0.582 |        182 | True        |
| control_dep        |           19 |      0.729 |         0.213 | 0.804 |              0.515 |        122 | True        |
| defuse_edge        |            3 |      0.904 |         0.330 | 0.981 |              0.575 |        199 | True        |
| lexical_token_type |           -1 |      0.997 |         0.838 | 0.000 |              0.159 |        108 | True        |
