# Static probes — deepseek-coder-6.7b_core

| task               |   peak_layer |   accuracy |   selectivity |   auc |   control_accuracy |   n_groups | converged   |
|:-------------------|-------------:|-----------:|--------------:|------:|-------------------:|-----------:|:------------|
| binding            |            0 |      0.976 |         0.401 | 0.998 |              0.576 |        466 | True        |
| control_dep        |            7 |      1.000 |         0.495 | 1.000 |              0.505 |        209 | True        |
| defuse_edge        |            0 |      0.974 |         0.429 | 0.998 |              0.545 |        580 | True        |
| lexical_token_type |           -1 |      1.000 |         0.871 | 0.000 |              0.129 |        360 | False       |
| taint_state        |            0 |      1.000 |         0.522 | 1.000 |              0.477 |        200 | True        |
