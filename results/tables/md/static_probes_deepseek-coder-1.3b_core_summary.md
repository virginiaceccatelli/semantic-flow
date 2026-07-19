# Static probes — deepseek-coder-1.3b_core

| task               |   peak_layer |   accuracy |   selectivity |   auc |   control_accuracy |   n_groups | converged   |
|:-------------------|-------------:|-----------:|--------------:|------:|-------------------:|-----------:|:------------|
| binding            |            0 |      0.979 |         0.414 | 0.998 |              0.565 |        466 | True        |
| control_dep        |           11 |      1.000 |         0.526 | 1.000 |              0.474 |        209 | True        |
| defuse_edge        |            7 |      0.994 |         0.432 | 1.000 |              0.562 |        580 | True        |
| lexical_token_type |           -1 |      1.000 |         0.896 | 0.000 |              0.104 |        360 | False       |
| taint_state        |            0 |      1.000 |         0.525 | 1.000 |              0.475 |        200 | True        |
