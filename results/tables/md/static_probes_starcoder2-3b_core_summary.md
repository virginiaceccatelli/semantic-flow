# Static probes — starcoder2-3b_core

| task               |   peak_layer |   accuracy |   selectivity |   auc |   control_accuracy |   n_groups | converged   |
|:-------------------|-------------:|-----------:|--------------:|------:|-------------------:|-----------:|:------------|
| binding            |            0 |      0.983 |         0.388 | 0.999 |              0.595 |        421 | True        |
| control_dep        |           19 |      0.987 |         0.378 | 1.000 |              0.609 |        289 | True        |
| defuse_edge        |            0 |      0.986 |         0.412 | 0.999 |              0.575 |        601 | True        |
| lexical_token_type |           -1 |      1.000 |         0.839 | 0.000 |              0.161 |        371 | True        |
| taint_state        |            3 |      1.000 |         0.580 | 1.000 |              0.420 |        200 | True        |
