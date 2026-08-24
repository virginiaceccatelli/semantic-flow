# Surface-shortcut baseline (no hidden states) — deepseek-coder-6.7b_csn_python_200

## What this experiment asks

This table tests whether a linear classifier can recover a program relation from frozen model hidden states. Read the model score against the measured surface baseline and shuffled-label control: a higher hidden-state score supports representation of the relation, but does not by itself show that the model uses it.

| task        |   overall |   diff_name |   distance_matched |   positive |   same_name_diff_binding |
|:------------|----------:|------------:|-------------------:|-----------:|-------------------------:|
| binding     |     0.685 |       0.743 |              0.745 |      0.464 |                    0.767 |
| defuse_edge |     0.599 |       0.641 |              0.558 |      0.514 |                    0.619 |
