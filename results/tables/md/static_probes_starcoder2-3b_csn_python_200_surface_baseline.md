# Surface-shortcut baseline (no hidden states) — starcoder2-3b_csn_python_200

## What this experiment asks

This table tests whether a linear classifier can recover a program relation from frozen model hidden states. Read the model score against the measured surface baseline and shuffled-label control: a higher hidden-state score supports representation of the relation, but does not by itself show that the model uses it.

| task        |   overall |   diff_name |   distance_matched |   indent_matched |   non_dependent |   positive |   same_name_diff_binding |
|:------------|----------:|------------:|-------------------:|-----------------:|----------------:|-----------:|-------------------------:|
| binding     |     0.687 |       0.747 |              0.758 |          nan     |         nan     |      0.440 |                    0.823 |
| defuse_edge |     0.586 |       0.628 |              0.525 |          nan     |         nan     |      0.504 |                    0.712 |
| control_dep |     0.727 |     nan     |            nan     |            0.775 |           0.840 |      0.525 |                  nan     |
