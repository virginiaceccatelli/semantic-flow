# Surface-shortcut baseline (no hidden states) — starcoder2-3b_core

## What this experiment asks

This table tests whether a linear classifier can recover a program relation from frozen model hidden states. Read the model score against the measured surface baseline and shuffled-label control: a higher hidden-state score supports representation of the relation, but does not by itself show that the model uses it.

| task        |   overall |   context_matched |   diff_name |   distance_matched |   indent_matched |   non_dependent |   positive |   same_name_diff_binding |
|:------------|----------:|------------------:|------------:|-------------------:|-----------------:|----------------:|-----------:|-------------------------:|
| binding     |     0.772 |             0.500 |       0.714 |              0.737 |          nan     |         nan     |      0.909 |                    0.866 |
| defuse_edge |     0.658 |             0.500 |       0.637 |              0.570 |          nan     |         nan     |      0.776 |                    0.705 |
| control_dep |     0.926 |           nan     |     nan     |            nan     |            0.671 |           1.000 |      0.960 |                  nan     |
