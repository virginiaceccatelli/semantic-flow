# Binding per-stratum accuracy — deepseek-coder-1.3b_csn_python_200

## What this experiment asks

This table tests whether a linear classifier can recover a program relation from frozen model hidden states. Read the model score against the measured surface baseline and shuffled-label control: a higher hidden-state score supports representation of the relation, but does not by itself show that the model uses it.

|   layer |   diff_name |   distance_matched |   positive |   same_name_diff_binding |
|--------:|------------:|-------------------:|-----------:|-------------------------:|
|  -1.000 |       0.957 |              0.922 |      0.945 |                    0.082 |
|   0.000 |       0.935 |              0.897 |      0.938 |                    0.138 |
|   3.000 |       0.946 |              0.921 |      0.957 |                    0.318 |
|   7.000 |       0.918 |              0.904 |      0.934 |                    0.473 |
|  11.000 |       0.906 |              0.894 |      0.881 |                    0.463 |
|  15.000 |       0.896 |              0.881 |      0.870 |                    0.484 |
|  19.000 |       0.905 |              0.893 |      0.850 |                    0.516 |
|  23.000 |       0.875 |              0.868 |      0.803 |                    0.493 |
