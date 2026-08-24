# Binding per-stratum accuracy — deepseek-coder-1.3b_core

## What this experiment asks

This table tests whether a linear classifier can recover a program relation from frozen model hidden states. Read the model score against the measured surface baseline and shuffled-label control: a higher hidden-state score supports representation of the relation, but does not by itself show that the model uses it.

|   layer |   context_matched |   diff_name |   distance_matched |   positive |   same_name_diff_binding |
|--------:|------------------:|------------:|-------------------:|-----------:|-------------------------:|
|  -1.000 |             0.500 |       1.000 |              0.929 |      0.999 |                    0.002 |
|   0.000 |             0.557 |       0.995 |              0.981 |      0.988 |                    0.914 |
|   3.000 |             0.962 |       0.998 |              0.988 |      1.000 |                    0.987 |
|   7.000 |             0.981 |       0.992 |              0.990 |      0.997 |                    0.990 |
|  11.000 |             0.972 |       0.987 |              0.985 |      0.997 |                    0.981 |
|  15.000 |             0.953 |       0.989 |              0.985 |      0.995 |                    0.982 |
|  19.000 |             0.943 |       0.994 |              0.983 |      0.998 |                    0.979 |
|  23.000 |             0.915 |       0.987 |              0.975 |      0.992 |                    0.968 |
