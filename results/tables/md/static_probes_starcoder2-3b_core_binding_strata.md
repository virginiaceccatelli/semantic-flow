# Binding per-stratum accuracy — starcoder2-3b_core

## What this experiment asks

This table tests whether a linear classifier can recover a program relation from frozen model hidden states. Read the model score against the measured surface baseline and shuffled-label control: a higher hidden-state score supports representation of the relation, but does not by itself show that the model uses it.

|   layer |   context_matched |   diff_name |   distance_matched |   positive |   same_name_diff_binding |
|--------:|------------------:|------------:|-------------------:|-----------:|-------------------------:|
|  -1.000 |             0.500 |       1.000 |              0.812 |      0.999 |                    0.002 |
|   0.000 |             0.726 |       0.991 |              0.981 |      0.993 |                    0.955 |
|   3.000 |             0.972 |       0.997 |              0.984 |      0.999 |                    0.984 |
|   7.000 |             0.972 |       0.993 |              0.985 |      0.999 |                    0.985 |
|  11.000 |             0.981 |       0.991 |              0.987 |      0.999 |                    0.986 |
|  15.000 |             0.981 |       0.994 |              0.984 |      1.000 |                    0.989 |
|  19.000 |             0.972 |       0.997 |              0.987 |      1.000 |                    0.988 |
|  23.000 |             0.972 |       0.995 |              0.985 |      0.999 |                    0.981 |
|  27.000 |             0.972 |       0.991 |              0.984 |      0.999 |                    0.972 |
|  29.000 |             0.962 |       0.981 |              0.979 |      0.994 |                    0.973 |
