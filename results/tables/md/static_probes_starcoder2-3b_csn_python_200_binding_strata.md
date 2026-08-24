# Binding per-stratum accuracy — starcoder2-3b_csn_python_200

## What this experiment asks

This table tests whether a linear classifier can recover a program relation from frozen model hidden states. Read the model score against the measured surface baseline and shuffled-label control: a higher hidden-state score supports representation of the relation, but does not by itself show that the model uses it.

|   layer |   diff_name |   distance_matched |   positive |   same_name_diff_binding |
|--------:|------------:|-------------------:|-----------:|-------------------------:|
|  -1.000 |       0.966 |              0.933 |      0.959 |                    0.023 |
|   0.000 |       0.956 |              0.922 |      0.970 |                    0.109 |
|   3.000 |       0.938 |              0.908 |      0.979 |                    0.347 |
|   7.000 |       0.926 |              0.898 |      0.964 |                    0.372 |
|  11.000 |       0.907 |              0.879 |      0.959 |                    0.334 |
|  15.000 |       0.912 |              0.892 |      0.941 |                    0.410 |
|  19.000 |       0.911 |              0.878 |      0.931 |                    0.411 |
|  23.000 |       0.889 |              0.884 |      0.871 |                    0.466 |
|  27.000 |       0.875 |              0.870 |      0.831 |                    0.531 |
|  29.000 |       0.859 |              0.870 |      0.799 |                    0.551 |
