# Binding per-stratum accuracy — deepseek-coder-6.7b_csn_python_200

## What this experiment asks

This table tests whether a linear classifier can recover a program relation from frozen model hidden states. Read the model score against the measured surface baseline and shuffled-label control: a higher hidden-state score supports representation of the relation, but does not by itself show that the model uses it.

|   layer |   diff_name |   distance_matched |   positive |   same_name_diff_binding |
|--------:|------------:|-------------------:|-----------:|-------------------------:|
|  -1.000 |       0.958 |              0.925 |      0.938 |                    0.095 |
|   0.000 |       0.944 |              0.908 |      0.943 |                    0.122 |
|   3.000 |       0.931 |              0.886 |      0.972 |                    0.131 |
|   7.000 |       0.924 |              0.897 |      0.962 |                    0.411 |
|  11.000 |       0.892 |              0.878 |      0.932 |                    0.401 |
|  15.000 |       0.891 |              0.873 |      0.912 |                    0.387 |
|  19.000 |       0.878 |              0.858 |      0.890 |                    0.492 |
|  23.000 |       0.890 |              0.866 |      0.871 |                    0.435 |
|  27.000 |       0.880 |              0.869 |      0.860 |                    0.436 |
|  31.000 |       0.866 |              0.858 |      0.830 |                    0.494 |
