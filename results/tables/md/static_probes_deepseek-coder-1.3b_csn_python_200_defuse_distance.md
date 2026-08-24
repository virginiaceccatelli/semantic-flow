# Def-use accuracy by token distance — deepseek-coder-1.3b_csn_python_200

## What this experiment asks

This table tests whether a linear classifier can recover a program relation from frozen model hidden states. Read the model score against the measured surface baseline and shuffled-label control: a higher hidden-state score supports representation of the relation, but does not by itself show that the model uses it.

|   layer |   dist_0_10 |   dist_10_50 |   dist_200_100000 |   dist_50_200 |
|--------:|------------:|-------------:|------------------:|--------------:|
|  -1.000 |       0.841 |        0.912 |             0.931 |         0.903 |
|   0.000 |       0.824 |        0.911 |             0.926 |         0.904 |
|   3.000 |       0.867 |        0.914 |             0.916 |         0.913 |
|   7.000 |       0.812 |        0.881 |             0.896 |         0.889 |
|  11.000 |       0.771 |        0.867 |             0.887 |         0.882 |
|  15.000 |       0.787 |        0.865 |             0.871 |         0.873 |
|  19.000 |       0.797 |        0.878 |             0.891 |         0.877 |
|  23.000 |       0.787 |        0.827 |             0.842 |         0.836 |
