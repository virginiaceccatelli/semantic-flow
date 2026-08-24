# Def-use accuracy by token distance — starcoder2-3b_csn_python_200

## What this experiment asks

This table tests whether a linear classifier can recover a program relation from frozen model hidden states. Read the model score against the measured surface baseline and shuffled-label control: a higher hidden-state score supports representation of the relation, but does not by itself show that the model uses it.

|   layer |   dist_0_10 |   dist_10_50 |   dist_200_100000 |   dist_50_200 |
|--------:|------------:|-------------:|------------------:|--------------:|
|  -1.000 |       0.812 |        0.930 |             0.922 |         0.922 |
|   0.000 |       0.810 |        0.927 |             0.893 |         0.903 |
|   3.000 |       0.826 |        0.918 |             0.909 |         0.904 |
|   7.000 |       0.812 |        0.902 |             0.902 |         0.895 |
|  11.000 |       0.792 |        0.895 |             0.870 |         0.875 |
|  15.000 |       0.770 |        0.897 |             0.885 |         0.887 |
|  19.000 |       0.814 |        0.900 |             0.879 |         0.889 |
|  23.000 |       0.794 |        0.888 |             0.867 |         0.879 |
|  27.000 |       0.778 |        0.845 |             0.842 |         0.843 |
|  29.000 |       0.774 |        0.820 |             0.821 |         0.824 |
