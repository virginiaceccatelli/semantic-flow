# Def-use accuracy by token distance — deepseek-coder-6.7b_csn_python_200

## What this experiment asks

This table tests whether a linear classifier can recover a program relation from frozen model hidden states. Read the model score against the measured surface baseline and shuffled-label control: a higher hidden-state score supports representation of the relation, but does not by itself show that the model uses it.

|   layer |   dist_0_10 |   dist_10_50 |   dist_200_100000 |   dist_50_200 |
|--------:|------------:|-------------:|------------------:|--------------:|
|  -1.000 |       0.843 |        0.906 |             0.932 |         0.910 |
|   0.000 |       0.831 |        0.908 |             0.916 |         0.909 |
|   3.000 |       0.843 |        0.921 |             0.912 |         0.911 |
|   7.000 |       0.821 |        0.899 |             0.912 |         0.897 |
|  11.000 |       0.785 |        0.879 |             0.874 |         0.873 |
|  15.000 |       0.771 |        0.870 |             0.894 |         0.873 |
|  19.000 |       0.766 |        0.875 |             0.870 |         0.864 |
|  23.000 |       0.783 |        0.867 |             0.868 |         0.872 |
|  27.000 |       0.785 |        0.849 |             0.846 |         0.857 |
|  31.000 |       0.795 |        0.837 |             0.835 |         0.844 |
