# Def-use accuracy by token distance — deepseek-coder-1.3b_core

## What this experiment asks

This table tests whether a linear classifier can recover a program relation from frozen model hidden states. Read the model score against the measured surface baseline and shuffled-label control: a higher hidden-state score supports representation of the relation, but does not by itself show that the model uses it.

|   layer |   dist_0_10 |   dist_10_50 |   dist_50_200 |
|--------:|------------:|-------------:|--------------:|
|  -1.000 |       0.835 |        0.891 |         0.923 |
|   0.000 |       0.973 |        0.975 |         0.943 |
|   3.000 |       0.996 |        0.993 |         0.993 |
|   7.000 |       0.996 |        0.993 |         0.990 |
|  11.000 |       0.995 |        0.990 |         0.974 |
|  15.000 |       0.992 |        0.988 |         0.982 |
|  19.000 |       0.991 |        0.989 |         0.989 |
|  23.000 |       0.983 |        0.982 |         0.974 |
