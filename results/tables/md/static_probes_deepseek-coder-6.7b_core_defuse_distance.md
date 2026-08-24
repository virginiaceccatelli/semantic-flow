# Def-use accuracy by token distance — deepseek-coder-6.7b_core

## What this experiment asks

This table tests whether a linear classifier can recover a program relation from frozen model hidden states. Read the model score against the measured surface baseline and shuffled-label control: a higher hidden-state score supports representation of the relation, but does not by itself show that the model uses it.

|   layer |   dist_0_10 |   dist_10_50 |   dist_50_200 |
|--------:|------------:|-------------:|--------------:|
|  -1.000 |       0.835 |        0.891 |         0.923 |
|   0.000 |       0.982 |        0.976 |         0.948 |
|   3.000 |       0.995 |        0.989 |         0.985 |
|   7.000 |       0.994 |        0.989 |         0.974 |
|  11.000 |       0.996 |        0.987 |         0.965 |
|  15.000 |       0.986 |        0.986 |         0.968 |
|  19.000 |       0.992 |        0.988 |         0.979 |
|  23.000 |       0.990 |        0.986 |         0.984 |
|  27.000 |       0.987 |        0.985 |         0.967 |
|  31.000 |       0.984 |        0.982 |         0.962 |
