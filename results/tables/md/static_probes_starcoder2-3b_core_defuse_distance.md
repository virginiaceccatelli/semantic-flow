# Def-use accuracy by token distance — starcoder2-3b_core

## What this experiment asks

This table tests whether a linear classifier can recover a program relation from frozen model hidden states. Read the model score against the measured surface baseline and shuffled-label control: a higher hidden-state score supports representation of the relation, but does not by itself show that the model uses it.

|   layer |   dist_0_10 |   dist_10_50 |   dist_50_200 |
|--------:|------------:|-------------:|--------------:|
|  -1.000 |       0.861 |        0.852 |         0.963 |
|   0.000 |       0.991 |        0.985 |         0.977 |
|   3.000 |       0.996 |        0.991 |         0.991 |
|   7.000 |       0.996 |        0.991 |         0.985 |
|  11.000 |       0.996 |        0.989 |         0.990 |
|  15.000 |       0.997 |        0.993 |         0.992 |
|  19.000 |       0.997 |        0.993 |         0.998 |
|  23.000 |       0.993 |        0.990 |         0.991 |
|  27.000 |       0.992 |        0.988 |         0.991 |
|  29.000 |       0.990 |        0.983 |         0.988 |
