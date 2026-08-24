# E6 lead time — deepseek-coder-6.7b

## What this experiment asks

This table measures the model’s own generated answer, rather than what a separate probe can decode. It is used as a behavioral check and, where applicable, to compare when hidden-state evidence appears with when the output becomes correct.

|   layer | readout   |   n_model_wrong |   per_prefix_error_rate |   early_warning_rate |   analytic_null |   early_warning_excess | constant_readout   | beats_position_floor   |   readout_never_wrong |   mean_lead |
|--------:|:----------|----------------:|------------------------:|---------------------:|----------------:|-----------------------:|:-------------------|:-----------------------|----------------------:|------------:|
|      -1 | position  |              19 |                   0.233 |                0.842 |           0.729 |                  0.113 | False              | False                  |                     3 |       2.375 |
|      -1 | probe     |              19 |                   0.174 |                0.895 |           0.612 |                  0.282 | True               | True                   |                     0 |       0.895 |
|      -1 | random    |              19 |                   0.702 |                1.000 |           0.995 |                  0.005 | False              | False                  |                     0 |       5.053 |
|       0 | position  |              19 |                   0.233 |                0.842 |           0.729 |                  0.113 | False              | False                  |                     3 |       2.375 |
|       0 | probe     |              19 |                   0.005 |                0.000 |           0.024 |                 -0.024 | False              | True                   |                    19 |     nan     |
|       0 | random    |              19 |                   0.154 |                0.632 |           0.565 |                  0.067 | False              | True                   |                     7 |       2.167 |
|       3 | position  |              19 |                   0.233 |                0.842 |           0.729 |                  0.113 | False              | False                  |                     3 |       2.375 |
|       3 | probe     |              19 |                   0.559 |                0.842 |           0.977 |                 -0.135 | False              | False                  |                     3 |       3.688 |
|       3 | random    |              19 |                   0.297 |                0.474 |           0.820 |                 -0.346 | False              | False                  |                    10 |       3.889 |
|       7 | position  |              19 |                   0.233 |                0.842 |           0.729 |                  0.113 | False              | False                  |                     3 |       2.375 |
|       7 | probe     |              19 |                   0.008 |                0.000 |           0.041 |                 -0.041 | False              | True                   |                    19 |     nan     |
|       7 | random    |              19 |                   0.124 |                0.474 |           0.482 |                 -0.008 | False              | True                   |                     7 |       1.833 |
|      11 | position  |              19 |                   0.233 |                0.842 |           0.729 |                  0.113 | False              | False                  |                     3 |       2.375 |
|      11 | probe     |              19 |                   0.027 |                0.105 |           0.128 |                 -0.023 | False              | True                   |                    17 |       1.000 |
|      11 | random    |              19 |                   0.197 |                0.579 |           0.662 |                 -0.083 | False              | True                   |                     8 |       3.091 |
|      15 | position  |              19 |                   0.233 |                0.842 |           0.729 |                  0.113 | False              | False                  |                     3 |       2.375 |
|      15 | probe     |              19 |                   0.013 |                0.211 |           0.062 |                  0.149 | False              | True                   |                    13 |       0.667 |
|      15 | random    |              19 |                   0.174 |                0.895 |           0.612 |                  0.282 | True               | True                   |                     0 |       0.895 |
|      19 | position  |              19 |                   0.233 |                0.842 |           0.729 |                  0.113 | False              | False                  |                     3 |       2.375 |
|      19 | probe     |              19 |                   0.014 |                0.000 |           0.069 |                 -0.069 | False              | True                   |                    19 |     nan     |
|      19 | random    |              19 |                   0.715 |                1.000 |           0.996 |                  0.004 | False              | False                  |                     0 |       5.053 |
|      23 | position  |              19 |                   0.233 |                0.842 |           0.729 |                  0.113 | False              | False                  |                     3 |       2.375 |
|      23 | probe     |              19 |                   0.016 |                0.000 |           0.080 |                 -0.080 | False              | True                   |                    19 |     nan     |
|      23 | random    |              19 |                   0.542 |                1.000 |           0.973 |                  0.027 | False              | False                  |                     0 |       5.053 |
|      27 | position  |              19 |                   0.233 |                0.842 |           0.729 |                  0.113 | False              | False                  |                     3 |       2.375 |
|      27 | probe     |              19 |                   0.010 |                0.000 |           0.049 |                 -0.049 | False              | True                   |                    19 |     nan     |
|      27 | random    |              19 |                   0.586 |                1.000 |           0.982 |                  0.018 | False              | False                  |                     0 |       5.053 |
|      31 | position  |              19 |                   0.233 |                0.842 |           0.729 |                  0.113 | False              | False                  |                     3 |       2.375 |
|      31 | probe     |              19 |                   0.172 |                0.789 |           0.607 |                  0.183 | False              | True                   |                     4 |       4.933 |
|      31 | random    |              19 |                   0.174 |                0.895 |           0.612 |                  0.282 | True               | True                   |                     0 |       0.895 |
