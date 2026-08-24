# J-lens taint lead time — deepseek-coder-6.7b

## What this experiment asks

This table applies the J-lens to the source-to-sink security distinction. It asks whether that distinction is visible in output-aligned coordinates at different layers; it is observational and therefore does not establish causal use.

|   layer | readout   |   n_test |   n_model_wrong |   latent_first |   early_warning_rate |   readout_never_wrong |   n_both_fail |   mean_lead |
|--------:|:----------|---------:|----------------:|---------------:|---------------------:|----------------------:|--------------:|------------:|
|      -1 | jlens     |       70 |              19 |             17 |                0.895 |                     0 |            19 |       0.895 |
|      -1 | logit     |       70 |              19 |             17 |                0.895 |                     0 |            19 |       0.895 |
|      -1 | probe     |       70 |              19 |             17 |                0.895 |                     0 |            19 |       0.895 |
|      -1 | random    |       70 |              19 |             17 |                0.895 |                     0 |            19 |       0.895 |
|       0 | jlens     |       70 |              19 |             19 |                1.000 |                     0 |            19 |       4.368 |
|       0 | logit     |       70 |              19 |             17 |                0.895 |                     1 |            18 |       1.944 |
|       0 | probe     |       70 |              19 |              0 |                0.000 |                    19 |             0 |     nan     |
|       0 | random    |       70 |              19 |              4 |                0.211 |                    15 |             4 |       2.000 |
|       3 | jlens     |       70 |              19 |             19 |                1.000 |                     0 |            19 |       3.368 |
|       3 | logit     |       70 |              19 |             17 |                0.895 |                     0 |            19 |       0.895 |
|       3 | probe     |       70 |              19 |             16 |                0.842 |                     3 |            16 |       3.688 |
|       3 | random    |       70 |              19 |             19 |                1.000 |                     0 |            19 |       1.947 |
|       7 | jlens     |       70 |              19 |             11 |                0.579 |                     0 |            19 |       0.579 |
|       7 | logit     |       70 |              19 |             17 |                0.895 |                     0 |            19 |       0.895 |
|       7 | probe     |       70 |              19 |              0 |                0.000 |                    19 |             0 |     nan     |
|       7 | random    |       70 |              19 |             17 |                0.895 |                     0 |            19 |       0.895 |
|      11 | jlens     |       70 |              19 |             12 |                0.632 |                     0 |            19 |       0.895 |
|      11 | logit     |       70 |              19 |             19 |                1.000 |                     0 |            19 |       5.053 |
|      11 | probe     |       70 |              19 |              2 |                0.105 |                    17 |             2 |       1.000 |
|      11 | random    |       70 |              19 |             19 |                1.000 |                     0 |            19 |       4.211 |
|      15 | jlens     |       70 |              19 |              2 |                0.105 |                    11 |             8 |       0.500 |
|      15 | logit     |       70 |              19 |              5 |                0.263 |                    13 |             6 |       4.833 |
|      15 | probe     |       70 |              19 |              4 |                0.211 |                    13 |             6 |       0.667 |
|      15 | random    |       70 |              19 |             19 |                1.000 |                     0 |            19 |       4.842 |
|      19 | jlens     |       70 |              19 |              0 |                0.000 |                    19 |             0 |     nan     |
|      19 | logit     |       70 |              19 |              5 |                0.263 |                     7 |            12 |       1.917 |
|      19 | probe     |       70 |              19 |              0 |                0.000 |                    19 |             0 |     nan     |
|      19 | random    |       70 |              19 |             19 |                1.000 |                     0 |            19 |       4.421 |
|      23 | jlens     |       70 |              19 |              0 |                0.000 |                    19 |             0 |     nan     |
|      23 | logit     |       70 |              19 |              5 |                0.263 |                     7 |            12 |       1.917 |
|      23 | probe     |       70 |              19 |              0 |                0.000 |                    19 |             0 |     nan     |
|      23 | random    |       70 |              19 |             10 |                0.526 |                     6 |            13 |       1.692 |
|      27 | jlens     |       70 |              19 |              0 |                0.000 |                    19 |             0 |     nan     |
|      27 | logit     |       70 |              19 |              0 |                0.000 |                    18 |             1 |       0.000 |
|      27 | probe     |       70 |              19 |              0 |                0.000 |                    19 |             0 |     nan     |
|      27 | random    |       70 |              19 |              8 |                0.421 |                     7 |            12 |       2.167 |
|      31 | jlens     |       70 |              19 |              0 |                0.000 |                    19 |             0 |     nan     |
|      31 | logit     |       70 |              19 |              0 |                0.000 |                    19 |             0 |     nan     |
|      31 | probe     |       70 |              19 |             15 |                0.789 |                     4 |            15 |       4.933 |
|      31 | random    |       70 |              19 |              5 |                0.263 |                     5 |            14 |       1.357 |
