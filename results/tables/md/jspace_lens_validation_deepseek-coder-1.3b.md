# E11 lens validation — deepseek-coder-1.3b

## What this experiment asks

This table checks whether the low-rank J-space used by the intervention retains the relevant lens signal and whether that readout is stable. These are prerequisite diagnostics, not standalone causal results.

| check                     |   layer | lens        |    top1 |     mrr |       n |   cosine_to_logit_lens |   is_last_layer |
|:--------------------------|--------:|:------------|--------:|--------:|--------:|-----------------------:|----------------:|
| V2_next_token             |       6 | clens       |   0.385 |   0.608 |  52.000 |                nan     |         nan     |
| V2_next_token             |       6 | logit       |   0.462 |   0.633 |  52.000 |                nan     |         nan     |
| V2_next_token             |       6 | gram_random |   0.077 |   0.293 |  52.000 |                nan     |         nan     |
| V2_next_token             |       6 | random      |   0.115 |   0.311 |  52.000 |                nan     |         nan     |
| V1_identity_at_last_layer |       6 | clens       | nan     | nan     | nan     |                  0.060 |           0.000 |
| V2_next_token             |      12 | clens       |   0.615 |   0.760 |  52.000 |                nan     |         nan     |
| V2_next_token             |      12 | logit       |   0.615 |   0.751 |  52.000 |                nan     |         nan     |
| V2_next_token             |      12 | gram_random |   0.135 |   0.309 |  52.000 |                nan     |         nan     |
| V2_next_token             |      12 | random      |   0.096 |   0.276 |  52.000 |                nan     |         nan     |
| V1_identity_at_last_layer |      12 | clens       | nan     | nan     | nan     |                  0.118 |           0.000 |
| V2_next_token             |      18 | clens       |   0.769 |   0.850 |  52.000 |                nan     |         nan     |
| V2_next_token             |      18 | logit       |   0.769 |   0.852 |  52.000 |                nan     |         nan     |
| V2_next_token             |      18 | gram_random |   0.096 |   0.262 |  52.000 |                nan     |         nan     |
| V2_next_token             |      18 | random      |   0.154 |   0.338 |  52.000 |                nan     |         nan     |
| V1_identity_at_last_layer |      18 | clens       | nan     | nan     | nan     |                  0.208 |           0.000 |
| V2_next_token             |      23 | clens       |   0.808 |   0.880 |  52.000 |                nan     |         nan     |
| V2_next_token             |      23 | logit       |   0.808 |   0.880 |  52.000 |                nan     |         nan     |
| V2_next_token             |      23 | gram_random |   0.038 |   0.236 |  52.000 |                nan     |         nan     |
| V2_next_token             |      23 | random      |   0.192 |   0.374 |  52.000 |                nan     |         nan     |
| V1_identity_at_last_layer |      23 | clens       | nan     | nan     | nan     |                  1.000 |           1.000 |
