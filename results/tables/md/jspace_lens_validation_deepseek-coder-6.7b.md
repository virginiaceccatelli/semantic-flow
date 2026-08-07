# E11 lens validation — deepseek-coder-6.7b

| check                     |   layer | lens        |    top1 |     mrr |       n |   cosine_to_logit_lens |   is_last_layer |
|:--------------------------|--------:|:------------|--------:|--------:|--------:|-----------------------:|----------------:|
| V2_next_token             |       8 | jlens       |   0.365 |   0.560 |  52.000 |                nan     |         nan     |
| V2_next_token             |       8 | logit       |   0.385 |   0.581 |  52.000 |                nan     |         nan     |
| V2_next_token             |       8 | gram_random |   0.135 |   0.328 |  52.000 |                nan     |         nan     |
| V2_next_token             |       8 | random      |   0.212 |   0.365 |  52.000 |                nan     |         nan     |
| V1_identity_at_last_layer |       8 | jlens       | nan     | nan     | nan     |                  0.124 |           0.000 |
| V2_next_token             |      16 | jlens       |   0.538 |   0.678 |  52.000 |                nan     |         nan     |
| V2_next_token             |      16 | logit       |   0.577 |   0.731 |  52.000 |                nan     |         nan     |
| V2_next_token             |      16 | gram_random |   0.135 |   0.312 |  52.000 |                nan     |         nan     |
| V2_next_token             |      16 | random      |   0.115 |   0.334 |  52.000 |                nan     |         nan     |
| V1_identity_at_last_layer |      16 | jlens       | nan     | nan     | nan     |                  0.367 |           0.000 |
| V2_next_token             |      24 | jlens       |   0.788 |   0.844 |  52.000 |                nan     |         nan     |
| V2_next_token             |      24 | logit       |   0.788 |   0.860 |  52.000 |                nan     |         nan     |
| V2_next_token             |      24 | gram_random |   0.077 |   0.253 |  52.000 |                nan     |         nan     |
| V2_next_token             |      24 | random      |   0.058 |   0.274 |  52.000 |                nan     |         nan     |
| V1_identity_at_last_layer |      24 | jlens       | nan     | nan     | nan     |                  0.494 |           0.000 |
| V2_next_token             |      31 | jlens       |   0.827 |   0.888 |  52.000 |                nan     |         nan     |
| V2_next_token             |      31 | logit       |   0.827 |   0.888 |  52.000 |                nan     |         nan     |
| V2_next_token             |      31 | gram_random |   0.096 |   0.245 |  52.000 |                nan     |         nan     |
| V2_next_token             |      31 | random      |   0.077 |   0.278 |  52.000 |                nan     |         nan     |
| V1_identity_at_last_layer |      31 | jlens       | nan     | nan     | nan     |                  1.000 |           1.000 |
