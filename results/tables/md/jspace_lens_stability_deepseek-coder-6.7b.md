# E11 lens stability — deepseek-coder-6.7b

## What this experiment asks

This table checks whether the low-rank J-space used by the intervention retains the relevant lens signal and whether that readout is stable. These are prerequisite diagnostics, not standalone causal results.

|   layer |   n_seeds |   cosine_mean |   cosine_min |   margin_sign_agreement |   pooled_vs_seed_cosine |   n_build_per_seed |   n_probe_states |
|--------:|----------:|--------------:|-------------:|------------------------:|------------------------:|-------------------:|-----------------:|
|   8.000 |     3.000 |         0.194 |        0.162 |                   0.470 |                   0.679 |            150.000 |           52.000 |
|  16.000 |     3.000 |         0.644 |        0.585 |                   0.823 |                   0.873 |            150.000 |           52.000 |
|  24.000 |     3.000 |         0.851 |        0.825 |                   0.909 |                   0.949 |            150.000 |           52.000 |
|  31.000 |     3.000 |         1.000 |        1.000 |                   1.000 |                   1.000 |            150.000 |           52.000 |
