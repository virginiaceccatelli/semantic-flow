# E11 behavioural accuracy — deepseek-coder-1.3b

| split   | variant   | op_family   |   accuracy |   argmax_accuracy |   n |
|:--------|:----------|:------------|-----------:|------------------:|----:|
| calib   | source    | affine      |      0.800 |             0.567 |  30 |
| calib   | source    | threshold   |      0.300 |             0.300 |  30 |
| calib   | target    | affine      |      0.467 |             0.333 |  30 |
| calib   | target    | threshold   |      0.533 |             0.533 |  30 |
| test    | source    | affine      |      0.871 |             0.714 |  70 |
| test    | source    | threshold   |      0.457 |             0.543 |  70 |
| test    | target    | affine      |      0.329 |             0.214 |  70 |
| test    | target    | threshold   |      0.471 |             0.486 |  70 |
