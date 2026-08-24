# E11 behavioural accuracy — deepseek-coder-6.7b

## What this experiment asks

This table records model behavior for the programs used in the J-space experiment. It verifies that the underlying task is measurable before interpreting a hidden-state intervention.

| split   | variant   | op_family   |   accuracy |   argmax_accuracy |   n |
|:--------|:----------|:------------|-----------:|------------------:|----:|
| calib   | source    | affine      |      0.833 |             0.833 |  30 |
| calib   | source    | threshold   |      0.500 |             0.500 |  30 |
| calib   | target    | affine      |      0.933 |             0.900 |  30 |
| calib   | target    | threshold   |      0.633 |             0.633 |  30 |
| test    | source    | affine      |      0.900 |             0.843 |  70 |
| test    | source    | threshold   |      0.714 |             0.714 |  70 |
| test    | target    | affine      |      0.943 |             0.871 |  70 |
| test    | target    | threshold   |      0.529 |             0.529 |  70 |
