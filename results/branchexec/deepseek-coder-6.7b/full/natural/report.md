# Stage 259: branch-following in the CruxEval order

1168 real members scored (0 dropped: the other branch's output is not a prefix-stable continuation of the CruxEval-order prompt). Accuracy in this order: 0.485.

| | CruxEval order (input after code) | input first (stage 253, same members) |
|---|---:|---:|
| answers that are one of the two branch outputs | 785 | 710 |
| follow the branch that runs | 0.721 | 0.511 |
| are the `if`-body branch's output | 0.580 | 0.694 |
| same branch for both inputs of a pair | 0.450 | 0.877 |
| … expected from body bias alone | 0.513 | 0.576 |
| pairs whose answer follows the input | 0.532 | 0.110 |

Members decided in both orders: 649; the same branch is chosen in both orders for 0.713 of them.

Follows the branch that runs, CruxEval order, by source: cruxeval 0.702, humaneval 0.825, mbpp 0.694
