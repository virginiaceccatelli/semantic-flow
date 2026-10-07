# Stage 259: branch-following in the CruxEval order

1168 real members scored (0 dropped: the other branch's output is not a prefix-stable continuation of the CruxEval-order prompt). Accuracy in this order: 0.375.

| | CruxEval order (input after code) | input first (stage 253, same members) |
|---|---:|---:|
| answers that are one of the two branch outputs | 715 | 690 |
| follow the branch that runs | 0.613 | 0.533 |
| are the `if`-body branch's output | 0.624 | 0.626 |
| same branch for both inputs of a pair | 0.684 | 0.926 |
| … expected from body bias alone | 0.531 | 0.532 |
| pairs whose answer follows the input | 0.293 | 0.055 |

Members decided in both orders: 626; the same branch is chosen in both orders for 0.613 of them.

Follows the branch that runs, CruxEval order, by source: cruxeval 0.635, humaneval 0.691, mbpp 0.588
