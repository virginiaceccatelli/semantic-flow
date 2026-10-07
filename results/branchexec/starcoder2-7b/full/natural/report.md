# Stage 259: branch-following in the CruxEval order

1184 real members scored (0 dropped: the other branch's output is not a prefix-stable continuation of the CruxEval-order prompt). Accuracy in this order: 0.441.

| | CruxEval order (input after code) | input first (stage 253, same members) |
|---|---:|---:|
| answers that are one of the two branch outputs | 806 | 763 |
| follow the branch that runs | 0.648 | 0.514 |
| are the `if`-body branch's output | 0.615 | 0.671 |
| same branch for both inputs of a pair | 0.579 | 0.886 |
| … expected from body bias alone | 0.527 | 0.559 |
| pairs whose answer follows the input | 0.390 | 0.096 |

Members decided in both orders: 700; the same branch is chosen in both orders for 0.741 of them.

Follows the branch that runs, CruxEval order, by source: cruxeval 0.612, humaneval 0.719, mbpp 0.631
