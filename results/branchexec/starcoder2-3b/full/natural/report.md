# Stage 259: branch-following in the CruxEval order

1184 real members scored (0 dropped: the other branch's output is not a prefix-stable continuation of the CruxEval-order prompt). Accuracy in this order: 0.413.

| | CruxEval order (input after code) | input first (stage 253, same members) |
|---|---:|---:|
| answers that are one of the two branch outputs | 781 | 727 |
| follow the branch that runs | 0.626 | 0.508 |
| are the `if`-body branch's output | 0.598 | 0.651 |
| same branch for both inputs of a pair | 0.624 | 0.913 |
| … expected from body bias alone | 0.519 | 0.545 |
| pairs whose answer follows the input | 0.337 | 0.066 |

Members decided in both orders: 643; the same branch is chosen in both orders for 0.666 of them.

Follows the branch that runs, CruxEval order, by source: cruxeval 0.589, humaneval 0.677, mbpp 0.615
