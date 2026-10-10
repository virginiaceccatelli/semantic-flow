# Definition survival: probe results

Balanced accuracy on held-out test repositories/pairs, chance 0.5, 95% group-bootstrap intervals. Each probe trained on 3592 candidates; layer chosen on its own domain's validation split (synthetic: 9, real: 8).

| train → test | n (groups) | probe | surface | heuristic | embedding | shuffled | probe − surface | probe − heuristic |
|---|---|---|---|---|---|---|---|---|
| synthetic → synthetic | 1168 (584) | 0.995 [0.991, 0.998] | 0.500 | 0.634 | 0.500 | 0.499 | 0.495 [0.491, 0.498] | 0.360 [0.342, 0.380] |
| synthetic → real | 2139 (346) | 0.640 [0.611, 0.669] | 0.500 | 0.856 | 0.448 | 0.509 | 0.140 [0.111, 0.169] | -0.216 [-0.261, -0.175] |
| real → synthetic | 1168 (584) | 0.588 [0.565, 0.613] | 0.500 | 0.634 | 0.500 | 0.461 | 0.088 [0.065, 0.113] | -0.046 [-0.063, -0.030] |
| real → real | 2139 (346) | 0.930 [0.905, 0.952] | 0.644 | 0.856 | 0.596 | 0.521 | 0.286 [0.237, 0.336] | 0.074 [0.041, 0.106] |

**surface**: logistic regression on token ids within 3 of d and u plus bucketed distance. **heuristic**: killed iff a redefinition is indented no deeper than the use. **embedding**: the same probe on layer −1. **shuffled**: the same probe trained on permuted labels.

## Accuracy by stratum (selected layer, test split)

| train_domain   | test_domain   | stratum         | family      |    n |   n_groups |   survive_fraction |   probe_accuracy |   surface_accuracy |   embedding_accuracy |
|:---------------|:--------------|:----------------|:------------|-----:|-----------:|-------------------:|-----------------:|-------------------:|---------------------:|
| synthetic      | synthetic     | heuristic_right | else_branch |  134 |        134 |              1.000 |            0.993 |              0.985 |                1.000 |
| synthetic      | synthetic     | heuristic_right | if_vs_with  |  150 |        150 |              1.000 |            0.993 |              0.967 |                1.000 |
| synthetic      | synthetic     | heuristic_right | indent_swap |  314 |        157 |              0.500 |            1.000 |              0.500 |                0.500 |
| synthetic      | synthetic     | heuristic_right | loop_target |  143 |        143 |              0.000 |            1.000 |              0.000 |                0.000 |
| synthetic      | synthetic     | heuristic_wrong | else_branch |  134 |        134 |              0.000 |            0.978 |              0.015 |                0.000 |
| synthetic      | synthetic     | heuristic_wrong | if_vs_with  |  150 |        150 |              0.000 |            0.993 |              0.033 |                0.000 |
| synthetic      | synthetic     | heuristic_wrong | loop_target |  143 |        143 |              1.000 |            1.000 |              1.000 |                1.000 |
| synthetic      | real          | heuristic_right | nan         | 1786 |        337 |              0.524 |            0.630 |              0.524 |                0.462 |
| synthetic      | real          | heuristic_wrong | nan         |  353 |        110 |              0.926 |            0.504 |              0.926 |                0.487 |
| real           | synthetic     | heuristic_right | else_branch |  134 |        134 |              1.000 |            0.993 |              0.925 |                0.515 |
| real           | synthetic     | heuristic_right | if_vs_with  |  150 |        150 |              1.000 |            0.953 |              0.953 |                0.527 |
| real           | synthetic     | heuristic_right | indent_swap |  314 |        157 |              0.500 |            0.930 |              0.500 |                0.500 |
| real           | synthetic     | heuristic_right | loop_target |  143 |        143 |              0.000 |            0.119 |              0.056 |                0.462 |
| real           | synthetic     | heuristic_wrong | else_branch |  134 |        134 |              0.000 |            0.022 |              0.075 |                0.485 |
| real           | synthetic     | heuristic_wrong | if_vs_with  |  150 |        150 |              0.000 |            0.167 |              0.047 |                0.473 |
| real           | synthetic     | heuristic_wrong | loop_target |  143 |        143 |              1.000 |            0.517 |              0.944 |                0.538 |
| real           | real          | heuristic_right | nan         | 1786 |        337 |              0.524 |            0.942 |              0.670 |                0.601 |
| real           | real          | heuristic_wrong | nan         |  353 |        110 |              0.926 |            0.892 |              0.669 |                0.663 |

Non-converged fits: 0 of 134.
