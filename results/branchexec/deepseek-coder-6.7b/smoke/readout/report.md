# Stage 252: branch readout

Layer selected on synthetic validation at `first/colon`: **24** (synthetic val pair accuracy 0.765).

Pairs: {'syn_train': 132, 'syn_val': 34, 'real': 37}

## Primary readout on real code (frozen synthetic direction)

| order | position | pair accuracy | 95% CI |
|---|---|---:|---|
| first | cond_last | 0.622 | [0.500, 0.778] |
| first | colon | 0.649 | [0.500, 0.833] |
| first | body_first | 0.568 | [0.250, 0.790] |
| first | def_colon | 0.378 | [0.194, 0.550] |
| first | answer | 0.405 | [0.175, 0.643] |
| last | cond_last | 0.500 | [0.500, 0.500] |
| last | colon | 0.500 | [0.500, 0.500] |
| last | answer | 0.514 | [0.286, 0.750] |

Structural zero (input-last, at the `if`): max |Δh| = 0 (exact: True) — pair accuracy there is 0.5 by construction.

Logistic probe at l* (frozen): real pair accuracy 0.811 [0.667, 0.970].
Model-free input-token floor (grouped CV): pair accuracy 0.284.
Member-level AUROC of the projection (descriptive, not pinned): 0.549.

## Real in-domain ceiling (grouped CV, same estimator)

| order | position | pair accuracy |
|---|---|---:|
| first | colon | 0.486 |
| first | answer | 0.405 |
| last | colon | 0.500 |
| last | answer | 0.378 |

## Best layer per read on real code

| order | position | best layer | pair accuracy |
|---|---|---:|---:|
| first | answer | 4 | 0.730 |
| first | body_first | 18 | 0.730 |
| first | colon | 14 | 0.811 |
| first | cond_last | 15 | 0.811 |
| first | def_colon | -1 | 0.500 |
| last | answer | 31 | 0.595 |
| last | colon | -1 | 0.500 |
| last | cond_last | -1 | 0.500 |

Full layer profiles: `readout_layers.csv`. The best-layer column is descriptive (selected on real data); the primary row above is not.
