# Stage 252: branch readout

Layer selected on synthetic validation at `first/colon`: **8** (synthetic val pair accuracy 0.826).

Pairs: {'syn_train': 1061, 'syn_val': 241, 'real': 735}

## Primary readout on real code (frozen synthetic direction)

| order | position | pair accuracy | 95% CI |
|---|---|---:|---|
| first | cond_last | 0.469 | [0.427, 0.510] |
| first | colon | 0.595 | [0.551, 0.638] |
| first | body_first | 0.637 | [0.594, 0.682] |
| first | def_colon | 0.525 | [0.481, 0.569] |
| first | answer | 0.556 | [0.515, 0.596] |
| last | cond_last | 0.500 | [0.500, 0.500] |
| last | colon | 0.500 | [0.500, 0.500] |
| last | answer | 0.512 | [0.463, 0.562] |

Structural zero (input-last, at the `if`): max |Δh| = 0 (exact: True) — pair accuracy there is 0.5 by construction.

Logistic probe at l* (frozen): real pair accuracy 0.637 [0.593, 0.683].
Model-free input-token floor (grouped CV): pair accuracy 0.473.
Member-level AUROC of the projection (descriptive, not pinned): 0.510.

## Real in-domain ceiling (grouped CV, same estimator)

| order | position | pair accuracy |
|---|---|---:|
| first | colon | 0.682 |
| first | answer | 0.562 |
| last | colon | 0.500 |
| last | answer | 0.567 |

## Best layer per read on real code

| order | position | best layer | pair accuracy |
|---|---|---:|---:|
| first | answer | 8 | 0.556 |
| first | body_first | 8 | 0.637 |
| first | colon | 18 | 0.612 |
| first | cond_last | 17 | 0.533 |
| first | def_colon | 12 | 0.535 |
| last | answer | 12 | 0.582 |
| last | colon | -1 | 0.500 |
| last | cond_last | -1 | 0.500 |

Full layer profiles: `readout_layers.csv`. The best-layer column is descriptive (selected on real data); the primary row above is not.
