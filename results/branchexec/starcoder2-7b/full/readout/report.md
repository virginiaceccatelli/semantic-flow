# Stage 252: branch readout

Layer selected on synthetic validation at `first/colon`: **21** (synthetic val pair accuracy 0.898).

Pairs: {'syn_train': 1086, 'syn_val': 244, 'real': 743}

## Primary readout on real code (frozen synthetic direction)

| order | position | pair accuracy | 95% CI |
|---|---|---:|---|
| first | cond_last | 0.583 | [0.539, 0.628] |
| first | colon | 0.680 | [0.637, 0.726] |
| first | body_first | 0.585 | [0.542, 0.631] |
| first | def_colon | 0.514 | [0.467, 0.562] |
| first | answer | 0.485 | [0.438, 0.532] |
| last | cond_last | 0.500 | [0.500, 0.500] |
| last | colon | 0.500 | [0.500, 0.500] |
| last | answer | 0.592 | [0.542, 0.644] |

Structural zero (input-last, at the `if`): max |Δh| = 0 (exact: True) — pair accuracy there is 0.5 by construction.

Logistic probe at l* (frozen): real pair accuracy 0.643 [0.601, 0.686].
Model-free input-token floor (grouped CV): pair accuracy 0.514.
Member-level AUROC of the projection (descriptive, not pinned): 0.533.

## Real in-domain ceiling (grouped CV, same estimator)

| order | position | pair accuracy |
|---|---|---:|
| first | colon | 0.651 |
| first | answer | 0.602 |
| last | colon | 0.500 |
| last | answer | 0.624 |

## Best layer per read on real code

| order | position | best layer | pair accuracy |
|---|---|---:|---:|
| first | answer | 31 | 0.568 |
| first | body_first | 16 | 0.661 |
| first | colon | 15 | 0.727 |
| first | cond_last | 16 | 0.610 |
| first | def_colon | 12 | 0.550 |
| last | answer | 23 | 0.599 |
| last | colon | -1 | 0.500 |
| last | cond_last | -1 | 0.500 |

Full layer profiles: `readout_layers.csv`. The best-layer column is descriptive (selected on real data); the primary row above is not.
