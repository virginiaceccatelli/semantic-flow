# Stage 252: branch readout

Layer selected on synthetic validation at `first/colon`: **13** (synthetic val pair accuracy 0.848).

Pairs: {'syn_train': 1086, 'syn_val': 244, 'real': 743}

## Primary readout on real code (frozen synthetic direction)

| order | position | pair accuracy | 95% CI |
|---|---|---:|---|
| first | cond_last | 0.534 | [0.487, 0.579] |
| first | colon | 0.678 | [0.639, 0.721] |
| first | body_first | 0.629 | [0.581, 0.674] |
| first | def_colon | 0.561 | [0.518, 0.608] |
| first | answer | 0.556 | [0.511, 0.606] |
| last | cond_last | 0.500 | [0.500, 0.500] |
| last | colon | 0.500 | [0.500, 0.500] |
| last | answer | 0.483 | [0.433, 0.534] |

Structural zero (input-last, at the `if`): max |Δh| = 0 (exact: True) — pair accuracy there is 0.5 by construction.

Logistic probe at l* (frozen): real pair accuracy 0.665 [0.625, 0.707].
Model-free input-token floor (grouped CV): pair accuracy 0.514.
Member-level AUROC of the projection (descriptive, not pinned): 0.531.

## Real in-domain ceiling (grouped CV, same estimator)

| order | position | pair accuracy |
|---|---|---:|
| first | colon | 0.673 |
| first | answer | 0.563 |
| last | colon | 0.500 |
| last | answer | 0.591 |

## Best layer per read on real code

| order | position | best layer | pair accuracy |
|---|---|---:|---:|
| first | answer | 15 | 0.559 |
| first | body_first | 13 | 0.629 |
| first | colon | 14 | 0.685 |
| first | cond_last | 27 | 0.622 |
| first | def_colon | 15 | 0.564 |
| last | answer | 22 | 0.579 |
| last | colon | -1 | 0.500 |
| last | cond_last | -1 | 0.500 |

Full layer profiles: `readout_layers.csv`. The best-layer column is descriptive (selected on real data); the primary row above is not.
