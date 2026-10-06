# Stage 252: branch readout

Layer selected on synthetic validation at `first/colon`: **16** (synthetic val pair accuracy 0.863).

Pairs: {'syn_train': 1061, 'syn_val': 241, 'real': 735}

## Primary readout on real code (frozen synthetic direction)

| order | position | pair accuracy | 95% CI |
|---|---|---:|---|
| first | cond_last | 0.671 | [0.621, 0.720] |
| first | colon | 0.746 | [0.700, 0.791] |
| first | body_first | 0.712 | [0.671, 0.751] |
| first | def_colon | 0.487 | [0.445, 0.529] |
| first | answer | 0.582 | [0.534, 0.632] |
| last | cond_last | 0.500 | [0.500, 0.500] |
| last | colon | 0.500 | [0.500, 0.500] |
| last | answer | 0.581 | [0.525, 0.637] |

Structural zero (input-last, at the `if`): max |Δh| = 0 (exact: True) — pair accuracy there is 0.5 by construction.

Logistic probe at l* (frozen): real pair accuracy 0.731 [0.684, 0.776].
Model-free input-token floor (grouped CV): pair accuracy 0.473.
Member-level AUROC of the projection (descriptive, not pinned): 0.544.

## Real in-domain ceiling (grouped CV, same estimator)

| order | position | pair accuracy |
|---|---|---:|
| first | colon | 0.796 |
| first | answer | 0.638 |
| last | colon | 0.500 |
| last | answer | 0.612 |

## Best layer per read on real code

| order | position | best layer | pair accuracy |
|---|---|---:|---:|
| first | answer | 31 | 0.592 |
| first | body_first | 16 | 0.712 |
| first | colon | 15 | 0.759 |
| first | cond_last | 15 | 0.673 |
| first | def_colon | 11 | 0.561 |
| last | answer | 30 | 0.599 |
| last | colon | -1 | 0.500 |
| last | cond_last | -1 | 0.500 |

Full layer profiles: `readout_layers.csv`. The best-layer column is descriptive (selected on real data); the primary row above is not.
