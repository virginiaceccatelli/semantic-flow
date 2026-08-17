# E15 — source→sink readout under obfuscation (deepseek-coder-6.7b)

**Verdict.** GATES PASS — the track is measurable; the numbers below are reported, not yet claimed

## Gates

| gate | passed | value | detail |
|---|---|---|---|
| S0 | PASS | 480.0000 | every validity gate passed: 480 clean programs, balanced across 3 families x 4 structures x 2 labels, split {'heldout': 72, 'train': 168} with no base leakage,  |
| S1 | PASS | 1200.0000 | 1200 programs extracted across ['train', 'heldout', 'heldout_obf'] with no skips and every source/sink anchor covered exactly by stored token positions |
| S2 | PASS | 1.0000 | fitted on 336 clean training programs (168 bases, digest 0b5fcb12614bab1e); best hidden accuracy 1.0000 at site sink_arg layer 15 with selectivity 0.4291; surfa |
| S3 | PASS | 1056.0000 | 1056 result rows over conditions ['clean_heldout', 'obf0', 'obf1', 'obf2', 'obf3', 'obf4'], both classes present in every reported cell, evaluated with a probe  |

## Clean training programs (grouped CV, site `sink_arg`)

- best hidden-state layer: 15 at accuracy 1.0
- selectivity at best: 0.4290552584670232
- measured surface baseline (token ids only): 0.4912655971479501

## Frozen readout on held-out programs (layer 11)

| condition | level | transformation | hidden | surface | n |
|---|---:|---|---:|---:|---:|
| clean_heldout | -1 | clean | 1.000 | 0.444 | 144 |
| obf0 | 0 | normalize | 1.000 | 0.444 | 144 |
| obf1 | 1 | rename | 0.910 | 0.479 | 144 |
| obf2 | 2 | opaque | 0.917 | 0.500 | 144 |
| obf3 | 3 | encode | 0.896 | 0.479 | 144 |
| obf4 | 4 | flatten | 0.604 | 0.507 | 144 |

Read the per-family and per-structure rows in `sinkflow_obfuscation.csv` before quoting the pooled number: a readout can hold on `direct` flows and fail across the helper boundary, and the pooled row hides that.

