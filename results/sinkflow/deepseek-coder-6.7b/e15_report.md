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

## Frozen readout on held-out programs (layer 15)

Intervals are cluster-bootstrapped over base programs. `pairs same` is the fraction of matched pairs given the *same* label — the two members differ only at the sink argument, so it rises only when the position has stopped carrying the distinction at all.

| condition | level | transformation | hidden [95% CI] | surface | pairs same | pred. unsafe | n |
|---|---:|---|---:|---:|---:|---:|---:|
| clean_heldout | -1 | clean | 1.000 [1.000, 1.000] | 0.444 | 0.000 | 0.500 | 144 |
| obf0 | 0 | normalize | 1.000 [1.000, 1.000] | 0.444 | 0.000 | 0.500 | 144 |
| obf1 | 1 | rename | 0.965 [0.938, 0.993] | 0.479 | 0.069 | 0.465 | 144 |
| obf2 | 2 | opaque | 0.979 [0.951, 1.000] | 0.500 | 0.042 | 0.507 | 144 |
| obf3 | 3 | encode | 0.958 [0.924, 0.986] | 0.479 | 0.083 | 0.486 | 144 |
| obf4 | 4 | flatten | 0.562 [0.500, 0.625] | 0.507 | 0.653 | 0.729 | 144 |

Read the per-family and per-structure rows in `sinkflow_obfuscation.csv` before quoting the pooled number: a readout can hold on `direct` flows and fail across the helper boundary, and the pooled row hides that.

