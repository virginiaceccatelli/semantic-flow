# E15 — source→sink readout under obfuscation (starcoder2-3b)

**Verdict.** GATES PASS — the track is measurable; the numbers below are reported, not yet claimed

## Gates

| gate | passed | value | detail |
|---|---|---|---|
| S0 | PASS | 480.0000 | every validity gate passed: 480 clean programs, balanced across 3 families x 4 structures x 2 labels, split {'heldout': 72, 'train': 168} with no base leakage,  |
| S1 | PASS | 1776.0000 | 1776 programs extracted across ['train', 'heldout', 'heldout_obf'] with no skips and every source/sink anchor covered exactly by stored token positions |
| S2 | PASS | 0.9971 | fitted on 336 clean training programs (168 bases, digest 0b5fcb12614bab1e); best hidden accuracy 0.9971 at site sink_arg layer 15 with selectivity 0.4799; surfa |
| S3 | PASS | 1920.0000 | 1920 result rows over conditions ['clean_heldout', 'normalize', 'rename_only', 'opaque_only', 'encode_only', 'flatten_only', 'rename_cumulative', 'rename_opaque |
| J0 | FAIL |  | not recorded |
| J1 | FAIL |  | not recorded |

## Clean training programs (grouped CV, site `sink_arg`)

- best hidden-state layer: 15 at accuracy 0.9970588235294118
- selectivity at best: 0.4798573975044563
- measured surface baseline (token ids only): 0.4879679144385027

## Frozen readout on held-out programs (layer 15)

Intervals are cluster-bootstrapped over base programs. `pairs same` is the fraction of matched pairs given the *same* label — the two members differ only at the sink argument, so it rises only when the position has stopped carrying the distinction at all.

| condition | level | transformation | hidden [95% CI] | surface | pairs same | pred. unsafe | n |
|---|---:|---|---:|---:|---:|---:|---:|
| clean_heldout | -1 | clean | 1.000 [1.000, 1.000] | 0.451 | 0.000 | 0.500 | 144 |
| normalize | 0 | normalize | 1.000 [1.000, 1.000] | 0.458 | 0.000 | 0.500 | 144 |
| rename_only | 11 | rename_only | 0.882 [0.833, 0.931] | 0.521 | 0.236 | 0.382 | 144 |
| opaque_only | 12 | opaque_only | 1.000 [1.000, 1.000] | 0.472 | 0.000 | 0.500 | 144 |
| encode_only | 13 | encode_only | 1.000 [1.000, 1.000] | 0.458 | 0.000 | 0.500 | 144 |
| flatten_only | 14 | flatten_only | 0.660 [0.583, 0.736] | 0.431 | 0.458 | 0.507 | 144 |
| rename_cumulative | 21 | rename_cumulative | 0.910 [0.861, 0.951] | 0.486 | 0.181 | 0.410 | 144 |
| rename_opaque | 22 | rename_opaque | 0.931 [0.889, 0.965] | 0.521 | 0.139 | 0.431 | 144 |
| rename_opaque_encode | 23 | rename_opaque_encode | 0.938 [0.896, 0.972] | 0.500 | 0.125 | 0.451 | 144 |
| rename_opaque_encode_flatten | 24 | rename_opaque_encode_flatten | 0.674 [0.604, 0.743] | 0.500 | 0.486 | 0.396 | 144 |

### Table 1 — atomic transformations (each applied alone)

What each transformation costs **on its own**. `normalize` is an ast round-trip, so it is the reference row: anything it costs is an unparse artifact, not a transformation.

| condition | accuracy | ci_lo | ci_hi | delta_clean | acc_unsafe | acc_safe | false_negative_rate | false_positive_rate | frac_predicted_unsafe | pairs_same_label | n |
|---|---|---|---|---|---|---|---|---|---|---|---|
| clean_heldout | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.500 | 0.000 | 144 |
| normalize | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.500 | 0.000 | 144 |
| rename_only | 0.882 | 0.833 | 0.931 | -0.118 | 0.764 | 1.000 | 0.236 | 0.000 | 0.382 | 0.236 | 144 |
| opaque_only | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.500 | 0.000 | 144 |
| encode_only | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.500 | 0.000 | 144 |
| flatten_only | 0.660 | 0.583 | 0.736 | -0.340 | 0.667 | 0.653 | 0.333 | 0.347 | 0.507 | 0.458 | 144 |


### Table 2 — cumulative ladder (adversarial composition)

`delta_previous` is the MARGINAL cost of the step this condition adds to the one above it. This is the only column that supports a sentence of the form 'adding X costs Y'.

| condition | accuracy | ci_lo | ci_hi | delta_clean | delta_previous | acc_unsafe | acc_safe | false_negative_rate | false_positive_rate | frac_predicted_unsafe | pairs_same_label | n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clean_heldout | 1.000 | 1.000 | 1.000 | 0.000 |  | 1.000 | 1.000 | 0.000 | 0.000 | 0.500 | 0.000 | 144 |
| normalize | 1.000 | 1.000 | 1.000 | 0.000 |  | 1.000 | 1.000 | 0.000 | 0.000 | 0.500 | 0.000 | 144 |
| rename_cumulative | 0.910 | 0.861 | 0.951 | -0.090 | -0.090 | 0.819 | 1.000 | 0.181 | 0.000 | 0.410 | 0.181 | 144 |
| rename_opaque | 0.931 | 0.889 | 0.965 | -0.069 | 0.021 | 0.861 | 1.000 | 0.139 | 0.000 | 0.431 | 0.139 | 144 |
| rename_opaque_encode | 0.938 | 0.896 | 0.972 | -0.062 | 0.007 | 0.889 | 0.986 | 0.111 | 0.014 | 0.451 | 0.125 | 144 |
| rename_opaque_encode_flatten | 0.674 | 0.604 | 0.743 | -0.326 | -0.264 | 0.569 | 0.778 | 0.431 | 0.222 | 0.396 | 0.486 | 144 |


### Table 3 — atomic versus cumulative (the interaction)

`interaction` = cumulative − atomic: the part of the cumulative failure the transformation does not produce on its own. The `rename` row is a draw-noise floor by construction (identical transformations, independent draws); read every other row against it. **Attribute a failure to a transformation only where its atomic row supports it** — otherwise it is a cumulative effect.

| transformation | atomic | atomic_accuracy | cumulative | cumulative_accuracy | interaction | marginal_in_ladder | atomic_fnr | cumulative_fnr | note |
|---|---|---|---|---|---|---|---|---|---|
| rename | rename_only | 0.882 | rename_cumulative | 0.910 | 0.028 | -0.090 | 0.236 | 0.181 | draw-noise floor: identical transformations, independent draws |
| opaque | opaque_only | 1.000 | rename_opaque | 0.931 | -0.069 | 0.021 | 0.000 | 0.139 |  |
| encode | encode_only | 1.000 | rename_opaque_encode | 0.938 | -0.062 | 0.007 | 0.000 | 0.111 |  |
| flatten | flatten_only | 0.660 | rename_opaque_encode_flatten | 0.674 | 0.014 | -0.264 | 0.333 | 0.431 |  |


### Table 4 — per-class accuracy and matched-pair collapse

Pooled accuracy conceals the failure the threat model is about. `false_negative_rate` is the fraction of genuinely unsafe programs called safe; `pairs_same_label` is the fraction of matched pairs given the SAME prediction, which rises only when the position has stopped carrying the distinction at all.

| condition | condition_kind | accuracy | acc_unsafe | acc_safe | false_negative_rate | false_positive_rate | frac_predicted_unsafe | pairs_same_label | n |
|---|---|---|---|---|---|---|---|---|---|
| clean_heldout | clean | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.500 | 0.000 | 144 |
| normalize | baseline | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.500 | 0.000 | 144 |
| rename_only | atomic | 0.882 | 0.764 | 1.000 | 0.236 | 0.000 | 0.382 | 0.236 | 144 |
| opaque_only | atomic | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.500 | 0.000 | 144 |
| encode_only | atomic | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.500 | 0.000 | 144 |
| flatten_only | atomic | 0.660 | 0.667 | 0.653 | 0.333 | 0.347 | 0.507 | 0.458 | 144 |
| rename_cumulative | cumulative | 0.910 | 0.819 | 1.000 | 0.181 | 0.000 | 0.410 | 0.181 | 144 |
| rename_opaque | cumulative | 0.931 | 0.861 | 1.000 | 0.139 | 0.000 | 0.431 | 0.139 | 144 |
| rename_opaque_encode | cumulative | 0.938 | 0.889 | 0.986 | 0.111 | 0.014 | 0.451 | 0.125 | 144 |
| rename_opaque_encode_flatten | cumulative | 0.674 | 0.569 | 0.778 | 0.431 | 0.222 | 0.396 | 0.486 | 144 |


### Table 5 — the four arms

`hidden_state` at the reported layer against its three floors. `whole_program_lexical` reads the entire program text (token n-grams, no hidden states, frozen on clean training programs): it bounds what a generator-level textual shortcut could achieve, which the ±3-token `local_surface` window cannot see.

| condition | hidden_state | embedding | local_surface | whole_program_lexical |
|---|---|---|---|---|
| clean_heldout | 1.000 | 0.507 | 0.451 | 0.465 |
| normalize | 1.000 | 0.507 | 0.458 | 0.500 |
| rename_only | 0.882 | 0.549 | 0.521 | 0.535 |
| opaque_only | 1.000 | 0.507 | 0.472 | 0.507 |
| encode_only | 1.000 | 0.507 | 0.458 | 0.493 |
| flatten_only | 0.660 | 0.507 | 0.431 | 0.472 |
| rename_cumulative | 0.910 | 0.514 | 0.486 | 0.486 |
| rename_opaque | 0.931 | 0.569 | 0.521 | 0.521 |
| rename_opaque_encode | 0.938 | 0.507 | 0.500 | 0.500 |
| rename_opaque_encode_flatten | 0.674 | 0.451 | 0.500 | 0.493 |


Read the per-family and per-structure rows in `sinkflow_obfuscation.csv` before quoting the pooled number: a readout can hold on `direct` flows and fail across the helper boundary, and the pooled row hides that.

