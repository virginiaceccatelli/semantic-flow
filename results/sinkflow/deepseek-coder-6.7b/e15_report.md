# E15 — source→sink readout under obfuscation (deepseek-coder-6.7b)

**Verdict.** GATES PASS — the track is measurable; the numbers below are reported, not yet claimed

## Gates

| gate | passed | value | detail |
|---|---|---|---|
| S0 | PASS | 480.0000 | every validity gate passed: 480 clean programs, balanced across 3 families x 4 structures x 2 labels, split {'heldout': 72, 'train': 168} with no base leakage,  |
| S1 | PASS | 1776.0000 | 1776 programs extracted across ['train', 'heldout', 'heldout_obf'] with no skips and every source/sink anchor covered exactly by stored token positions |
| S2 | PASS | 1.0000 | fitted on 336 clean training programs (168 bases, digest 0b5fcb12614bab1e); best hidden accuracy 1.0000 at site sink_arg layer 15 with selectivity 0.4291; surfa |
| S3 | PASS | 1920.0000 | 1920 result rows over conditions ['clean_heldout', 'normalize', 'rename_only', 'opaque_only', 'encode_only', 'flatten_only', 'rename_cumulative', 'rename_opaque |
| J0 | FAIL |  | not recorded |
| J1 | FAIL |  | not recorded |

## Clean training programs (grouped CV, site `sink_arg`)

- best hidden-state layer: 15 at accuracy 1.0
- selectivity at best: 0.4290552584670232
- measured surface baseline (token ids only): 0.4912655971479501

## Frozen readout on held-out programs (layer 11)

Intervals are cluster-bootstrapped over base programs. `pairs same` is the fraction of matched pairs given the *same* label — the two members differ only at the sink argument, so it rises only when the position has stopped carrying the distinction at all.

| condition | level | transformation | hidden [95% CI] | surface | pairs same | pred. unsafe | n |
|---|---:|---|---:|---:|---:|---:|---:|
| clean_heldout | -1 | clean | 1.000 [1.000, 1.000] | 0.444 | 0.000 | 0.500 | 144 |
| normalize | 0 | normalize | 1.000 [1.000, 1.000] | 0.444 | 0.000 | 0.500 | 144 |
| rename_only | 11 | rename_only | 0.951 [0.917, 0.979] | 0.486 | 0.097 | 0.521 | 144 |
| opaque_only | 12 | opaque_only | 0.986 [0.965, 1.000] | 0.444 | 0.028 | 0.500 | 144 |
| encode_only | 13 | encode_only | 0.993 [0.979, 1.000] | 0.444 | 0.014 | 0.507 | 144 |
| flatten_only | 14 | flatten_only | 0.715 [0.653, 0.778] | 0.444 | 0.514 | 0.688 | 144 |
| rename_cumulative | 21 | rename_cumulative | 0.910 [0.861, 0.958] | 0.507 | 0.125 | 0.521 | 144 |
| rename_opaque | 22 | rename_opaque | 0.903 [0.861, 0.944] | 0.472 | 0.194 | 0.569 | 144 |
| rename_opaque_encode | 23 | rename_opaque_encode | 0.868 [0.806, 0.924] | 0.514 | 0.208 | 0.604 | 144 |
| rename_opaque_encode_flatten | 24 | rename_opaque_encode_flatten | 0.625 [0.583, 0.674] | 0.514 | 0.750 | 0.875 | 144 |

### Table 1 — atomic transformations (each applied alone)

What each transformation costs **on its own**. `normalize` is an ast round-trip, so it is the reference row: anything it costs is an unparse artifact, not a transformation.

| condition | accuracy | ci_lo | ci_hi | delta_clean | acc_unsafe | acc_safe | false_negative_rate | false_positive_rate | frac_predicted_unsafe | pairs_same_label | n |
|---|---|---|---|---|---|---|---|---|---|---|---|
| clean_heldout | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.500 | 0.000 | 144 |
| normalize | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.500 | 0.000 | 144 |
| rename_only | 0.951 | 0.917 | 0.979 | -0.049 | 0.972 | 0.931 | 0.028 | 0.069 | 0.521 | 0.097 | 144 |
| opaque_only | 0.986 | 0.965 | 1.000 | -0.014 | 0.986 | 0.986 | 0.014 | 0.014 | 0.500 | 0.028 | 144 |
| encode_only | 0.993 | 0.979 | 1.000 | -0.007 | 1.000 | 0.986 | 0.000 | 0.014 | 0.507 | 0.014 | 144 |
| flatten_only | 0.715 | 0.653 | 0.778 | -0.285 | 0.903 | 0.528 | 0.097 | 0.472 | 0.688 | 0.514 | 144 |


### Table 2 — cumulative ladder (adversarial composition)

`delta_previous` is the MARGINAL cost of the step this condition adds to the one above it. This is the only column that supports a sentence of the form 'adding X costs Y'.

| condition | accuracy | ci_lo | ci_hi | delta_clean | delta_previous | acc_unsafe | acc_safe | false_negative_rate | false_positive_rate | frac_predicted_unsafe | pairs_same_label | n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clean_heldout | 1.000 | 1.000 | 1.000 | 0.000 |  | 1.000 | 1.000 | 0.000 | 0.000 | 0.500 | 0.000 | 144 |
| normalize | 1.000 | 1.000 | 1.000 | 0.000 |  | 1.000 | 1.000 | 0.000 | 0.000 | 0.500 | 0.000 | 144 |
| rename_cumulative | 0.910 | 0.861 | 0.958 | -0.090 | -0.090 | 0.931 | 0.889 | 0.069 | 0.111 | 0.521 | 0.125 | 144 |
| rename_opaque | 0.903 | 0.861 | 0.944 | -0.097 | -0.007 | 0.972 | 0.833 | 0.028 | 0.167 | 0.569 | 0.194 | 144 |
| rename_opaque_encode | 0.868 | 0.806 | 0.924 | -0.132 | -0.035 | 0.972 | 0.764 | 0.028 | 0.236 | 0.604 | 0.208 | 144 |
| rename_opaque_encode_flatten | 0.625 | 0.583 | 0.674 | -0.375 | -0.243 | 1.000 | 0.250 | 0.000 | 0.750 | 0.875 | 0.750 | 144 |


### Table 3 — atomic versus cumulative (the interaction)

`interaction` = cumulative − atomic: the part of the cumulative failure the transformation does not produce on its own. The `rename` row is a draw-noise floor by construction (identical transformations, independent draws); read every other row against it. **Attribute a failure to a transformation only where its atomic row supports it** — otherwise it is a cumulative effect.

| transformation | atomic | atomic_accuracy | cumulative | cumulative_accuracy | interaction | marginal_in_ladder | atomic_fnr | cumulative_fnr | note |
|---|---|---|---|---|---|---|---|---|---|
| rename | rename_only | 0.951 | rename_cumulative | 0.910 | -0.042 | -0.090 | 0.028 | 0.069 | draw-noise floor: identical transformations, independent draws |
| opaque | opaque_only | 0.986 | rename_opaque | 0.903 | -0.083 | -0.007 | 0.014 | 0.028 |  |
| encode | encode_only | 0.993 | rename_opaque_encode | 0.868 | -0.125 | -0.035 | 0.000 | 0.028 |  |
| flatten | flatten_only | 0.715 | rename_opaque_encode_flatten | 0.625 | -0.090 | -0.243 | 0.097 | 0.000 |  |


### Table 4 — per-class accuracy and matched-pair collapse

Pooled accuracy conceals the failure the threat model is about. `false_negative_rate` is the fraction of genuinely unsafe programs called safe; `pairs_same_label` is the fraction of matched pairs given the SAME prediction, which rises only when the position has stopped carrying the distinction at all.

| condition | condition_kind | accuracy | acc_unsafe | acc_safe | false_negative_rate | false_positive_rate | frac_predicted_unsafe | pairs_same_label | n |
|---|---|---|---|---|---|---|---|---|---|
| clean_heldout | clean | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.500 | 0.000 | 144 |
| normalize | baseline | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.500 | 0.000 | 144 |
| rename_only | atomic | 0.951 | 0.972 | 0.931 | 0.028 | 0.069 | 0.521 | 0.097 | 144 |
| opaque_only | atomic | 0.986 | 0.986 | 0.986 | 0.014 | 0.014 | 0.500 | 0.028 | 144 |
| encode_only | atomic | 0.993 | 1.000 | 0.986 | 0.000 | 0.014 | 0.507 | 0.014 | 144 |
| flatten_only | atomic | 0.715 | 0.903 | 0.528 | 0.097 | 0.472 | 0.688 | 0.514 | 144 |
| rename_cumulative | cumulative | 0.910 | 0.931 | 0.889 | 0.069 | 0.111 | 0.521 | 0.125 | 144 |
| rename_opaque | cumulative | 0.903 | 0.972 | 0.833 | 0.028 | 0.167 | 0.569 | 0.194 | 144 |
| rename_opaque_encode | cumulative | 0.868 | 0.972 | 0.764 | 0.028 | 0.236 | 0.604 | 0.208 | 144 |
| rename_opaque_encode_flatten | cumulative | 0.625 | 1.000 | 0.250 | 0.000 | 0.750 | 0.875 | 0.750 | 144 |


### Table 5 — the four arms

`hidden_state` at the reported layer against its three floors. `whole_program_lexical` reads the entire program text (token n-grams, no hidden states, frozen on clean training programs): it bounds what a generator-level textual shortcut could achieve, which the ±3-token `local_surface` window cannot see.

| condition | hidden_state | embedding | local_surface | whole_program_lexical |
|---|---|---|---|---|
| clean_heldout | 1.000 | 0.507 | 0.444 | 0.465 |
| normalize | 1.000 | 0.507 | 0.444 | 0.500 |
| rename_only | 0.951 | 0.549 | 0.486 | 0.535 |
| opaque_only | 0.986 | 0.507 | 0.444 | 0.507 |
| encode_only | 0.993 | 0.507 | 0.444 | 0.493 |
| flatten_only | 0.715 | 0.507 | 0.444 | 0.472 |
| rename_cumulative | 0.910 | 0.514 | 0.507 | 0.486 |
| rename_opaque | 0.903 | 0.569 | 0.472 | 0.521 |
| rename_opaque_encode | 0.868 | 0.507 | 0.514 | 0.500 |
| rename_opaque_encode_flatten | 0.625 | 0.451 | 0.514 | 0.493 |


Read the per-family and per-structure rows in `sinkflow_obfuscation.csv` before quoting the pooled number: a readout can hold on `direct` flows and fail across the helper boundary, and the pooled row hides that.

