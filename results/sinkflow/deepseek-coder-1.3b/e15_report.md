# E15 — source→sink readout under obfuscation (deepseek-coder-1.3b)

**Verdict.** GATES PASS — the track is measurable; the numbers below are reported, not yet claimed

## Gates

| gate | passed | value | detail |
|---|---|---|---|
| S0 | PASS | 480.0000 | every validity gate passed: 480 clean programs, balanced across 3 families x 4 structures x 2 labels, split {'heldout': 72, 'train': 168} with no base leakage,  |
| S1 | PASS | 1776.0000 | 1776 programs extracted across ['train', 'heldout', 'heldout_obf'] with no skips and every source/sink anchor covered exactly by stored token positions |
| S2 | PASS | 1.0000 | fitted on 336 clean training programs (168 bases, digest 0b5fcb12614bab1e); best hidden accuracy 1.0000 at site sink_arg layer 11 with selectivity 0.3874; surfa |
| S3 | PASS | 1600.0000 | 1600 result rows over conditions ['clean_heldout', 'normalize', 'rename_only', 'opaque_only', 'encode_only', 'flatten_only', 'rename_cumulative', 'rename_opaque |
| J0 | FAIL |  | not recorded |
| J1 | FAIL |  | not recorded |

## Clean training programs (grouped CV, site `sink_arg`)

- best hidden-state layer: 11 at accuracy 1.0
- selectivity at best: 0.4325311942959002
- measured surface baseline (token ids only): 0.4912655971479501

## Frozen readout on held-out programs (layer 11)

Intervals are cluster-bootstrapped over base programs. `pairs same` is the fraction of matched pairs given the *same* label — the two members differ only at the sink argument, so it rises only when the position has stopped carrying the distinction at all.

| condition | level | transformation | hidden [95% CI] | surface | pairs same | pred. unsafe | n |
|---|---:|---|---:|---:|---:|---:|---:|
| clean_heldout | -1 | clean | 1.000 [1.000, 1.000] | 0.444 | 0.000 | 0.500 | 144 |
| normalize | 0 | normalize | 1.000 [1.000, 1.000] | 0.444 | 0.000 | 0.500 | 144 |
| rename_only | 11 | rename_only | 0.938 [0.889, 0.972] | 0.486 | 0.097 | 0.479 | 144 |
| opaque_only | 12 | opaque_only | 1.000 [1.000, 1.000] | 0.444 | 0.000 | 0.500 | 144 |
| encode_only | 13 | encode_only | 1.000 [1.000, 1.000] | 0.444 | 0.000 | 0.500 | 144 |
| flatten_only | 14 | flatten_only | 0.688 [0.618, 0.750] | 0.444 | 0.514 | 0.438 | 144 |
| rename_cumulative | 21 | rename_cumulative | 0.958 [0.924, 0.986] | 0.507 | 0.083 | 0.514 | 144 |
| rename_opaque | 22 | rename_opaque | 0.944 [0.910, 0.979] | 0.472 | 0.111 | 0.514 | 144 |
| rename_opaque_encode | 23 | rename_opaque_encode | 0.951 [0.910, 0.986] | 0.514 | 0.069 | 0.535 | 144 |
| rename_opaque_encode_flatten | 24 | rename_opaque_encode_flatten | 0.729 [0.660, 0.799] | 0.514 | 0.431 | 0.438 | 144 |

### Table 1 — atomic transformations (each applied alone)

What each transformation costs **on its own**. `normalize` is an ast round-trip, so it is the reference row: anything it costs is an unparse artifact, not a transformation.

| condition | accuracy | ci_lo | ci_hi | delta_clean | acc_unsafe | acc_safe | false_negative_rate | false_positive_rate | frac_predicted_unsafe | pairs_same_label | n |
|---|---|---|---|---|---|---|---|---|---|---|---|
| clean_heldout | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.500 | 0.000 | 144 |
| normalize | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.500 | 0.000 | 144 |
| rename_only | 0.938 | 0.889 | 0.972 | -0.062 | 0.917 | 0.958 | 0.083 | 0.042 | 0.479 | 0.097 | 144 |
| opaque_only | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.500 | 0.000 | 144 |
| encode_only | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.500 | 0.000 | 144 |
| flatten_only | 0.688 | 0.618 | 0.750 | -0.312 | 0.625 | 0.750 | 0.375 | 0.250 | 0.438 | 0.514 | 144 |


### Table 2 — cumulative ladder (adversarial composition)

`delta_previous` is the MARGINAL cost of the step this condition adds to the one above it. This is the only column that supports a sentence of the form 'adding X costs Y'.

| condition | accuracy | ci_lo | ci_hi | delta_clean | delta_previous | acc_unsafe | acc_safe | false_negative_rate | false_positive_rate | frac_predicted_unsafe | pairs_same_label | n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clean_heldout | 1.000 | 1.000 | 1.000 | 0.000 |  | 1.000 | 1.000 | 0.000 | 0.000 | 0.500 | 0.000 | 144 |
| normalize | 1.000 | 1.000 | 1.000 | 0.000 |  | 1.000 | 1.000 | 0.000 | 0.000 | 0.500 | 0.000 | 144 |
| rename_cumulative | 0.958 | 0.924 | 0.986 | -0.042 | -0.042 | 0.972 | 0.944 | 0.028 | 0.056 | 0.514 | 0.083 | 144 |
| rename_opaque | 0.944 | 0.910 | 0.979 | -0.056 | -0.014 | 0.958 | 0.931 | 0.042 | 0.069 | 0.514 | 0.111 | 144 |
| rename_opaque_encode | 0.951 | 0.910 | 0.986 | -0.049 | 0.007 | 0.986 | 0.917 | 0.014 | 0.083 | 0.535 | 0.069 | 144 |
| rename_opaque_encode_flatten | 0.729 | 0.660 | 0.799 | -0.271 | -0.222 | 0.667 | 0.792 | 0.333 | 0.208 | 0.438 | 0.431 | 144 |


### Table 3 — atomic versus cumulative (the interaction)

`interaction` = cumulative − atomic: the part of the cumulative failure the transformation does not produce on its own. The `rename` row is a draw-noise floor by construction (identical transformations, independent draws); read every other row against it. **Attribute a failure to a transformation only where its atomic row supports it** — otherwise it is a cumulative effect.

| transformation | atomic | atomic_accuracy | cumulative | cumulative_accuracy | interaction | marginal_in_ladder | atomic_fnr | cumulative_fnr | note |
|---|---|---|---|---|---|---|---|---|---|
| rename | rename_only | 0.938 | rename_cumulative | 0.958 | 0.021 | -0.042 | 0.083 | 0.028 | draw-noise floor: identical transformations, independent draws |
| opaque | opaque_only | 1.000 | rename_opaque | 0.944 | -0.056 | -0.014 | 0.000 | 0.042 |  |
| encode | encode_only | 1.000 | rename_opaque_encode | 0.951 | -0.049 | 0.007 | 0.000 | 0.014 |  |
| flatten | flatten_only | 0.688 | rename_opaque_encode_flatten | 0.729 | 0.042 | -0.222 | 0.375 | 0.333 |  |


### Table 4 — per-class accuracy and matched-pair collapse

Pooled accuracy conceals the failure the threat model is about. `false_negative_rate` is the fraction of genuinely unsafe programs called safe; `pairs_same_label` is the fraction of matched pairs given the SAME prediction, which rises only when the position has stopped carrying the distinction at all.

| condition | condition_kind | accuracy | acc_unsafe | acc_safe | false_negative_rate | false_positive_rate | frac_predicted_unsafe | pairs_same_label | n |
|---|---|---|---|---|---|---|---|---|---|
| clean_heldout | clean | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.500 | 0.000 | 144 |
| normalize | baseline | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.500 | 0.000 | 144 |
| rename_only | atomic | 0.938 | 0.917 | 0.958 | 0.083 | 0.042 | 0.479 | 0.097 | 144 |
| opaque_only | atomic | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.500 | 0.000 | 144 |
| encode_only | atomic | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.500 | 0.000 | 144 |
| flatten_only | atomic | 0.688 | 0.625 | 0.750 | 0.375 | 0.250 | 0.438 | 0.514 | 144 |
| rename_cumulative | cumulative | 0.958 | 0.972 | 0.944 | 0.028 | 0.056 | 0.514 | 0.083 | 144 |
| rename_opaque | cumulative | 0.944 | 0.958 | 0.931 | 0.042 | 0.069 | 0.514 | 0.111 | 144 |
| rename_opaque_encode | cumulative | 0.951 | 0.986 | 0.917 | 0.014 | 0.083 | 0.535 | 0.069 | 144 |
| rename_opaque_encode_flatten | cumulative | 0.729 | 0.667 | 0.792 | 0.333 | 0.208 | 0.438 | 0.431 | 144 |


### Table 5 — the four arms

`hidden_state` at the reported layer against its three floors. `whole_program_lexical` reads the entire program text (token n-grams, no hidden states, frozen on clean training programs): it bounds what a generator-level textual shortcut could achieve, which the ±3-token `local_surface` window cannot see.

| condition | hidden_state | embedding | local_surface | whole_program_lexical |
|---|---|---|---|---|
| clean_heldout | 1.000 | 0.507 | 0.444 | 0.465 |
| normalize | 1.000 | 0.507 | 0.444 | 0.500 |
| rename_only | 0.938 | 0.549 | 0.486 | 0.535 |
| opaque_only | 1.000 | 0.507 | 0.444 | 0.507 |
| encode_only | 1.000 | 0.507 | 0.444 | 0.493 |
| flatten_only | 0.688 | 0.507 | 0.444 | 0.472 |
| rename_cumulative | 0.958 | 0.514 | 0.507 | 0.486 |
| rename_opaque | 0.944 | 0.569 | 0.472 | 0.521 |
| rename_opaque_encode | 0.951 | 0.507 | 0.514 | 0.500 |
| rename_opaque_encode_flatten | 0.729 | 0.451 | 0.514 | 0.493 |


Read the per-family and per-structure rows in `sinkflow_obfuscation.csv` before quoting the pooled number: a readout can hold on `direct` flows and fail across the helper boundary, and the pooled row hides that.

