# E15 — source→sink readout under obfuscation (deepseek-coder-6.7b)

**Verdict.** GATES PASS — the track is measurable; the numbers below are reported, not yet claimed

## Gates

| gate | passed | value | detail |
|---|---|---|---|
| S0 | PASS | 480.0000 | every validity gate passed: 480 clean programs, balanced across 3 families x 4 structures x 2 labels, split {'heldout': 72, 'train': 168} with no base leakage,  |
| S1 | PASS | 1776.0000 | 1776 programs extracted across ['train', 'heldout', 'heldout_obf'] with no skips and every source/sink anchor covered exactly by stored token positions |
| S2 | PASS | 1.0000 | fitted on 336 clean training programs (168 bases, digest 0b5fcb12614bab1e); best hidden accuracy 1.0000 at site sink_arg layer 15 with selectivity 0.4291; surfa |
| S3 | PASS | 1920.0000 | 1920 result rows over conditions ['clean_heldout', 'normalize', 'rename_only', 'opaque_only', 'encode_only', 'flatten_only', 'rename_cumulative', 'rename_opaque |
| J0 | PASS | 196.0000 | 196 frozen candidate tokens (1 unsafe-oriented, 3 safe-oriented, 3 words omitted by the tokenizer check), three lenses at each of 10 layers, forward logits unch |
| J1 | PASS | 43200.0000 | 43200 pair rows over 3 lenses x 10 layers x 2 sites x 10 conditions on 72 held-out bases, oriented unsafe-minus-safe, tokens frozen at 2026-08-19 10:52:50 on tr |

## Clean training programs (grouped CV, site `sink_arg`)

- best hidden-state layer: 15 at accuracy 1.0
- selectivity at best: 0.4290552584670232
- measured surface baseline (token ids only): 0.4912655971479501

## Frozen readout on held-out programs (layer 15)

Intervals are cluster-bootstrapped over base programs. `pairs same` is the fraction of matched pairs given the *same* label — the two members differ only at the sink argument, so it rises only when the position has stopped carrying the distinction at all.

| condition | level | transformation | hidden [95% CI] | surface | pairs same | pred. unsafe | n |
|---|---:|---|---:|---:|---:|---:|---:|
| clean_heldout | -1 | clean | 1.000 [1.000, 1.000] | 0.444 | 0.000 | 0.500 | 144 |
| normalize | 0 | normalize | 1.000 [1.000, 1.000] | 0.444 | 0.000 | 0.500 | 144 |
| rename_only | 11 | rename_only | 0.986 [0.965, 1.000] | 0.486 | 0.028 | 0.486 | 144 |
| opaque_only | 12 | opaque_only | 1.000 [1.000, 1.000] | 0.444 | 0.000 | 0.500 | 144 |
| encode_only | 13 | encode_only | 1.000 [1.000, 1.000] | 0.444 | 0.000 | 0.500 | 144 |
| flatten_only | 14 | flatten_only | 0.667 [0.597, 0.729] | 0.444 | 0.556 | 0.667 | 144 |
| rename_cumulative | 21 | rename_cumulative | 0.951 [0.916, 0.979] | 0.507 | 0.097 | 0.451 | 144 |
| rename_opaque | 22 | rename_opaque | 0.965 [0.931, 0.986] | 0.472 | 0.069 | 0.521 | 144 |
| rename_opaque_encode | 23 | rename_opaque_encode | 0.965 [0.931, 0.993] | 0.514 | 0.069 | 0.507 | 144 |
| rename_opaque_encode_flatten | 24 | rename_opaque_encode_flatten | 0.653 [0.590, 0.715] | 0.514 | 0.583 | 0.708 | 144 |

### Table 1 — atomic transformations (each applied alone)

What each transformation costs **on its own**. `normalize` is an ast round-trip, so it is the reference row: anything it costs is an unparse artifact, not a transformation.

| condition | accuracy | ci_lo | ci_hi | delta_clean | acc_unsafe | acc_safe | false_negative_rate | false_positive_rate | frac_predicted_unsafe | pairs_same_label | n |
|---|---|---|---|---|---|---|---|---|---|---|---|
| clean_heldout | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.500 | 0.000 | 144 |
| normalize | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.500 | 0.000 | 144 |
| rename_only | 0.986 | 0.965 | 1.000 | -0.014 | 0.972 | 1.000 | 0.028 | 0.000 | 0.486 | 0.028 | 144 |
| opaque_only | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.500 | 0.000 | 144 |
| encode_only | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.500 | 0.000 | 144 |
| flatten_only | 0.667 | 0.597 | 0.729 | -0.333 | 0.833 | 0.500 | 0.167 | 0.500 | 0.667 | 0.556 | 144 |


### Table 2 — cumulative ladder (adversarial composition)

`delta_previous` is the MARGINAL cost of the step this condition adds to the one above it. This is the only column that supports a sentence of the form 'adding X costs Y'.

| condition | accuracy | ci_lo | ci_hi | delta_clean | delta_previous | acc_unsafe | acc_safe | false_negative_rate | false_positive_rate | frac_predicted_unsafe | pairs_same_label | n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clean_heldout | 1.000 | 1.000 | 1.000 | 0.000 |  | 1.000 | 1.000 | 0.000 | 0.000 | 0.500 | 0.000 | 144 |
| normalize | 1.000 | 1.000 | 1.000 | 0.000 |  | 1.000 | 1.000 | 0.000 | 0.000 | 0.500 | 0.000 | 144 |
| rename_cumulative | 0.951 | 0.916 | 0.979 | -0.049 | -0.049 | 0.903 | 1.000 | 0.097 | 0.000 | 0.451 | 0.097 | 144 |
| rename_opaque | 0.965 | 0.931 | 0.986 | -0.035 | 0.014 | 0.986 | 0.944 | 0.014 | 0.056 | 0.521 | 0.069 | 144 |
| rename_opaque_encode | 0.965 | 0.931 | 0.993 | -0.035 | 0.000 | 0.972 | 0.958 | 0.028 | 0.042 | 0.507 | 0.069 | 144 |
| rename_opaque_encode_flatten | 0.653 | 0.590 | 0.715 | -0.347 | -0.312 | 0.861 | 0.444 | 0.139 | 0.556 | 0.708 | 0.583 | 144 |


### Table 3 — atomic versus cumulative (the interaction)

`interaction` = cumulative − atomic: the part of the cumulative failure the transformation does not produce on its own. The `rename` row is a draw-noise floor by construction (identical transformations, independent draws); read every other row against it. **Attribute a failure to a transformation only where its atomic row supports it** — otherwise it is a cumulative effect.

| transformation | atomic | atomic_accuracy | cumulative | cumulative_accuracy | interaction | marginal_in_ladder | atomic_fnr | cumulative_fnr | note |
|---|---|---|---|---|---|---|---|---|---|
| rename | rename_only | 0.986 | rename_cumulative | 0.951 | -0.035 | -0.049 | 0.028 | 0.097 | draw-noise floor: identical transformations, independent draws |
| opaque | opaque_only | 1.000 | rename_opaque | 0.965 | -0.035 | 0.014 | 0.000 | 0.014 |  |
| encode | encode_only | 1.000 | rename_opaque_encode | 0.965 | -0.035 | 0.000 | 0.000 | 0.028 |  |
| flatten | flatten_only | 0.667 | rename_opaque_encode_flatten | 0.653 | -0.014 | -0.312 | 0.167 | 0.139 |  |


### Table 4 — per-class accuracy and matched-pair collapse

Pooled accuracy conceals the failure the threat model is about. `false_negative_rate` is the fraction of genuinely unsafe programs called safe; `pairs_same_label` is the fraction of matched pairs given the SAME prediction, which rises only when the position has stopped carrying the distinction at all.

| condition | condition_kind | accuracy | acc_unsafe | acc_safe | false_negative_rate | false_positive_rate | frac_predicted_unsafe | pairs_same_label | n |
|---|---|---|---|---|---|---|---|---|---|
| clean_heldout | clean | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.500 | 0.000 | 144 |
| normalize | baseline | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.500 | 0.000 | 144 |
| rename_only | atomic | 0.986 | 0.972 | 1.000 | 0.028 | 0.000 | 0.486 | 0.028 | 144 |
| opaque_only | atomic | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.500 | 0.000 | 144 |
| encode_only | atomic | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.500 | 0.000 | 144 |
| flatten_only | atomic | 0.667 | 0.833 | 0.500 | 0.167 | 0.500 | 0.667 | 0.556 | 144 |
| rename_cumulative | cumulative | 0.951 | 0.903 | 1.000 | 0.097 | 0.000 | 0.451 | 0.097 | 144 |
| rename_opaque | cumulative | 0.965 | 0.986 | 0.944 | 0.014 | 0.056 | 0.521 | 0.069 | 144 |
| rename_opaque_encode | cumulative | 0.965 | 0.972 | 0.958 | 0.028 | 0.042 | 0.507 | 0.069 | 144 |
| rename_opaque_encode_flatten | cumulative | 0.653 | 0.861 | 0.444 | 0.139 | 0.556 | 0.708 | 0.583 | 144 |


### Table 5 — the four arms

`hidden_state` at the reported layer against its three floors. `whole_program_lexical` reads the entire program text (token n-grams, no hidden states, frozen on clean training programs): it bounds what a generator-level textual shortcut could achieve, which the ±3-token `local_surface` window cannot see.

| condition | hidden_state | embedding | local_surface | whole_program_lexical |
|---|---|---|---|---|
| clean_heldout | 1.000 | 0.507 | 0.444 | 0.465 |
| normalize | 1.000 | 0.507 | 0.444 | 0.500 |
| rename_only | 0.986 | 0.549 | 0.486 | 0.535 |
| opaque_only | 1.000 | 0.507 | 0.444 | 0.507 |
| encode_only | 1.000 | 0.507 | 0.444 | 0.493 |
| flatten_only | 0.667 | 0.507 | 0.444 | 0.472 |
| rename_cumulative | 0.951 | 0.514 | 0.507 | 0.486 |
| rename_opaque | 0.965 | 0.569 | 0.472 | 0.521 |
| rename_opaque_encode | 0.965 | 0.507 | 0.514 | 0.500 |
| rename_opaque_encode_flatten | 0.653 | 0.451 | 0.514 | 0.493 |


Read the per-family and per-structure rows in `sinkflow_obfuscation.csv` before quoting the pooled number: a readout can hold on `direct` flows and fail across the helper boundary, and the pooled row hides that.

