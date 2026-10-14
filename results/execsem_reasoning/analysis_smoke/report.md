# Stage 293: execution uncertainty and execution semantics in reasoning models

Upstream U-Space pipeline unmodified (prompt: explain_result_v1). Higher score = more likely wrong; AUROC .5 = chance. Intervals resample programs (all seeds) or pairs. See docs/EXECSEM_REASONING.md.

## qwen4b

### Experiment 1: U-Lens and execution errors

### cruxeval: N=772 answers (772 programs x seeds), error rate 0.053

- seed 41: 800 traces, 772 labelled, excluded {'truncated': 23, 'multi_marker': 5}, answer status {'parsed': 772}, accuracy 0.947 (all items, exclusions wrong: 0.914), mean length 4927

| Method | AUROC [95% CI] | per-seed mean ± sd | length-matched | AUROC by length bin (low → high) |
|---|---|---|---:|---|
| ulens | 0.628 [0.521, 0.734] | 0.628 ± – | 0.628 | 0.875 / 0.532 / 0.655 / 0.534 |
| a_cone | 0.346 [0.239, 0.458] | 0.346 ± – | 0.452 | 0.511 / 0.576 / 0.451 / 0.312 |
| predictive_entropy | 0.649 [0.547, 0.746] | 0.649 ± – | 0.631 | 0.861 / 0.522 / 0.654 / 0.555 |
| think_entropy | 0.644 [0.539, 0.743] | 0.644 ± – | 0.631 | 0.868 / 0.519 / 0.652 / 0.553 |
| gen_length | 0.733 [0.649, 0.817] | 0.733 ± – | 0.585 | 0.596 / 0.676 / 0.491 / 0.710 |
| lexical_think | 0.707 [0.631, 0.785] | 0.707 ± – | 0.605 | 0.542 / 0.584 / 0.571 / 0.654 |
| lexical_explanation | 0.499 [0.498, 0.500] | 0.499 ± – | 0.499 | 0.497 / 0.500 / 0.500 / 0.500 |
| msp | 0.524 [0.436, 0.608] | 0.524 ± – | 0.469 |  |
| max_entropy | 0.683 [0.605, 0.757] | 0.683 ± – | 0.560 |  |
| mean_nll | 0.645 [0.543, 0.741] | 0.645 ± – | 0.622 |  |
| self_certainty | 0.682 [0.585, 0.775] | 0.682 ± – | 0.652 |  |
| deepconf | 0.660 [0.559, 0.759] | 0.660 ± – | 0.647 |  |

- ulens_minus_predictive_entropy: ΔAUROC -0.0210 [-0.0414, -0.0028]
- ulens_minus_think_entropy: ΔAUROC -0.0158 [-0.0335, 0.0005]
- ulens_minus_gen_length: ΔAUROC -0.1049 [-0.2492, 0.0290]
- ulens_minus_lexical_think: ΔAUROC -0.0785 [-0.1813, 0.0208]
- A_cone AUROC matched on thinking entropy 0.360, on length 0.452
- log A_cone beyond log thinking-entropy, log length, lexical: standardised coefficient -0.391 [-0.764, -0.027], LR 5.89 (p≈0.0152; rows not independent, use the interval)
- Check: E-Space pass vs upstream end-of-thinking U coordinates, max |diff| 0.0000

### mbpp: N=199 answers (199 programs x seeds), error rate 0.030

- seed 41: 200 traces, 199 labelled, excluded {'truncated': 1}, answer status {'parsed': 199}, accuracy 0.970 (all items, exclusions wrong: 0.965), mean length 4939

| Method | AUROC [95% CI] | per-seed mean ± sd | length-matched | AUROC by length bin (low → high) |
|---|---|---|---:|---|
| ulens | 0.739 [0.538, 0.916] | 0.739 ± – | 0.671 | 0.673 / – / 0.787 / 0.708 |
| a_cone | 0.639 [0.394, 0.815] | 0.639 ± – | 0.791 | 0.531 / – / 0.887 / 0.573 |
| predictive_entropy | 0.731 [0.520, 0.917] | 0.731 ± – | 0.630 | 0.673 / – / 0.752 / 0.708 |
| think_entropy | 0.731 [0.527, 0.915] | 0.731 ± – | 0.643 | 0.694 / – / 0.752 / 0.698 |
| gen_length | 0.686 [0.408, 0.913] | 0.686 ± – | 0.436 | 0.571 / – / 0.695 / 0.875 |
| lexical_think | 0.748 [0.479, 0.953] | 0.748 ± – | 0.700 | 0.571 / – / 0.766 / 0.844 |
| lexical_explanation | 0.500 [0.500, 0.500] | 0.500 ± – | 0.500 | 0.500 / – / 0.500 / 0.500 |
| msp | 0.406 [0.252, 0.562] | 0.406 ± – | 0.316 |  |
| max_entropy | 0.657 [0.428, 0.808] | 0.657 ± – | 0.525 |  |
| mean_nll | 0.724 [0.519, 0.911] | 0.724 ± – | 0.623 |  |
| self_certainty | 0.808 [0.686, 0.929] | 0.808 ± – | 0.757 |  |
| deepconf | 0.747 [0.595, 0.892] | 0.747 ± – | 0.723 |  |

- ulens_minus_predictive_entropy: ΔAUROC 0.0086 [-0.0320, 0.0408]
- ulens_minus_think_entropy: ΔAUROC 0.0086 [-0.0268, 0.0392]
- ulens_minus_gen_length: ΔAUROC 0.0535 [-0.1166, 0.2245]
- ulens_minus_lexical_think: ΔAUROC -0.0086 [-0.1201, 0.1035]
- A_cone AUROC matched on thinking entropy 0.709, on length 0.791
- log A_cone beyond log thinking-entropy, log length, lexical: standardised coefficient 0.643 [-0.252, 4.400], LR 1.49 (p≈0.2219; rows not independent, use the interval)
- Check: E-Space pass vs upstream end-of-thinking U coordinates, max |diff| 0.0000

### Experiment 1, token level (thinking trace)

### cruxeval: 772 traces, 718 with checkable claims, 12171 claims, 954 wrong

- Traces with >= 1 wrong claim: incorrect answers 0.512, correct answers 0.286
- Wrong vs right claims, window mean: cone AUROC 0.633 [0.596, 0.667]; entropy AUROC 0.598 [0.548, 0.653]; cone matched on entropy 0.638
- Within traces with both: cone wrong − right 0.0760, positive in 0.658 of 222 (sign test p 0.0000)
- Cone in the 16 tokens before revision markers minus trace mean (correct_traces): -0.0310 [-0.0387, -0.0235] over 17752 markers
- Cone in the 16 tokens before revision markers minus trace mean (incorrect_traces): 0.0460 [0.0221, 0.0680] over 1872 markers
- Top-1% cone tokens near a wrong claim: 0.031 vs 0.021 of all tokens (ratio 1.47; 230 traces)
- Cone by thinking decile, correct_traces: 1.798 1.723 1.641 1.616 1.591 1.588 1.572 1.557 1.554 1.553
- Cone by thinking decile, incorrect_traces: 1.836 1.829 1.809 1.792 1.728 1.726 1.707 1.662 1.654 1.640

### mbpp: 199 traces, 179 with checkable claims, 4927 claims, 293 wrong

- Traces with >= 1 wrong claim: incorrect answers 0.333, correct answers 0.358
- Wrong vs right claims, window mean: cone AUROC 0.621 [0.535, 0.699]; entropy AUROC 0.644 [0.573, 0.722]; cone matched on entropy 0.628
- Within traces with both: cone wrong − right 0.0593, positive in 0.609 of 69 (sign test p 0.0912)
- Cone in the 16 tokens before revision markers minus trace mean (correct_traces): -0.0330 [-0.0467, -0.0191] over 4574 markers
- Cone in the 16 tokens before revision markers minus trace mean (incorrect_traces): 0.0178 [-0.0518, 0.0894] over 235 markers
- Top-1% cone tokens near a wrong claim: 0.067 vs 0.021 of all tokens (ratio 3.21; 71 traces)
- Cone by thinking decile, correct_traces: 1.763 1.660 1.584 1.560 1.545 1.540 1.534 1.508 1.504 1.490
- Cone by thinking decile, incorrect_traces: 1.798 1.761 1.795 1.727 1.616 1.693 1.610 1.656 1.667 1.583

### Experiment 2: E-Space

anchor lists not frozen yet: run `293 freeze` first (test results withheld)

