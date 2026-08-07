# E11 pilot go/no-go — deepseek-coder-1.3b

**Verdict: NO-GO**

- **FAIL** `behavioural_balanced_accuracy` — test balanced accuracy 0.532 (all pairs 0.530); per variant {'source': 0.664, 'target': 0.4}; threshold 0.75
- **FAIL** `readout_beats_random_control` — layer 23 (chosen on calibration): accuracy advantage -0.014 [-0.043, +0.014]
- **FAIL** `swap_moves_logits_toward_swapped_value` — site L6 (chosen on calibration): paired logit shift +0.002 [-0.001, +0.005], flip rate 0.007

## Cross-operation consistency

- families: 2, min per-family shift +0.001, all positive: True, all CIs positive: False

## No-op control

- max |Δ logit-diff| = 0.00e+00 (passes: True)
