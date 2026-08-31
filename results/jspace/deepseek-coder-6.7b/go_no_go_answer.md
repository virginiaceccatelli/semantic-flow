# E11 pilot go/no-go — deepseek-coder-6.7b (position: answer)

## What this experiment asks

This is a gate decision for the archived J-space intervention. “GO” or “NO-GO” summarizes whether every predeclared requirement was met; the individual checks explain which inference is or is not licensed. It should be read with the J-space section of `docs/ARCHIVE.md`.

**Verdict: NO-GO**

- **PASS** `behavioural_balanced_accuracy` — test balanced accuracy 0.771 (all pairs 0.758); per variant {'source': 0.807, 'target': 0.736}; threshold 0.75
- **PASS** `readout_beats_random_control` — layer 31 (chosen on calibration): accuracy advantage +0.218 [+0.179, +0.257]
- **PASS** `swap_moves_logits_toward_swapped_value` — site L24 (chosen on calibration): paired logit shift +0.280 [+0.213, +0.357], flip rate 0.011
- **FAIL** `swap_is_specific_to_the_value_subspace` — vs logit_value: +0.003 [-0.008, +0.016] (the Jacobian correction, not the unembedding); vs clens_offvalue: +0.265 [+0.194, +0.345] (these values, not the digit subspace at large)

## Cross-operation consistency

- families: 2, min per-family shift +0.000, all positive: True, all CIs positive: False

## No-op control

- max |Δ logit-diff| = 0.00e+00 (passes: True)
