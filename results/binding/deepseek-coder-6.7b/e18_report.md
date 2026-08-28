# E18 — is the binding expressible in scope vocabulary? (deepseek-coder-6.7b)

**Verdict: `not_verbalised`.** The probe succeeds on these very states — binding is linearly present at this position and layer, and E13/R10 shows it is causally used — while no predeclared word pair is both strong and exceptional relative to Gram-matched directions in both value arms at consecutive tested layers. Binding is REPRESENTED AND CAUSALLY USED BUT NOT DETECTABLY VERBALISED in this lexicon at this position. Large raw reversals are not enough: matched random directions produce them frequently too.

Read at the unchanged `x` of `return x` in the **unprompted** E13 program — no answer suffix, no question, no generation — at the binding probe's own layer grid. E17 asks the prompted-behaviour version of this question and is reported separately.

## The checklist (declared before the run)

| check | passed | detail |
|---|---|---|
| ran | True | stage 160 produced scored rows for this model |
| mechanically_valid | True | H10 recorded no violations |
| probe_succeeds | True | the binding probe clears 0.80 on held-out bases at layers [8, 12, 16, 20, 24] — the positive control that makes a null here informative |
| scope_reverses_in_both_arms | False | clear scope pairs at two adjacent tested layers: none (clear layers none) |
| scope_beats_logit_lens | False | and beats the plain logit lens at layers none |
| scope_one_arm_only | False | scope-family reversal fires in exactly one arm at layers none — the literal-tracking signature |
| control_family_reverses | False | a positional or action family fires in both arms at positional: none, action: none |

## The positive control: E13's binding probe, calibration-trained

Fitted on the frozen calibration bases at these very states, read on the frozen test bases. Bar 0.80. It establishes that binding information is present at this position and layer and nothing else; its binary output is never expressed in word coordinates.

| layer | accuracy | f1 | auc | control_accuracy | selectivity | n_calib_bases | n_test_bases | succeeds |
|---|---|---|---|---|---|---|---|---|
| 8 | 1.00000 | 1.00000 | 1.00000 | 0.50268 | 0.49732 | 120 | 280 | True |
| 12 | 1.00000 | 1.00000 | 1.00000 | 0.50804 | 0.49196 | 120 | 280 | True |
| 16 | 1.00000 | 1.00000 | 1.00000 | 0.38304 | 0.61696 | 120 | 280 | True |
| 20 | 1.00000 | 1.00000 | 1.00000 | 0.47411 | 0.52589 | 120 | 280 | True |
| 24 | 1.00000 | 1.00000 | 1.00000 | 0.46875 | 0.53125 | 120 | 280 | True |

E13's own H2, for reference and not merged with the row above: `best layer 8: binding decodable at 1.000 (selectivity 0.524) against a MEASURED surface baseline of 0.500; margin +0.500. Thresholds 0.8 and 0.1. The floor is pinned by construction here: the anchor token is identical across the counterfactual and the mutation is outside the baseline's window.`

## Every pair, before any pooling

Signs are pooled within a predeclared family only after every pair has been reported, so a family mean can always be read against the pairs that produced it. J-lens rows; the other two readouts are in `lexlens_summary.csv`.

| layer | arm | family | inner_word | outer_word | reversal | reversal_ci_lo | reversal_ci_hi | beats_chance | mean_delta |
|---|---|---|---|---|---|---|---|---|---|
| 8 | ab | action | replaced | kept | 0.71786 | 0.66786 | 0.77143 | True | 0.00957 |
| 8 | ab | action | changed | unchanged | 0.00000 | 0.00000 | 0.00000 | False | -0.09843 |
| 8 | ab | positional | later | earlier | 1.00000 | 1.00000 | 1.00000 | True | 0.06225 |
| 8 | ab | positional | second | first | 0.71429 | 0.66071 | 0.76786 | True | 0.01645 |
| 8 | ab | positional | new | original | 0.59643 | 0.53571 | 0.65357 | True | 0.00510 |
| 8 | ab | scope | local | global | 0.96429 | 0.94277 | 0.98571 | True | 0.05741 |
| 8 | ab | scope | inner | outer | 0.12143 | 0.08563 | 0.16071 | False | -0.02401 |
| 8 | ab | scope | inside | outside | 0.11429 | 0.07857 | 0.15357 | False | -0.02634 |
| 8 | ab | scope | nested | module | 0.70000 | 0.64643 | 0.75357 | True | 0.01533 |
| 12 | ab | action | replaced | kept | 0.55714 | 0.50000 | 0.61786 | False | 0.00108 |
| 12 | ab | action | changed | unchanged | 0.72857 | 0.67500 | 0.77857 | True | 0.04075 |
| 12 | ab | positional | later | earlier | 1.00000 | 1.00000 | 1.00000 | True | 0.21524 |
| 12 | ab | positional | second | first | 0.63929 | 0.58571 | 0.69286 | True | 0.01883 |
| 12 | ab | positional | new | original | 0.87500 | 0.83571 | 0.91429 | True | 0.06821 |
| 12 | ab | scope | local | global | 0.78929 | 0.74286 | 0.83571 | True | 0.03693 |
| 12 | ab | scope | inner | outer | 0.62857 | 0.57143 | 0.68571 | True | 0.01018 |
| 12 | ab | scope | inside | outside | 0.90357 | 0.86786 | 0.93571 | True | 0.06198 |
| 12 | ab | scope | nested | module | 0.94286 | 0.91429 | 0.96786 | True | 0.11712 |
| 16 | ab | action | replaced | kept | 0.92500 | 0.89286 | 0.95357 | True | 0.21112 |
| 16 | ab | action | changed | unchanged | 0.55000 | 0.48929 | 0.60357 | False | 0.01130 |
| 16 | ab | positional | later | earlier | 1.00000 | 1.00000 | 1.00000 | True | 0.29866 |
| 16 | ab | positional | second | first | 0.56786 | 0.50714 | 0.62143 | True | 0.03144 |
| 16 | ab | positional | new | original | 0.16786 | 0.12500 | 0.21429 | False | -0.08661 |
| 16 | ab | scope | local | global | 0.96429 | 0.94286 | 0.98571 | True | 0.17582 |
| 16 | ab | scope | inner | outer | 0.66786 | 0.61429 | 0.72143 | True | 0.05516 |
| 16 | ab | scope | inside | outside | 0.82500 | 0.78214 | 0.87143 | True | 0.09998 |
| 16 | ab | scope | nested | module | 1.00000 | 1.00000 | 1.00000 | True | 0.41735 |
| 20 | ab | action | replaced | kept | 0.98214 | 0.96429 | 0.99643 | True | 0.55684 |
| 20 | ab | action | changed | unchanged | 0.10000 | 0.06786 | 0.13571 | False | -0.22924 |
| 20 | ab | positional | later | earlier | 0.73571 | 0.68214 | 0.78571 | True | 0.13704 |
| 20 | ab | positional | second | first | 0.06786 | 0.03929 | 0.10000 | False | -0.25450 |
| 20 | ab | positional | new | original | 0.03571 | 0.01429 | 0.05714 | False | -0.36973 |
| 20 | ab | scope | local | global | 0.99643 | 0.98929 | 1.00000 | True | 0.91741 |
| 20 | ab | scope | inner | outer | 0.50000 | 0.43929 | 0.56071 | False | 0.01633 |
| 20 | ab | scope | inside | outside | 0.97143 | 0.95357 | 0.98929 | True | 0.41879 |
| 20 | ab | scope | nested | module | 0.96429 | 0.93929 | 0.98571 | True | 0.48956 |
| 24 | ab | action | replaced | kept | 0.81429 | 0.76786 | 0.85714 | True | 0.25598 |
| 24 | ab | action | changed | unchanged | 0.06786 | 0.03929 | 0.09643 | False | -0.51026 |
| 24 | ab | positional | later | earlier | 0.88214 | 0.84286 | 0.91786 | True | 0.38152 |
| 24 | ab | positional | second | first | 0.17500 | 0.12857 | 0.22143 | False | -0.23217 |
| 24 | ab | positional | new | original | 0.22143 | 0.17500 | 0.27143 | False | -0.23577 |
| 24 | ab | scope | local | global | 0.98571 | 0.97143 | 0.99643 | True | 0.81393 |
| 24 | ab | scope | inner | outer | 0.51429 | 0.45714 | 0.57143 | False | -0.01514 |
| 24 | ab | scope | inside | outside | 0.85000 | 0.80714 | 0.89286 | True | 0.27536 |
| 24 | ab | scope | nested | module | 0.99286 | 0.98214 | 1.00000 | True | 0.79827 |
| 8 | ba | action | replaced | kept | 0.71429 | 0.66071 | 0.76429 | True | 0.01009 |
| 8 | ba | action | changed | unchanged | 0.00000 | 0.00000 | 0.00000 | False | -0.09789 |
| 8 | ba | positional | later | earlier | 1.00000 | 1.00000 | 1.00000 | True | 0.06183 |
| 8 | ba | positional | second | first | 0.71071 | 0.65714 | 0.76786 | True | 0.01628 |
| 8 | ba | positional | new | original | 0.59643 | 0.53929 | 0.65366 | True | 0.00477 |
| 8 | ba | scope | local | global | 0.95000 | 0.92143 | 0.97500 | True | 0.05678 |
| 8 | ba | scope | inner | outer | 0.12500 | 0.08929 | 0.16429 | False | -0.02408 |
| 8 | ba | scope | inside | outside | 0.14286 | 0.10357 | 0.18571 | False | -0.02571 |
| 8 | ba | scope | nested | module | 0.68214 | 0.62857 | 0.73571 | True | 0.01533 |
| 12 | ba | action | replaced | kept | 0.58571 | 0.52857 | 0.64286 | True | 0.00445 |
| 12 | ba | action | changed | unchanged | 0.73571 | 0.67857 | 0.78571 | True | 0.04166 |
| 12 | ba | positional | later | earlier | 1.00000 | 1.00000 | 1.00000 | True | 0.21682 |
| 12 | ba | positional | second | first | 0.59286 | 0.53920 | 0.64643 | True | 0.01666 |
| 12 | ba | positional | new | original | 0.86071 | 0.82143 | 0.90000 | True | 0.06821 |
| 12 | ba | scope | local | global | 0.78929 | 0.74286 | 0.83571 | True | 0.03672 |
| 12 | ba | scope | inner | outer | 0.58571 | 0.52857 | 0.63937 | True | 0.00835 |
| 12 | ba | scope | inside | outside | 0.89643 | 0.86071 | 0.93214 | True | 0.06125 |
| 12 | ba | scope | nested | module | 0.96071 | 0.93571 | 0.98214 | True | 0.11973 |
| 16 | ba | action | replaced | kept | 0.94643 | 0.91786 | 0.97143 | True | 0.21810 |
| 16 | ba | action | changed | unchanged | 0.56429 | 0.50714 | 0.62143 | True | 0.01355 |
| 16 | ba | positional | later | earlier | 1.00000 | 1.00000 | 1.00000 | True | 0.30294 |
| 16 | ba | positional | second | first | 0.55000 | 0.49286 | 0.60714 | False | 0.02564 |
| 16 | ba | positional | new | original | 0.20000 | 0.15357 | 0.24643 | False | -0.07860 |
| 16 | ba | scope | local | global | 0.96786 | 0.94643 | 0.98571 | True | 0.17688 |
| 16 | ba | scope | inner | outer | 0.67500 | 0.62143 | 0.72857 | True | 0.05521 |
| 16 | ba | scope | inside | outside | 0.81071 | 0.76429 | 0.85714 | True | 0.09750 |
| 16 | ba | scope | nested | module | 1.00000 | 1.00000 | 1.00000 | True | 0.41971 |
| 20 | ba | action | replaced | kept | 0.98214 | 0.96429 | 0.99643 | True | 0.56224 |
| 20 | ba | action | changed | unchanged | 0.10357 | 0.06786 | 0.13929 | False | -0.23399 |
| 20 | ba | positional | later | earlier | 0.75000 | 0.70000 | 0.80000 | True | 0.14138 |
| 20 | ba | positional | second | first | 0.10000 | 0.06429 | 0.13571 | False | -0.25636 |
| 20 | ba | positional | new | original | 0.03929 | 0.01786 | 0.06080 | False | -0.36323 |
| 20 | ba | scope | local | global | 1.00000 | 1.00000 | 1.00000 | True | 0.92669 |
| 20 | ba | scope | inner | outer | 0.51429 | 0.45357 | 0.57500 | False | 0.01806 |
| 20 | ba | scope | inside | outside | 0.98571 | 0.97134 | 0.99643 | True | 0.41522 |
| 20 | ba | scope | nested | module | 0.98214 | 0.96429 | 0.99643 | True | 0.49172 |
| 24 | ba | action | replaced | kept | 0.79286 | 0.74286 | 0.83929 | True | 0.26707 |
| 24 | ba | action | changed | unchanged | 0.06429 | 0.03571 | 0.09286 | False | -0.51232 |
| 24 | ba | positional | later | earlier | 0.89286 | 0.85357 | 0.92500 | True | 0.38054 |
| 24 | ba | positional | second | first | 0.19643 | 0.15000 | 0.24286 | False | -0.23062 |
| 24 | ba | positional | new | original | 0.22500 | 0.17857 | 0.27857 | False | -0.22386 |
| 24 | ba | scope | local | global | 0.99286 | 0.98214 | 1.00000 | True | 0.82170 |
| 24 | ba | scope | inner | outer | 0.51071 | 0.45357 | 0.57143 | False | -0.01616 |
| 24 | ba | scope | inside | outside | 0.87500 | 0.83571 | 0.91071 | True | 0.27510 |
| 24 | ba | scope | nested | module | 0.99643 | 0.98929 | 1.00000 | True | 0.79428 |
| 8 | both | action | replaced | kept | 0.71607 | 0.66786 | 0.76429 | True | 0.00983 |
| 8 | both | action | changed | unchanged | 0.00000 | 0.00000 | 0.00000 | False | -0.09816 |
| 8 | both | positional | later | earlier | 1.00000 | 1.00000 | 1.00000 | True | 0.06204 |
| 8 | both | positional | second | first | 0.71250 | 0.66250 | 0.76433 | True | 0.01637 |
| 8 | both | positional | new | original | 0.59643 | 0.54286 | 0.64826 | True | 0.00493 |
| 8 | both | scope | local | global | 0.95714 | 0.93393 | 0.97857 | True | 0.05710 |
| 8 | both | scope | inner | outer | 0.12321 | 0.08750 | 0.16250 | False | -0.02404 |
| 8 | both | scope | inside | outside | 0.12857 | 0.09643 | 0.16429 | False | -0.02602 |
| 8 | both | scope | nested | module | 0.69107 | 0.63750 | 0.74286 | True | 0.01533 |
| 12 | both | action | replaced | kept | 0.57143 | 0.51786 | 0.62679 | True | 0.00276 |
| 12 | both | action | changed | unchanged | 0.73214 | 0.68214 | 0.78036 | True | 0.04120 |
| 12 | both | positional | later | earlier | 1.00000 | 1.00000 | 1.00000 | True | 0.21603 |
| 12 | both | positional | second | first | 0.61607 | 0.56607 | 0.66607 | True | 0.01774 |
| 12 | both | positional | new | original | 0.86786 | 0.83214 | 0.90183 | True | 0.06821 |
| 12 | both | scope | local | global | 0.78929 | 0.74464 | 0.83214 | True | 0.03683 |
| 12 | both | scope | inner | outer | 0.60714 | 0.55536 | 0.66071 | True | 0.00926 |
| 12 | both | scope | inside | outside | 0.90000 | 0.87143 | 0.93036 | True | 0.06161 |
| 12 | both | scope | nested | module | 0.95179 | 0.92857 | 0.97143 | True | 0.11842 |
| 16 | both | action | replaced | kept | 0.93571 | 0.90714 | 0.96071 | True | 0.21461 |
| 16 | both | action | changed | unchanged | 0.55714 | 0.50357 | 0.60893 | True | 0.01243 |
| 16 | both | positional | later | earlier | 1.00000 | 1.00000 | 1.00000 | True | 0.30080 |
| 16 | both | positional | second | first | 0.55893 | 0.51250 | 0.60179 | True | 0.02854 |
| 16 | both | positional | new | original | 0.18393 | 0.14643 | 0.22321 | False | -0.08261 |
| 16 | both | scope | local | global | 0.96607 | 0.94643 | 0.98393 | True | 0.17635 |
| 16 | both | scope | inner | outer | 0.67143 | 0.61964 | 0.72326 | True | 0.05519 |
| 16 | both | scope | inside | outside | 0.81786 | 0.78393 | 0.85357 | True | 0.09874 |
| 16 | both | scope | nested | module | 1.00000 | 1.00000 | 1.00000 | True | 0.41853 |
| 20 | both | action | replaced | kept | 0.98214 | 0.96607 | 0.99464 | True | 0.55954 |
| 20 | both | action | changed | unchanged | 0.10179 | 0.07321 | 0.13393 | False | -0.23162 |
| 20 | both | positional | later | earlier | 0.74286 | 0.70179 | 0.78214 | True | 0.13921 |
| 20 | both | positional | second | first | 0.08393 | 0.05893 | 0.11071 | False | -0.25543 |
| 20 | both | positional | new | original | 0.03750 | 0.01786 | 0.05893 | False | -0.36648 |
| 20 | both | scope | local | global | 0.99821 | 0.99464 | 1.00000 | True | 0.92205 |
| 20 | both | scope | inner | outer | 0.50714 | 0.45710 | 0.55897 | False | 0.01719 |
| 20 | both | scope | inside | outside | 0.97857 | 0.96786 | 0.98929 | True | 0.41701 |
| 20 | both | scope | nested | module | 0.97321 | 0.95714 | 0.98750 | True | 0.49064 |
| 24 | both | action | replaced | kept | 0.80357 | 0.76246 | 0.84286 | True | 0.26153 |
| 24 | both | action | changed | unchanged | 0.06607 | 0.04107 | 0.09286 | False | -0.51129 |
| 24 | both | positional | later | earlier | 0.88750 | 0.85357 | 0.91964 | True | 0.38103 |
| 24 | both | positional | second | first | 0.18571 | 0.14821 | 0.22500 | False | -0.23140 |
| 24 | both | positional | new | original | 0.22321 | 0.18036 | 0.26964 | False | -0.22982 |
| 24 | both | scope | local | global | 0.98929 | 0.97857 | 0.99821 | True | 0.81782 |
| 24 | both | scope | inner | outer | 0.51250 | 0.45893 | 0.56786 | False | -0.01565 |
| 24 | both | scope | inside | outside | 0.86250 | 0.82857 | 0.89464 | True | 0.27523 |
| 24 | both | scope | nested | module | 0.99464 | 0.98571 | 1.00000 | True | 0.79628 |

## Which word contrasts are distinctly readable?

Each pair is kept separate. `random_percentile_*` places its J-lens reversal against independent Gram-matched directions on the same test bases. `clear_at_layer` requires at least 0.80 reversal in both value arms and at least the 99th random-direction percentile in both. The verdict requires the same scope pair to be clear at two adjacent tested layers; otherwise the honest result is no consistent verbalisation.

| layer | family | inner_word | outer_word | reversal_ab | reversal_ba | random_percentile_ab | random_percentile_ba | both_arm_percentile | min_arm_reversal | logit_min_arm_reversal | beats_logit | clear_at_layer | n_random_directions |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | action | replaced | kept | 0.71786 | 0.71429 | 0.60500 | 0.60300 | 0.60300 | 0.71429 | 0.96786 | False | False | 500 |
| 8 | action | changed | unchanged | 0.00000 | 0.00000 | 0.03600 | 0.03800 | 0.03600 | 0.00000 | 0.49286 | False | False | 500 |
| 8 | positional | later | earlier | 1.00000 | 1.00000 | 0.95600 | 0.95900 | 0.95600 | 1.00000 | 1.00000 | False | False | 500 |
| 8 | positional | second | first | 0.71429 | 0.71071 | 0.64700 | 0.64900 | 0.64700 | 0.71071 | 0.66071 | True | False | 500 |
| 8 | positional | new | original | 0.59643 | 0.59643 | 0.53300 | 0.53600 | 0.53300 | 0.59643 | 0.42500 | True | False | 500 |
| 8 | scope | local | global | 0.96429 | 0.95000 | 0.82400 | 0.80000 | 0.80000 | 0.95000 | 0.27857 | True | False | 500 |
| 8 | scope | inner | outer | 0.12143 | 0.12500 | 0.29400 | 0.30000 | 0.29400 | 0.12143 | 0.55357 | False | False | 500 |
| 8 | scope | inside | outside | 0.11429 | 0.14286 | 0.27600 | 0.30500 | 0.27600 | 0.11429 | 0.70000 | False | False | 500 |
| 8 | scope | nested | module | 0.70000 | 0.68214 | 0.59800 | 0.58800 | 0.58800 | 0.68214 | 0.08214 | True | False | 500 |
| 12 | action | replaced | kept | 0.55714 | 0.58571 | 0.51300 | 0.52500 | 0.51300 | 0.55714 | 0.95357 | False | False | 500 |
| 12 | action | changed | unchanged | 0.72857 | 0.73571 | 0.63200 | 0.63900 | 0.63200 | 0.72857 | 0.99286 | False | False | 500 |
| 12 | positional | later | earlier | 1.00000 | 1.00000 | 0.95500 | 0.95800 | 0.95500 | 1.00000 | 1.00000 | False | False | 500 |
| 12 | positional | second | first | 0.63929 | 0.59286 | 0.59500 | 0.57200 | 0.57200 | 0.59286 | 0.00000 | True | False | 500 |
| 12 | positional | new | original | 0.87500 | 0.86071 | 0.76200 | 0.74800 | 0.74800 | 0.86071 | 0.97143 | False | False | 500 |
| 12 | scope | local | global | 0.78929 | 0.78929 | 0.64700 | 0.64900 | 0.64700 | 0.78929 | 1.00000 | False | False | 500 |
| 12 | scope | inner | outer | 0.62857 | 0.58571 | 0.58700 | 0.56200 | 0.56200 | 0.58571 | 0.47500 | True | False | 500 |
| 12 | scope | inside | outside | 0.90357 | 0.89643 | 0.71900 | 0.71300 | 0.71300 | 0.89643 | 0.65357 | True | False | 500 |
| 12 | scope | nested | module | 0.94286 | 0.96071 | 0.75700 | 0.78500 | 0.75700 | 0.94286 | 0.89643 | True | False | 500 |
| 16 | action | replaced | kept | 0.92500 | 0.94643 | 0.76400 | 0.79000 | 0.76400 | 0.92500 | 1.00000 | False | False | 500 |
| 16 | action | changed | unchanged | 0.55000 | 0.56429 | 0.52600 | 0.53800 | 0.52600 | 0.55000 | 0.96071 | False | False | 500 |
| 16 | positional | later | earlier | 1.00000 | 1.00000 | 0.96400 | 0.96400 | 0.96400 | 1.00000 | 0.99643 | True | False | 500 |
| 16 | positional | second | first | 0.56786 | 0.55000 | 0.53700 | 0.53000 | 0.53000 | 0.55000 | 0.13214 | True | False | 500 |
| 16 | positional | new | original | 0.16786 | 0.20000 | 0.31200 | 0.34000 | 0.31200 | 0.16786 | 0.60714 | False | False | 500 |
| 16 | scope | local | global | 0.96429 | 0.96786 | 0.79100 | 0.79800 | 0.79100 | 0.96429 | 1.00000 | False | False | 500 |
| 16 | scope | inner | outer | 0.66786 | 0.67500 | 0.56800 | 0.58100 | 0.56800 | 0.66786 | 0.42500 | True | False | 500 |
| 16 | scope | inside | outside | 0.82500 | 0.81071 | 0.65900 | 0.64400 | 0.64400 | 0.81071 | 0.43929 | True | False | 500 |
| 16 | scope | nested | module | 1.00000 | 1.00000 | 0.95300 | 0.96100 | 0.95300 | 1.00000 | 1.00000 | False | False | 500 |
| 20 | action | replaced | kept | 0.98214 | 0.98214 | 0.89000 | 0.89100 | 0.89000 | 0.98214 | 0.98214 | False | False | 500 |
| 20 | action | changed | unchanged | 0.10000 | 0.10357 | 0.20700 | 0.20700 | 0.20700 | 0.10000 | 0.30714 | False | False | 500 |
| 20 | positional | later | earlier | 0.73571 | 0.75000 | 0.65800 | 0.66800 | 0.65800 | 0.73571 | 0.61071 | True | False | 500 |
| 20 | positional | second | first | 0.06786 | 0.10000 | 0.20500 | 0.24100 | 0.20500 | 0.06786 | 0.19286 | False | False | 500 |
| 20 | positional | new | original | 0.03571 | 0.03929 | 0.17300 | 0.18000 | 0.17300 | 0.03571 | 0.01071 | True | False | 500 |
| 20 | scope | local | global | 0.99643 | 1.00000 | 0.93600 | 0.97200 | 0.93600 | 0.99643 | 1.00000 | False | False | 500 |
| 20 | scope | inner | outer | 0.50000 | 0.51429 | 0.46800 | 0.47400 | 0.46800 | 0.50000 | 0.22500 | True | False | 500 |
| 20 | scope | inside | outside | 0.97143 | 0.98571 | 0.81800 | 0.85200 | 0.81800 | 0.97143 | 0.90714 | True | False | 500 |
| 20 | scope | nested | module | 0.96429 | 0.98214 | 0.83900 | 0.87800 | 0.83900 | 0.96429 | 0.97143 | False | False | 500 |
| 24 | action | replaced | kept | 0.81429 | 0.79286 | 0.72200 | 0.70300 | 0.70300 | 0.79286 | 0.83214 | False | False | 500 |
| 24 | action | changed | unchanged | 0.06786 | 0.06429 | 0.17100 | 0.16400 | 0.16400 | 0.06429 | 0.07143 | False | False | 500 |
| 24 | positional | later | earlier | 0.88214 | 0.89286 | 0.77400 | 0.79000 | 0.77400 | 0.88214 | 0.67857 | True | False | 500 |
| 24 | positional | second | first | 0.17500 | 0.19643 | 0.22800 | 0.25700 | 0.22800 | 0.17500 | 0.10000 | True | False | 500 |
| 24 | positional | new | original | 0.22143 | 0.22500 | 0.28100 | 0.29100 | 0.28100 | 0.22143 | 0.13214 | True | False | 500 |
| 24 | scope | local | global | 0.98571 | 0.99286 | 0.91500 | 0.92600 | 0.91500 | 0.98571 | 0.98929 | False | False | 500 |
| 24 | scope | inner | outer | 0.51429 | 0.51071 | 0.49500 | 0.49800 | 0.49500 | 0.51071 | 0.32857 | True | False | 500 |
| 24 | scope | inside | outside | 0.85000 | 0.87500 | 0.75900 | 0.78400 | 0.75900 | 0.85000 | 0.58571 | True | False | 500 |
| 24 | scope | nested | module | 0.99286 | 0.99643 | 0.92700 | 0.94200 | 0.92700 | 0.99286 | 0.98571 | True | False | 500 |

## Reversal by family, per arm

`reversal` is the share of base programs whose inner-minus-outer margin moves in the predicted direction when the one differing token flips the binding. Chance is 0.500. Intervals are cluster bootstraps over base programs. Only `reversal` is comparable across readouts — the three lenses put out scores on different scales — so `mean_delta` is reported within a readout and never across.

| layer | readout | family | arm | reversal | reversal_ci_lo | reversal_ci_hi | beats_chance | mean_delta | delta_ci_lo | delta_ci_hi | n_bases |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | gram_random | action | ab | 0.50750 | 0.50624 | 0.50878 | True | 0.00105 | 0.00094 | 0.00115 | 280 |
| 8 | gram_random | positional | ab | 0.50800 | 0.50710 | 0.50892 | True | 0.00149 | 0.00141 | 0.00157 | 280 |
| 8 | gram_random | scope | ab | 0.49979 | 0.49878 | 0.50077 | False | -0.00054 | -0.00062 | -0.00047 | 280 |
| 8 | jlens | action | ab | 0.35893 | 0.33393 | 0.38571 | False | -0.04443 | -0.04654 | -0.04230 | 280 |
| 8 | jlens | positional | ab | 0.77024 | 0.74286 | 0.79762 | True | 0.02794 | 0.02601 | 0.02985 | 280 |
| 8 | jlens | scope | ab | 0.47500 | 0.45625 | 0.49464 | False | 0.00560 | 0.00415 | 0.00711 | 280 |
| 8 | logit | action | ab | 0.73036 | 0.69821 | 0.76071 | True | 1.28641 | 1.17335 | 1.39335 | 280 |
| 8 | logit | positional | ab | 0.69881 | 0.67262 | 0.72619 | True | 1.82734 | 1.73660 | 1.92224 | 280 |
| 8 | logit | scope | ab | 0.40536 | 0.37679 | 0.43214 | False | -0.41181 | -0.49353 | -0.32660 | 280 |
| 12 | gram_random | action | ab | 0.50057 | 0.49941 | 0.50181 | False | 0.00049 | 0.00024 | 0.00073 | 280 |
| 12 | gram_random | positional | ab | 0.48403 | 0.48302 | 0.48501 | False | -0.00433 | -0.00454 | -0.00413 | 280 |
| 12 | gram_random | scope | ab | 0.50194 | 0.50113 | 0.50282 | True | 0.00104 | 0.00091 | 0.00116 | 280 |
| 12 | jlens | action | ab | 0.64286 | 0.59643 | 0.68750 | True | 0.02091 | 0.01466 | 0.02720 | 280 |
| 12 | jlens | positional | ab | 0.83810 | 0.81667 | 0.85952 | True | 0.10076 | 0.09694 | 0.10450 | 280 |
| 12 | jlens | scope | ab | 0.81607 | 0.79375 | 0.83929 | True | 0.05655 | 0.05297 | 0.06025 | 280 |
| 12 | logit | action | ab | 0.97321 | 0.95893 | 0.98571 | True | 4.63979 | 4.43216 | 4.83796 | 280 |
| 12 | logit | positional | ab | 0.65833 | 0.65119 | 0.66548 | True | 2.87656 | 2.74412 | 3.00492 | 280 |
| 12 | logit | scope | ab | 0.76071 | 0.73661 | 0.78484 | True | 3.09150 | 2.94405 | 3.24662 | 280 |
| 16 | gram_random | action | ab | 0.50260 | 0.50146 | 0.50377 | True | 0.00026 | -0.00007 | 0.00060 | 280 |
| 16 | gram_random | positional | ab | 0.49841 | 0.49726 | 0.49955 | False | -0.00125 | -0.00168 | -0.00079 | 280 |
| 16 | gram_random | scope | ab | 0.50546 | 0.50469 | 0.50623 | True | 0.00162 | 0.00129 | 0.00195 | 280 |
| 16 | jlens | action | ab | 0.73750 | 0.70000 | 0.77143 | True | 0.11121 | 0.09959 | 0.12231 | 280 |
| 16 | jlens | positional | ab | 0.57857 | 0.55357 | 0.60238 | True | 0.08116 | 0.07354 | 0.08837 | 280 |
| 16 | jlens | scope | ab | 0.86429 | 0.84375 | 0.88482 | True | 0.18708 | 0.17875 | 0.19543 | 280 |
| 16 | logit | action | ab | 0.98036 | 0.96786 | 0.99107 | True | 7.24069 | 6.96356 | 7.49949 | 280 |
| 16 | logit | positional | ab | 0.57976 | 0.55595 | 0.60357 | True | 1.25420 | 1.04558 | 1.45983 | 280 |
| 16 | logit | scope | ab | 0.71607 | 0.69196 | 0.74018 | True | 5.42194 | 5.21292 | 5.63362 | 280 |
| 20 | gram_random | action | ab | 0.51129 | 0.50998 | 0.51264 | True | 0.00261 | 0.00193 | 0.00334 | 280 |
| 20 | gram_random | positional | ab | 0.49749 | 0.49655 | 0.49845 | False | -0.00602 | -0.00668 | -0.00532 | 280 |
| 20 | gram_random | scope | ab | 0.50385 | 0.50288 | 0.50477 | True | -0.00104 | -0.00168 | -0.00037 | 280 |
| 20 | jlens | action | ab | 0.54107 | 0.52317 | 0.56071 | True | 0.16380 | 0.14291 | 0.18359 | 280 |
| 20 | jlens | positional | ab | 0.27976 | 0.25714 | 0.30119 | False | -0.16240 | -0.17691 | -0.14829 | 280 |
| 20 | jlens | scope | ab | 0.85804 | 0.83929 | 0.87679 | True | 0.46052 | 0.44097 | 0.47953 | 280 |
| 20 | logit | action | ab | 0.64643 | 0.61786 | 0.67500 | True | 5.00812 | 4.57252 | 5.41040 | 280 |
| 20 | logit | positional | ab | 0.27143 | 0.24405 | 0.29765 | False | -3.17908 | -3.47209 | -2.88862 | 280 |
| 20 | logit | scope | ab | 0.77589 | 0.75980 | 0.79196 | True | 8.90416 | 8.51345 | 9.29108 | 280 |
| 24 | gram_random | action | ab | 0.49288 | 0.49144 | 0.49429 | False | -0.00347 | -0.00458 | -0.00236 | 280 |
| 24 | gram_random | positional | ab | 0.51375 | 0.51260 | 0.51494 | True | 0.01719 | 0.01625 | 0.01810 | 280 |
| 24 | gram_random | scope | ab | 0.51509 | 0.51377 | 0.51640 | True | 0.01673 | 0.01574 | 0.01774 | 280 |
| 24 | jlens | action | ab | 0.44107 | 0.41250 | 0.46786 | False | -0.12714 | -0.15793 | -0.10020 | 280 |
| 24 | jlens | positional | ab | 0.42619 | 0.39643 | 0.45595 | False | -0.02881 | -0.05161 | -0.00663 | 280 |
| 24 | jlens | scope | ab | 0.83571 | 0.81339 | 0.85714 | True | 0.46811 | 0.44525 | 0.49054 | 280 |
| 24 | logit | action | ab | 0.45179 | 0.42500 | 0.47679 | False | -2.23883 | -2.84269 | -1.68390 | 280 |
| 24 | logit | positional | ab | 0.31667 | 0.28690 | 0.34286 | False | -2.85715 | -3.27623 | -2.43533 | 280 |
| 24 | logit | scope | ab | 0.72857 | 0.70536 | 0.75179 | True | 7.67665 | 7.20673 | 8.13653 | 280 |
| 8 | gram_random | action | ba | 0.50773 | 0.50641 | 0.50902 | True | 0.00106 | 0.00096 | 0.00116 | 280 |
| 8 | gram_random | positional | ba | 0.50742 | 0.50649 | 0.50834 | True | 0.00150 | 0.00142 | 0.00158 | 280 |
| 8 | gram_random | scope | ba | 0.50014 | 0.49922 | 0.50103 | False | -0.00053 | -0.00061 | -0.00046 | 280 |
| 8 | jlens | action | ba | 0.35714 | 0.33036 | 0.38214 | False | -0.04390 | -0.04598 | -0.04174 | 280 |
| 8 | jlens | positional | ba | 0.76905 | 0.74167 | 0.79646 | True | 0.02763 | 0.02570 | 0.02952 | 280 |
| 8 | jlens | scope | ba | 0.47500 | 0.45536 | 0.49554 | False | 0.00558 | 0.00408 | 0.00709 | 280 |
| 8 | logit | action | ba | 0.73571 | 0.70536 | 0.76607 | True | 1.28903 | 1.17669 | 1.39533 | 280 |
| 8 | logit | positional | ba | 0.69524 | 0.67143 | 0.72143 | True | 1.80117 | 1.71371 | 1.89280 | 280 |
| 8 | logit | scope | ba | 0.40804 | 0.38036 | 0.43661 | False | -0.39827 | -0.47693 | -0.31269 | 280 |
| 12 | gram_random | action | ba | 0.50108 | 0.49987 | 0.50228 | False | 0.00055 | 0.00028 | 0.00081 | 280 |
| 12 | gram_random | positional | ba | 0.48428 | 0.48326 | 0.48527 | False | -0.00434 | -0.00455 | -0.00414 | 280 |
| 12 | gram_random | scope | ba | 0.50232 | 0.50159 | 0.50306 | True | 0.00104 | 0.00092 | 0.00117 | 280 |
| 12 | jlens | action | ba | 0.66071 | 0.61429 | 0.70536 | True | 0.02305 | 0.01658 | 0.02905 | 280 |
| 12 | jlens | positional | ba | 0.81786 | 0.79643 | 0.83929 | True | 0.10056 | 0.09671 | 0.10416 | 280 |
| 12 | jlens | scope | ba | 0.80804 | 0.78571 | 0.83125 | True | 0.05651 | 0.05292 | 0.06023 | 280 |
| 12 | logit | action | ba | 0.97857 | 0.96607 | 0.98929 | True | 4.65942 | 4.44989 | 4.85555 | 280 |
| 12 | logit | positional | ba | 0.65714 | 0.65000 | 0.66310 | True | 2.83592 | 2.71094 | 2.96711 | 280 |
| 12 | logit | scope | ba | 0.76339 | 0.73929 | 0.79018 | True | 3.12548 | 2.97868 | 3.28550 | 280 |
| 16 | gram_random | action | ba | 0.50191 | 0.50072 | 0.50306 | True | 0.00017 | -0.00018 | 0.00051 | 280 |
| 16 | gram_random | positional | ba | 0.49889 | 0.49771 | 0.49998 | False | -0.00112 | -0.00160 | -0.00063 | 280 |
| 16 | gram_random | scope | ba | 0.50539 | 0.50458 | 0.50623 | True | 0.00165 | 0.00132 | 0.00197 | 280 |
| 16 | jlens | action | ba | 0.75536 | 0.72143 | 0.79107 | True | 0.11583 | 0.10489 | 0.12636 | 280 |
| 16 | jlens | positional | ba | 0.58333 | 0.55952 | 0.60714 | True | 0.08332 | 0.07612 | 0.09098 | 280 |
| 16 | jlens | scope | ba | 0.86339 | 0.84286 | 0.88482 | True | 0.18733 | 0.17955 | 0.19535 | 280 |
| 16 | logit | action | ba | 0.98214 | 0.97143 | 0.99107 | True | 7.34505 | 7.07896 | 7.60958 | 280 |
| 16 | logit | positional | ba | 0.59167 | 0.56786 | 0.61548 | True | 1.27862 | 1.07270 | 1.49309 | 280 |
| 16 | logit | scope | ba | 0.73125 | 0.70625 | 0.75627 | True | 5.44931 | 5.24695 | 5.66422 | 280 |
| 20 | gram_random | action | ba | 0.51114 | 0.50971 | 0.51254 | True | 0.00270 | 0.00201 | 0.00346 | 280 |
| 20 | gram_random | positional | ba | 0.49709 | 0.49612 | 0.49818 | False | -0.00587 | -0.00653 | -0.00518 | 280 |
| 20 | gram_random | scope | ba | 0.50425 | 0.50323 | 0.50521 | True | -0.00102 | -0.00168 | -0.00034 | 280 |
| 20 | jlens | action | ba | 0.54286 | 0.52496 | 0.56250 | True | 0.16412 | 0.14204 | 0.18376 | 280 |
| 20 | jlens | positional | ba | 0.29643 | 0.27262 | 0.31786 | False | -0.15941 | -0.17373 | -0.14485 | 280 |
| 20 | jlens | scope | ba | 0.87054 | 0.85446 | 0.88661 | True | 0.46292 | 0.44353 | 0.48183 | 280 |
| 20 | logit | action | ba | 0.65000 | 0.62143 | 0.67857 | True | 4.95765 | 4.49450 | 5.35249 | 280 |
| 20 | logit | positional | ba | 0.28333 | 0.25711 | 0.30952 | False | -3.12071 | -3.40475 | -2.81073 | 280 |
| 20 | logit | scope | ba | 0.78125 | 0.76429 | 0.79821 | True | 8.92284 | 8.52210 | 9.32008 | 280 |
| 24 | gram_random | action | ba | 0.49248 | 0.49101 | 0.49386 | False | -0.00385 | -0.00497 | -0.00274 | 280 |
| 24 | gram_random | positional | ba | 0.51477 | 0.51367 | 0.51592 | True | 0.01750 | 0.01654 | 0.01846 | 280 |
| 24 | gram_random | scope | ba | 0.51527 | 0.51403 | 0.51645 | True | 0.01702 | 0.01602 | 0.01797 | 280 |
| 24 | jlens | action | ba | 0.42857 | 0.39643 | 0.45714 | False | -0.12262 | -0.15321 | -0.09404 | 280 |
| 24 | jlens | positional | ba | 0.43810 | 0.41190 | 0.46429 | False | -0.02465 | -0.04543 | -0.00375 | 280 |
| 24 | jlens | scope | ba | 0.84375 | 0.82411 | 0.86339 | True | 0.46873 | 0.44721 | 0.48921 | 280 |
| 24 | logit | action | ba | 0.45357 | 0.42321 | 0.48036 | False | -2.13806 | -2.75095 | -1.54475 | 280 |
| 24 | logit | positional | ba | 0.30357 | 0.27619 | 0.33095 | False | -2.79967 | -3.19055 | -2.40049 | 280 |
| 24 | logit | scope | ba | 0.73125 | 0.70804 | 0.75268 | True | 7.65686 | 7.19866 | 8.09164 | 280 |
| 8 | gram_random | action | both | 0.50762 | 0.50643 | 0.50882 | True | 0.00105 | 0.00095 | 0.00115 | 280 |
| 8 | gram_random | positional | both | 0.50771 | 0.50688 | 0.50853 | True | 0.00150 | 0.00142 | 0.00158 | 280 |
| 8 | gram_random | scope | both | 0.49996 | 0.49905 | 0.50087 | False | -0.00054 | -0.00061 | -0.00047 | 280 |
| 8 | jlens | action | both | 0.35804 | 0.33393 | 0.38214 | False | -0.04417 | -0.04623 | -0.04206 | 280 |
| 8 | jlens | positional | both | 0.76964 | 0.74403 | 0.79524 | True | 0.02778 | 0.02586 | 0.02966 | 280 |
| 8 | jlens | scope | both | 0.47500 | 0.45670 | 0.49375 | False | 0.00559 | 0.00412 | 0.00707 | 280 |
| 8 | logit | action | both | 0.73304 | 0.70446 | 0.76071 | True | 1.28772 | 1.17629 | 1.39315 | 280 |
| 8 | logit | positional | both | 0.69702 | 0.67262 | 0.72262 | True | 1.81426 | 1.72605 | 1.90525 | 280 |
| 8 | logit | scope | both | 0.40670 | 0.38080 | 0.43171 | False | -0.40504 | -0.48542 | -0.32096 | 280 |
| 12 | gram_random | action | both | 0.50083 | 0.49971 | 0.50195 | False | 0.00052 | 0.00027 | 0.00077 | 280 |
| 12 | gram_random | positional | both | 0.48415 | 0.48326 | 0.48504 | False | -0.00434 | -0.00455 | -0.00414 | 280 |
| 12 | gram_random | scope | both | 0.50213 | 0.50144 | 0.50283 | True | 0.00104 | 0.00092 | 0.00116 | 280 |
| 12 | jlens | action | both | 0.65179 | 0.60804 | 0.69464 | True | 0.02198 | 0.01577 | 0.02789 | 280 |
| 12 | jlens | positional | both | 0.82798 | 0.80833 | 0.84702 | True | 0.10066 | 0.09690 | 0.10419 | 280 |
| 12 | jlens | scope | both | 0.81205 | 0.79152 | 0.83304 | True | 0.05653 | 0.05305 | 0.06011 | 280 |
| 12 | logit | action | both | 0.97589 | 0.96339 | 0.98661 | True | 4.64961 | 4.45139 | 4.83505 | 280 |
| 12 | logit | positional | both | 0.65774 | 0.65238 | 0.66250 | True | 2.85624 | 2.73137 | 2.97804 | 280 |
| 12 | logit | scope | both | 0.76205 | 0.73929 | 0.78661 | True | 3.10849 | 2.96624 | 3.25845 | 280 |
| 16 | gram_random | action | both | 0.50226 | 0.50122 | 0.50329 | True | 0.00021 | -0.00011 | 0.00055 | 280 |
| 16 | gram_random | positional | both | 0.49865 | 0.49762 | 0.49964 | False | -0.00118 | -0.00159 | -0.00076 | 280 |
| 16 | gram_random | scope | both | 0.50543 | 0.50474 | 0.50615 | True | 0.00164 | 0.00132 | 0.00195 | 280 |
| 16 | jlens | action | both | 0.74643 | 0.71250 | 0.77946 | True | 0.11352 | 0.10259 | 0.12398 | 280 |
| 16 | jlens | positional | both | 0.58095 | 0.56131 | 0.60060 | True | 0.08224 | 0.07546 | 0.08919 | 280 |
| 16 | jlens | scope | both | 0.86384 | 0.84508 | 0.88259 | True | 0.18720 | 0.17972 | 0.19485 | 280 |
| 16 | logit | action | both | 0.98125 | 0.97232 | 0.98929 | True | 7.29287 | 7.02905 | 7.54380 | 280 |
| 16 | logit | positional | both | 0.58571 | 0.56548 | 0.60655 | True | 1.26641 | 1.06790 | 1.46779 | 280 |
| 16 | logit | scope | both | 0.72366 | 0.70134 | 0.74598 | True | 5.43562 | 5.23805 | 5.63878 | 280 |
| 20 | gram_random | action | both | 0.51122 | 0.50993 | 0.51248 | True | 0.00266 | 0.00198 | 0.00338 | 280 |
| 20 | gram_random | positional | both | 0.49729 | 0.49650 | 0.49811 | False | -0.00594 | -0.00655 | -0.00532 | 280 |
| 20 | gram_random | scope | both | 0.50405 | 0.50318 | 0.50488 | True | -0.00103 | -0.00166 | -0.00040 | 280 |
| 20 | jlens | action | both | 0.54196 | 0.52500 | 0.55982 | True | 0.16396 | 0.14279 | 0.18334 | 280 |
| 20 | jlens | positional | both | 0.28810 | 0.26964 | 0.30536 | False | -0.16090 | -0.17398 | -0.14773 | 280 |
| 20 | jlens | scope | both | 0.86429 | 0.84955 | 0.87991 | True | 0.46172 | 0.44284 | 0.48006 | 280 |
| 20 | logit | action | both | 0.64821 | 0.62232 | 0.67411 | True | 4.98289 | 4.54780 | 5.37052 | 280 |
| 20 | logit | positional | both | 0.27738 | 0.25356 | 0.29940 | False | -3.14989 | -3.41275 | -2.86108 | 280 |
| 20 | logit | scope | both | 0.77857 | 0.76429 | 0.79286 | True | 8.91350 | 8.53012 | 9.29259 | 280 |
| 24 | gram_random | action | both | 0.49268 | 0.49139 | 0.49399 | False | -0.00366 | -0.00472 | -0.00262 | 280 |
| 24 | gram_random | positional | both | 0.51426 | 0.51326 | 0.51524 | True | 0.01735 | 0.01644 | 0.01826 | 280 |
| 24 | gram_random | scope | both | 0.51518 | 0.51405 | 0.51632 | True | 0.01688 | 0.01593 | 0.01780 | 280 |
| 24 | jlens | action | both | 0.43482 | 0.40804 | 0.45982 | False | -0.12488 | -0.15477 | -0.09708 | 280 |
| 24 | jlens | positional | both | 0.43214 | 0.40952 | 0.45537 | False | -0.02673 | -0.04702 | -0.00689 | 280 |
| 24 | jlens | scope | both | 0.83973 | 0.82054 | 0.85805 | True | 0.46842 | 0.44662 | 0.48934 | 280 |
| 24 | logit | action | both | 0.45268 | 0.42768 | 0.47589 | False | -2.18845 | -2.76895 | -1.61827 | 280 |
| 24 | logit | positional | both | 0.31012 | 0.28629 | 0.33452 | False | -2.82841 | -3.19782 | -2.45106 | 280 |
| 24 | logit | scope | both | 0.72991 | 0.71070 | 0.74866 | True | 7.66675 | 7.21160 | 8.09954 | 280 |

## J-lens against its controls, paired on the same rows

`gram_random` matches the J-lens norms AND angles, so the only thing that differs is which residual-stream directions the rows point at; `logit` is `g * W_U[w]` with no Jacobian, so it answers whether the correction added anything the unembedding did not already have.

| layer | family | arm | control | reversal_jlens | reversal_control | difference | ci_lo | ci_hi | beats_control | n_bases |
|---|---|---|---|---|---|---|---|---|---|---|
| 8 | action | ab | gram_random | 0.35893 | 0.50750 | -0.14857 | -0.17441 | -0.12231 | False | 280 |
| 8 | positional | ab | gram_random | 0.77024 | 0.50800 | 0.26224 | 0.23500 | 0.28996 | True | 280 |
| 8 | scope | ab | gram_random | 0.47500 | 0.49979 | -0.02479 | -0.04368 | -0.00549 | False | 280 |
| 12 | action | ab | gram_random | 0.64286 | 0.50057 | 0.14228 | 0.09588 | 0.18737 | True | 280 |
| 12 | positional | ab | gram_random | 0.83810 | 0.48403 | 0.35406 | 0.33309 | 0.37482 | True | 280 |
| 12 | scope | ab | gram_random | 0.81607 | 0.50194 | 0.31413 | 0.29167 | 0.33707 | True | 280 |
| 16 | action | ab | gram_random | 0.73750 | 0.50260 | 0.23490 | 0.19743 | 0.26914 | True | 280 |
| 16 | positional | ab | gram_random | 0.57857 | 0.49841 | 0.08016 | 0.05530 | 0.10398 | True | 280 |
| 16 | scope | ab | gram_random | 0.86429 | 0.50546 | 0.35882 | 0.33834 | 0.37903 | True | 280 |
| 20 | action | ab | gram_random | 0.54107 | 0.51129 | 0.02978 | 0.01063 | 0.04885 | True | 280 |
| 20 | positional | ab | gram_random | 0.27976 | 0.49749 | -0.21773 | -0.24001 | -0.19636 | False | 280 |
| 20 | scope | ab | gram_random | 0.85804 | 0.50385 | 0.35419 | 0.33588 | 0.37275 | True | 280 |
| 24 | action | ab | gram_random | 0.44107 | 0.49288 | -0.05181 | -0.08097 | -0.02427 | False | 280 |
| 24 | positional | ab | gram_random | 0.42619 | 0.51375 | -0.08756 | -0.11750 | -0.05813 | False | 280 |
| 24 | scope | ab | gram_random | 0.83571 | 0.51509 | 0.32063 | 0.29834 | 0.34243 | True | 280 |
| 8 | action | ab | logit | 0.35893 | 0.73036 | -0.37143 | -0.40714 | -0.33571 | False | 280 |
| 8 | positional | ab | logit | 0.77024 | 0.69881 | 0.07143 | 0.03571 | 0.10714 | True | 280 |
| 8 | scope | ab | logit | 0.47500 | 0.40536 | 0.06964 | 0.04196 | 0.09821 | True | 280 |
| 12 | action | ab | logit | 0.64286 | 0.97321 | -0.33036 | -0.37500 | -0.28750 | False | 280 |
| 12 | positional | ab | logit | 0.83810 | 0.65833 | 0.17976 | 0.15949 | 0.20000 | True | 280 |
| 12 | scope | ab | logit | 0.81607 | 0.76071 | 0.05536 | 0.03214 | 0.07946 | True | 280 |
| 16 | action | ab | logit | 0.73750 | 0.98036 | -0.24286 | -0.27857 | -0.20893 | False | 280 |
| 16 | positional | ab | logit | 0.57857 | 0.57976 | -0.00119 | -0.02976 | 0.02619 | False | 280 |
| 16 | scope | ab | logit | 0.86429 | 0.71607 | 0.14821 | 0.12768 | 0.16964 | True | 280 |
| 20 | action | ab | logit | 0.54107 | 0.64643 | -0.10536 | -0.13036 | -0.08214 | False | 280 |
| 20 | positional | ab | logit | 0.27976 | 0.27143 | 0.00833 | -0.01429 | 0.02976 | False | 280 |
| 20 | scope | ab | logit | 0.85804 | 0.77589 | 0.08214 | 0.06696 | 0.09821 | True | 280 |
| 24 | action | ab | logit | 0.44107 | 0.45179 | -0.01071 | -0.02857 | 0.00714 | False | 280 |
| 24 | positional | ab | logit | 0.42619 | 0.31667 | 0.10952 | 0.08810 | 0.13333 | True | 280 |
| 24 | scope | ab | logit | 0.83571 | 0.72857 | 0.10714 | 0.08929 | 0.12679 | True | 280 |
| 8 | action | ba | gram_random | 0.35714 | 0.50773 | -0.15059 | -0.17675 | -0.12446 | False | 280 |
| 8 | positional | ba | gram_random | 0.76905 | 0.50742 | 0.26163 | 0.23404 | 0.28949 | True | 280 |
| 8 | scope | ba | gram_random | 0.47500 | 0.50014 | -0.02514 | -0.04523 | -0.00439 | False | 280 |
| 12 | action | ba | gram_random | 0.66071 | 0.50108 | 0.15964 | 0.11367 | 0.20486 | True | 280 |
| 12 | positional | ba | gram_random | 0.81786 | 0.48428 | 0.33358 | 0.31242 | 0.35502 | True | 280 |
| 12 | scope | ba | gram_random | 0.80804 | 0.50232 | 0.30571 | 0.28311 | 0.32880 | True | 280 |
| 16 | action | ba | gram_random | 0.75536 | 0.50191 | 0.25345 | 0.21928 | 0.28828 | True | 280 |
| 16 | positional | ba | gram_random | 0.58333 | 0.49889 | 0.08445 | 0.06038 | 0.10845 | True | 280 |
| 16 | scope | ba | gram_random | 0.86339 | 0.50539 | 0.35800 | 0.33680 | 0.37951 | True | 280 |
| 20 | action | ba | gram_random | 0.54286 | 0.51114 | 0.03171 | 0.01276 | 0.05153 | True | 280 |
| 20 | positional | ba | gram_random | 0.29643 | 0.49709 | -0.20066 | -0.22361 | -0.17943 | False | 280 |
| 20 | scope | ba | gram_random | 0.87054 | 0.50425 | 0.36629 | 0.35048 | 0.38236 | True | 280 |
| 24 | action | ba | gram_random | 0.42857 | 0.49248 | -0.06391 | -0.09579 | -0.03529 | False | 280 |
| 24 | positional | ba | gram_random | 0.43810 | 0.51477 | -0.07667 | -0.10318 | -0.05083 | False | 280 |
| 24 | scope | ba | gram_random | 0.84375 | 0.51527 | 0.32848 | 0.30861 | 0.34788 | True | 280 |
| 8 | action | ba | logit | 0.35714 | 0.73571 | -0.37857 | -0.41429 | -0.34464 | False | 280 |
| 8 | positional | ba | logit | 0.76905 | 0.69524 | 0.07381 | 0.03807 | 0.11071 | True | 280 |
| 8 | scope | ba | logit | 0.47500 | 0.40804 | 0.06696 | 0.03929 | 0.09464 | True | 280 |
| 12 | action | ba | logit | 0.66071 | 0.97857 | -0.31786 | -0.36250 | -0.27321 | False | 280 |
| 12 | positional | ba | logit | 0.81786 | 0.65714 | 0.16071 | 0.14048 | 0.18098 | True | 280 |
| 12 | scope | ba | logit | 0.80804 | 0.76339 | 0.04464 | 0.02143 | 0.06699 | True | 280 |
| 16 | action | ba | logit | 0.75536 | 0.98214 | -0.22679 | -0.26076 | -0.19464 | False | 280 |
| 16 | positional | ba | logit | 0.58333 | 0.59167 | -0.00833 | -0.03690 | 0.02024 | False | 280 |
| 16 | scope | ba | logit | 0.86339 | 0.73125 | 0.13214 | 0.11161 | 0.15357 | True | 280 |
| 20 | action | ba | logit | 0.54286 | 0.65000 | -0.10714 | -0.13214 | -0.08214 | False | 280 |
| 20 | positional | ba | logit | 0.29643 | 0.28333 | 0.01310 | -0.00833 | 0.03571 | False | 280 |
| 20 | scope | ba | logit | 0.87054 | 0.78125 | 0.08929 | 0.07500 | 0.10446 | True | 280 |
| 24 | action | ba | logit | 0.42857 | 0.45357 | -0.02500 | -0.04107 | -0.01071 | False | 280 |
| 24 | positional | ba | logit | 0.43810 | 0.30357 | 0.13452 | 0.11071 | 0.15952 | True | 280 |
| 24 | scope | ba | logit | 0.84375 | 0.73125 | 0.11250 | 0.09286 | 0.13217 | True | 280 |
| 8 | action | both | gram_random | 0.35804 | 0.50762 | -0.14958 | -0.17386 | -0.12505 | False | 280 |
| 8 | positional | both | gram_random | 0.76964 | 0.50771 | 0.26193 | 0.23581 | 0.28747 | True | 280 |
| 8 | scope | both | gram_random | 0.47500 | 0.49996 | -0.02496 | -0.04302 | -0.00622 | False | 280 |
| 12 | action | both | gram_random | 0.65179 | 0.50083 | 0.15096 | 0.10711 | 0.19391 | True | 280 |
| 12 | positional | both | gram_random | 0.82798 | 0.48415 | 0.34382 | 0.32409 | 0.36302 | True | 280 |
| 12 | scope | both | gram_random | 0.81205 | 0.50213 | 0.30992 | 0.28941 | 0.33072 | True | 280 |
| 16 | action | both | gram_random | 0.74643 | 0.50226 | 0.24417 | 0.21023 | 0.27707 | True | 280 |
| 16 | positional | both | gram_random | 0.58095 | 0.49865 | 0.08230 | 0.06291 | 0.10169 | True | 280 |
| 16 | scope | both | gram_random | 0.86384 | 0.50543 | 0.35841 | 0.33940 | 0.37722 | True | 280 |
| 20 | action | both | gram_random | 0.54196 | 0.51122 | 0.03075 | 0.01355 | 0.04903 | True | 280 |
| 20 | positional | both | gram_random | 0.28810 | 0.49729 | -0.20920 | -0.22750 | -0.19184 | False | 280 |
| 20 | scope | both | gram_random | 0.86429 | 0.50405 | 0.36024 | 0.34584 | 0.37554 | True | 280 |
| 24 | action | both | gram_random | 0.43482 | 0.49268 | -0.05786 | -0.08499 | -0.03304 | False | 280 |
| 24 | positional | both | gram_random | 0.43214 | 0.51426 | -0.08212 | -0.10527 | -0.05834 | False | 280 |
| 24 | scope | both | gram_random | 0.83973 | 0.51518 | 0.32456 | 0.30521 | 0.34325 | True | 280 |
| 8 | action | both | logit | 0.35804 | 0.73304 | -0.37500 | -0.40714 | -0.34286 | False | 280 |
| 8 | positional | both | logit | 0.76964 | 0.69702 | 0.07262 | 0.03810 | 0.10714 | True | 280 |
| 8 | scope | both | logit | 0.47500 | 0.40670 | 0.06830 | 0.04330 | 0.09331 | True | 280 |
| 12 | action | both | logit | 0.65179 | 0.97589 | -0.32411 | -0.36609 | -0.28482 | False | 280 |
| 12 | positional | both | logit | 0.82798 | 0.65774 | 0.17024 | 0.15119 | 0.18929 | True | 280 |
| 12 | scope | both | logit | 0.81205 | 0.76205 | 0.05000 | 0.03036 | 0.07009 | True | 280 |
| 16 | action | both | logit | 0.74643 | 0.98125 | -0.23482 | -0.26696 | -0.20444 | False | 280 |
| 16 | positional | both | logit | 0.58095 | 0.58571 | -0.00476 | -0.02500 | 0.01607 | False | 280 |
| 16 | scope | both | logit | 0.86384 | 0.72366 | 0.14018 | 0.12455 | 0.15671 | True | 280 |
| 20 | action | both | logit | 0.54196 | 0.64821 | -0.10625 | -0.12768 | -0.08661 | False | 280 |
| 20 | positional | both | logit | 0.28810 | 0.27738 | 0.01071 | -0.00714 | 0.02857 | False | 280 |
| 20 | scope | both | logit | 0.86429 | 0.77857 | 0.08571 | 0.07411 | 0.09777 | True | 280 |
| 24 | action | both | logit | 0.43482 | 0.45268 | -0.01786 | -0.03125 | -0.00536 | False | 280 |
| 24 | positional | both | logit | 0.43214 | 0.31012 | 0.12202 | 0.10417 | 0.14167 | True | 280 |
| 24 | scope | both | logit | 0.83973 | 0.72991 | 0.10982 | 0.09598 | 0.12500 | True | 280 |

## The three conditions, by family and layer

| family | layer | arm | reversal | reversal_ci_lo | reversal_ci_hi | beats_chance | beats_random | beats_logit | probe_succeeds |
|---|---|---|---|---|---|---|---|---|---|
| action | 8 | ab | 0.35893 | 0.33393 | 0.38571 | False | False | False | True |
| positional | 8 | ab | 0.77024 | 0.74286 | 0.79762 | True | True | True | True |
| scope | 8 | ab | 0.47500 | 0.45625 | 0.49464 | False | False | True | True |
| action | 12 | ab | 0.64286 | 0.59643 | 0.68750 | True | True | False | True |
| positional | 12 | ab | 0.83810 | 0.81667 | 0.85952 | True | True | True | True |
| scope | 12 | ab | 0.81607 | 0.79375 | 0.83929 | True | True | True | True |
| action | 16 | ab | 0.73750 | 0.70000 | 0.77143 | True | True | False | True |
| positional | 16 | ab | 0.57857 | 0.55357 | 0.60238 | True | True | False | True |
| scope | 16 | ab | 0.86429 | 0.84375 | 0.88482 | True | True | True | True |
| action | 20 | ab | 0.54107 | 0.52317 | 0.56071 | True | True | False | True |
| positional | 20 | ab | 0.27976 | 0.25714 | 0.30119 | False | False | False | True |
| scope | 20 | ab | 0.85804 | 0.83929 | 0.87679 | True | True | True | True |
| action | 24 | ab | 0.44107 | 0.41250 | 0.46786 | False | False | False | True |
| positional | 24 | ab | 0.42619 | 0.39643 | 0.45595 | False | False | True | True |
| scope | 24 | ab | 0.83571 | 0.81339 | 0.85714 | True | True | True | True |
| action | 8 | ba | 0.35714 | 0.33036 | 0.38214 | False | False | False | True |
| positional | 8 | ba | 0.76905 | 0.74167 | 0.79646 | True | True | True | True |
| scope | 8 | ba | 0.47500 | 0.45536 | 0.49554 | False | False | True | True |
| action | 12 | ba | 0.66071 | 0.61429 | 0.70536 | True | True | False | True |
| positional | 12 | ba | 0.81786 | 0.79643 | 0.83929 | True | True | True | True |
| scope | 12 | ba | 0.80804 | 0.78571 | 0.83125 | True | True | True | True |
| action | 16 | ba | 0.75536 | 0.72143 | 0.79107 | True | True | False | True |
| positional | 16 | ba | 0.58333 | 0.55952 | 0.60714 | True | True | False | True |
| scope | 16 | ba | 0.86339 | 0.84286 | 0.88482 | True | True | True | True |
| action | 20 | ba | 0.54286 | 0.52496 | 0.56250 | True | True | False | True |
| positional | 20 | ba | 0.29643 | 0.27262 | 0.31786 | False | False | False | True |
| scope | 20 | ba | 0.87054 | 0.85446 | 0.88661 | True | True | True | True |
| action | 24 | ba | 0.42857 | 0.39643 | 0.45714 | False | False | False | True |
| positional | 24 | ba | 0.43810 | 0.41190 | 0.46429 | False | False | True | True |
| scope | 24 | ba | 0.84375 | 0.82411 | 0.86339 | True | True | True | True |
| action | 8 | both | 0.35804 | 0.33393 | 0.38214 | False | False | False | True |
| positional | 8 | both | 0.76964 | 0.74403 | 0.79524 | True | True | True | True |
| scope | 8 | both | 0.47500 | 0.45670 | 0.49375 | False | False | True | True |
| action | 12 | both | 0.65179 | 0.60804 | 0.69464 | True | True | False | True |
| positional | 12 | both | 0.82798 | 0.80833 | 0.84702 | True | True | True | True |
| scope | 12 | both | 0.81205 | 0.79152 | 0.83304 | True | True | True | True |
| action | 16 | both | 0.74643 | 0.71250 | 0.77946 | True | True | False | True |
| positional | 16 | both | 0.58095 | 0.56131 | 0.60060 | True | True | False | True |
| scope | 16 | both | 0.86384 | 0.84508 | 0.88259 | True | True | True | True |
| action | 20 | both | 0.54196 | 0.52500 | 0.55982 | True | True | False | True |
| positional | 20 | both | 0.28810 | 0.26964 | 0.30536 | False | False | False | True |
| scope | 20 | both | 0.86429 | 0.84955 | 0.87991 | True | True | True | True |
| action | 24 | both | 0.43482 | 0.40804 | 0.45982 | False | False | False | True |
| positional | 24 | both | 0.43214 | 0.40952 | 0.45537 | False | False | True | True |
| scope | 24 | both | 0.83973 | 0.82054 | 0.85805 | True | True | True | True |

## The two value arms

The scored word is identical in both arms while the returned literal swaps, so a reversal caused by the binding has the same sign in `ab` and `ba` and one caused by the literal has opposite signs.

| layer | readout | family | reversal_ab | reversal_ba | beats_chance_ab | beats_chance_ba | agree | both_beat_chance |
|---|---|---|---|---|---|---|---|---|
| 8 | gram_random | action | 0.50750 | 0.50773 | True | True | True | True |
| 8 | gram_random | positional | 0.50800 | 0.50742 | True | True | True | True |
| 8 | gram_random | scope | 0.49979 | 0.50014 | False | False | False | False |
| 8 | jlens | action | 0.35893 | 0.35714 | False | False | True | False |
| 8 | jlens | positional | 0.77024 | 0.76905 | True | True | True | True |
| 8 | jlens | scope | 0.47500 | 0.47500 | False | False | True | False |
| 8 | logit | action | 0.73036 | 0.73571 | True | True | True | True |
| 8 | logit | positional | 0.69881 | 0.69524 | True | True | True | True |
| 8 | logit | scope | 0.40536 | 0.40804 | False | False | True | False |
| 12 | gram_random | action | 0.50057 | 0.50108 | False | False | True | False |
| 12 | gram_random | positional | 0.48403 | 0.48428 | False | False | True | False |
| 12 | gram_random | scope | 0.50194 | 0.50232 | True | True | True | True |
| 12 | jlens | action | 0.64286 | 0.66071 | True | True | True | True |
| 12 | jlens | positional | 0.83810 | 0.81786 | True | True | True | True |
| 12 | jlens | scope | 0.81607 | 0.80804 | True | True | True | True |
| 12 | logit | action | 0.97321 | 0.97857 | True | True | True | True |
| 12 | logit | positional | 0.65833 | 0.65714 | True | True | True | True |
| 12 | logit | scope | 0.76071 | 0.76339 | True | True | True | True |
| 16 | gram_random | action | 0.50260 | 0.50191 | True | True | True | True |
| 16 | gram_random | positional | 0.49841 | 0.49889 | False | False | True | False |
| 16 | gram_random | scope | 0.50546 | 0.50539 | True | True | True | True |
| 16 | jlens | action | 0.73750 | 0.75536 | True | True | True | True |
| 16 | jlens | positional | 0.57857 | 0.58333 | True | True | True | True |
| 16 | jlens | scope | 0.86429 | 0.86339 | True | True | True | True |
| 16 | logit | action | 0.98036 | 0.98214 | True | True | True | True |
| 16 | logit | positional | 0.57976 | 0.59167 | True | True | True | True |
| 16 | logit | scope | 0.71607 | 0.73125 | True | True | True | True |
| 20 | gram_random | action | 0.51129 | 0.51114 | True | True | True | True |
| 20 | gram_random | positional | 0.49749 | 0.49709 | False | False | True | False |
| 20 | gram_random | scope | 0.50385 | 0.50425 | True | True | True | True |
| 20 | jlens | action | 0.54107 | 0.54286 | True | True | True | True |
| 20 | jlens | positional | 0.27976 | 0.29643 | False | False | True | False |
| 20 | jlens | scope | 0.85804 | 0.87054 | True | True | True | True |
| 20 | logit | action | 0.64643 | 0.65000 | True | True | True | True |
| 20 | logit | positional | 0.27143 | 0.28333 | False | False | True | False |
| 20 | logit | scope | 0.77589 | 0.78125 | True | True | True | True |
| 24 | gram_random | action | 0.49288 | 0.49248 | False | False | True | False |
| 24 | gram_random | positional | 0.51375 | 0.51477 | True | True | True | True |
| 24 | gram_random | scope | 0.51509 | 0.51527 | True | True | True | True |
| 24 | jlens | action | 0.44107 | 0.42857 | False | False | True | False |
| 24 | jlens | positional | 0.42619 | 0.43810 | False | False | True | False |
| 24 | jlens | scope | 0.83571 | 0.84375 | True | True | True | True |
| 24 | logit | action | 0.45179 | 0.45357 | False | False | True | False |
| 24 | logit | positional | 0.31667 | 0.30357 | False | False | True | False |
| 24 | logit | scope | 0.72857 | 0.73125 | True | True | True | True |

## Pooled over every kept pair

| layer | readout | arm | reversal | reversal_ci_lo | reversal_ci_hi | beats_chance | mean_delta | n_bases |
|---|---|---|---|---|---|---|---|---|
| 8 | gram_random | ab | 0.50424 | 0.50363 | 0.50486 | True | 0.00049 | 280 |
| 8 | jlens | ab | 0.54762 | 0.53333 | 0.56111 | True | 0.00193 | 280 |
| 8 | logit | ab | 0.57540 | 0.55674 | 0.59286 | True | 0.71196 | 280 |
| 12 | gram_random | ab | 0.49567 | 0.49512 | 0.49624 | False | -0.00088 | 280 |
| 12 | jlens | ab | 0.78492 | 0.76706 | 0.80278 | True | 0.06337 | 280 |
| 12 | logit | ab | 0.77381 | 0.76230 | 0.78571 | True | 3.36392 | 280 |
| 16 | gram_random | ab | 0.50248 | 0.50193 | 0.50302 | True | 0.00036 | 280 |
| 16 | jlens | ab | 0.74087 | 0.72619 | 0.75476 | True | 0.13491 | 280 |
| 16 | logit | ab | 0.72937 | 0.71429 | 0.74484 | True | 4.43686 | 280 |
| 20 | gram_random | ab | 0.50338 | 0.50278 | 0.50399 | True | -0.00188 | 280 |
| 20 | jlens | ab | 0.59484 | 0.58294 | 0.60714 | True | 0.18694 | 280 |
| 20 | logit | ab | 0.57897 | 0.56548 | 0.59286 | True | 4.01063 | 280 |
| 24 | gram_random | ab | 0.50971 | 0.50893 | 0.51047 | True | 0.01240 | 280 |
| 24 | jlens | ab | 0.61151 | 0.59484 | 0.62779 | True | 0.17019 | 280 |
| 24 | logit | ab | 0.52976 | 0.51270 | 0.54643 | True | 1.96194 | 280 |
| 8 | gram_random | ba | 0.50425 | 0.50364 | 0.50486 | True | 0.00050 | 280 |
| 8 | jlens | ba | 0.54683 | 0.53294 | 0.56072 | True | 0.00193 | 280 |
| 8 | logit | ba | 0.57659 | 0.55873 | 0.59484 | True | 0.70984 | 280 |
| 12 | gram_random | ba | 0.49603 | 0.49553 | 0.49653 | False | -0.00086 | 280 |
| 12 | jlens | ba | 0.77857 | 0.76111 | 0.79643 | True | 0.06376 | 280 |
| 12 | logit | ba | 0.77579 | 0.76468 | 0.78770 | True | 3.36984 | 280 |
| 16 | gram_random | ba | 0.50245 | 0.50190 | 0.50301 | True | 0.00040 | 280 |
| 16 | jlens | ba | 0.74603 | 0.73135 | 0.76111 | True | 0.13677 | 280 |
| 16 | logit | ba | 0.74048 | 0.72578 | 0.75635 | True | 4.48036 | 280 |
| 20 | gram_random | ba | 0.50339 | 0.50271 | 0.50406 | True | -0.00181 | 280 |
| 20 | jlens | ba | 0.60635 | 0.59524 | 0.61786 | True | 0.18908 | 280 |
| 20 | logit | ba | 0.58611 | 0.57261 | 0.59961 | True | 4.02717 | 280 |
| 24 | gram_random | ba | 0.51004 | 0.50932 | 0.51077 | True | 0.01254 | 280 |
| 24 | jlens | ba | 0.61627 | 0.60039 | 0.63214 | True | 0.17286 | 280 |
| 24 | logit | ba | 0.52698 | 0.51032 | 0.54365 | True | 1.99470 | 280 |
| 8 | gram_random | both | 0.50425 | 0.50368 | 0.50480 | True | 0.00049 | 280 |
| 8 | jlens | both | 0.54722 | 0.53413 | 0.56012 | True | 0.00193 | 280 |
| 8 | logit | both | 0.57599 | 0.55873 | 0.59226 | True | 0.71090 | 280 |
| 12 | gram_random | both | 0.49585 | 0.49538 | 0.49630 | False | -0.00087 | 280 |
| 12 | jlens | both | 0.78175 | 0.76548 | 0.79861 | True | 0.06356 | 280 |
| 12 | logit | both | 0.77480 | 0.76448 | 0.78571 | True | 3.36688 | 280 |
| 16 | gram_random | both | 0.50246 | 0.50198 | 0.50295 | True | 0.00038 | 280 |
| 16 | jlens | both | 0.74345 | 0.73115 | 0.75635 | True | 0.13584 | 280 |
| 16 | logit | both | 0.73492 | 0.72202 | 0.74901 | True | 4.45861 | 280 |
| 20 | gram_random | both | 0.50339 | 0.50283 | 0.50394 | True | -0.00185 | 280 |
| 20 | jlens | both | 0.60060 | 0.59087 | 0.61052 | True | 0.18801 | 280 |
| 20 | logit | both | 0.58254 | 0.57123 | 0.59405 | True | 4.01890 | 280 |
| 24 | gram_random | both | 0.50987 | 0.50917 | 0.51055 | True | 0.01247 | 280 |
| 24 | jlens | both | 0.61389 | 0.59940 | 0.62877 | True | 0.17153 | 280 |
| 24 | logit | both | 0.52837 | 0.51448 | 0.54266 | True | 1.97832 | 280 |

## The instrument

The J-lens is the repository's corpus-built lens: same estimator, same third-party Python corpus, same build/held-out split, same stability probe and the same V1/V2 validations as E11's. The only thing E18 changes is which candidate rows are built, because a J-lens row is a per-token object and the frozen value lens has no row for `local`. No binding program is seen during the build and the 9 pairs were declared in `src/experiments/binding_lexlens.py` before any state was read.

### Stability across independent builds

Reported, never used to select a layer: a layer whose independently built lenses disagree on the DECISIONS they produce cannot carry a claim about that layer however large its reversal looks.

| layer | n_seeds | cosine_mean | cosine_min | margin_sign_agreement | pooled_vs_seed_cosine | n_build_per_seed | n_probe_states |
|---|---|---|---|---|---|---|---|
| 8 | 5 | 0.26234 | 0.18626 | 0.60868 | 0.63382 | 200 | 13 |
| 12 | 5 | 0.58222 | 0.51600 | 0.59677 | 0.81517 | 200 | 13 |
| 16 | 5 | 0.79159 | 0.74076 | 0.79851 | 0.91226 | 200 | 13 |
| 20 | 5 | 0.88921 | 0.85575 | 0.88040 | 0.95427 | 200 | 13 |
| 24 | 5 | 0.94227 | 0.92003 | 0.92134 | 0.97641 | 200 | 13 |
| 31 | 5 | 1.00000 | 1.00000 | 1.00000 | 1.00000 | 200 | 13 |

### V1 / V2

V1: at the last decoder layer `J` is the identity, so the J-lens must reproduce the logit lens exactly. V2: next-token recovery on held-out corpus positions whose true next token is one of these very words.

| check | layer | lens | top1 | mrr | n | cosine_to_logit_lens | is_last_layer |
|---|---|---|---|---|---|---|---|
| V2_next_token | 8 | jlens | 0.38462 | 0.56184 | 13.00000 |  |  |
| V2_next_token | 8 | logit | 0.30769 | 0.52393 | 13.00000 |  |  |
| V2_next_token | 8 | gram_random | 0.00000 | 0.16564 | 13.00000 |  |  |
| V1_identity_at_last_layer | 8 | jlens |  |  |  | 0.33723 | False |
| V2_next_token | 12 | jlens | 0.30769 | 0.42909 | 13.00000 |  |  |
| V2_next_token | 12 | logit | 0.23077 | 0.40058 | 13.00000 |  |  |
| V2_next_token | 12 | gram_random | 0.07692 | 0.22404 | 13.00000 |  |  |
| V1_identity_at_last_layer | 12 | jlens |  |  |  | 0.54428 | False |
| V2_next_token | 16 | jlens | 0.23077 | 0.47179 | 13.00000 |  |  |
| V2_next_token | 16 | logit | 0.38462 | 0.60330 | 13.00000 |  |  |
| V2_next_token | 16 | gram_random | 0.07692 | 0.16180 | 13.00000 |  |  |
| V1_identity_at_last_layer | 16 | jlens |  |  |  | 0.72052 | False |
| V2_next_token | 20 | jlens | 0.53846 | 0.68077 | 13.00000 |  |  |
| V2_next_token | 20 | logit | 0.69231 | 0.80128 | 13.00000 |  |  |
| V2_next_token | 20 | gram_random | 0.00000 | 0.13286 | 13.00000 |  |  |
| V1_identity_at_last_layer | 20 | jlens |  |  |  | 0.80299 | False |
| V2_next_token | 24 | jlens | 0.76923 | 0.83462 | 13.00000 |  |  |
| V2_next_token | 24 | logit | 0.76923 | 0.87179 | 13.00000 |  |  |
| V2_next_token | 24 | gram_random | 0.00000 | 0.17012 | 13.00000 |  |  |
| V1_identity_at_last_layer | 24 | jlens |  |  |  | 0.83743 | False |
| V2_next_token | 31 | jlens | 1.00000 | 1.00000 | 13.00000 |  |  |
| V2_next_token | 31 | logit | 1.00000 | 1.00000 | 13.00000 |  |  |
| V2_next_token | 31 | gram_random | 0.07692 | 0.28974 | 13.00000 |  |  |
| V1_identity_at_last_layer | 31 | jlens |  |  |  | 1.00000 | True |

### The Gram-matched control, per seed

The CSV retains every split and direction separately; only a short preview is shown here.

| split | layer | arm | seed | family | inner_word | outer_word | reversal | mean_delta | n_bases |
|---|---|---|---|---|---|---|---|---|---|
| calib | 8 | ab | 42 | scope | local | global | 1.00000 | 0.08235 | 120 |
| test | 8 | ab | 42 | scope | local | global | 0.98214 | 0.07940 | 280 |
| calib | 8 | ab | 42 | scope | inner | outer | 0.79167 | 0.01350 | 120 |
| test | 8 | ab | 42 | scope | inner | outer | 0.77857 | 0.01211 | 280 |
| calib | 8 | ab | 42 | scope | inside | outside | 0.54167 | -0.00094 | 120 |
| test | 8 | ab | 42 | scope | inside | outside | 0.50714 | -0.00290 | 280 |
| calib | 8 | ab | 42 | scope | nested | module | 0.63333 | 0.01518 | 120 |
| test | 8 | ab | 42 | scope | nested | module | 0.62857 | 0.01298 | 280 |
| calib | 8 | ab | 42 | positional | later | earlier | 0.05000 | -0.04191 | 120 |
| test | 8 | ab | 42 | positional | later | earlier | 0.06071 | -0.03667 | 280 |
| calib | 8 | ab | 42 | positional | second | first | 0.93333 | 0.03140 | 120 |
| test | 8 | ab | 42 | positional | second | first | 0.92500 | 0.03378 | 280 |
| calib | 8 | ab | 42 | positional | new | original | 0.65000 | 0.01233 | 120 |
| test | 8 | ab | 42 | positional | new | original | 0.70000 | 0.01454 | 280 |
| calib | 8 | ab | 42 | action | replaced | kept | 0.10000 | -0.03754 | 120 |
| test | 8 | ab | 42 | action | replaced | kept | 0.10000 | -0.03564 | 280 |
| calib | 8 | ab | 42 | action | changed | unchanged | 0.99167 | 0.06300 | 120 |
| test | 8 | ab | 42 | action | changed | unchanged | 0.98929 | 0.06512 | 280 |
| calib | 8 | ab | 43 | scope | local | global | 1.00000 | 0.09760 | 120 |
| test | 8 | ab | 43 | scope | local | global | 1.00000 | 0.09598 | 280 |
| calib | 8 | ab | 43 | scope | inner | outer | 0.35000 | -0.00853 | 120 |
| test | 8 | ab | 43 | scope | inner | outer | 0.38929 | -0.00726 | 280 |
| calib | 8 | ab | 43 | scope | inside | outside | 0.05000 | -0.03739 | 120 |
| test | 8 | ab | 43 | scope | inside | outside | 0.03214 | -0.03467 | 280 |
| calib | 8 | ab | 43 | scope | nested | module | 0.78333 | 0.02613 | 120 |
| test | 8 | ab | 43 | scope | nested | module | 0.71786 | 0.02503 | 280 |
| calib | 8 | ab | 43 | positional | later | earlier | 0.00000 | -0.08498 | 120 |
| test | 8 | ab | 43 | positional | later | earlier | 0.00000 | -0.08780 | 280 |
| calib | 8 | ab | 43 | positional | second | first | 0.03333 | -0.05498 | 120 |
| test | 8 | ab | 43 | positional | second | first | 0.04286 | -0.05176 | 280 |
| calib | 8 | ab | 43 | positional | new | original | 0.95833 | 0.03980 | 120 |
| test | 8 | ab | 43 | positional | new | original | 0.92857 | 0.03775 | 280 |
| calib | 8 | ab | 43 | action | replaced | kept | 0.00000 | -0.05740 | 120 |
| test | 8 | ab | 43 | action | replaced | kept | 0.01429 | -0.05969 | 280 |
| calib | 8 | ab | 43 | action | changed | unchanged | 0.24167 | -0.02020 | 120 |
| test | 8 | ab | 43 | action | changed | unchanged | 0.18571 | -0.02535 | 280 |
| calib | 8 | ab | 44 | scope | local | global | 0.49167 | 0.00184 | 120 |
| test | 8 | ab | 44 | scope | local | global | 0.45000 | -0.00137 | 280 |
| calib | 8 | ab | 44 | scope | inner | outer | 0.18333 | -0.01602 | 120 |
| test | 8 | ab | 44 | scope | inner | outer | 0.22857 | -0.01441 | 280 |

## The lexicon

Predeclared, 9 matched opposing pairs over `scope` and the control families `positional`, `action`. Both control families predict the SAME sign as `scope` — under the inner binding the winning definition is the local one, the later one, and the one that replaced the other — so they are controls in what a positive result would MEAN, not in which direction it would point. A pair whose either side is not one stable token is dropped WHOLE.

| family | inner_word | outer_word | inner_id | outer_id | inner_variant | outer_variant | kept | reason |
|---|---|---|---|---|---|---|---|---|
| scope | local | global | 2291 | 5160 |  local |  global | 1 |  |
| scope | inner | outer | 9526 | 12915 |  inner |  outer | 1 |  |
| scope | inside | outside | 4640 | 4871 |  inside |  outside | 1 |  |
| scope | nested | module | 28919 | 6230 |  nested |  module | 1 |  |
| positional | later | earlier | 3455 | 7239 |  later |  earlier | 1 |  |
| positional | second | first | 1856 | 1019 |  second |  first | 1 |  |
| positional | new | original | 756 | 3620 |  new |  original | 1 |  |
| action | replaced | kept | 10900 | 5976 |  replaced |  kept | 1 |  |
| action | changed | unchanged | 5452 | 31940 |  changed |  unchanged | 1 |  |

## The exactness conditions of the read

| check | cells | holds | bases_failing |
|---|---|---|---|
| scored_text_is_program | 1600 | 1600 | 0 |
| bare_prefix_matches_prompt | 1600 | 1600 | 0 |
| use_is_last_bare | 1600 | 1600 | 0 |
| use_token_identical | 1600 | 1600 | 0 |
| one_token_mutation | 1600 | 1600 | 0 |
| mutation_distance_ok | 1600 | 1600 | 0 |

Distances, over 1600 cells: the mutation sits 6-6 tokens before the use; the bare program is [16] tokens and the use anchor is its last, against E13's answer prompt at [21].

## Gates

| gate | passed | recorded | value | owner_stage | detail |
|---|---|---|---|---|---|
| H0 | True | True | 1.00000 | 101_binding_verify | 1.0000 of 400 bases agree with an independent interpreter and satisfy every invariant, including the arm crossing that makes the held-out test a falsification (threshold 0.999). Per check: semantics_agree 1.0000, arms_crossed 1.0000, mutation_distance_ok 1.0000, anchors_ordered 1.0000, values_distinct 1.0000, tokens_distinct 1.0000 |
| H1 | True | True | 1.00000 | 102_binding_behaviour | overall 1.000 [1.000, 1.000] against 0.85; weakest cell ab_source 1.000 against 0.75 |
| H2 | True | True | 1.00000 | 104_binding_decode | best layer 8: binding decodable at 1.000 (selectivity 0.524) against a MEASURED surface baseline of 0.500; margin +0.500. Thresholds 0.8 and 0.1. The floor is pinned by construction here: the anchor token is identical across the counterfactual and the mutation is outside the baseline's window. |
| H3 | True | True | 4.79016 | 105_binding_ceiling | site use, layer 8 (both chosen on calibration): ab: +4.781 [+4.683, +4.878], flip rate 0.857; ba: +4.799 [+4.694, +4.903], flip rate 0.879 (thresholds: CI above 0.0, flip rate 0.25). Both arms must be measurable or an H5 null says nothing. |
| H4 | True | True | 1.88846 | 106_binding_interchange | ab @ use L8 r1: +9.029 [+8.952, +9.108] = 189% of the whole-state ceiling +4.781 (threshold 50%); controls cleared: True; edit moved 0.479 of ||h|| |
| H5 | True | True | 1.13821 | 106_binding_interchange | ba @ use L8 r1: das_binding installed 100.0% = 114% of the held-out ceiling (threshold 50%); margin +9.009 [+8.933, +9.089] = 188% of it; discriminator — answer_direction ab +2.322 [+2.157, +2.482], installed 27.9% (passes: True); ba/ab argmax ratio 0.154 against transport's 1.025 (bar 0.513) (fails: True) |
| H6 | True | True | 51200.00000 | 140_binding_relevance | 25600 readings and 51200 paired contrasts over 4 contrasts x 8 layers x 4 target conditions; median |rho-1| 8.61e-08; conserving layers [0, 3, 7, 11, 15, 19, 23, 27]; LRP rules bound {'ln': 65, 'mlp': 32, 'attn': 32} |
| H7 | True | True | 221.00000 | 150_binding_verbal_discover | 10 lexicon pairs from 4 families, 15 mechanism words, 221 candidates (160 discovered, 32 random) over 120 calib bases |
| H8 | True | True | 14400.00000 | 151_binding_verbal_behaviour | 14400 scored choices over 9 questions x 400 bases x 4 cells; report split test |
| H9 | True | True | 64000.00000 | 152_binding_verbal_relevance | 38400 readings and 64000 paired contrasts over 4 contrasts x 8 layers x 5 target conditions for scope/direct; median |rho-1| 1.27e-07; readable layers [0, 3, 7, 11, 15, 19, 23, 27]; LRP rules bound {'ln': 65, 'mlp': 32, 'attn': 32} |
| H10 | True | True | 108000.00000 | 160_binding_lexlens | 108000 reversal rows over 400 bases x 2 arms x 5 layers x 3 readouts x 9 pairs; probe succeeds at layers [8, 12, 16, 20, 24]; report split test |

## Do not claim

- that a null here shows the model cannot verbalise binding — it shows this lexicon, at this position, under these three readouts, does not separate; E17 asks the prompted-behaviour version of the question and answers it differently
- that a reversal here shows the model USES the word — a lens reading is a readout of a state, it intervenes on nothing, and E13/R10's DAS interchange is the causal result
- that the probe's accuracy is a J-lens result; it is the positive control, fitted in its own basis for its own label, and it is never expressed in word coordinates
- that a layer profile locates where binding is COMPUTED; it is where a fixed vocabulary contrast is readable
- anything about real code, other templates, other languages, or model families outside the ones the lens was built and validated on
