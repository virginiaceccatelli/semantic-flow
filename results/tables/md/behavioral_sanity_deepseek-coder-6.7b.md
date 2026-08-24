# E6 behavioural-signal sanity — deepseek-coder-6.7b

## What this experiment asks

This table measures the model’s own generated answer, rather than what a separate probe can decode. It is used as a behavioral check and, where applicable, to compare when hidden-state evidence appears with when the output becomes correct.

|   n_prefixes |   says_tainted_rate |   base_rate_tainted |   accuracy |   balanced_accuracy |   acc_truth_1 |   acc_truth_0 | constant_responder   | usable   |
|-------------:|--------------------:|--------------------:|-----------:|--------------------:|--------------:|--------------:|:---------------------|:---------|
|          342 |               0.874 |               0.813 |      0.939 |               0.836 |         1.000 |         0.672 | False                | True     |
