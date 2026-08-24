# Obfuscation robustness — deepseek-coder-1.3b

## What this experiment asks

This table tests a frozen probe on meaning-preserving program rewrites. Because the probe is not retrained, a lower score means the original readout no longer transfers cleanly; it does not prove that all semantic information has vanished from the model.

| task        |   obf_level | obf_name   |   accuracy |      n |
|:------------|------------:|:-----------|-----------:|-------:|
| binding     |           0 | normalize  |      0.966 |  22216 |
| binding     |           1 | rename     |      0.729 |  22144 |
| binding     |           2 | opaque     |      0.707 |  79392 |
| binding     |           3 | encode     |      0.718 | 139112 |
| binding     |           4 | flatten    |      0.582 | 203208 |
| defuse_edge |           0 | normalize  |      0.971 |  14024 |
| defuse_edge |           1 | rename     |      0.723 |  13776 |
| defuse_edge |           2 | opaque     |      0.708 |  30920 |
| defuse_edge |           3 | encode     |      0.716 |  40576 |
| defuse_edge |           4 | flatten    |      0.600 |  99680 |
