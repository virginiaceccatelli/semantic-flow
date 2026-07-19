# Obfuscation robustness — deepseek-coder-1.3b

| task        |   obf_level | obf_name   |   accuracy |      n |
|:------------|------------:|:-----------|-----------:|-------:|
| binding     |           0 | normalize  |      0.967 |  22216 |
| binding     |           1 | rename     |      0.725 |  22144 |
| binding     |           2 | opaque     |      0.716 |  79392 |
| binding     |           3 | encode     |      0.725 | 139112 |
| binding     |           4 | flatten    |      0.593 | 203208 |
| defuse_edge |           0 | normalize  |      0.972 |  14024 |
| defuse_edge |           1 | rename     |      0.731 |  13776 |
| defuse_edge |           2 | opaque     |      0.720 |  30920 |
| defuse_edge |           3 | encode     |      0.728 |  40576 |
| defuse_edge |           4 | flatten    |      0.615 |  99680 |
