# Obfuscation robustness — deepseek-coder-1.3b

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
