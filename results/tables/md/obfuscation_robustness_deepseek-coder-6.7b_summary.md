# Obfuscation robustness — deepseek-coder-6.7b

| task        |   obf_level | obf_name   |   accuracy |      n |
|:------------|------------:|:-----------|-----------:|-------:|
| binding     |           0 | normalize  |      0.974 |  27770 |
| binding     |           1 | rename     |      0.726 |  27680 |
| binding     |           2 | opaque     |      0.722 |  99240 |
| binding     |           3 | encode     |      0.738 | 173890 |
| binding     |           4 | flatten    |      0.590 | 254010 |
| defuse_edge |           0 | normalize  |      0.976 |  17530 |
| defuse_edge |           1 | rename     |      0.701 |  17220 |
| defuse_edge |           2 | opaque     |      0.703 |  38650 |
| defuse_edge |           3 | encode     |      0.717 |  50720 |
| defuse_edge |           4 | flatten    |      0.598 | 124600 |
