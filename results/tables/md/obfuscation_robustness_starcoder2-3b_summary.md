# Obfuscation robustness — starcoder2-3b

| task        |   obf_level | obf_name   |   accuracy |      n |
|:------------|------------:|:-----------|-----------:|-------:|
| binding     |           0 | normalize  |      0.972 |  27770 |
| binding     |           1 | rename     |      0.703 |  27770 |
| binding     |           2 | opaque     |      0.673 |  99240 |
| binding     |           3 | encode     |      0.690 | 173890 |
| binding     |           4 | flatten    |      0.503 | 254010 |
| defuse_edge |           0 | normalize  |      0.974 |  17930 |
| defuse_edge |           1 | rename     |      0.698 |  17760 |
| defuse_edge |           2 | opaque     |      0.668 |  38650 |
| defuse_edge |           3 | encode     |      0.680 |  50730 |
| defuse_edge |           4 | flatten    |      0.439 | 124600 |
