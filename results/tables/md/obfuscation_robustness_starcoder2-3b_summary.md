# Obfuscation robustness — starcoder2-3b

## What this experiment asks

This table tests a frozen probe on meaning-preserving program rewrites. Because the probe is not retrained, a lower score means the original readout no longer transfers cleanly; it does not prove that all semantic information has vanished from the model.

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
