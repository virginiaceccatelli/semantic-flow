# Context degradation — starcoder2-3b

## What this experiment asks

This table asks what happens when more context is inserted between semantically related program elements. Compare each condition with the clean condition: inert filler tests distance alone, while name-reusing filler tests semantic interference. The comparison separates a long-context problem from confusion caused by competing references.

| task        | filler_type      |   filler_target |   accuracy |       n |
|:------------|:-----------------|----------------:|-----------:|--------:|
| binding     | comment_prose    |               0 |      0.972 |   19670 |
| binding     | comment_prose    |              50 |      0.920 |   19630 |
| binding     | comment_prose    |             100 |      0.913 |   19660 |
| binding     | comment_prose    |             200 |      0.905 |   19660 |
| binding     | comment_prose    |             500 |      0.895 |   19650 |
| binding     | comment_prose    |            1000 |      0.885 |   19660 |
| binding     | competing_update |               0 |      0.972 |   19710 |
| binding     | competing_update |              50 |      0.879 |   19310 |
| binding     | competing_update |             100 |      0.876 |   19710 |
| binding     | competing_update |             200 |      0.865 |   19710 |
| binding     | competing_update |             500 |      0.856 |   19710 |
| binding     | competing_update |            1000 |      0.844 |   19710 |
| binding     | dead_code        |               0 |      0.972 |   19710 |
| binding     | dead_code        |              50 |      0.904 |   24010 |
| binding     | dead_code        |             100 |      0.902 |   28010 |
| binding     | dead_code        |             200 |      0.856 |   36010 |
| binding     | dead_code        |             500 |      0.787 |   58010 |
| binding     | dead_code        |            1000 |      0.756 |   92010 |
| binding     | lexical_decoy    |               0 |      0.971 |   19710 |
| binding     | lexical_decoy    |              50 |      0.890 |   26010 |
| binding     | lexical_decoy    |             100 |      0.882 |   30010 |
| binding     | lexical_decoy    |             200 |      0.848 |   38010 |
| binding     | lexical_decoy    |             500 |      0.815 |   60010 |
| binding     | lexical_decoy    |            1000 |      0.789 |   96010 |
| binding     | scope_shadow     |               0 |      0.972 |   19710 |
| binding     | scope_shadow     |              50 |      0.708 |   52330 |
| binding     | scope_shadow     |             100 |      0.619 |   93370 |
| binding     | scope_shadow     |             200 |      0.513 |  218650 |
| binding     | scope_shadow     |             500 |      0.386 |  940090 |
| binding     | scope_shadow     |            1000 |      0.311 | 3192590 |
| defuse_edge | comment_prose    |               0 |      0.979 |   14630 |
| defuse_edge | comment_prose    |              50 |      0.908 |   14480 |
| defuse_edge | comment_prose    |             100 |      0.912 |   14480 |
| defuse_edge | comment_prose    |             200 |      0.909 |   14480 |
| defuse_edge | comment_prose    |             500 |      0.901 |   14480 |
| defuse_edge | comment_prose    |            1000 |      0.888 |   14450 |
| defuse_edge | competing_update |               0 |      0.979 |   14630 |
| defuse_edge | competing_update |              50 |      0.902 |   13780 |
| defuse_edge | competing_update |             100 |      0.911 |   14630 |
| defuse_edge | competing_update |             200 |      0.903 |   14630 |
| defuse_edge | competing_update |             500 |      0.897 |   14630 |
| defuse_edge | competing_update |            1000 |      0.874 |   13510 |
| defuse_edge | dead_code        |               0 |      0.978 |   14630 |
| defuse_edge | dead_code        |              50 |      0.910 |   19210 |
| defuse_edge | dead_code        |             100 |      0.894 |   23030 |
| defuse_edge | dead_code        |             200 |      0.864 |   30630 |
| defuse_edge | dead_code        |             500 |      0.834 |   53210 |
| defuse_edge | dead_code        |            1000 |      0.818 |   87210 |
| defuse_edge | lexical_decoy    |               0 |      0.979 |   14630 |
| defuse_edge | lexical_decoy    |              50 |      0.900 |   20580 |
| defuse_edge | lexical_decoy    |             100 |      0.892 |   24170 |
| defuse_edge | lexical_decoy    |             200 |      0.863 |   30780 |
| defuse_edge | lexical_decoy    |             500 |      0.845 |   54530 |
| defuse_edge | lexical_decoy    |            1000 |      0.826 |   88870 |
| defuse_edge | scope_shadow     |               0 |      0.979 |   14620 |
| defuse_edge | scope_shadow     |              50 |      0.728 |   34860 |
| defuse_edge | scope_shadow     |             100 |      0.667 |   57500 |
| defuse_edge | scope_shadow     |             200 |      0.565 |  125690 |
| defuse_edge | scope_shadow     |             500 |      0.394 |  504630 |
| defuse_edge | scope_shadow     |            1000 |      0.292 | 1655890 |
