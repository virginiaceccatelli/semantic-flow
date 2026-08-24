# Context degradation — deepseek-coder-6.7b

## What this experiment asks

This table asks what happens when more context is inserted between semantically related program elements. Compare each condition with the clean condition: inert filler tests distance alone, while name-reusing filler tests semantic interference. The comparison separates a long-context problem from confusion caused by competing references.

| task        | filler_type      |   filler_target |   accuracy |       n |
|:------------|:-----------------|----------------:|-----------:|--------:|
| binding     | comment_prose    |               0 |      0.974 |   19710 |
| binding     | comment_prose    |              50 |      0.932 |   19700 |
| binding     | comment_prose    |             100 |      0.922 |   19710 |
| binding     | comment_prose    |             200 |      0.918 |   19710 |
| binding     | comment_prose    |             500 |      0.914 |   19690 |
| binding     | comment_prose    |            1000 |      0.902 |   19700 |
| binding     | competing_update |               0 |      0.975 |   19710 |
| binding     | competing_update |              50 |      0.881 |   19710 |
| binding     | competing_update |             100 |      0.854 |   19710 |
| binding     | competing_update |             200 |      0.866 |   19700 |
| binding     | competing_update |             500 |      0.862 |   19700 |
| binding     | competing_update |            1000 |      0.844 |   19710 |
| binding     | dead_code        |               0 |      0.974 |   19710 |
| binding     | dead_code        |              50 |      0.928 |   24010 |
| binding     | dead_code        |             100 |      0.908 |   28010 |
| binding     | dead_code        |             200 |      0.847 |   36010 |
| binding     | dead_code        |             500 |      0.786 |   58010 |
| binding     | dead_code        |            1000 |      0.759 |   90010 |
| binding     | lexical_decoy    |               0 |      0.974 |   19710 |
| binding     | lexical_decoy    |              50 |      0.884 |   26010 |
| binding     | lexical_decoy    |             100 |      0.877 |   30010 |
| binding     | lexical_decoy    |             200 |      0.846 |   37980 |
| binding     | lexical_decoy    |             500 |      0.797 |   59290 |
| binding     | lexical_decoy    |            1000 |      0.769 |   91310 |
| binding     | scope_shadow     |               0 |      0.974 |   19710 |
| binding     | scope_shadow     |              50 |      0.816 |   41850 |
| binding     | scope_shadow     |             100 |      0.745 |   78090 |
| binding     | scope_shadow     |             200 |      0.666 |  193770 |
| binding     | scope_shadow     |             500 |      0.576 |  734970 |
| binding     | scope_shadow     |            1000 |      0.519 | 2433460 |
| defuse_edge | comment_prose    |               0 |      0.980 |   14100 |
| defuse_edge | comment_prose    |              50 |      0.924 |   13700 |
| defuse_edge | comment_prose    |             100 |      0.908 |   13710 |
| defuse_edge | comment_prose    |             200 |      0.897 |   13720 |
| defuse_edge | comment_prose    |             500 |      0.892 |   13730 |
| defuse_edge | comment_prose    |            1000 |      0.881 |   13700 |
| defuse_edge | competing_update |               0 |      0.980 |   14130 |
| defuse_edge | competing_update |              50 |      0.895 |   13760 |
| defuse_edge | competing_update |             100 |      0.846 |   13760 |
| defuse_edge | competing_update |             200 |      0.847 |   13770 |
| defuse_edge | competing_update |             500 |      0.868 |   13730 |
| defuse_edge | competing_update |            1000 |      0.875 |   14630 |
| defuse_edge | dead_code        |               0 |      0.981 |   14100 |
| defuse_edge | dead_code        |              50 |      0.918 |   19210 |
| defuse_edge | dead_code        |             100 |      0.876 |   23210 |
| defuse_edge | dead_code        |             200 |      0.807 |   31210 |
| defuse_edge | dead_code        |             500 |      0.748 |   53210 |
| defuse_edge | dead_code        |            1000 |      0.712 |   85210 |
| defuse_edge | lexical_decoy    |               0 |      0.981 |   14140 |
| defuse_edge | lexical_decoy    |              50 |      0.887 |   21100 |
| defuse_edge | lexical_decoy    |             100 |      0.880 |   24900 |
| defuse_edge | lexical_decoy    |             200 |      0.843 |   32070 |
| defuse_edge | lexical_decoy    |             500 |      0.810 |   53170 |
| defuse_edge | lexical_decoy    |            1000 |      0.769 |   83660 |
| defuse_edge | scope_shadow     |               0 |      0.981 |   14130 |
| defuse_edge | scope_shadow     |              50 |      0.761 |   28730 |
| defuse_edge | scope_shadow     |             100 |      0.722 |   49850 |
| defuse_edge | scope_shadow     |             200 |      0.689 |  113690 |
| defuse_edge | scope_shadow     |             500 |      0.643 |  395520 |
| defuse_edge | scope_shadow     |            1000 |      0.596 | 1258100 |
