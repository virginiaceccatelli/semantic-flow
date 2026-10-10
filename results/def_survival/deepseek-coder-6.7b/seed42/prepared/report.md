# Definition survival: prepared data

| domain    | split      |   programs |   groups |   candidates |   survive |   heuristic_accuracy |   killed |
|:----------|:-----------|-----------:|---------:|-------------:|----------:|---------------------:|---------:|
| real      | test       |        623 |      346 |         2139 |      1263 |                0.835 |      876 |
| real      | train      |       1764 |     1043 |         6068 |      3672 |                0.829 |     2396 |
| real      | validation |        613 |      335 |         2045 |      1202 |                0.840 |      843 |
| synthetic | test       |       1168 |      584 |         1168 |       584 |                0.634 |      584 |
| synthetic | train      |       3592 |     1796 |         3592 |      1796 |                0.627 |     1796 |
| synthetic | validation |       1240 |      620 |         1240 |       620 |                0.611 |      620 |

`heuristic_accuracy`: killed iff some redefinition is indented no deeper than the use.

| domain   | level     | reason                                              |   count |
|:---------|:----------|:----------------------------------------------------|--------:|
| real     | program   | AssertionError: Cannot locate definition identifier |      60 |
| real     | program   | SyntaxError                                         |     182 |
| real     | program   | ValueError: global_or_nonlocal                      |     100 |
| real     | program   | ValueError: over_length                             |     801 |
| real     | program   | ValueError: tokenizer_round_trip                    |      18 |
| real     | program   | no_supported_candidate                              |   16514 |
| real     | candidate | binding_not_complete_before_next_event              |     344 |
| real     | candidate | constant_condition_in_region                        |     223 |
| real     | candidate | jump_or_nested_scope_in_region                      |    5108 |
| real     | candidate | loop_else_in_region                                 |      75 |
| real     | candidate | same_line_redefinition                              |   12190 |
| real     | candidate | self_referential_killer                             |   16501 |
| real     | candidate | try_or_match_overlaps_region                        |    2886 |
