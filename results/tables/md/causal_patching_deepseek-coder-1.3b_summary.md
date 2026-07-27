# Causal patching — deepseek-coder-1.3b

|   layer | position      |   mean_recovery | causal_class                                                         |
|--------:|:--------------|----------------:|:---------------------------------------------------------------------|
|      -1 | last_token    |           0.000 | {'not_encoded': 40}                                                  |
|      -1 | sanitizer_def |           0.000 | {'not_encoded': 40}                                                  |
|      -1 | sink_arg      |           0.000 | {'not_encoded': 40}                                                  |
|       0 | last_token    |           0.107 | {'not_encoded': 40}                                                  |
|       0 | sanitizer_def |           0.000 | {'not_encoded': 40}                                                  |
|       0 | sink_arg      |           1.087 | {'not_encoded': 40}                                                  |
|       3 | last_token    |           0.129 | {'not_encoded': 40}                                                  |
|       3 | sanitizer_def |           0.000 | {'not_encoded': 40}                                                  |
|       3 | sink_arg      |           0.872 | {'not_encoded': 40}                                                  |
|       7 | last_token    |           0.561 | {'encoded_and_used': 27, 'encoded_but_unused': 12, 'not_encoded': 1} |
|       7 | sanitizer_def |           0.000 | {'encoded_but_unused': 39, 'not_encoded': 1}                         |
|       7 | sink_arg      |           0.208 | {'encoded_but_unused': 31, 'encoded_and_used': 8, 'not_encoded': 1}  |
|      11 | last_token    |           0.771 | {'encoded_and_used': 21, 'encoded_but_unused': 13, 'not_encoded': 6} |
|      11 | sanitizer_def |           0.000 | {'encoded_but_unused': 34, 'not_encoded': 6}                         |
|      11 | sink_arg      |           0.106 | {'encoded_but_unused': 30, 'not_encoded': 6, 'encoded_and_used': 4}  |
|      15 | last_token    |           0.992 | {'not_encoded': 22, 'encoded_and_used': 12, 'encoded_but_unused': 6} |
|      15 | sanitizer_def |           0.000 | {'not_encoded': 22, 'encoded_but_unused': 18}                        |
|      15 | sink_arg      |           0.180 | {'not_encoded': 22, 'encoded_but_unused': 15, 'encoded_and_used': 3} |
|      19 | last_token    |           0.899 | {'not_encoded': 38, 'encoded_but_unused': 2}                         |
|      19 | sanitizer_def |           0.000 | {'not_encoded': 38, 'encoded_but_unused': 2}                         |
|      19 | sink_arg      |           0.095 | {'not_encoded': 38, 'encoded_and_used': 1, 'encoded_but_unused': 1}  |
|      23 | last_token    |           1.000 | {'not_encoded': 37, 'encoded_and_used': 3}                           |
|      23 | sanitizer_def |           0.000 | {'not_encoded': 37, 'encoded_but_unused': 3}                         |
|      23 | sink_arg      |           0.000 | {'not_encoded': 37, 'encoded_but_unused': 3}                         |
