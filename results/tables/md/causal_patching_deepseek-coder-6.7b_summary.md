# Causal patching — deepseek-coder-6.7b

|   layer | position      |   mean_recovery | causal_class                                                         |
|--------:|:--------------|----------------:|:---------------------------------------------------------------------|
|      -1 | last_token    |           0.000 | {'not_encoded': 40}                                                  |
|      -1 | sanitizer_def |           0.000 | {'not_encoded': 40}                                                  |
|      -1 | sink_arg      |           0.000 | {'not_encoded': 40}                                                  |
|       0 | last_token    |          -0.007 | {'not_encoded': 38, 'encoded_but_unused': 2}                         |
|       0 | sanitizer_def |           0.000 | {'not_encoded': 38, 'encoded_but_unused': 2}                         |
|       0 | sink_arg      |           0.985 | {'not_encoded': 38, 'encoded_and_used': 2}                           |
|       3 | last_token    |           0.014 | {'not_encoded': 40}                                                  |
|       3 | sanitizer_def |           0.000 | {'not_encoded': 40}                                                  |
|       3 | sink_arg      |           0.912 | {'not_encoded': 40}                                                  |
|       7 | last_token    |           0.074 | {'encoded_but_unused': 29, 'not_encoded': 11}                        |
|       7 | sanitizer_def |           0.000 | {'encoded_but_unused': 29, 'not_encoded': 11}                        |
|       7 | sink_arg      |           0.708 | {'encoded_and_used': 26, 'not_encoded': 11, 'encoded_but_unused': 3} |
|      11 | last_token    |           0.145 | {'encoded_but_unused': 31, 'not_encoded': 9}                         |
|      11 | sanitizer_def |           0.000 | {'encoded_but_unused': 31, 'not_encoded': 9}                         |
|      11 | sink_arg      |           0.500 | {'encoded_and_used': 16, 'encoded_but_unused': 15, 'not_encoded': 9} |
|      15 | last_token    |           0.308 | {'not_encoded': 21, 'encoded_but_unused': 17, 'encoded_and_used': 2} |
|      15 | sanitizer_def |           0.000 | {'not_encoded': 21, 'encoded_but_unused': 19}                        |
|      15 | sink_arg      |           0.235 | {'not_encoded': 21, 'encoded_but_unused': 18, 'encoded_and_used': 1} |
|      19 | last_token    |           0.650 | {'encoded_and_used': 23, 'not_encoded': 9, 'encoded_but_unused': 8}  |
|      19 | sanitizer_def |           0.000 | {'encoded_but_unused': 31, 'not_encoded': 9}                         |
|      19 | sink_arg      |           0.038 | {'encoded_but_unused': 31, 'not_encoded': 9}                         |
|      23 | last_token    |           0.759 | {'not_encoded': 24, 'encoded_and_used': 13, 'encoded_but_unused': 3} |
|      23 | sanitizer_def |           0.000 | {'not_encoded': 24, 'encoded_but_unused': 16}                        |
|      23 | sink_arg      |           0.049 | {'not_encoded': 24, 'encoded_but_unused': 16}                        |
|      27 | last_token    |           0.786 | {'not_encoded': 27, 'encoded_and_used': 11, 'encoded_but_unused': 2} |
|      27 | sanitizer_def |           0.000 | {'not_encoded': 27, 'encoded_but_unused': 13}                        |
|      27 | sink_arg      |           0.045 | {'not_encoded': 27, 'encoded_but_unused': 13}                        |
|      31 | last_token    |           1.000 | {'encoded_and_used': 31, 'not_encoded': 9}                           |
|      31 | sanitizer_def |           0.000 | {'encoded_but_unused': 31, 'not_encoded': 9}                         |
|      31 | sink_arg      |           0.000 | {'encoded_but_unused': 31, 'not_encoded': 9}                         |
