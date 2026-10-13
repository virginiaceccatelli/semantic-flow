# Execution prediction and U-Space

Model: `deepseek-coder-1.3b`. Block index: **6** (zero-based, post-block).
Read location: `prompt_end`. No fine-tuning or steering.

## Execution

Evaluated: 726 / 726 calls.
Program-macro accuracy: 0.2809917355371901; call accuracy: 0.2809917355371901; invalid answers: 56.

## Uncertainty

Higher scores indicate greater measured uncertainty.

| Score | Eligible calls | Error AUROC | Actual retained coverage | Retained error rate |
|---|---:|---:|---:|---:|
| ulens_score | 726 | 0.6361937495304635 | 0.800275482093664 | 0.6833046471600689 |
| a_cone | 726 | 0.42668657501314705 | 0.800275482093664 | 0.7314974182444062 |
| mean_answer_entropy | 726 | 0.639518067763504 | 0.800275482093664 | 0.6815834767641996 |
| mean_answer_nll | 726 | 0.6579332882578318 | 0.800275482093664 | 0.693631669535284 |

U-Space alignment (`a_cone`) uses the published formula. `ulens_score` is `a_cone * mean_answer_entropy`: a short-answer adaptation, not the published end-of-thinking protocol. Entropy covers tokens producing the answer, excluding EOS and newline-only termination. It is not a probability of error.

`top_uspace_tokens` decodes the signed U-Space projection through the J-lens (an exploratory extension). `top_jlens_tokens` decodes the complete residual. Both lists contain literal tokenizer tokens, including whitespace/fragments; neither establishes the cause of a failure.

## First ten evaluated examples

```json
{
  "case_id": "sample_0",
  "call": "f([1, 1, 3, 1, 3, 1])",
  "expected": "[(4, 1), (4, 1), (4, 1), (4, 1), (2, 3), (2, 3)]",
  "answer": "[(3, 3), (1, 1), (1, 1)]",
  "correct": false,
  "location": "prompt_end",
  "read_token_index": 89,
  "read_token_text": " ==",
  "a_cone": 1.5747642625794678,
  "ulens_score": 0.47216405083525126,
  "category_cosines": {
    "ambiguity": 0.051842086017131805,
    "incompleteness": -0.07428687065839767,
    "conflicting_evidence": -0.02648535557091236,
    "general_uncertainty": -0.011673768982291222
  },
  "top_uspace_tokens": [
    {
      "rank": 1,
      "token_id": 10047,
      "text": "\ufffd",
      "score": 24.5
    },
    {
      "rank": 2,
      "token_id": 12394,
      "text": " \ufffd",
      "score": 24.25
    },
    {
      "rank": 3,
      "token_id": 24122,
      "text": " gonna",
      "score": 19.875
    },
    {
      "rank": 4,
      "token_id": 22709,
      "text": " \ufffd",
      "score": 18.0
    },
    {
      "rank": 5,
      "token_id": 14176,
      "text": " anymore",
      "score": 13.8125
    },
    {
      "rank": 6,
      "token_id": 30052,
      "text": " lol",
      "score": 13.8125
    },
    {
      "rank": 7,
      "token_id": 24331,
      "text": " fuck",
      "score": 12.875
    },
    {
      "rank": 8,
      "token_id": 7678,
      "text": " strugg",
      "score": 12.25
    },
    {
      "rank": 9,
      "token_id": 24717,
      "text": " COVID",
      "score": 11.6875
    },
    {
      "rank": 10,
      "token_id": 30475,
      "text": " musul",
      "score": 10.5625
    },
    {
      "rank": 11,
      "token_id": 19149,
      "text": " Instagram",
      "score": 10.5
    },
    {
      "rank": 12,
      "token_id": 30727,
      "text": "\u5360\u6bd4",
      "score": 10.5
    },
    {
      "rank": 13,
      "token_id": 22199,
      "text": "\u6210\u4e3a\u4e86",
      "score": 10.0
    },
    {
      "rank": 14,
      "token_id": 31804,
      "text": " Nigeria",
      "score": 9.875
    },
    {
      "rank": 15,
      "token_id": 29881,
      "text": " anytime",
      "score": 9.5625
    },
    {
      "rank": 16,
      "token_id": 30603,
      "text": " fulfilled",
      "score": 9.5
    },
    {
      "rank": 17,
      "token_id": 31586,
      "text": " germans",
      "score": 9.375
    },
    {
      "rank": 18,
      "token_id": 9482,
      "text": " => ",
      "score": 9.3125
    },
    {
      "rank": 19,
      "token_id": 7045,
      "text": " couldn",
      "score": 9.3125
    },
    {
      "rank": 20,
      "token_id": 7361,
      "text": " wouldn",
      "score": 9.1875
    }
  ],
  "top_jlens_tokens": [
    {
      "rank": 1,
      "token_id": 11181,
      "text": " somewhere",
      "score": 6.96875
    },
    {
      "rank": 2,
      "token_id": 2220,
      "text": " `",
      "score": 6.75
    },
    {
      "rank": 3,
      "token_id": 8528,
      "text": " +=",
      "score": 6.6875
    },
    {
      "rank": 4,
      "token_id": 16352,
      "text": " elsewhere",
      "score": 6.4375
    },
    {
      "rank": 5,
      "token_id": 1585,
      "text": " --",
      "score": 6.34375
    },
    {
      "rank": 6,
      "token_id": 6236,
      "text": " ``",
      "score": 6.25
    },
    {
      "rank": 7,
      "token_id": 1368,
      "text": " because",
      "score": 6.09375
    },
    {
      "rank": 8,
      "token_id": 25,
      "text": ":",
      "score": 5.84375
    },
    {
      "rank": 9,
      "token_id": 4052,
      "text": " expected",
      "score": 5.65625
    },
    {
      "rank": 10,
      "token_id": 440,
      "text": " \"",
      "score": 5.59375
    },
    {
      "rank": 11,
      "token_id": 26022,
      "text": " \"#",
      "score": 5.5
    },
    {
      "rank": 12,
      "token_id": 1866,
      "text": " something",
      "score": 5.4375
    },
    {
      "rank": 13,
      "token_id": 26702,
      "text": " (#",
      "score": 5.40625
    },
    {
      "rank": 14,
      "token_id": 24531,
      "text": " foo",
      "score": 5.28125
    },
    {
      "rank": 15,
      "token_id": 23323,
      "text": " += ",
      "score": 5.25
    },
    {
      "rank": 16,
      "token_id": 32014,
      "text": "<\uff5cend\u2581of\u2581sentence\uff5c>",
      "score": 5.15625
    },
    {
      "rank": 17,
      "token_id": 19423,
      "text": " TODO",
      "score": 5.125
    },
    {
      "rank": 18,
      "token_id": 19788,
      "text": " {\"",
      "score": 5.125
    },
    {
      "rank": 19,
      "token_id": 945,
      "text": " +",
      "score": 5.09375
    },
    {
      "rank": 20,
      "token_id": 4341,
      "text": " implement",
      "score": 4.84375
    }
  ]
}
```

```json
{
  "case_id": "sample_2",
  "call": "f('hbtofdeiequ')",
  "expected": "'hbtofdeiequ'",
  "answer": "'hbtofdei'",
  "correct": false,
  "location": "prompt_end",
  "read_token_index": 84,
  "read_token_text": " ==",
  "a_cone": 1.6068002777541819,
  "ulens_score": 2.5356351374173647,
  "category_cosines": {
    "ambiguity": 0.04971829801797867,
    "incompleteness": -0.06775348633527756,
    "conflicting_evidence": -0.028896039351820946,
    "general_uncertainty": -0.0051497649401426315
  },
  "top_uspace_tokens": [
    {
      "rank": 1,
      "token_id": 10047,
      "text": "\ufffd",
      "score": 25.0
    },
    {
      "rank": 2,
      "token_id": 12394,
      "text": " \ufffd",
      "score": 24.875
    },
    {
      "rank": 3,
      "token_id": 24122,
      "text": " gonna",
      "score": 19.75
    },
    {
      "rank": 4,
      "token_id": 22709,
      "text": " \ufffd",
      "score": 18.25
    },
    {
      "rank": 5,
      "token_id": 30052,
      "text": " lol",
      "score": 13.9375
    },
    {
      "rank": 6,
      "token_id": 14176,
      "text": " anymore",
      "score": 13.3125
    },
    {
      "rank": 7,
      "token_id": 24331,
      "text": " fuck",
      "score": 13.1875
    },
    {
      "rank": 8,
      "token_id": 7678,
      "text": " strugg",
      "score": 12.4375
    },
    {
      "rank": 9,
      "token_id": 24717,
      "text": " COVID",
      "score": 11.9375
    },
    {
      "rank": 10,
      "token_id": 19149,
      "text": " Instagram",
      "score": 10.6875
    },
    {
      "rank": 11,
      "token_id": 30727,
      "text": "\u5360\u6bd4",
      "score": 10.625
    },
    {
      "rank": 12,
      "token_id": 30475,
      "text": " musul",
      "score": 10.3125
    },
    {
      "rank": 13,
      "token_id": 31586,
      "text": " germans",
      "score": 10.1875
    },
    {
      "rank": 14,
      "token_id": 22028,
      "text": " dumpster",
      "score": 9.75
    },
    {
      "rank": 15,
      "token_id": 22199,
      "text": "\u6210\u4e3a\u4e86",
      "score": 9.6875
    },
    {
      "rank": 16,
      "token_id": 7045,
      "text": " couldn",
      "score": 9.375
    },
    {
      "rank": 17,
      "token_id": 18268,
      "text": " scenarios",
      "score": 9.375
    },
    {
      "rank": 18,
      "token_id": 26754,
      "text": "\u65b0\u51a0",
      "score": 9.3125
    },
    {
      "rank": 19,
      "token_id": 9482,
      "text": " => ",
      "score": 9.25
    },
    {
      "rank": 20,
      "token_id": 7361,
      "text": " wouldn",
      "score": 9.25
    }
  ],
  "top_jlens_tokens": [
    {
      "rank": 1,
      "token_id": 2220,
      "text": " `",
      "score": 8.375
    },
    {
      "rank": 2,
      "token_id": 6236,
      "text": " ``",
      "score": 6.0625
    },
    {
      "rank": 3,
      "token_id": 440,
      "text": " \"",
      "score": 5.96875
    },
    {
      "rank": 4,
      "token_id": 1368,
      "text": " because",
      "score": 5.78125
    },
    {
      "rank": 5,
      "token_id": 8528,
      "text": " +=",
      "score": 5.78125
    },
    {
      "rank": 6,
      "token_id": 26022,
      "text": " \"#",
      "score": 5.75
    },
    {
      "rank": 7,
      "token_id": 1585,
      "text": " --",
      "score": 5.59375
    },
    {
      "rank": 8,
      "token_id": 11181,
      "text": " somewhere",
      "score": 5.09375
    },
    {
      "rank": 9,
      "token_id": 19788,
      "text": " {\"",
      "score": 5.03125
    },
    {
      "rank": 10,
      "token_id": 9678,
      "text": " implementation",
      "score": 4.9375
    },
    {
      "rank": 11,
      "token_id": 4052,
      "text": " expected",
      "score": 4.9375
    },
    {
      "rank": 12,
      "token_id": 945,
      "text": " +",
      "score": 4.8125
    },
    {
      "rank": 13,
      "token_id": 2044,
      "text": " =>",
      "score": 4.78125
    },
    {
      "rank": 14,
      "token_id": 371,
      "text": " $",
      "score": 4.6875
    },
    {
      "rank": 15,
      "token_id": 17079,
      "text": " {{",
      "score": 4.6875
    },
    {
      "rank": 16,
      "token_id": 32014,
      "text": "<\uff5cend\u2581of\u2581sentence\uff5c>",
      "score": 4.625
    },
    {
      "rank": 17,
      "token_id": 1866,
      "text": " something",
      "score": 4.53125
    },
    {
      "rank": 18,
      "token_id": 8690,
      "text": " API",
      "score": 4.53125
    },
    {
      "rank": 19,
      "token_id": 17625,
      "text": " [\"",
      "score": 4.46875
    },
    {
      "rank": 20,
      "token_id": 23323,
      "text": " += ",
      "score": 4.375
    }
  ]
}
```

```json
{
  "case_id": "sample_3",
  "call": "f('bcksrut', 'q')",
  "expected": "'bcksrutq'",
  "answer": "'bcksrutq'",
  "correct": true,
  "location": "prompt_end",
  "read_token_index": 69,
  "read_token_text": " ==",
  "a_cone": 1.6402062261204817,
  "ulens_score": 0.9658855962426415,
  "category_cosines": {
    "ambiguity": 0.054086487740278244,
    "incompleteness": -0.06914165616035461,
    "conflicting_evidence": -0.028145747259259224,
    "general_uncertainty": -0.006905172020196915
  },
  "top_uspace_tokens": [
    {
      "rank": 1,
      "token_id": 10047,
      "text": "\ufffd",
      "score": 25.125
    },
    {
      "rank": 2,
      "token_id": 12394,
      "text": " \ufffd",
      "score": 25.125
    },
    {
      "rank": 3,
      "token_id": 24122,
      "text": " gonna",
      "score": 19.75
    },
    {
      "rank": 4,
      "token_id": 22709,
      "text": " \ufffd",
      "score": 18.625
    },
    {
      "rank": 5,
      "token_id": 30052,
      "text": " lol",
      "score": 14.0625
    },
    {
      "rank": 6,
      "token_id": 14176,
      "text": " anymore",
      "score": 13.4375
    },
    {
      "rank": 7,
      "token_id": 24331,
      "text": " fuck",
      "score": 13.3125
    },
    {
      "rank": 8,
      "token_id": 7678,
      "text": " strugg",
      "score": 12.375
    },
    {
      "rank": 9,
      "token_id": 24717,
      "text": " COVID",
      "score": 12.0625
    },
    {
      "rank": 10,
      "token_id": 19149,
      "text": " Instagram",
      "score": 10.875
    },
    {
      "rank": 11,
      "token_id": 30475,
      "text": " musul",
      "score": 10.75
    },
    {
      "rank": 12,
      "token_id": 30727,
      "text": "\u5360\u6bd4",
      "score": 10.5
    },
    {
      "rank": 13,
      "token_id": 31586,
      "text": " germans",
      "score": 10.0625
    },
    {
      "rank": 14,
      "token_id": 22199,
      "text": "\u6210\u4e3a\u4e86",
      "score": 9.6875
    },
    {
      "rank": 15,
      "token_id": 7045,
      "text": " couldn",
      "score": 9.6875
    },
    {
      "rank": 16,
      "token_id": 22028,
      "text": " dumpster",
      "score": 9.625
    },
    {
      "rank": 17,
      "token_id": 26754,
      "text": "\u65b0\u51a0",
      "score": 9.5
    },
    {
      "rank": 18,
      "token_id": 8632,
      "text": " impossible",
      "score": 9.4375
    },
    {
      "rank": 19,
      "token_id": 7361,
      "text": " wouldn",
      "score": 9.4375
    },
    {
      "rank": 20,
      "token_id": 6740,
      "text": " wasn",
      "score": 9.375
    }
  ],
  "top_jlens_tokens": [
    {
      "rank": 1,
      "token_id": 2220,
      "text": " `",
      "score": 7.9375
    },
    {
      "rank": 2,
      "token_id": 6236,
      "text": " ``",
      "score": 6.5625
    },
    {
      "rank": 3,
      "token_id": 1368,
      "text": " because",
      "score": 6.1875
    },
    {
      "rank": 4,
      "token_id": 440,
      "text": " \"",
      "score": 5.9375
    },
    {
      "rank": 5,
      "token_id": 8528,
      "text": " +=",
      "score": 5.46875
    },
    {
      "rank": 6,
      "token_id": 1585,
      "text": " --",
      "score": 5.4375
    },
    {
      "rank": 7,
      "token_id": 26022,
      "text": " \"#",
      "score": 5.34375
    },
    {
      "rank": 8,
      "token_id": 11181,
      "text": " somewhere",
      "score": 5.3125
    },
    {
      "rank": 9,
      "token_id": 4052,
      "text": " expected",
      "score": 5.15625
    },
    {
      "rank": 10,
      "token_id": 19788,
      "text": " {\"",
      "score": 5.0
    },
    {
      "rank": 11,
      "token_id": 1866,
      "text": " something",
      "score": 4.75
    },
    {
      "rank": 12,
      "token_id": 371,
      "text": " $",
      "score": 4.71875
    },
    {
      "rank": 13,
      "token_id": 945,
      "text": " +",
      "score": 4.625
    },
    {
      "rank": 14,
      "token_id": 2044,
      "text": " =>",
      "score": 4.5625
    },
    {
      "rank": 15,
      "token_id": 19423,
      "text": " TODO",
      "score": 4.40625
    },
    {
      "rank": 16,
      "token_id": 9678,
      "text": " implementation",
      "score": 4.375
    },
    {
      "rank": 17,
      "token_id": 17079,
      "text": " {{",
      "score": 4.3125
    },
    {
      "rank": 18,
      "token_id": 25,
      "text": ":",
      "score": 4.1875
    },
    {
      "rank": 19,
      "token_id": 17625,
      "text": " [\"",
      "score": 4.125
    },
    {
      "rank": 20,
      "token_id": 24531,
      "text": " foo",
      "score": 4.0625
    }
  ]
}
```

```json
{
  "case_id": "sample_4",
  "call": "f([' ', '  ', '    ', '   '])",
  "expected": "'           '",
  "answer": "'  '",
  "correct": false,
  "location": "prompt_end",
  "read_token_index": 59,
  "read_token_text": " ==",
  "a_cone": 1.6610269994214553,
  "ulens_score": 3.2096568764848747,
  "category_cosines": {
    "ambiguity": 0.0526493564248085,
    "incompleteness": -0.0655352845788002,
    "conflicting_evidence": -0.019549628719687462,
    "general_uncertainty": -0.012291152961552143
  },
  "top_uspace_tokens": [
    {
      "rank": 1,
      "token_id": 10047,
      "text": "\ufffd",
      "score": 25.125
    },
    {
      "rank": 2,
      "token_id": 12394,
      "text": " \ufffd",
      "score": 25.0
    },
    {
      "rank": 3,
      "token_id": 24122,
      "text": " gonna",
      "score": 19.625
    },
    {
      "rank": 4,
      "token_id": 22709,
      "text": " \ufffd",
      "score": 18.875
    },
    {
      "rank": 5,
      "token_id": 30052,
      "text": " lol",
      "score": 14.0625
    },
    {
      "rank": 6,
      "token_id": 14176,
      "text": " anymore",
      "score": 13.9375
    },
    {
      "rank": 7,
      "token_id": 24331,
      "text": " fuck",
      "score": 13.1875
    },
    {
      "rank": 8,
      "token_id": 24717,
      "text": " COVID",
      "score": 12.0
    },
    {
      "rank": 9,
      "token_id": 7678,
      "text": " strugg",
      "score": 12.0
    },
    {
      "rank": 10,
      "token_id": 30475,
      "text": " musul",
      "score": 11.5625
    },
    {
      "rank": 11,
      "token_id": 19149,
      "text": " Instagram",
      "score": 10.9375
    },
    {
      "rank": 12,
      "token_id": 30727,
      "text": "\u5360\u6bd4",
      "score": 10.375
    },
    {
      "rank": 13,
      "token_id": 8632,
      "text": " impossible",
      "score": 10.125
    },
    {
      "rank": 14,
      "token_id": 7045,
      "text": " couldn",
      "score": 10.0
    },
    {
      "rank": 15,
      "token_id": 22199,
      "text": "\u6210\u4e3a\u4e86",
      "score": 9.875
    },
    {
      "rank": 16,
      "token_id": 31804,
      "text": " Nigeria",
      "score": 9.8125
    },
    {
      "rank": 17,
      "token_id": 30603,
      "text": " fulfilled",
      "score": 9.6875
    },
    {
      "rank": 18,
      "token_id": 7361,
      "text": " wouldn",
      "score": 9.5
    },
    {
      "rank": 19,
      "token_id": 9482,
      "text": " => ",
      "score": 9.4375
    },
    {
      "rank": 20,
      "token_id": 31586,
      "text": " germans",
      "score": 9.375
    }
  ],
  "top_jlens_tokens": [
    {
      "rank": 1,
      "token_id": 32014,
      "text": "<\uff5cend\u2581of\u2581sentence\uff5c>",
      "score": 8.625
    },
    {
      "rank": 2,
      "token_id": 2220,
      "text": " `",
      "score": 6.15625
    },
    {
      "rank": 3,
      "token_id": 25,
      "text": ":",
      "score": 5.53125
    },
    {
      "rank": 4,
      "token_id": 440,
      "text": " \"",
      "score": 5.34375
    },
    {
      "rank": 5,
      "token_id": 11181,
      "text": " somewhere",
      "score": 5.28125
    },
    {
      "rank": 6,
      "token_id": 6236,
      "text": " ``",
      "score": 5.1875
    },
    {
      "rank": 7,
      "token_id": 1368,
      "text": " because",
      "score": 5.0625
    },
    {
      "rank": 8,
      "token_id": 1866,
      "text": " something",
      "score": 4.9375
    },
    {
      "rank": 9,
      "token_id": 1585,
      "text": " --",
      "score": 4.875
    },
    {
      "rank": 10,
      "token_id": 3552,
      "text": " ...",
      "score": 4.71875
    },
    {
      "rank": 11,
      "token_id": 8528,
      "text": " +=",
      "score": 4.71875
    },
    {
      "rank": 12,
      "token_id": 4052,
      "text": " expected",
      "score": 4.34375
    },
    {
      "rank": 13,
      "token_id": 16352,
      "text": " elsewhere",
      "score": 4.28125
    },
    {
      "rank": 14,
      "token_id": 19423,
      "text": " TODO",
      "score": 4.1875
    },
    {
      "rank": 15,
      "token_id": 371,
      "text": " $",
      "score": 3.96875
    },
    {
      "rank": 16,
      "token_id": 9678,
      "text": " implementation",
      "score": 3.859375
    },
    {
      "rank": 17,
      "token_id": 26022,
      "text": " \"#",
      "score": 3.734375
    },
    {
      "rank": 18,
      "token_id": 26,
      "text": ";",
      "score": 3.734375
    },
    {
      "rank": 19,
      "token_id": 185,
      "text": "\n",
      "score": 3.703125
    },
    {
      "rank": 20,
      "token_id": 945,
      "text": " +",
      "score": 3.671875
    }
  ]
}
```

```json
{
  "case_id": "sample_5",
  "call": "f('DSUWeqExTQdCMGpqur', 'a', 'x')",
  "expected": "(0, 'xxxxxxxxxxxxxxxxxx')",
  "answer": "(2, 'DSUWeqExTQdCMGx')",
  "correct": false,
  "location": "prompt_end",
  "read_token_index": 126,
  "read_token_text": " ==",
  "a_cone": 1.4806906626877052,
  "ulens_score": 2.087289507521462,
  "category_cosines": {
    "ambiguity": 0.04134243726730347,
    "incompleteness": -0.07456691563129425,
    "conflicting_evidence": -0.025899333879351616,
    "general_uncertainty": -0.007159245666116476
  },
  "top_uspace_tokens": [
    {
      "rank": 1,
      "token_id": 10047,
      "text": "\ufffd",
      "score": 23.75
    },
    {
      "rank": 2,
      "token_id": 12394,
      "text": " \ufffd",
      "score": 23.5
    },
    {
      "rank": 3,
      "token_id": 24122,
      "text": " gonna",
      "score": 20.0
    },
    {
      "rank": 4,
      "token_id": 22709,
      "text": " \ufffd",
      "score": 17.125
    },
    {
      "rank": 5,
      "token_id": 14176,
      "text": " anymore",
      "score": 13.5625
    },
    {
      "rank": 6,
      "token_id": 30052,
      "text": " lol",
      "score": 13.3125
    },
    {
      "rank": 7,
      "token_id": 24331,
      "text": " fuck",
      "score": 12.3125
    },
    {
      "rank": 8,
      "token_id": 7678,
      "text": " strugg",
      "score": 12.3125
    },
    {
      "rank": 9,
      "token_id": 24717,
      "text": " COVID",
      "score": 11.1875
    },
    {
      "rank": 10,
      "token_id": 30727,
      "text": "\u5360\u6bd4",
      "score": 10.8125
    },
    {
      "rank": 11,
      "token_id": 31804,
      "text": " Nigeria",
      "score": 9.9375
    },
    {
      "rank": 12,
      "token_id": 19149,
      "text": " Instagram",
      "score": 9.8125
    },
    {
      "rank": 13,
      "token_id": 22199,
      "text": "\u6210\u4e3a\u4e86",
      "score": 9.8125
    },
    {
      "rank": 14,
      "token_id": 30475,
      "text": " musul",
      "score": 9.75
    },
    {
      "rank": 15,
      "token_id": 29881,
      "text": " anytime",
      "score": 9.6875
    },
    {
      "rank": 16,
      "token_id": 26696,
      "text": " utilizing",
      "score": 9.4375
    },
    {
      "rank": 17,
      "token_id": 9482,
      "text": " => ",
      "score": 9.3125
    },
    {
      "rank": 18,
      "token_id": 31586,
      "text": " germans",
      "score": 9.25
    },
    {
      "rank": 19,
      "token_id": 10273,
      "text": "!!!",
      "score": 9.25
    },
    {
      "rank": 20,
      "token_id": 18268,
      "text": " scenarios",
      "score": 9.25
    }
  ],
  "top_jlens_tokens": [
    {
      "rank": 1,
      "token_id": 32014,
      "text": "<\uff5cend\u2581of\u2581sentence\uff5c>",
      "score": 7.375
    },
    {
      "rank": 2,
      "token_id": 1585,
      "text": " --",
      "score": 6.21875
    },
    {
      "rank": 3,
      "token_id": 440,
      "text": " \"",
      "score": 5.90625
    },
    {
      "rank": 4,
      "token_id": 1368,
      "text": " because",
      "score": 5.71875
    },
    {
      "rank": 5,
      "token_id": 25,
      "text": ":",
      "score": 5.0625
    },
    {
      "rank": 6,
      "token_id": 3552,
      "text": " ...",
      "score": 4.84375
    },
    {
      "rank": 7,
      "token_id": 6236,
      "text": " ``",
      "score": 4.8125
    },
    {
      "rank": 8,
      "token_id": 334,
      "text": " (",
      "score": 4.78125
    },
    {
      "rank": 9,
      "token_id": 8528,
      "text": " +=",
      "score": 4.75
    },
    {
      "rank": 10,
      "token_id": 185,
      "text": "\n",
      "score": 4.4375
    },
    {
      "rank": 11,
      "token_id": 26,
      "text": ";",
      "score": 4.40625
    },
    {
      "rank": 12,
      "token_id": 26022,
      "text": " \"#",
      "score": 4.375
    },
    {
      "rank": 13,
      "token_id": 945,
      "text": " +",
      "score": 4.3125
    },
    {
      "rank": 14,
      "token_id": 2220,
      "text": " `",
      "score": 4.3125
    },
    {
      "rank": 15,
      "token_id": 23323,
      "text": " += ",
      "score": 4.21875
    },
    {
      "rank": 16,
      "token_id": 11181,
      "text": " somewhere",
      "score": 4.21875
    },
    {
      "rank": 17,
      "token_id": 26702,
      "text": " (#",
      "score": 4.21875
    },
    {
      "rank": 18,
      "token_id": 1866,
      "text": " something",
      "score": 4.15625
    },
    {
      "rank": 19,
      "token_id": 8507,
      "text": " (\"",
      "score": 3.953125
    },
    {
      "rank": 20,
      "token_id": 750,
      "text": " when",
      "score": 3.921875
    }
  ]
}
```

```json
{
  "case_id": "sample_6",
  "call": "f({'11': 52, '65': 34, 'a': 12, '4': 52, '74': 31})",
  "expected": "[('74', 31)]",
  "answer": "[('a', 12), ('4', 52), ('11', 52), ('65', 34), ('74', 31)]",
  "correct": false,
  "location": "prompt_end",
  "read_token_index": 111,
  "read_token_text": " ==",
  "a_cone": 1.746557215503199,
  "ulens_score": 0.3883553370777868,
  "category_cosines": {
    "ambiguity": 0.05790967121720314,
    "incompleteness": -0.06828852742910385,
    "conflicting_evidence": -0.018404467031359673,
    "general_uncertainty": -0.0019163353135809302
  },
  "top_uspace_tokens": [
    {
      "rank": 1,
      "token_id": 12394,
      "text": " \ufffd",
      "score": 26.125
    },
    {
      "rank": 2,
      "token_id": 10047,
      "text": "\ufffd",
      "score": 26.0
    },
    {
      "rank": 3,
      "token_id": 22709,
      "text": " \ufffd",
      "score": 19.75
    },
    {
      "rank": 4,
      "token_id": 24122,
      "text": " gonna",
      "score": 19.125
    },
    {
      "rank": 5,
      "token_id": 30052,
      "text": " lol",
      "score": 13.9375
    },
    {
      "rank": 6,
      "token_id": 24331,
      "text": " fuck",
      "score": 13.25
    },
    {
      "rank": 7,
      "token_id": 14176,
      "text": " anymore",
      "score": 13.125
    },
    {
      "rank": 8,
      "token_id": 24717,
      "text": " COVID",
      "score": 12.1875
    },
    {
      "rank": 9,
      "token_id": 30475,
      "text": " musul",
      "score": 12.125
    },
    {
      "rank": 10,
      "token_id": 7678,
      "text": " strugg",
      "score": 12.0625
    },
    {
      "rank": 11,
      "token_id": 19149,
      "text": " Instagram",
      "score": 11.125
    },
    {
      "rank": 12,
      "token_id": 30727,
      "text": "\u5360\u6bd4",
      "score": 10.625
    },
    {
      "rank": 13,
      "token_id": 26754,
      "text": "\u65b0\u51a0",
      "score": 10.4375
    },
    {
      "rank": 14,
      "token_id": 31586,
      "text": " germans",
      "score": 10.375
    },
    {
      "rank": 15,
      "token_id": 7045,
      "text": " couldn",
      "score": 9.9375
    },
    {
      "rank": 16,
      "token_id": 24950,
      "text": " medalla",
      "score": 9.875
    },
    {
      "rank": 17,
      "token_id": 22028,
      "text": " dumpster",
      "score": 9.875
    },
    {
      "rank": 18,
      "token_id": 6740,
      "text": " wasn",
      "score": 9.625
    },
    {
      "rank": 19,
      "token_id": 7361,
      "text": " wouldn",
      "score": 9.5625
    },
    {
      "rank": 20,
      "token_id": 9482,
      "text": " => ",
      "score": 9.375
    }
  ],
  "top_jlens_tokens": [
    {
      "rank": 1,
      "token_id": 185,
      "text": "\n",
      "score": 8.5
    },
    {
      "rank": 2,
      "token_id": 32014,
      "text": "<\uff5cend\u2581of\u2581sentence\uff5c>",
      "score": 8.5
    },
    {
      "rank": 3,
      "token_id": 25,
      "text": ":",
      "score": 5.40625
    },
    {
      "rank": 4,
      "token_id": 16352,
      "text": " elsewhere",
      "score": 5.40625
    },
    {
      "rank": 5,
      "token_id": 11181,
      "text": " somewhere",
      "score": 5.09375
    },
    {
      "rank": 6,
      "token_id": 8528,
      "text": " +=",
      "score": 5.09375
    },
    {
      "rank": 7,
      "token_id": 1866,
      "text": " something",
      "score": 5.0
    },
    {
      "rank": 8,
      "token_id": 1585,
      "text": " --",
      "score": 4.90625
    },
    {
      "rank": 9,
      "token_id": 1368,
      "text": " because",
      "score": 4.875
    },
    {
      "rank": 10,
      "token_id": 16603,
      "text": " [[",
      "score": 4.84375
    },
    {
      "rank": 11,
      "token_id": 17079,
      "text": " {{",
      "score": 4.53125
    },
    {
      "rank": 12,
      "token_id": 17625,
      "text": " [\"",
      "score": 4.40625
    },
    {
      "rank": 13,
      "token_id": 26702,
      "text": " (#",
      "score": 4.375
    },
    {
      "rank": 14,
      "token_id": 440,
      "text": " \"",
      "score": 4.34375
    },
    {
      "rank": 15,
      "token_id": 821,
      "text": " [",
      "score": 4.3125
    },
    {
      "rank": 16,
      "token_id": 2220,
      "text": " `",
      "score": 4.15625
    },
    {
      "rank": 17,
      "token_id": 19423,
      "text": " TODO",
      "score": 4.125
    },
    {
      "rank": 18,
      "token_id": 4052,
      "text": " expected",
      "score": 4.125
    },
    {
      "rank": 19,
      "token_id": 8507,
      "text": " (\"",
      "score": 3.984375
    },
    {
      "rank": 20,
      "token_id": 31421,
      "text": " \"[",
      "score": 3.953125
    }
  ]
}
```

```json
{
  "case_id": "sample_7",
  "call": "f([])",
  "expected": "[]",
  "answer": "[]",
  "correct": true,
  "location": "prompt_end",
  "read_token_index": 105,
  "read_token_text": " ==",
  "a_cone": 1.6632721877605954,
  "ulens_score": 1.2931174481106746,
  "category_cosines": {
    "ambiguity": 0.05823887139558792,
    "incompleteness": -0.06389059871435165,
    "conflicting_evidence": -0.040357016026973724,
    "general_uncertainty": -0.005902155768126249
  },
  "top_uspace_tokens": [
    {
      "rank": 1,
      "token_id": 12394,
      "text": " \ufffd",
      "score": 25.5
    },
    {
      "rank": 2,
      "token_id": 10047,
      "text": "\ufffd",
      "score": 25.25
    },
    {
      "rank": 3,
      "token_id": 24122,
      "text": " gonna",
      "score": 19.5
    },
    {
      "rank": 4,
      "token_id": 22709,
      "text": " \ufffd",
      "score": 18.375
    },
    {
      "rank": 5,
      "token_id": 30052,
      "text": " lol",
      "score": 14.4375
    },
    {
      "rank": 6,
      "token_id": 24331,
      "text": " fuck",
      "score": 13.875
    },
    {
      "rank": 7,
      "token_id": 14176,
      "text": " anymore",
      "score": 12.9375
    },
    {
      "rank": 8,
      "token_id": 7678,
      "text": " strugg",
      "score": 12.625
    },
    {
      "rank": 9,
      "token_id": 24717,
      "text": " COVID",
      "score": 12.375
    },
    {
      "rank": 10,
      "token_id": 19149,
      "text": " Instagram",
      "score": 11.1875
    },
    {
      "rank": 11,
      "token_id": 22028,
      "text": " dumpster",
      "score": 10.75
    },
    {
      "rank": 12,
      "token_id": 31586,
      "text": " germans",
      "score": 10.6875
    },
    {
      "rank": 13,
      "token_id": 8632,
      "text": " impossible",
      "score": 10.5625
    },
    {
      "rank": 14,
      "token_id": 26754,
      "text": "\u65b0\u51a0",
      "score": 10.125
    },
    {
      "rank": 15,
      "token_id": 30727,
      "text": "\u5360\u6bd4",
      "score": 10.125
    },
    {
      "rank": 16,
      "token_id": 6740,
      "text": " wasn",
      "score": 10.0625
    },
    {
      "rank": 17,
      "token_id": 7045,
      "text": " couldn",
      "score": 10.0625
    },
    {
      "rank": 18,
      "token_id": 22199,
      "text": "\u6210\u4e3a\u4e86",
      "score": 9.75
    },
    {
      "rank": 19,
      "token_id": 30475,
      "text": " musul",
      "score": 9.75
    },
    {
      "rank": 20,
      "token_id": 7361,
      "text": " wouldn",
      "score": 9.5625
    }
  ],
  "top_jlens_tokens": [
    {
      "rank": 1,
      "token_id": 6236,
      "text": " ``",
      "score": 6.65625
    },
    {
      "rank": 2,
      "token_id": 2220,
      "text": " `",
      "score": 6.5625
    },
    {
      "rank": 3,
      "token_id": 1585,
      "text": " --",
      "score": 6.34375
    },
    {
      "rank": 4,
      "token_id": 8528,
      "text": " +=",
      "score": 5.8125
    },
    {
      "rank": 5,
      "token_id": 4052,
      "text": " expected",
      "score": 5.28125
    },
    {
      "rank": 6,
      "token_id": 440,
      "text": " \"",
      "score": 5.125
    },
    {
      "rank": 7,
      "token_id": 19423,
      "text": " TODO",
      "score": 5.125
    },
    {
      "rank": 8,
      "token_id": 25,
      "text": ":",
      "score": 5.09375
    },
    {
      "rank": 9,
      "token_id": 11181,
      "text": " somewhere",
      "score": 5.0625
    },
    {
      "rank": 10,
      "token_id": 945,
      "text": " +",
      "score": 4.90625
    },
    {
      "rank": 11,
      "token_id": 4341,
      "text": " implement",
      "score": 4.5625
    },
    {
      "rank": 12,
      "token_id": 19748,
      "text": " canonical",
      "score": 4.5
    },
    {
      "rank": 13,
      "token_id": 9678,
      "text": " implementation",
      "score": 4.3125
    },
    {
      "rank": 14,
      "token_id": 371,
      "text": " $",
      "score": 4.0625
    },
    {
      "rank": 15,
      "token_id": 4813,
      "text": " ~",
      "score": 3.90625
    },
    {
      "rank": 16,
      "token_id": 24531,
      "text": " foo",
      "score": 3.84375
    },
    {
      "rank": 17,
      "token_id": 1368,
      "text": " because",
      "score": 3.75
    },
    {
      "rank": 18,
      "token_id": 8507,
      "text": " (\"",
      "score": 3.75
    },
    {
      "rank": 19,
      "token_id": 334,
      "text": " (",
      "score": 3.484375
    },
    {
      "rank": 20,
      "token_id": 19788,
      "text": " {\"",
      "score": 3.40625
    }
  ]
}
```

```json
{
  "case_id": "sample_8",
  "call": "f('UppEr', 0)",
  "expected": "'UppEr'",
  "answer": "'UppEr'",
  "correct": true,
  "location": "prompt_end",
  "read_token_index": 65,
  "read_token_text": " ==",
  "a_cone": 1.5913199821697552,
  "ulens_score": 1.9085253428312996,
  "category_cosines": {
    "ambiguity": 0.04734841361641884,
    "incompleteness": -0.07485522329807281,
    "conflicting_evidence": -0.023131202906370163,
    "general_uncertainty": 0.0003669123980216682
  },
  "top_uspace_tokens": [
    {
      "rank": 1,
      "token_id": 10047,
      "text": "\ufffd",
      "score": 24.875
    },
    {
      "rank": 2,
      "token_id": 12394,
      "text": " \ufffd",
      "score": 24.75
    },
    {
      "rank": 3,
      "token_id": 24122,
      "text": " gonna",
      "score": 19.75
    },
    {
      "rank": 4,
      "token_id": 22709,
      "text": " \ufffd",
      "score": 18.25
    },
    {
      "rank": 5,
      "token_id": 30052,
      "text": " lol",
      "score": 13.375
    },
    {
      "rank": 6,
      "token_id": 14176,
      "text": " anymore",
      "score": 13.0625
    },
    {
      "rank": 7,
      "token_id": 24331,
      "text": " fuck",
      "score": 12.5625
    },
    {
      "rank": 8,
      "token_id": 7678,
      "text": " strugg",
      "score": 12.3125
    },
    {
      "rank": 9,
      "token_id": 24717,
      "text": " COVID",
      "score": 11.5
    },
    {
      "rank": 10,
      "token_id": 30727,
      "text": "\u5360\u6bd4",
      "score": 11.0
    },
    {
      "rank": 11,
      "token_id": 30475,
      "text": " musul",
      "score": 10.6875
    },
    {
      "rank": 12,
      "token_id": 19149,
      "text": " Instagram",
      "score": 10.25
    },
    {
      "rank": 13,
      "token_id": 31586,
      "text": " germans",
      "score": 10.125
    },
    {
      "rank": 14,
      "token_id": 22028,
      "text": " dumpster",
      "score": 9.5
    },
    {
      "rank": 15,
      "token_id": 9482,
      "text": " => ",
      "score": 9.375
    },
    {
      "rank": 16,
      "token_id": 18268,
      "text": " scenarios",
      "score": 9.375
    },
    {
      "rank": 17,
      "token_id": 22199,
      "text": "\u6210\u4e3a\u4e86",
      "score": 9.125
    },
    {
      "rank": 18,
      "token_id": 26754,
      "text": "\u65b0\u51a0",
      "score": 9.125
    },
    {
      "rank": 19,
      "token_id": 29881,
      "text": " anytime",
      "score": 9.0625
    },
    {
      "rank": 20,
      "token_id": 24950,
      "text": " medalla",
      "score": 9.0625
    }
  ],
  "top_jlens_tokens": [
    {
      "rank": 1,
      "token_id": 6236,
      "text": " ``",
      "score": 8.0
    },
    {
      "rank": 2,
      "token_id": 2220,
      "text": " `",
      "score": 7.90625
    },
    {
      "rank": 3,
      "token_id": 1585,
      "text": " --",
      "score": 7.3125
    },
    {
      "rank": 4,
      "token_id": 440,
      "text": " \"",
      "score": 6.375
    },
    {
      "rank": 5,
      "token_id": 11181,
      "text": " somewhere",
      "score": 6.3125
    },
    {
      "rank": 6,
      "token_id": 1368,
      "text": " because",
      "score": 5.75
    },
    {
      "rank": 7,
      "token_id": 8528,
      "text": " +=",
      "score": 5.6875
    },
    {
      "rank": 8,
      "token_id": 19423,
      "text": " TODO",
      "score": 5.3125
    },
    {
      "rank": 9,
      "token_id": 1866,
      "text": " something",
      "score": 5.28125
    },
    {
      "rank": 10,
      "token_id": 25,
      "text": ":",
      "score": 5.28125
    },
    {
      "rank": 11,
      "token_id": 945,
      "text": " +",
      "score": 5.0
    },
    {
      "rank": 12,
      "token_id": 9678,
      "text": " implementation",
      "score": 5.0
    },
    {
      "rank": 13,
      "token_id": 3552,
      "text": " ...",
      "score": 4.90625
    },
    {
      "rank": 14,
      "token_id": 24531,
      "text": " foo",
      "score": 4.875
    },
    {
      "rank": 15,
      "token_id": 4052,
      "text": " expected",
      "score": 4.78125
    },
    {
      "rank": 16,
      "token_id": 4341,
      "text": " implement",
      "score": 4.75
    },
    {
      "rank": 17,
      "token_id": 371,
      "text": " $",
      "score": 4.71875
    },
    {
      "rank": 18,
      "token_id": 2044,
      "text": " =>",
      "score": 4.71875
    },
    {
      "rank": 19,
      "token_id": 738,
      "text": " some",
      "score": 4.4375
    },
    {
      "rank": 20,
      "token_id": 8507,
      "text": " (\"",
      "score": 4.4375
    }
  ]
}
```

```json
{
  "case_id": "sample_9",
  "call": "f('#284376598')",
  "expected": "False",
  "answer": "False",
  "correct": true,
  "location": "prompt_end",
  "read_token_index": 63,
  "read_token_text": " ==",
  "a_cone": 1.5966411326378247,
  "ulens_score": 1.4052390915899189,
  "category_cosines": {
    "ambiguity": 0.05130260810256004,
    "incompleteness": -0.07665887475013733,
    "conflicting_evidence": -0.027732424437999725,
    "general_uncertainty": -0.0012518108123913407
  },
  "top_uspace_tokens": [
    {
      "rank": 1,
      "token_id": 10047,
      "text": "\ufffd",
      "score": 25.0
    },
    {
      "rank": 2,
      "token_id": 12394,
      "text": " \ufffd",
      "score": 24.875
    },
    {
      "rank": 3,
      "token_id": 24122,
      "text": " gonna",
      "score": 19.75
    },
    {
      "rank": 4,
      "token_id": 22709,
      "text": " \ufffd",
      "score": 18.25
    },
    {
      "rank": 5,
      "token_id": 30052,
      "text": " lol",
      "score": 13.625
    },
    {
      "rank": 6,
      "token_id": 14176,
      "text": " anymore",
      "score": 13.125
    },
    {
      "rank": 7,
      "token_id": 24331,
      "text": " fuck",
      "score": 12.875
    },
    {
      "rank": 8,
      "token_id": 7678,
      "text": " strugg",
      "score": 12.375
    },
    {
      "rank": 9,
      "token_id": 24717,
      "text": " COVID",
      "score": 11.6875
    },
    {
      "rank": 10,
      "token_id": 30727,
      "text": "\u5360\u6bd4",
      "score": 10.875
    },
    {
      "rank": 11,
      "token_id": 30475,
      "text": " musul",
      "score": 10.5
    },
    {
      "rank": 12,
      "token_id": 19149,
      "text": " Instagram",
      "score": 10.4375
    },
    {
      "rank": 13,
      "token_id": 31586,
      "text": " germans",
      "score": 10.25
    },
    {
      "rank": 14,
      "token_id": 22028,
      "text": " dumpster",
      "score": 9.75
    },
    {
      "rank": 15,
      "token_id": 18268,
      "text": " scenarios",
      "score": 9.375
    },
    {
      "rank": 16,
      "token_id": 9482,
      "text": " => ",
      "score": 9.3125
    },
    {
      "rank": 17,
      "token_id": 22199,
      "text": "\u6210\u4e3a\u4e86",
      "score": 9.3125
    },
    {
      "rank": 18,
      "token_id": 26754,
      "text": "\u65b0\u51a0",
      "score": 9.25
    },
    {
      "rank": 19,
      "token_id": 29881,
      "text": " anytime",
      "score": 9.0625
    },
    {
      "rank": 20,
      "token_id": 7361,
      "text": " wouldn",
      "score": 9.0625
    }
  ],
  "top_jlens_tokens": [
    {
      "rank": 1,
      "token_id": 2220,
      "text": " `",
      "score": 7.96875
    },
    {
      "rank": 2,
      "token_id": 6236,
      "text": " ``",
      "score": 5.6875
    },
    {
      "rank": 3,
      "token_id": 1585,
      "text": " --",
      "score": 5.25
    },
    {
      "rank": 4,
      "token_id": 440,
      "text": " \"",
      "score": 5.03125
    },
    {
      "rank": 5,
      "token_id": 1368,
      "text": " because",
      "score": 4.8125
    },
    {
      "rank": 6,
      "token_id": 11181,
      "text": " somewhere",
      "score": 4.59375
    },
    {
      "rank": 7,
      "token_id": 26022,
      "text": " \"#",
      "score": 4.59375
    },
    {
      "rank": 8,
      "token_id": 1866,
      "text": " something",
      "score": 4.375
    },
    {
      "rank": 9,
      "token_id": 19788,
      "text": " {\"",
      "score": 4.09375
    },
    {
      "rank": 10,
      "token_id": 8528,
      "text": " +=",
      "score": 4.0625
    },
    {
      "rank": 11,
      "token_id": 4052,
      "text": " expected",
      "score": 3.890625
    },
    {
      "rank": 12,
      "token_id": 8690,
      "text": " API",
      "score": 3.84375
    },
    {
      "rank": 13,
      "token_id": 945,
      "text": " +",
      "score": 3.75
    },
    {
      "rank": 14,
      "token_id": 371,
      "text": " $",
      "score": 3.734375
    },
    {
      "rank": 15,
      "token_id": 26702,
      "text": " (#",
      "score": 3.65625
    },
    {
      "rank": 16,
      "token_id": 334,
      "text": " (",
      "score": 3.5625
    },
    {
      "rank": 17,
      "token_id": 1228,
      "text": " result",
      "score": 3.546875
    },
    {
      "rank": 18,
      "token_id": 17079,
      "text": " {{",
      "score": 3.515625
    },
    {
      "rank": 19,
      "token_id": 9678,
      "text": " implementation",
      "score": 3.515625
    },
    {
      "rank": 20,
      "token_id": 4813,
      "text": " ~",
      "score": 3.390625
    }
  ]
}
```

```json
{
  "case_id": "sample_10",
  "call": "f('')",
  "expected": "''",
  "answer": "''",
  "correct": true,
  "location": "prompt_end",
  "read_token_index": 86,
  "read_token_text": " ==",
  "a_cone": 1.6577659527939816,
  "ulens_score": 1.1817344649733899,
  "category_cosines": {
    "ambiguity": 0.06327053159475327,
    "incompleteness": -0.07386493682861328,
    "conflicting_evidence": -0.037273772060871124,
    "general_uncertainty": -0.008689546957612038
  },
  "top_uspace_tokens": [
    {
      "rank": 1,
      "token_id": 12394,
      "text": " \ufffd",
      "score": 25.375
    },
    {
      "rank": 2,
      "token_id": 10047,
      "text": "\ufffd",
      "score": 25.25
    },
    {
      "rank": 3,
      "token_id": 24122,
      "text": " gonna",
      "score": 19.625
    },
    {
      "rank": 4,
      "token_id": 22709,
      "text": " \ufffd",
      "score": 18.625
    },
    {
      "rank": 5,
      "token_id": 30052,
      "text": " lol",
      "score": 14.3125
    },
    {
      "rank": 6,
      "token_id": 24331,
      "text": " fuck",
      "score": 13.625
    },
    {
      "rank": 7,
      "token_id": 14176,
      "text": " anymore",
      "score": 13.3125
    },
    {
      "rank": 8,
      "token_id": 7678,
      "text": " strugg",
      "score": 12.5
    },
    {
      "rank": 9,
      "token_id": 24717,
      "text": " COVID",
      "score": 12.25
    },
    {
      "rank": 10,
      "token_id": 19149,
      "text": " Instagram",
      "score": 11.125
    },
    {
      "rank": 11,
      "token_id": 30475,
      "text": " musul",
      "score": 10.375
    },
    {
      "rank": 12,
      "token_id": 30727,
      "text": "\u5360\u6bd4",
      "score": 10.3125
    },
    {
      "rank": 13,
      "token_id": 31586,
      "text": " germans",
      "score": 10.3125
    },
    {
      "rank": 14,
      "token_id": 8632,
      "text": " impossible",
      "score": 10.25
    },
    {
      "rank": 15,
      "token_id": 22028,
      "text": " dumpster",
      "score": 10.0625
    },
    {
      "rank": 16,
      "token_id": 7045,
      "text": " couldn",
      "score": 10.0
    },
    {
      "rank": 17,
      "token_id": 22199,
      "text": "\u6210\u4e3a\u4e86",
      "score": 9.8125
    },
    {
      "rank": 18,
      "token_id": 6740,
      "text": " wasn",
      "score": 9.75
    },
    {
      "rank": 19,
      "token_id": 26754,
      "text": "\u65b0\u51a0",
      "score": 9.75
    },
    {
      "rank": 20,
      "token_id": 30603,
      "text": " fulfilled",
      "score": 9.5
    }
  ],
  "top_jlens_tokens": [
    {
      "rank": 1,
      "token_id": 2220,
      "text": " `",
      "score": 6.5
    },
    {
      "rank": 2,
      "token_id": 440,
      "text": " \"",
      "score": 5.4375
    },
    {
      "rank": 3,
      "token_id": 945,
      "text": " +",
      "score": 5.125
    },
    {
      "rank": 4,
      "token_id": 9678,
      "text": " implementation",
      "score": 5.0
    },
    {
      "rank": 5,
      "token_id": 1866,
      "text": " something",
      "score": 4.96875
    },
    {
      "rank": 6,
      "token_id": 6236,
      "text": " ``",
      "score": 4.96875
    },
    {
      "rank": 7,
      "token_id": 11181,
      "text": " somewhere",
      "score": 4.84375
    },
    {
      "rank": 8,
      "token_id": 1585,
      "text": " --",
      "score": 4.78125
    },
    {
      "rank": 9,
      "token_id": 8528,
      "text": " +=",
      "score": 4.78125
    },
    {
      "rank": 10,
      "token_id": 4052,
      "text": " expected",
      "score": 4.46875
    },
    {
      "rank": 11,
      "token_id": 4341,
      "text": " implement",
      "score": 4.21875
    },
    {
      "rank": 12,
      "token_id": 19423,
      "text": " TODO",
      "score": 4.1875
    },
    {
      "rank": 13,
      "token_id": 2044,
      "text": " =>",
      "score": 4.09375
    },
    {
      "rank": 14,
      "token_id": 19788,
      "text": " {\"",
      "score": 3.984375
    },
    {
      "rank": 15,
      "token_id": 32014,
      "text": "<\uff5cend\u2581of\u2581sentence\uff5c>",
      "score": 3.84375
    },
    {
      "rank": 16,
      "token_id": 1368,
      "text": " because",
      "score": 3.8125
    },
    {
      "rank": 17,
      "token_id": 26022,
      "text": " \"#",
      "score": 3.75
    },
    {
      "rank": 18,
      "token_id": 25,
      "text": ":",
      "score": 3.65625
    },
    {
      "rank": 19,
      "token_id": 371,
      "text": " $",
      "score": 3.53125
    },
    {
      "rank": 20,
      "token_id": 2956,
      "text": " &&",
      "score": 3.453125
    }
  ]
}
```
