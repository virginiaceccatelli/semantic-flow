# Execution prediction and U-Space

Model: `deepseek-coder-1.3b`. Block index: **6** (zero-based, post-block).
Read location: `prompt_end`. No fine-tuning or steering.

## Execution

Evaluated: 1345 / 1345 calls.
Program-macro accuracy: 0.3158482142857143; call accuracy: 0.316728624535316; invalid answers: 147.

## Uncertainty

Higher scores indicate greater measured uncertainty.

| Score | Eligible calls | Error AUROC | Actual retained coverage | Retained error rate |
|---|---:|---:|---:|---:|
| ulens_score | 1345 | 0.6400711122009533 | 0.8 | 0.6589219330855018 |
| a_cone | 1345 | 0.4991698467920326 | 0.8 | 0.6775092936802974 |
| mean_answer_entropy | 1345 | 0.6424926052506552 | 0.8 | 0.654275092936803 |
| mean_answer_nll | 1345 | 0.6306329088057543 | 0.8 | 0.6617100371747212 |

U-Space alignment (`a_cone`) uses the published formula. `ulens_score` is `a_cone * mean_answer_entropy`: a short-answer adaptation, not the published end-of-thinking protocol. Entropy covers tokens producing the answer, excluding EOS and newline-only termination. It is not a probability of error.

`top_uspace_tokens` decodes the signed U-Space projection through the J-lens (an exploratory extension). `top_jlens_tokens` decodes the complete residual. Both lists contain literal tokenizer tokens, including whitespace/fragments; neither establishes the cause of a failure.

## First ten evaluated examples

```json
{
  "case_id": "11:0",
  "call": "remove_Occ(\"hello\",\"l\")",
  "expected": "'heo'",
  "answer": "\"heo\"",
  "correct": true,
  "location": "prompt_end",
  "read_token_index": 151,
  "read_token_text": " ==",
  "a_cone": 1.955569952356661,
  "ulens_score": 0.780752366101428,
  "category_cosines": {
    "ambiguity": 0.07392416149377823,
    "incompleteness": -0.0631411150097847,
    "conflicting_evidence": -0.02402949146926403,
    "general_uncertainty": 0.01610925979912281
  },
  "top_uspace_tokens": [
    {
      "rank": 1,
      "token_id": 12394,
      "text": " \ufffd",
      "score": 27.75
    },
    {
      "rank": 2,
      "token_id": 10047,
      "text": "\ufffd",
      "score": 26.875
    },
    {
      "rank": 3,
      "token_id": 22709,
      "text": " \ufffd",
      "score": 20.875
    },
    {
      "rank": 4,
      "token_id": 24122,
      "text": " gonna",
      "score": 17.375
    },
    {
      "rank": 5,
      "token_id": 30052,
      "text": " lol",
      "score": 13.8125
    },
    {
      "rank": 6,
      "token_id": 24331,
      "text": " fuck",
      "score": 13.6875
    },
    {
      "rank": 7,
      "token_id": 26754,
      "text": "\u65b0\u51a0",
      "score": 13.25
    },
    {
      "rank": 8,
      "token_id": 22028,
      "text": " dumpster",
      "score": 12.75
    },
    {
      "rank": 9,
      "token_id": 24717,
      "text": " COVID",
      "score": 12.5625
    },
    {
      "rank": 10,
      "token_id": 30475,
      "text": " musul",
      "score": 12.5
    },
    {
      "rank": 11,
      "token_id": 31586,
      "text": " germans",
      "score": 12.3125
    },
    {
      "rank": 12,
      "token_id": 19149,
      "text": " Instagram",
      "score": 11.75
    },
    {
      "rank": 13,
      "token_id": 7678,
      "text": " strugg",
      "score": 11.75
    },
    {
      "rank": 14,
      "token_id": 6740,
      "text": " wasn",
      "score": 11.0
    },
    {
      "rank": 15,
      "token_id": 14176,
      "text": " anymore",
      "score": 10.9375
    },
    {
      "rank": 16,
      "token_id": 24950,
      "text": " medalla",
      "score": 10.9375
    },
    {
      "rank": 17,
      "token_id": 7045,
      "text": " couldn",
      "score": 10.375
    },
    {
      "rank": 18,
      "token_id": 30727,
      "text": "\u5360\u6bd4",
      "score": 10.1875
    },
    {
      "rank": 19,
      "token_id": 10193,
      "text": "\u75ab\u60c5",
      "score": 9.9375
    },
    {
      "rank": 20,
      "token_id": 3306,
      "text": " didn",
      "score": 9.875
    }
  ],
  "top_jlens_tokens": [
    {
      "rank": 1,
      "token_id": 19423,
      "text": " TODO",
      "score": 7.46875
    },
    {
      "rank": 2,
      "token_id": 2220,
      "text": " `",
      "score": 7.03125
    },
    {
      "rank": 3,
      "token_id": 1585,
      "text": " --",
      "score": 5.84375
    },
    {
      "rank": 4,
      "token_id": 17625,
      "text": " [\"",
      "score": 5.84375
    },
    {
      "rank": 5,
      "token_id": 8528,
      "text": " +=",
      "score": 5.75
    },
    {
      "rank": 6,
      "token_id": 8507,
      "text": " (\"",
      "score": 5.5625
    },
    {
      "rank": 7,
      "token_id": 440,
      "text": " \"",
      "score": 5.3125
    },
    {
      "rank": 8,
      "token_id": 945,
      "text": " +",
      "score": 4.90625
    },
    {
      "rank": 9,
      "token_id": 572,
      "text": " *",
      "score": 4.5625
    },
    {
      "rank": 10,
      "token_id": 4813,
      "text": " ~",
      "score": 4.375
    },
    {
      "rank": 11,
      "token_id": 31421,
      "text": " \"[",
      "score": 4.34375
    },
    {
      "rank": 12,
      "token_id": 6236,
      "text": " ``",
      "score": 4.34375
    },
    {
      "rank": 13,
      "token_id": 19788,
      "text": " {\"",
      "score": 4.25
    },
    {
      "rank": 14,
      "token_id": 11181,
      "text": " somewhere",
      "score": 4.0
    },
    {
      "rank": 15,
      "token_id": 21713,
      "text": " runtime",
      "score": 3.90625
    },
    {
      "rank": 16,
      "token_id": 23323,
      "text": " += ",
      "score": 3.671875
    },
    {
      "rank": 17,
      "token_id": 8690,
      "text": " API",
      "score": 3.640625
    },
    {
      "rank": 18,
      "token_id": 16352,
      "text": " elsewhere",
      "score": 3.625
    },
    {
      "rank": 19,
      "token_id": 19476,
      "text": " ([",
      "score": 3.609375
    },
    {
      "rank": 20,
      "token_id": 25,
      "text": ":",
      "score": 3.59375
    }
  ]
}
```

```json
{
  "case_id": "11:1",
  "call": "remove_Occ(\"abcda\",\"a\")",
  "expected": "'bcd'",
  "answer": "\"cd\"",
  "correct": false,
  "location": "prompt_end",
  "read_token_index": 152,
  "read_token_text": " ==",
  "a_cone": 1.9774473493578226,
  "ulens_score": 1.387283387582094,
  "category_cosines": {
    "ambiguity": 0.07308423519134521,
    "incompleteness": -0.06136290356516838,
    "conflicting_evidence": -0.021746275946497917,
    "general_uncertainty": 0.01764489710330963
  },
  "top_uspace_tokens": [
    {
      "rank": 1,
      "token_id": 12394,
      "text": " \ufffd",
      "score": 27.875
    },
    {
      "rank": 2,
      "token_id": 10047,
      "text": "\ufffd",
      "score": 27.0
    },
    {
      "rank": 3,
      "token_id": 22709,
      "text": " \ufffd",
      "score": 21.0
    },
    {
      "rank": 4,
      "token_id": 24122,
      "text": " gonna",
      "score": 17.125
    },
    {
      "rank": 5,
      "token_id": 30052,
      "text": " lol",
      "score": 13.75
    },
    {
      "rank": 6,
      "token_id": 24331,
      "text": " fuck",
      "score": 13.625
    },
    {
      "rank": 7,
      "token_id": 26754,
      "text": "\u65b0\u51a0",
      "score": 13.4375
    },
    {
      "rank": 8,
      "token_id": 22028,
      "text": " dumpster",
      "score": 12.9375
    },
    {
      "rank": 9,
      "token_id": 30475,
      "text": " musul",
      "score": 12.6875
    },
    {
      "rank": 10,
      "token_id": 24717,
      "text": " COVID",
      "score": 12.5
    },
    {
      "rank": 11,
      "token_id": 31586,
      "text": " germans",
      "score": 12.4375
    },
    {
      "rank": 12,
      "token_id": 19149,
      "text": " Instagram",
      "score": 11.75
    },
    {
      "rank": 13,
      "token_id": 7678,
      "text": " strugg",
      "score": 11.625
    },
    {
      "rank": 14,
      "token_id": 24950,
      "text": " medalla",
      "score": 11.125
    },
    {
      "rank": 15,
      "token_id": 6740,
      "text": " wasn",
      "score": 11.0625
    },
    {
      "rank": 16,
      "token_id": 14176,
      "text": " anymore",
      "score": 10.75
    },
    {
      "rank": 17,
      "token_id": 7045,
      "text": " couldn",
      "score": 10.375
    },
    {
      "rank": 18,
      "token_id": 30727,
      "text": "\u5360\u6bd4",
      "score": 10.1875
    },
    {
      "rank": 19,
      "token_id": 10193,
      "text": "\u75ab\u60c5",
      "score": 10.0
    },
    {
      "rank": 20,
      "token_id": 3306,
      "text": " didn",
      "score": 10.0
    }
  ],
  "top_jlens_tokens": [
    {
      "rank": 1,
      "token_id": 19423,
      "text": " TODO",
      "score": 7.21875
    },
    {
      "rank": 2,
      "token_id": 2220,
      "text": " `",
      "score": 7.125
    },
    {
      "rank": 3,
      "token_id": 17625,
      "text": " [\"",
      "score": 6.34375
    },
    {
      "rank": 4,
      "token_id": 8528,
      "text": " +=",
      "score": 6.03125
    },
    {
      "rank": 5,
      "token_id": 8507,
      "text": " (\"",
      "score": 5.84375
    },
    {
      "rank": 6,
      "token_id": 1585,
      "text": " --",
      "score": 5.8125
    },
    {
      "rank": 7,
      "token_id": 440,
      "text": " \"",
      "score": 5.59375
    },
    {
      "rank": 8,
      "token_id": 945,
      "text": " +",
      "score": 4.96875
    },
    {
      "rank": 9,
      "token_id": 31421,
      "text": " \"[",
      "score": 4.96875
    },
    {
      "rank": 10,
      "token_id": 6236,
      "text": " ``",
      "score": 4.71875
    },
    {
      "rank": 11,
      "token_id": 572,
      "text": " *",
      "score": 4.65625
    },
    {
      "rank": 12,
      "token_id": 19788,
      "text": " {\"",
      "score": 4.65625
    },
    {
      "rank": 13,
      "token_id": 4813,
      "text": " ~",
      "score": 4.375
    },
    {
      "rank": 14,
      "token_id": 19476,
      "text": " ([",
      "score": 4.25
    },
    {
      "rank": 15,
      "token_id": 21713,
      "text": " runtime",
      "score": 4.1875
    },
    {
      "rank": 16,
      "token_id": 6317,
      "text": "[\"",
      "score": 4.125
    },
    {
      "rank": 17,
      "token_id": 8690,
      "text": " API",
      "score": 4.0
    },
    {
      "rank": 18,
      "token_id": 23323,
      "text": " += ",
      "score": 3.96875
    },
    {
      "rank": 19,
      "token_id": 11181,
      "text": " somewhere",
      "score": 3.78125
    },
    {
      "rank": 20,
      "token_id": 16352,
      "text": " elsewhere",
      "score": 3.703125
    }
  ]
}
```

```json
{
  "case_id": "11:2",
  "call": "remove_Occ(\"PHP\",\"P\")",
  "expected": "'H'",
  "answer": "\"HPH\"",
  "correct": false,
  "location": "prompt_end",
  "read_token_index": 151,
  "read_token_text": " ==",
  "a_cone": 1.8483816254398344,
  "ulens_score": 1.4033875227309978,
  "category_cosines": {
    "ambiguity": 0.0646219402551651,
    "incompleteness": -0.06867395341396332,
    "conflicting_evidence": -0.02023553103208542,
    "general_uncertainty": 0.01102738082408905
  },
  "top_uspace_tokens": [
    {
      "rank": 1,
      "token_id": 12394,
      "text": " \ufffd",
      "score": 27.125
    },
    {
      "rank": 2,
      "token_id": 10047,
      "text": "\ufffd",
      "score": 26.625
    },
    {
      "rank": 3,
      "token_id": 22709,
      "text": " \ufffd",
      "score": 20.375
    },
    {
      "rank": 4,
      "token_id": 24122,
      "text": " gonna",
      "score": 18.375
    },
    {
      "rank": 5,
      "token_id": 30052,
      "text": " lol",
      "score": 13.75
    },
    {
      "rank": 6,
      "token_id": 24331,
      "text": " fuck",
      "score": 13.375
    },
    {
      "rank": 7,
      "token_id": 30475,
      "text": " musul",
      "score": 12.3125
    },
    {
      "rank": 8,
      "token_id": 24717,
      "text": " COVID",
      "score": 12.3125
    },
    {
      "rank": 9,
      "token_id": 26754,
      "text": "\u65b0\u51a0",
      "score": 12.0
    },
    {
      "rank": 10,
      "token_id": 7678,
      "text": " strugg",
      "score": 11.9375
    },
    {
      "rank": 11,
      "token_id": 14176,
      "text": " anymore",
      "score": 11.875
    },
    {
      "rank": 12,
      "token_id": 22028,
      "text": " dumpster",
      "score": 11.625
    },
    {
      "rank": 13,
      "token_id": 31586,
      "text": " germans",
      "score": 11.625
    },
    {
      "rank": 14,
      "token_id": 19149,
      "text": " Instagram",
      "score": 11.3125
    },
    {
      "rank": 15,
      "token_id": 30727,
      "text": "\u5360\u6bd4",
      "score": 10.6875
    },
    {
      "rank": 16,
      "token_id": 24950,
      "text": " medalla",
      "score": 10.5625
    },
    {
      "rank": 17,
      "token_id": 6740,
      "text": " wasn",
      "score": 10.125
    },
    {
      "rank": 18,
      "token_id": 7045,
      "text": " couldn",
      "score": 9.875
    },
    {
      "rank": 19,
      "token_id": 10193,
      "text": "\u75ab\u60c5",
      "score": 9.8125
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
      "token_id": 19423,
      "text": " TODO",
      "score": 6.78125
    },
    {
      "rank": 2,
      "token_id": 1585,
      "text": " --",
      "score": 5.75
    },
    {
      "rank": 3,
      "token_id": 440,
      "text": " \"",
      "score": 5.65625
    },
    {
      "rank": 4,
      "token_id": 2220,
      "text": " `",
      "score": 5.5
    },
    {
      "rank": 5,
      "token_id": 8528,
      "text": " +=",
      "score": 4.96875
    },
    {
      "rank": 6,
      "token_id": 8507,
      "text": " (\"",
      "score": 4.84375
    },
    {
      "rank": 7,
      "token_id": 945,
      "text": " +",
      "score": 4.84375
    },
    {
      "rank": 8,
      "token_id": 17625,
      "text": " [\"",
      "score": 4.4375
    },
    {
      "rank": 9,
      "token_id": 4813,
      "text": " ~",
      "score": 4.125
    },
    {
      "rank": 10,
      "token_id": 6236,
      "text": " ``",
      "score": 3.984375
    },
    {
      "rank": 11,
      "token_id": 9678,
      "text": " implementation",
      "score": 3.75
    },
    {
      "rank": 12,
      "token_id": 11181,
      "text": " somewhere",
      "score": 3.703125
    },
    {
      "rank": 13,
      "token_id": 572,
      "text": " *",
      "score": 3.703125
    },
    {
      "rank": 14,
      "token_id": 334,
      "text": " (",
      "score": 3.515625
    },
    {
      "rank": 15,
      "token_id": 21713,
      "text": " runtime",
      "score": 3.46875
    },
    {
      "rank": 16,
      "token_id": 4341,
      "text": " implement",
      "score": 3.34375
    },
    {
      "rank": 17,
      "token_id": 24531,
      "text": " foo",
      "score": 3.296875
    },
    {
      "rank": 18,
      "token_id": 25,
      "text": ":",
      "score": 3.234375
    },
    {
      "rank": 19,
      "token_id": 16352,
      "text": " elsewhere",
      "score": 3.140625
    },
    {
      "rank": 20,
      "token_id": 371,
      "text": " $",
      "score": 3.109375
    }
  ]
}
```

```json
{
  "case_id": "12:0",
  "call": "sort_matrix([[1, 2, 3], [2, 4, 5], [1, 1, 1]])",
  "expected": "[[1, 1, 1], [1, 2, 3], [2, 4, 5]]",
  "answer": "[[1, 1, 1], [1, 2, 3], [2, 4, 5]]",
  "correct": true,
  "location": "prompt_end",
  "read_token_index": 78,
  "read_token_text": " ==",
  "a_cone": 1.7159979480226315,
  "ulens_score": 0.19732951922923497,
  "category_cosines": {
    "ambiguity": 0.04787827283143997,
    "incompleteness": -0.060408808290958405,
    "conflicting_evidence": -0.011924827471375465,
    "general_uncertainty": -0.00466985534876585
  },
  "top_uspace_tokens": [
    {
      "rank": 1,
      "token_id": 12394,
      "text": " \ufffd",
      "score": 25.625
    },
    {
      "rank": 2,
      "token_id": 10047,
      "text": "\ufffd",
      "score": 25.625
    },
    {
      "rank": 3,
      "token_id": 22709,
      "text": " \ufffd",
      "score": 19.625
    },
    {
      "rank": 4,
      "token_id": 24122,
      "text": " gonna",
      "score": 19.25
    },
    {
      "rank": 5,
      "token_id": 30052,
      "text": " lol",
      "score": 13.75
    },
    {
      "rank": 6,
      "token_id": 14176,
      "text": " anymore",
      "score": 13.5
    },
    {
      "rank": 7,
      "token_id": 24331,
      "text": " fuck",
      "score": 12.9375
    },
    {
      "rank": 8,
      "token_id": 30475,
      "text": " musul",
      "score": 12.3125
    },
    {
      "rank": 9,
      "token_id": 24717,
      "text": " COVID",
      "score": 11.9375
    },
    {
      "rank": 10,
      "token_id": 7678,
      "text": " strugg",
      "score": 11.875
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
      "score": 10.6875
    },
    {
      "rank": 13,
      "token_id": 7045,
      "text": " couldn",
      "score": 9.8125
    },
    {
      "rank": 14,
      "token_id": 31586,
      "text": " germans",
      "score": 9.75
    },
    {
      "rank": 15,
      "token_id": 24950,
      "text": " medalla",
      "score": 9.75
    },
    {
      "rank": 16,
      "token_id": 26754,
      "text": "\u65b0\u51a0",
      "score": 9.75
    },
    {
      "rank": 17,
      "token_id": 9482,
      "text": " => ",
      "score": 9.5625
    },
    {
      "rank": 18,
      "token_id": 7361,
      "text": " wouldn",
      "score": 9.4375
    },
    {
      "rank": 19,
      "token_id": 10193,
      "text": "\u75ab\u60c5",
      "score": 9.25
    },
    {
      "rank": 20,
      "token_id": 6740,
      "text": " wasn",
      "score": 9.1875
    }
  ],
  "top_jlens_tokens": [
    {
      "rank": 1,
      "token_id": 185,
      "text": "\n",
      "score": 9.4375
    },
    {
      "rank": 2,
      "token_id": 32014,
      "text": "<\uff5cend\u2581of\u2581sentence\uff5c>",
      "score": 9.3125
    },
    {
      "rank": 3,
      "token_id": 16352,
      "text": " elsewhere",
      "score": 8.5625
    },
    {
      "rank": 4,
      "token_id": 1368,
      "text": " because",
      "score": 7.53125
    },
    {
      "rank": 5,
      "token_id": 25,
      "text": ":",
      "score": 7.46875
    },
    {
      "rank": 6,
      "token_id": 11181,
      "text": " somewhere",
      "score": 7.0
    },
    {
      "rank": 7,
      "token_id": 1585,
      "text": " --",
      "score": 6.9375
    },
    {
      "rank": 8,
      "token_id": 182,
      "text": "\u0007",
      "score": 6.875
    },
    {
      "rank": 9,
      "token_id": 6236,
      "text": " ``",
      "score": 6.46875
    },
    {
      "rank": 10,
      "token_id": 16603,
      "text": " [[",
      "score": 6.46875
    },
    {
      "rank": 11,
      "token_id": 17079,
      "text": " {{",
      "score": 6.375
    },
    {
      "rank": 12,
      "token_id": 8528,
      "text": " +=",
      "score": 6.09375
    },
    {
      "rank": 13,
      "token_id": 2330,
      "text": " {\\",
      "score": 5.9375
    },
    {
      "rank": 14,
      "token_id": 19423,
      "text": " TODO",
      "score": 5.875
    },
    {
      "rank": 15,
      "token_id": 31421,
      "text": " \"[",
      "score": 5.8125
    },
    {
      "rank": 16,
      "token_id": 17625,
      "text": " [\"",
      "score": 5.65625
    },
    {
      "rank": 17,
      "token_id": 4052,
      "text": " expected",
      "score": 5.625
    },
    {
      "rank": 18,
      "token_id": 440,
      "text": " \"",
      "score": 5.625
    },
    {
      "rank": 19,
      "token_id": 1866,
      "text": " something",
      "score": 5.59375
    },
    {
      "rank": 20,
      "token_id": 2220,
      "text": " `",
      "score": 5.53125
    }
  ]
}
```

```json
{
  "case_id": "12:1",
  "call": "sort_matrix([[1, 2, 3], [-2, 4, -5], [1, -1, 1]])",
  "expected": "[[-2, 4, -5], [1, -1, 1], [1, 2, 3]]",
  "answer": "[[-2, 4, -5], [1, -1, 1], [1, 2, 3]]",
  "correct": true,
  "location": "prompt_end",
  "read_token_index": 78,
  "read_token_text": " ==",
  "a_cone": 1.7079187360079404,
  "ulens_score": 0.20114696572421745,
  "category_cosines": {
    "ambiguity": 0.04774751514196396,
    "incompleteness": -0.06075296178460121,
    "conflicting_evidence": -0.011078763753175735,
    "general_uncertainty": -0.006293441168963909
  },
  "top_uspace_tokens": [
    {
      "rank": 1,
      "token_id": 10047,
      "text": "\ufffd",
      "score": 25.5
    },
    {
      "rank": 2,
      "token_id": 12394,
      "text": " \ufffd",
      "score": 25.375
    },
    {
      "rank": 3,
      "token_id": 22709,
      "text": " \ufffd",
      "score": 19.5
    },
    {
      "rank": 4,
      "token_id": 24122,
      "text": " gonna",
      "score": 19.25
    },
    {
      "rank": 5,
      "token_id": 30052,
      "text": " lol",
      "score": 13.75
    },
    {
      "rank": 6,
      "token_id": 14176,
      "text": " anymore",
      "score": 13.6875
    },
    {
      "rank": 7,
      "token_id": 24331,
      "text": " fuck",
      "score": 12.875
    },
    {
      "rank": 8,
      "token_id": 30475,
      "text": " musul",
      "score": 12.375
    },
    {
      "rank": 9,
      "token_id": 24717,
      "text": " COVID",
      "score": 11.9375
    },
    {
      "rank": 10,
      "token_id": 7678,
      "text": " strugg",
      "score": 11.75
    },
    {
      "rank": 11,
      "token_id": 19149,
      "text": " Instagram",
      "score": 10.875
    },
    {
      "rank": 12,
      "token_id": 30727,
      "text": "\u5360\u6bd4",
      "score": 10.6875
    },
    {
      "rank": 13,
      "token_id": 7045,
      "text": " couldn",
      "score": 9.8125
    },
    {
      "rank": 14,
      "token_id": 24950,
      "text": " medalla",
      "score": 9.6875
    },
    {
      "rank": 15,
      "token_id": 9482,
      "text": " => ",
      "score": 9.625
    },
    {
      "rank": 16,
      "token_id": 26754,
      "text": "\u65b0\u51a0",
      "score": 9.5625
    },
    {
      "rank": 17,
      "token_id": 31586,
      "text": " germans",
      "score": 9.5
    },
    {
      "rank": 18,
      "token_id": 7361,
      "text": " wouldn",
      "score": 9.4375
    },
    {
      "rank": 19,
      "token_id": 31804,
      "text": " Nigeria",
      "score": 9.3125
    },
    {
      "rank": 20,
      "token_id": 22199,
      "text": "\u6210\u4e3a\u4e86",
      "score": 9.25
    }
  ],
  "top_jlens_tokens": [
    {
      "rank": 1,
      "token_id": 185,
      "text": "\n",
      "score": 10.1875
    },
    {
      "rank": 2,
      "token_id": 32014,
      "text": "<\uff5cend\u2581of\u2581sentence\uff5c>",
      "score": 9.6875
    },
    {
      "rank": 3,
      "token_id": 16352,
      "text": " elsewhere",
      "score": 8.375
    },
    {
      "rank": 4,
      "token_id": 1368,
      "text": " because",
      "score": 7.65625
    },
    {
      "rank": 5,
      "token_id": 25,
      "text": ":",
      "score": 7.5625
    },
    {
      "rank": 6,
      "token_id": 11181,
      "text": " somewhere",
      "score": 6.90625
    },
    {
      "rank": 7,
      "token_id": 182,
      "text": "\u0007",
      "score": 6.78125
    },
    {
      "rank": 8,
      "token_id": 1585,
      "text": " --",
      "score": 6.78125
    },
    {
      "rank": 9,
      "token_id": 16603,
      "text": " [[",
      "score": 6.5
    },
    {
      "rank": 10,
      "token_id": 17079,
      "text": " {{",
      "score": 6.5
    },
    {
      "rank": 11,
      "token_id": 6236,
      "text": " ``",
      "score": 6.15625
    },
    {
      "rank": 12,
      "token_id": 8528,
      "text": " +=",
      "score": 6.09375
    },
    {
      "rank": 13,
      "token_id": 2330,
      "text": " {\\",
      "score": 6.09375
    },
    {
      "rank": 14,
      "token_id": 31421,
      "text": " \"[",
      "score": 5.9375
    },
    {
      "rank": 15,
      "token_id": 19423,
      "text": " TODO",
      "score": 5.875
    },
    {
      "rank": 16,
      "token_id": 26,
      "text": ";",
      "score": 5.75
    },
    {
      "rank": 17,
      "token_id": 4052,
      "text": " expected",
      "score": 5.65625
    },
    {
      "rank": 18,
      "token_id": 440,
      "text": " \"",
      "score": 5.625
    },
    {
      "rank": 19,
      "token_id": 1866,
      "text": " something",
      "score": 5.5625
    },
    {
      "rank": 20,
      "token_id": 17625,
      "text": " [\"",
      "score": 5.5625
    }
  ]
}
```

```json
{
  "case_id": "12:2",
  "call": "sort_matrix([[5,8,9],[6,4,3],[2,1,4]])",
  "expected": "[[2, 1, 4], [6, 4, 3], [5, 8, 9]]",
  "answer": "[[2,1,4],[6,4,3],[5,8,9]]",
  "correct": true,
  "location": "prompt_end",
  "read_token_index": 72,
  "read_token_text": " ==",
  "a_cone": 1.7000756231812244,
  "ulens_score": 0.40973623172142687,
  "category_cosines": {
    "ambiguity": 0.047304537147283554,
    "incompleteness": -0.05954533442854881,
    "conflicting_evidence": -0.01592390425503254,
    "general_uncertainty": -0.003912868909537792
  },
  "top_uspace_tokens": [
    {
      "rank": 1,
      "token_id": 12394,
      "text": " \ufffd",
      "score": 25.625
    },
    {
      "rank": 2,
      "token_id": 10047,
      "text": "\ufffd",
      "score": 25.625
    },
    {
      "rank": 3,
      "token_id": 22709,
      "text": " \ufffd",
      "score": 19.375
    },
    {
      "rank": 4,
      "token_id": 24122,
      "text": " gonna",
      "score": 19.375
    },
    {
      "rank": 5,
      "token_id": 30052,
      "text": " lol",
      "score": 13.875
    },
    {
      "rank": 6,
      "token_id": 14176,
      "text": " anymore",
      "score": 13.375
    },
    {
      "rank": 7,
      "token_id": 24331,
      "text": " fuck",
      "score": 13.125
    },
    {
      "rank": 8,
      "token_id": 7678,
      "text": " strugg",
      "score": 12.0625
    },
    {
      "rank": 9,
      "token_id": 24717,
      "text": " COVID",
      "score": 12.0
    },
    {
      "rank": 10,
      "token_id": 30475,
      "text": " musul",
      "score": 11.8125
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
      "score": 10.6875
    },
    {
      "rank": 13,
      "token_id": 31586,
      "text": " germans",
      "score": 10.0
    },
    {
      "rank": 14,
      "token_id": 26754,
      "text": "\u65b0\u51a0",
      "score": 9.8125
    },
    {
      "rank": 15,
      "token_id": 7045,
      "text": " couldn",
      "score": 9.75
    },
    {
      "rank": 16,
      "token_id": 24950,
      "text": " medalla",
      "score": 9.5625
    },
    {
      "rank": 17,
      "token_id": 9482,
      "text": " => ",
      "score": 9.5
    },
    {
      "rank": 18,
      "token_id": 7361,
      "text": " wouldn",
      "score": 9.4375
    },
    {
      "rank": 19,
      "token_id": 22028,
      "text": " dumpster",
      "score": 9.375
    },
    {
      "rank": 20,
      "token_id": 6740,
      "text": " wasn",
      "score": 9.3125
    }
  ],
  "top_jlens_tokens": [
    {
      "rank": 1,
      "token_id": 16352,
      "text": " elsewhere",
      "score": 8.75
    },
    {
      "rank": 2,
      "token_id": 32014,
      "text": "<\uff5cend\u2581of\u2581sentence\uff5c>",
      "score": 8.4375
    },
    {
      "rank": 3,
      "token_id": 25,
      "text": ":",
      "score": 7.375
    },
    {
      "rank": 4,
      "token_id": 1585,
      "text": " --",
      "score": 7.3125
    },
    {
      "rank": 5,
      "token_id": 1368,
      "text": " because",
      "score": 7.21875
    },
    {
      "rank": 6,
      "token_id": 11181,
      "text": " somewhere",
      "score": 7.09375
    },
    {
      "rank": 7,
      "token_id": 185,
      "text": "\n",
      "score": 6.78125
    },
    {
      "rank": 8,
      "token_id": 6236,
      "text": " ``",
      "score": 6.625
    },
    {
      "rank": 9,
      "token_id": 8528,
      "text": " +=",
      "score": 6.28125
    },
    {
      "rank": 10,
      "token_id": 440,
      "text": " \"",
      "score": 6.21875
    },
    {
      "rank": 11,
      "token_id": 2220,
      "text": " `",
      "score": 6.125
    },
    {
      "rank": 12,
      "token_id": 182,
      "text": "\u0007",
      "score": 6.09375
    },
    {
      "rank": 13,
      "token_id": 19423,
      "text": " TODO",
      "score": 5.96875
    },
    {
      "rank": 14,
      "token_id": 8507,
      "text": " (\"",
      "score": 5.875
    },
    {
      "rank": 15,
      "token_id": 4052,
      "text": " expected",
      "score": 5.84375
    },
    {
      "rank": 16,
      "token_id": 17625,
      "text": " [\"",
      "score": 5.65625
    },
    {
      "rank": 17,
      "token_id": 1866,
      "text": " something",
      "score": 5.59375
    },
    {
      "rank": 18,
      "token_id": 16603,
      "text": " [[",
      "score": 5.5625
    },
    {
      "rank": 19,
      "token_id": 31421,
      "text": " \"[",
      "score": 5.53125
    },
    {
      "rank": 20,
      "token_id": 24531,
      "text": " foo",
      "score": 5.5
    }
  ]
}
```

```json
{
  "case_id": "13:0",
  "call": "count_common(['red','green','black','pink','black','white','black','eyes','white','black','orange','pink','pink','red','red','white','orange','white',\"black\",'pink','green','green','pink','green','pink','white','orange',\"orange\",'red'])",
  "expected": "[('pink', 6), ('black', 5), ('white', 5), ('red', 4)]",
  "answer": "[('pink', 4), ('white', 4), ('black', 4), ('red', 4)]",
  "correct": false,
  "location": "prompt_end",
  "read_token_index": 147,
  "read_token_text": " ==",
  "a_cone": 1.5656399516257125,
  "ulens_score": 0.6383642770193106,
  "category_cosines": {
    "ambiguity": 0.04832672327756882,
    "incompleteness": -0.07411137968301773,
    "conflicting_evidence": -0.023017829284071922,
    "general_uncertainty": -0.008489919826388359
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
      "score": 13.6875
    },
    {
      "rank": 6,
      "token_id": 30052,
      "text": " lol",
      "score": 13.625
    },
    {
      "rank": 7,
      "token_id": 24331,
      "text": " fuck",
      "score": 12.6875
    },
    {
      "rank": 8,
      "token_id": 7678,
      "text": " strugg",
      "score": 12.1875
    },
    {
      "rank": 9,
      "token_id": 24717,
      "text": " COVID",
      "score": 11.5625
    },
    {
      "rank": 10,
      "token_id": 30727,
      "text": "\u5360\u6bd4",
      "score": 10.6875
    },
    {
      "rank": 11,
      "token_id": 30475,
      "text": " musul",
      "score": 10.625
    },
    {
      "rank": 12,
      "token_id": 19149,
      "text": " Instagram",
      "score": 10.3125
    },
    {
      "rank": 13,
      "token_id": 22199,
      "text": "\u6210\u4e3a\u4e86",
      "score": 9.75
    },
    {
      "rank": 14,
      "token_id": 31804,
      "text": " Nigeria",
      "score": 9.75
    },
    {
      "rank": 15,
      "token_id": 29881,
      "text": " anytime",
      "score": 9.5
    },
    {
      "rank": 16,
      "token_id": 9482,
      "text": " => ",
      "score": 9.4375
    },
    {
      "rank": 17,
      "token_id": 31586,
      "text": " germans",
      "score": 9.375
    },
    {
      "rank": 18,
      "token_id": 30603,
      "text": " fulfilled",
      "score": 9.1875
    },
    {
      "rank": 19,
      "token_id": 18268,
      "text": " scenarios",
      "score": 9.1875
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
      "token_id": 32014,
      "text": "<\uff5cend\u2581of\u2581sentence\uff5c>",
      "score": 11.3125
    },
    {
      "rank": 2,
      "token_id": 185,
      "text": "\n",
      "score": 9.25
    },
    {
      "rank": 3,
      "token_id": 1368,
      "text": " because",
      "score": 5.3125
    },
    {
      "rank": 4,
      "token_id": 1866,
      "text": " something",
      "score": 5.25
    },
    {
      "rank": 5,
      "token_id": 11181,
      "text": " somewhere",
      "score": 4.96875
    },
    {
      "rank": 6,
      "token_id": 25,
      "text": ":",
      "score": 4.875
    },
    {
      "rank": 7,
      "token_id": 440,
      "text": " \"",
      "score": 4.59375
    },
    {
      "rank": 8,
      "token_id": 19423,
      "text": " TODO",
      "score": 4.5
    },
    {
      "rank": 9,
      "token_id": 9678,
      "text": " implementation",
      "score": 4.34375
    },
    {
      "rank": 10,
      "token_id": 16352,
      "text": " elsewhere",
      "score": 4.3125
    },
    {
      "rank": 11,
      "token_id": 4341,
      "text": " implement",
      "score": 4.125
    },
    {
      "rank": 12,
      "token_id": 334,
      "text": " (",
      "score": 4.03125
    },
    {
      "rank": 13,
      "token_id": 12766,
      "text": " implemented",
      "score": 3.84375
    },
    {
      "rank": 14,
      "token_id": 26,
      "text": ";",
      "score": 3.5
    },
    {
      "rank": 15,
      "token_id": 4052,
      "text": " expected",
      "score": 3.390625
    },
    {
      "rank": 16,
      "token_id": 3552,
      "text": " ...",
      "score": 3.296875
    },
    {
      "rank": 17,
      "token_id": 26702,
      "text": " (#",
      "score": 3.15625
    },
    {
      "rank": 18,
      "token_id": 8528,
      "text": " +=",
      "score": 3.125
    },
    {
      "rank": 19,
      "token_id": 945,
      "text": " +",
      "score": 3.015625
    },
    {
      "rank": 20,
      "token_id": 1585,
      "text": " --",
      "score": 2.921875
    }
  ]
}
```

```json
{
  "case_id": "13:1",
  "call": "count_common(['one', 'two', 'three', 'four', 'five', 'one', 'two', 'one', 'three', 'one'])",
  "expected": "[('one', 4), ('two', 2), ('three', 2), ('four', 1)]",
  "answer": "[('one', 4), ('two', 3), ('three', 2), ('four', 1)]",
  "correct": false,
  "location": "prompt_end",
  "read_token_index": 108,
  "read_token_text": " ==",
  "a_cone": 1.5645723276313297,
  "ulens_score": 0.3026272877497063,
  "category_cosines": {
    "ambiguity": 0.04663136601448059,
    "incompleteness": -0.07657942175865173,
    "conflicting_evidence": -0.01571686565876007,
    "general_uncertainty": -0.008341426961123943
  },
  "top_uspace_tokens": [
    {
      "rank": 1,
      "token_id": 10047,
      "text": "\ufffd",
      "score": 24.375
    },
    {
      "rank": 2,
      "token_id": 12394,
      "text": " \ufffd",
      "score": 24.0
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
      "score": 18.125
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
      "score": 13.25
    },
    {
      "rank": 7,
      "token_id": 24331,
      "text": " fuck",
      "score": 12.25
    },
    {
      "rank": 8,
      "token_id": 7678,
      "text": " strugg",
      "score": 11.9375
    },
    {
      "rank": 9,
      "token_id": 24717,
      "text": " COVID",
      "score": 11.25
    },
    {
      "rank": 10,
      "token_id": 30475,
      "text": " musul",
      "score": 11.25
    },
    {
      "rank": 11,
      "token_id": 30727,
      "text": "\u5360\u6bd4",
      "score": 10.875
    },
    {
      "rank": 12,
      "token_id": 19149,
      "text": " Instagram",
      "score": 10.0625
    },
    {
      "rank": 13,
      "token_id": 31804,
      "text": " Nigeria",
      "score": 9.875
    },
    {
      "rank": 14,
      "token_id": 22199,
      "text": "\u6210\u4e3a\u4e86",
      "score": 9.5625
    },
    {
      "rank": 15,
      "token_id": 29881,
      "text": " anytime",
      "score": 9.5625
    },
    {
      "rank": 16,
      "token_id": 9482,
      "text": " => ",
      "score": 9.5625
    },
    {
      "rank": 17,
      "token_id": 18268,
      "text": " scenarios",
      "score": 9.125
    },
    {
      "rank": 18,
      "token_id": 21409,
      "text": "\u00b4",
      "score": 9.0
    },
    {
      "rank": 19,
      "token_id": 31586,
      "text": " germans",
      "score": 9.0
    },
    {
      "rank": 20,
      "token_id": 30603,
      "text": " fulfilled",
      "score": 9.0
    }
  ],
  "top_jlens_tokens": [
    {
      "rank": 1,
      "token_id": 32014,
      "text": "<\uff5cend\u2581of\u2581sentence\uff5c>",
      "score": 11.8125
    },
    {
      "rank": 2,
      "token_id": 185,
      "text": "\n",
      "score": 11.4375
    },
    {
      "rank": 3,
      "token_id": 1368,
      "text": " because",
      "score": 5.25
    },
    {
      "rank": 4,
      "token_id": 25,
      "text": ":",
      "score": 4.84375
    },
    {
      "rank": 5,
      "token_id": 19423,
      "text": " TODO",
      "score": 4.8125
    },
    {
      "rank": 6,
      "token_id": 9678,
      "text": " implementation",
      "score": 4.78125
    },
    {
      "rank": 7,
      "token_id": 1866,
      "text": " something",
      "score": 4.65625
    },
    {
      "rank": 8,
      "token_id": 16352,
      "text": " elsewhere",
      "score": 4.65625
    },
    {
      "rank": 9,
      "token_id": 11181,
      "text": " somewhere",
      "score": 4.53125
    },
    {
      "rank": 10,
      "token_id": 440,
      "text": " \"",
      "score": 4.5
    },
    {
      "rank": 11,
      "token_id": 4341,
      "text": " implement",
      "score": 4.28125
    },
    {
      "rank": 12,
      "token_id": 12766,
      "text": " implemented",
      "score": 4.1875
    },
    {
      "rank": 13,
      "token_id": 26,
      "text": ";",
      "score": 4.09375
    },
    {
      "rank": 14,
      "token_id": 8528,
      "text": " +=",
      "score": 3.875
    },
    {
      "rank": 15,
      "token_id": 26702,
      "text": " (#",
      "score": 3.859375
    },
    {
      "rank": 16,
      "token_id": 16603,
      "text": " [[",
      "score": 3.796875
    },
    {
      "rank": 17,
      "token_id": 357,
      "text": " \\",
      "score": 3.59375
    },
    {
      "rank": 18,
      "token_id": 26022,
      "text": " \"#",
      "score": 3.578125
    },
    {
      "rank": 19,
      "token_id": 334,
      "text": " (",
      "score": 3.5625
    },
    {
      "rank": 20,
      "token_id": 4052,
      "text": " expected",
      "score": 3.515625
    }
  ]
}
```

```json
{
  "case_id": "13:2",
  "call": "count_common(['Facebook', 'Apple', 'Amazon', 'Netflix', 'Google', 'Apple', 'Netflix', 'Amazon'])",
  "expected": "[('Apple', 2), ('Amazon', 2), ('Netflix', 2), ('Facebook', 1)]",
  "answer": "[('Amazon', 2), ('Netflix', 2), ('Google', 1), ('Facebook', 1)]",
  "correct": false,
  "location": "prompt_end",
  "read_token_index": 107,
  "read_token_text": " ==",
  "a_cone": 1.5011963492967777,
  "ulens_score": 0.3794405805574731,
  "category_cosines": {
    "ambiguity": 0.043537843972444534,
    "incompleteness": -0.07552319020032883,
    "conflicting_evidence": -0.016640378162264824,
    "general_uncertainty": -0.015518114902079105
  },
  "top_uspace_tokens": [
    {
      "rank": 1,
      "token_id": 10047,
      "text": "\ufffd",
      "score": 23.625
    },
    {
      "rank": 2,
      "token_id": 12394,
      "text": " \ufffd",
      "score": 23.0
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
      "score": 17.375
    },
    {
      "rank": 5,
      "token_id": 14176,
      "text": " anymore",
      "score": 14.25
    },
    {
      "rank": 6,
      "token_id": 30052,
      "text": " lol",
      "score": 13.25
    },
    {
      "rank": 7,
      "token_id": 24331,
      "text": " fuck",
      "score": 12.0625
    },
    {
      "rank": 8,
      "token_id": 7678,
      "text": " strugg",
      "score": 11.8125
    },
    {
      "rank": 9,
      "token_id": 24717,
      "text": " COVID",
      "score": 11.0625
    },
    {
      "rank": 10,
      "token_id": 30475,
      "text": " musul",
      "score": 10.75
    },
    {
      "rank": 11,
      "token_id": 30727,
      "text": "\u5360\u6bd4",
      "score": 10.625
    },
    {
      "rank": 12,
      "token_id": 31804,
      "text": " Nigeria",
      "score": 10.5625
    },
    {
      "rank": 13,
      "token_id": 22199,
      "text": "\u6210\u4e3a\u4e86",
      "score": 10.125
    },
    {
      "rank": 14,
      "token_id": 29881,
      "text": " anytime",
      "score": 10.0
    },
    {
      "rank": 15,
      "token_id": 19149,
      "text": " Instagram",
      "score": 9.875
    },
    {
      "rank": 16,
      "token_id": 21409,
      "text": "\u00b4",
      "score": 9.8125
    },
    {
      "rank": 17,
      "token_id": 26696,
      "text": " utilizing",
      "score": 9.6875
    },
    {
      "rank": 18,
      "token_id": 10273,
      "text": "!!!",
      "score": 9.625
    },
    {
      "rank": 19,
      "token_id": 18985,
      "text": " dont",
      "score": 9.5625
    },
    {
      "rank": 20,
      "token_id": 9482,
      "text": " => ",
      "score": 9.5625
    }
  ],
  "top_jlens_tokens": [
    {
      "rank": 1,
      "token_id": 32014,
      "text": "<\uff5cend\u2581of\u2581sentence\uff5c>",
      "score": 12.6875
    },
    {
      "rank": 2,
      "token_id": 185,
      "text": "\n",
      "score": 11.875
    },
    {
      "rank": 3,
      "token_id": 1368,
      "text": " because",
      "score": 5.15625
    },
    {
      "rank": 4,
      "token_id": 9678,
      "text": " implementation",
      "score": 4.96875
    },
    {
      "rank": 5,
      "token_id": 1866,
      "text": " something",
      "score": 4.65625
    },
    {
      "rank": 6,
      "token_id": 19423,
      "text": " TODO",
      "score": 4.59375
    },
    {
      "rank": 7,
      "token_id": 440,
      "text": " \"",
      "score": 4.5
    },
    {
      "rank": 8,
      "token_id": 12766,
      "text": " implemented",
      "score": 4.5
    },
    {
      "rank": 9,
      "token_id": 4341,
      "text": " implement",
      "score": 4.40625
    },
    {
      "rank": 10,
      "token_id": 25,
      "text": ":",
      "score": 4.21875
    },
    {
      "rank": 11,
      "token_id": 11181,
      "text": " somewhere",
      "score": 4.15625
    },
    {
      "rank": 12,
      "token_id": 26,
      "text": ";",
      "score": 4.0625
    },
    {
      "rank": 13,
      "token_id": 16352,
      "text": " elsewhere",
      "score": 3.9375
    },
    {
      "rank": 14,
      "token_id": 357,
      "text": " \\",
      "score": 3.78125
    },
    {
      "rank": 15,
      "token_id": 16603,
      "text": " [[",
      "score": 3.65625
    },
    {
      "rank": 16,
      "token_id": 334,
      "text": " (",
      "score": 3.59375
    },
    {
      "rank": 17,
      "token_id": 8528,
      "text": " +=",
      "score": 3.0625
    },
    {
      "rank": 18,
      "token_id": 26022,
      "text": " \"#",
      "score": 3.03125
    },
    {
      "rank": 19,
      "token_id": 31421,
      "text": " \"[",
      "score": 2.90625
    },
    {
      "rank": 20,
      "token_id": 26702,
      "text": " (#",
      "score": 2.890625
    }
  ]
}
```

```json
{
  "case_id": "15:0",
  "call": "split_lowerstring(\"AbCd\")",
  "expected": "['bC', 'd']",
  "answer": "['Ab', 'Cd']",
  "correct": false,
  "location": "prompt_end",
  "read_token_index": 65,
  "read_token_text": " ==",
  "a_cone": 1.82468867360849,
  "ulens_score": 1.1013550558787066,
  "category_cosines": {
    "ambiguity": 0.06198107823729515,
    "incompleteness": -0.06372816860675812,
    "conflicting_evidence": -0.01956801861524582,
    "general_uncertainty": 0.0014131144853308797
  },
  "top_uspace_tokens": [
    {
      "rank": 1,
      "token_id": 12394,
      "text": " \ufffd",
      "score": 26.875
    },
    {
      "rank": 2,
      "token_id": 10047,
      "text": "\ufffd",
      "score": 26.5
    },
    {
      "rank": 3,
      "token_id": 22709,
      "text": " \ufffd",
      "score": 20.375
    },
    {
      "rank": 4,
      "token_id": 24122,
      "text": " gonna",
      "score": 18.75
    },
    {
      "rank": 5,
      "token_id": 30052,
      "text": " lol",
      "score": 14.125
    },
    {
      "rank": 6,
      "token_id": 24331,
      "text": " fuck",
      "score": 13.5625
    },
    {
      "rank": 7,
      "token_id": 14176,
      "text": " anymore",
      "score": 12.6875
    },
    {
      "rank": 8,
      "token_id": 24717,
      "text": " COVID",
      "score": 12.4375
    },
    {
      "rank": 9,
      "token_id": 30475,
      "text": " musul",
      "score": 12.375
    },
    {
      "rank": 10,
      "token_id": 7678,
      "text": " strugg",
      "score": 12.0
    },
    {
      "rank": 11,
      "token_id": 19149,
      "text": " Instagram",
      "score": 11.5
    },
    {
      "rank": 12,
      "token_id": 26754,
      "text": "\u65b0\u51a0",
      "score": 11.375
    },
    {
      "rank": 13,
      "token_id": 31586,
      "text": " germans",
      "score": 10.9375
    },
    {
      "rank": 14,
      "token_id": 22028,
      "text": " dumpster",
      "score": 10.75
    },
    {
      "rank": 15,
      "token_id": 30727,
      "text": "\u5360\u6bd4",
      "score": 10.5
    },
    {
      "rank": 16,
      "token_id": 7045,
      "text": " couldn",
      "score": 10.375
    },
    {
      "rank": 17,
      "token_id": 6740,
      "text": " wasn",
      "score": 10.3125
    },
    {
      "rank": 18,
      "token_id": 24950,
      "text": " medalla",
      "score": 10.25
    },
    {
      "rank": 19,
      "token_id": 8632,
      "text": " impossible",
      "score": 10.0
    },
    {
      "rank": 20,
      "token_id": 7361,
      "text": " wouldn",
      "score": 9.75
    }
  ],
  "top_jlens_tokens": [
    {
      "rank": 1,
      "token_id": 2220,
      "text": " `",
      "score": 7.0625
    },
    {
      "rank": 2,
      "token_id": 1585,
      "text": " --",
      "score": 7.03125
    },
    {
      "rank": 3,
      "token_id": 19423,
      "text": " TODO",
      "score": 6.6875
    },
    {
      "rank": 4,
      "token_id": 17625,
      "text": " [\"",
      "score": 6.59375
    },
    {
      "rank": 5,
      "token_id": 8528,
      "text": " +=",
      "score": 6.15625
    },
    {
      "rank": 6,
      "token_id": 8507,
      "text": " (\"",
      "score": 6.03125
    },
    {
      "rank": 7,
      "token_id": 440,
      "text": " \"",
      "score": 5.75
    },
    {
      "rank": 8,
      "token_id": 1368,
      "text": " because",
      "score": 5.71875
    },
    {
      "rank": 9,
      "token_id": 19788,
      "text": " {\"",
      "score": 5.5625
    },
    {
      "rank": 10,
      "token_id": 11181,
      "text": " somewhere",
      "score": 5.5625
    },
    {
      "rank": 11,
      "token_id": 6236,
      "text": " ``",
      "score": 5.375
    },
    {
      "rank": 12,
      "token_id": 32014,
      "text": "<\uff5cend\u2581of\u2581sentence\uff5c>",
      "score": 5.125
    },
    {
      "rank": 13,
      "token_id": 31421,
      "text": " \"[",
      "score": 5.0
    },
    {
      "rank": 14,
      "token_id": 26073,
      "text": " Git",
      "score": 4.96875
    },
    {
      "rank": 15,
      "token_id": 6317,
      "text": "[\"",
      "score": 4.90625
    },
    {
      "rank": 16,
      "token_id": 945,
      "text": " +",
      "score": 4.84375
    },
    {
      "rank": 17,
      "token_id": 25,
      "text": ":",
      "score": 4.8125
    },
    {
      "rank": 18,
      "token_id": 17079,
      "text": " {{",
      "score": 4.71875
    },
    {
      "rank": 19,
      "token_id": 9678,
      "text": " implementation",
      "score": 4.71875
    },
    {
      "rank": 20,
      "token_id": 26022,
      "text": " \"#",
      "score": 4.65625
    }
  ]
}
```
