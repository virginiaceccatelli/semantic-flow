# Execution prediction and U-Space

Model: `deepseek-coder-6.7b`. Block index: **6** (zero-based, post-block).
Read location: `prompt_end`. No fine-tuning or steering.

## Execution

Evaluated: 1345 / 1345 calls.
Program-macro accuracy: 0.44866071428571425; call accuracy: 0.44981412639405205; invalid answers: 228.

## Uncertainty

Higher scores indicate greater measured uncertainty.

| Score | Eligible calls | Error AUROC | Actual retained coverage | Retained error rate |
|---|---:|---:|---:|---:|
| ulens_score | 1344 | 0.6371844909918474 | 0.7998511904761905 | 0.5386046511627907 |
| a_cone | 1345 | 0.50282778646415 | 0.8 | 0.5566914498141264 |
| mean_answer_entropy | 1344 | 0.6346481172905087 | 0.8005952380952381 | 0.5371747211895911 |
| mean_answer_nll | 1344 | 0.6528120421834286 | 0.7998511904761905 | 0.5376744186046511 |

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
  "a_cone": 1.9713820405812066,
  "ulens_score": 0.8978595432444343,
  "category_cosines": {
    "ambiguity": 0.00513757299631834,
    "incompleteness": 0.004962120205163956,
    "conflicting_evidence": -0.020446758717298508,
    "general_uncertainty": 0.01731758750975132
  },
  "top_uspace_tokens": [
    {
      "rank": 1,
      "token_id": 16503,
      "text": " weird",
      "score": 26.625
    },
    {
      "rank": 2,
      "token_id": 10796,
      "text": " surprise",
      "score": 23.5
    },
    {
      "rank": 3,
      "token_id": 5397,
      "text": " surpr",
      "score": 20.25
    },
    {
      "rank": 4,
      "token_id": 6509,
      "text": " stuff",
      "score": 20.0
    },
    {
      "rank": 5,
      "token_id": 18971,
      "text": " mystery",
      "score": 18.625
    },
    {
      "rank": 6,
      "token_id": 8392,
      "text": " strange",
      "score": 18.125
    },
    {
      "rank": 7,
      "token_id": 15250,
      "text": " surprising",
      "score": 17.875
    },
    {
      "rank": 8,
      "token_id": 12394,
      "text": " \ufffd",
      "score": 17.75
    },
    {
      "rank": 9,
      "token_id": 2001,
      "text": " things",
      "score": 17.75
    },
    {
      "rank": 10,
      "token_id": 12933,
      "text": " suspect",
      "score": 17.75
    },
    {
      "rank": 11,
      "token_id": 22231,
      "text": " mysterious",
      "score": 17.375
    },
    {
      "rank": 12,
      "token_id": 6643,
      "text": " maybe",
      "score": 16.875
    },
    {
      "rank": 13,
      "token_id": 1782,
      "text": "thing",
      "score": 16.625
    },
    {
      "rank": 14,
      "token_id": 572,
      "text": " *",
      "score": 16.375
    },
    {
      "rank": 15,
      "token_id": 14522,
      "text": " unexpected",
      "score": 16.25
    },
    {
      "rank": 16,
      "token_id": 30499,
      "text": " tricky",
      "score": 15.9375
    },
    {
      "rank": 17,
      "token_id": 10054,
      "text": "\u60ca",
      "score": 15.875
    },
    {
      "rank": 18,
      "token_id": 2411,
      "text": " thing",
      "score": 15.6875
    },
    {
      "rank": 19,
      "token_id": 12081,
      "text": " surprised",
      "score": 15.5625
    },
    {
      "rank": 20,
      "token_id": 29437,
      "text": " suspicious",
      "score": 15.4375
    }
  ],
  "top_jlens_tokens": [
    {
      "rank": 1,
      "token_id": 572,
      "text": " *",
      "score": 11.875
    },
    {
      "rank": 2,
      "token_id": 19476,
      "text": " ([",
      "score": 11.375
    },
    {
      "rank": 3,
      "token_id": 821,
      "text": " [",
      "score": 10.8125
    },
    {
      "rank": 4,
      "token_id": 1585,
      "text": " --",
      "score": 10.125
    },
    {
      "rank": 5,
      "token_id": 4052,
      "text": " expected",
      "score": 9.0625
    },
    {
      "rank": 6,
      "token_id": 2220,
      "text": " `",
      "score": 8.5
    },
    {
      "rank": 7,
      "token_id": 32014,
      "text": "<\uff5cend\u2581of\u2581sentence\uff5c>",
      "score": 8.375
    },
    {
      "rank": 8,
      "token_id": 357,
      "text": " \\",
      "score": 8.375
    },
    {
      "rank": 9,
      "token_id": 31421,
      "text": " \"[",
      "score": 8.3125
    },
    {
      "rank": 10,
      "token_id": 1070,
      "text": " _",
      "score": 8.25
    },
    {
      "rank": 11,
      "token_id": 1938,
      "text": " >",
      "score": 8.25
    },
    {
      "rank": 12,
      "token_id": 16603,
      "text": " [[",
      "score": 8.1875
    },
    {
      "rank": 13,
      "token_id": 58,
      "text": "[",
      "score": 8.125
    },
    {
      "rank": 14,
      "token_id": 1013,
      "text": " <",
      "score": 8.0
    },
    {
      "rank": 15,
      "token_id": 6236,
      "text": " ``",
      "score": 7.84375
    },
    {
      "rank": 16,
      "token_id": 23691,
      "text": " defaults",
      "score": 7.78125
    },
    {
      "rank": 17,
      "token_id": 17794,
      "text": " [\\",
      "score": 7.75
    },
    {
      "rank": 18,
      "token_id": 12867,
      "text": " \\(",
      "score": 7.6875
    },
    {
      "rank": 19,
      "token_id": 17237,
      "text": " >=",
      "score": 7.4375
    },
    {
      "rank": 20,
      "token_id": 6617,
      "text": " otherwise",
      "score": 7.4375
    }
  ]
}
```

```json
{
  "case_id": "11:1",
  "call": "remove_Occ(\"abcda\",\"a\")",
  "expected": "'bcd'",
  "answer": "\"bcd\"",
  "correct": true,
  "location": "prompt_end",
  "read_token_index": 152,
  "read_token_text": " ==",
  "a_cone": 1.9579987982840166,
  "ulens_score": 1.0272767900685404,
  "category_cosines": {
    "ambiguity": 0.008248981088399887,
    "incompleteness": 0.005751447752118111,
    "conflicting_evidence": -0.023476339876651764,
    "general_uncertainty": 0.017672225832939148
  },
  "top_uspace_tokens": [
    {
      "rank": 1,
      "token_id": 16503,
      "text": " weird",
      "score": 28.125
    },
    {
      "rank": 2,
      "token_id": 10796,
      "text": " surprise",
      "score": 22.875
    },
    {
      "rank": 3,
      "token_id": 6509,
      "text": " stuff",
      "score": 20.625
    },
    {
      "rank": 4,
      "token_id": 5397,
      "text": " surpr",
      "score": 19.75
    },
    {
      "rank": 5,
      "token_id": 12394,
      "text": " \ufffd",
      "score": 19.75
    },
    {
      "rank": 6,
      "token_id": 18971,
      "text": " mystery",
      "score": 18.875
    },
    {
      "rank": 7,
      "token_id": 8392,
      "text": " strange",
      "score": 18.875
    },
    {
      "rank": 8,
      "token_id": 22231,
      "text": " mysterious",
      "score": 18.25
    },
    {
      "rank": 9,
      "token_id": 2001,
      "text": " things",
      "score": 18.25
    },
    {
      "rank": 10,
      "token_id": 14522,
      "text": " unexpected",
      "score": 18.0
    },
    {
      "rank": 11,
      "token_id": 15250,
      "text": " surprising",
      "score": 18.0
    },
    {
      "rank": 12,
      "token_id": 6643,
      "text": " maybe",
      "score": 17.25
    },
    {
      "rank": 13,
      "token_id": 10047,
      "text": "\ufffd",
      "score": 16.75
    },
    {
      "rank": 14,
      "token_id": 24331,
      "text": " fuck",
      "score": 16.75
    },
    {
      "rank": 15,
      "token_id": 1782,
      "text": "thing",
      "score": 16.5
    },
    {
      "rank": 16,
      "token_id": 2411,
      "text": " thing",
      "score": 16.375
    },
    {
      "rank": 17,
      "token_id": 30499,
      "text": " tricky",
      "score": 16.25
    },
    {
      "rank": 18,
      "token_id": 572,
      "text": " *",
      "score": 16.125
    },
    {
      "rank": 19,
      "token_id": 12081,
      "text": " surprised",
      "score": 15.75
    },
    {
      "rank": 20,
      "token_id": 17121,
      "text": " myster",
      "score": 15.6875
    }
  ],
  "top_jlens_tokens": [
    {
      "rank": 1,
      "token_id": 572,
      "text": " *",
      "score": 12.0625
    },
    {
      "rank": 2,
      "token_id": 19476,
      "text": " ([",
      "score": 12.0
    },
    {
      "rank": 3,
      "token_id": 821,
      "text": " [",
      "score": 11.125
    },
    {
      "rank": 4,
      "token_id": 1585,
      "text": " --",
      "score": 10.1875
    },
    {
      "rank": 5,
      "token_id": 4052,
      "text": " expected",
      "score": 9.25
    },
    {
      "rank": 6,
      "token_id": 31421,
      "text": " \"[",
      "score": 8.9375
    },
    {
      "rank": 7,
      "token_id": 2220,
      "text": " `",
      "score": 8.8125
    },
    {
      "rank": 8,
      "token_id": 16603,
      "text": " [[",
      "score": 8.5
    },
    {
      "rank": 9,
      "token_id": 357,
      "text": " \\",
      "score": 8.5
    },
    {
      "rank": 10,
      "token_id": 1013,
      "text": " <",
      "score": 8.4375
    },
    {
      "rank": 11,
      "token_id": 58,
      "text": "[",
      "score": 8.25
    },
    {
      "rank": 12,
      "token_id": 17794,
      "text": " [\\",
      "score": 8.25
    },
    {
      "rank": 13,
      "token_id": 1070,
      "text": " _",
      "score": 8.1875
    },
    {
      "rank": 14,
      "token_id": 6236,
      "text": " ``",
      "score": 8.1875
    },
    {
      "rank": 15,
      "token_id": 12867,
      "text": " \\(",
      "score": 8.0
    },
    {
      "rank": 16,
      "token_id": 1938,
      "text": " >",
      "score": 7.96875
    },
    {
      "rank": 17,
      "token_id": 23691,
      "text": " defaults",
      "score": 7.78125
    },
    {
      "rank": 18,
      "token_id": 32014,
      "text": "<\uff5cend\u2581of\u2581sentence\uff5c>",
      "score": 7.65625
    },
    {
      "rank": 19,
      "token_id": 25977,
      "text": " \\[",
      "score": 7.5625
    },
    {
      "rank": 20,
      "token_id": 25234,
      "text": " [_",
      "score": 7.53125
    }
  ]
}
```

```json
{
  "case_id": "11:2",
  "call": "remove_Occ(\"PHP\",\"P\")",
  "expected": "'H'",
  "answer": "\"HP\"",
  "correct": false,
  "location": "prompt_end",
  "read_token_index": 151,
  "read_token_text": " ==",
  "a_cone": 1.9681891736469568,
  "ulens_score": 1.4363458742039414,
  "category_cosines": {
    "ambiguity": 0.0036266196984797716,
    "incompleteness": 0.004938202910125256,
    "conflicting_evidence": -0.021454110741615295,
    "general_uncertainty": 0.019199127331376076
  },
  "top_uspace_tokens": [
    {
      "rank": 1,
      "token_id": 16503,
      "text": " weird",
      "score": 25.0
    },
    {
      "rank": 2,
      "token_id": 10796,
      "text": " surprise",
      "score": 23.375
    },
    {
      "rank": 3,
      "token_id": 5397,
      "text": " surpr",
      "score": 20.0
    },
    {
      "rank": 4,
      "token_id": 6509,
      "text": " stuff",
      "score": 19.375
    },
    {
      "rank": 5,
      "token_id": 12933,
      "text": " suspect",
      "score": 18.75
    },
    {
      "rank": 6,
      "token_id": 18971,
      "text": " mystery",
      "score": 17.875
    },
    {
      "rank": 7,
      "token_id": 2001,
      "text": " things",
      "score": 17.375
    },
    {
      "rank": 8,
      "token_id": 15250,
      "text": " surprising",
      "score": 17.125
    },
    {
      "rank": 9,
      "token_id": 8392,
      "text": " strange",
      "score": 17.125
    },
    {
      "rank": 10,
      "token_id": 572,
      "text": " *",
      "score": 16.75
    },
    {
      "rank": 11,
      "token_id": 1782,
      "text": "thing",
      "score": 16.375
    },
    {
      "rank": 12,
      "token_id": 6643,
      "text": " maybe",
      "score": 16.375
    },
    {
      "rank": 13,
      "token_id": 12394,
      "text": " \ufffd",
      "score": 16.25
    },
    {
      "rank": 14,
      "token_id": 22231,
      "text": " mysterious",
      "score": 16.125
    },
    {
      "rank": 15,
      "token_id": 26573,
      "text": " suspicion",
      "score": 15.9375
    },
    {
      "rank": 16,
      "token_id": 10054,
      "text": "\u60ca",
      "score": 15.75
    },
    {
      "rank": 17,
      "token_id": 1070,
      "text": " _",
      "score": 15.625
    },
    {
      "rank": 18,
      "token_id": 30499,
      "text": " tricky",
      "score": 15.375
    },
    {
      "rank": 19,
      "token_id": 29437,
      "text": " suspicious",
      "score": 15.125
    },
    {
      "rank": 20,
      "token_id": 25196,
      "text": " suspected",
      "score": 15.0625
    }
  ],
  "top_jlens_tokens": [
    {
      "rank": 1,
      "token_id": 572,
      "text": " *",
      "score": 11.5625
    },
    {
      "rank": 2,
      "token_id": 19476,
      "text": " ([",
      "score": 11.0
    },
    {
      "rank": 3,
      "token_id": 821,
      "text": " [",
      "score": 10.875
    },
    {
      "rank": 4,
      "token_id": 1585,
      "text": " --",
      "score": 10.0625
    },
    {
      "rank": 5,
      "token_id": 4052,
      "text": " expected",
      "score": 9.3125
    },
    {
      "rank": 6,
      "token_id": 2220,
      "text": " `",
      "score": 8.9375
    },
    {
      "rank": 7,
      "token_id": 32014,
      "text": "<\uff5cend\u2581of\u2581sentence\uff5c>",
      "score": 8.875
    },
    {
      "rank": 8,
      "token_id": 1013,
      "text": " <",
      "score": 8.3125
    },
    {
      "rank": 9,
      "token_id": 16603,
      "text": " [[",
      "score": 8.25
    },
    {
      "rank": 10,
      "token_id": 1938,
      "text": " >",
      "score": 8.25
    },
    {
      "rank": 11,
      "token_id": 58,
      "text": "[",
      "score": 8.25
    },
    {
      "rank": 12,
      "token_id": 31421,
      "text": " \"[",
      "score": 8.1875
    },
    {
      "rank": 13,
      "token_id": 357,
      "text": " \\",
      "score": 7.96875
    },
    {
      "rank": 14,
      "token_id": 6236,
      "text": " ``",
      "score": 7.96875
    },
    {
      "rank": 15,
      "token_id": 4077,
      "text": " http",
      "score": 7.8125
    },
    {
      "rank": 16,
      "token_id": 2312,
      "text": " ==",
      "score": 7.71875
    },
    {
      "rank": 17,
      "token_id": 440,
      "text": " \"",
      "score": 7.46875
    },
    {
      "rank": 18,
      "token_id": 17794,
      "text": " [\\",
      "score": 7.46875
    },
    {
      "rank": 19,
      "token_id": 3752,
      "text": " via",
      "score": 7.40625
    },
    {
      "rank": 20,
      "token_id": 17237,
      "text": " >=",
      "score": 7.34375
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
  "a_cone": 2.212863672703912,
  "ulens_score": 0.213994133370443,
  "category_cosines": {
    "ambiguity": 0.018598372116684914,
    "incompleteness": 0.00987919233739376,
    "conflicting_evidence": -0.02377631887793541,
    "general_uncertainty": 0.027323471382260323
  },
  "top_uspace_tokens": [
    {
      "rank": 1,
      "token_id": 16503,
      "text": " weird",
      "score": 33.75
    },
    {
      "rank": 2,
      "token_id": 10796,
      "text": " surprise",
      "score": 28.375
    },
    {
      "rank": 3,
      "token_id": 15250,
      "text": " surprising",
      "score": 26.25
    },
    {
      "rank": 4,
      "token_id": 5397,
      "text": " surpr",
      "score": 25.875
    },
    {
      "rank": 5,
      "token_id": 14522,
      "text": " unexpected",
      "score": 25.125
    },
    {
      "rank": 6,
      "token_id": 22231,
      "text": " mysterious",
      "score": 24.25
    },
    {
      "rank": 7,
      "token_id": 18971,
      "text": " mystery",
      "score": 23.875
    },
    {
      "rank": 8,
      "token_id": 8392,
      "text": " strange",
      "score": 23.5
    },
    {
      "rank": 9,
      "token_id": 12394,
      "text": " \ufffd",
      "score": 22.25
    },
    {
      "rank": 10,
      "token_id": 12081,
      "text": " surprised",
      "score": 21.875
    },
    {
      "rank": 11,
      "token_id": 17121,
      "text": " myster",
      "score": 21.375
    },
    {
      "rank": 12,
      "token_id": 22900,
      "text": "\u610f\u5916",
      "score": 21.0
    },
    {
      "rank": 13,
      "token_id": 30499,
      "text": " tricky",
      "score": 20.625
    },
    {
      "rank": 14,
      "token_id": 29437,
      "text": " suspicious",
      "score": 20.625
    },
    {
      "rank": 15,
      "token_id": 22709,
      "text": " \ufffd",
      "score": 20.0
    },
    {
      "rank": 16,
      "token_id": 6509,
      "text": " stuff",
      "score": 19.875
    },
    {
      "rank": 17,
      "token_id": 10047,
      "text": "\ufffd",
      "score": 19.875
    },
    {
      "rank": 18,
      "token_id": 28281,
      "text": " accidentally",
      "score": 19.625
    },
    {
      "rank": 19,
      "token_id": 24331,
      "text": " fuck",
      "score": 19.5
    },
    {
      "rank": 20,
      "token_id": 12365,
      "text": " Maybe",
      "score": 19.5
    }
  ],
  "top_jlens_tokens": [
    {
      "rank": 1,
      "token_id": 185,
      "text": "\n",
      "score": 14.375
    },
    {
      "rank": 2,
      "token_id": 19476,
      "text": " ([",
      "score": 13.3125
    },
    {
      "rank": 3,
      "token_id": 821,
      "text": " [",
      "score": 12.6875
    },
    {
      "rank": 4,
      "token_id": 32014,
      "text": "<\uff5cend\u2581of\u2581sentence\uff5c>",
      "score": 12.625
    },
    {
      "rank": 5,
      "token_id": 572,
      "text": " *",
      "score": 11.75
    },
    {
      "rank": 6,
      "token_id": 357,
      "text": " \\",
      "score": 11.5
    },
    {
      "rank": 7,
      "token_id": 58,
      "text": "[",
      "score": 11.1875
    },
    {
      "rank": 8,
      "token_id": 17794,
      "text": " [\\",
      "score": 11.0
    },
    {
      "rank": 9,
      "token_id": 16603,
      "text": " [[",
      "score": 10.3125
    },
    {
      "rank": 10,
      "token_id": 2220,
      "text": " `",
      "score": 10.0625
    },
    {
      "rank": 11,
      "token_id": 25234,
      "text": " [_",
      "score": 10.0
    },
    {
      "rank": 12,
      "token_id": 4052,
      "text": " expected",
      "score": 9.9375
    },
    {
      "rank": 13,
      "token_id": 188,
      "text": "\r",
      "score": 9.75
    },
    {
      "rank": 14,
      "token_id": 1013,
      "text": " <",
      "score": 9.25
    },
    {
      "rank": 15,
      "token_id": 5154,
      "text": "[\\",
      "score": 8.9375
    },
    {
      "rank": 16,
      "token_id": 17079,
      "text": " {{",
      "score": 8.875
    },
    {
      "rank": 17,
      "token_id": 31421,
      "text": " \"[",
      "score": 8.625
    },
    {
      "rank": 18,
      "token_id": 1938,
      "text": " >",
      "score": 8.3125
    },
    {
      "rank": 19,
      "token_id": 243,
      "text": "  ",
      "score": 8.1875
    },
    {
      "rank": 20,
      "token_id": 12867,
      "text": " \\(",
      "score": 8.1875
    }
  ]
}
```

```json
{
  "case_id": "12:1",
  "call": "sort_matrix([[1, 2, 3], [-2, 4, -5], [1, -1, 1]])",
  "expected": "[[-2, 4, -5], [1, -1, 1], [1, 2, 3]]",
  "answer": "[[1, -1, 1], [-2, 4, -5], [1, 2, 3]]",
  "correct": false,
  "location": "prompt_end",
  "read_token_index": 78,
  "read_token_text": " ==",
  "a_cone": 2.206500245711454,
  "ulens_score": 0.2258089051580134,
  "category_cosines": {
    "ambiguity": 0.018299540504813194,
    "incompleteness": 0.010537887923419476,
    "conflicting_evidence": -0.023950014263391495,
    "general_uncertainty": 0.026660321280360222
  },
  "top_uspace_tokens": [
    {
      "rank": 1,
      "token_id": 16503,
      "text": " weird",
      "score": 33.75
    },
    {
      "rank": 2,
      "token_id": 10796,
      "text": " surprise",
      "score": 28.375
    },
    {
      "rank": 3,
      "token_id": 15250,
      "text": " surprising",
      "score": 26.25
    },
    {
      "rank": 4,
      "token_id": 5397,
      "text": " surpr",
      "score": 25.875
    },
    {
      "rank": 5,
      "token_id": 14522,
      "text": " unexpected",
      "score": 25.125
    },
    {
      "rank": 6,
      "token_id": 22231,
      "text": " mysterious",
      "score": 24.875
    },
    {
      "rank": 7,
      "token_id": 18971,
      "text": " mystery",
      "score": 24.5
    },
    {
      "rank": 8,
      "token_id": 8392,
      "text": " strange",
      "score": 23.875
    },
    {
      "rank": 9,
      "token_id": 17121,
      "text": " myster",
      "score": 21.875
    },
    {
      "rank": 10,
      "token_id": 12081,
      "text": " surprised",
      "score": 21.75
    },
    {
      "rank": 11,
      "token_id": 12394,
      "text": " \ufffd",
      "score": 21.75
    },
    {
      "rank": 12,
      "token_id": 22900,
      "text": "\u610f\u5916",
      "score": 21.0
    },
    {
      "rank": 13,
      "token_id": 30499,
      "text": " tricky",
      "score": 20.625
    },
    {
      "rank": 14,
      "token_id": 29437,
      "text": " suspicious",
      "score": 20.625
    },
    {
      "rank": 15,
      "token_id": 6509,
      "text": " stuff",
      "score": 20.125
    },
    {
      "rank": 16,
      "token_id": 28281,
      "text": " accidentally",
      "score": 19.75
    },
    {
      "rank": 17,
      "token_id": 22709,
      "text": " \ufffd",
      "score": 19.625
    },
    {
      "rank": 18,
      "token_id": 12365,
      "text": " Maybe",
      "score": 19.5
    },
    {
      "rank": 19,
      "token_id": 24331,
      "text": " fuck",
      "score": 19.375
    },
    {
      "rank": 20,
      "token_id": 10047,
      "text": "\ufffd",
      "score": 19.375
    }
  ],
  "top_jlens_tokens": [
    {
      "rank": 1,
      "token_id": 185,
      "text": "\n",
      "score": 15.375
    },
    {
      "rank": 2,
      "token_id": 19476,
      "text": " ([",
      "score": 13.1875
    },
    {
      "rank": 3,
      "token_id": 32014,
      "text": "<\uff5cend\u2581of\u2581sentence\uff5c>",
      "score": 13.1875
    },
    {
      "rank": 4,
      "token_id": 821,
      "text": " [",
      "score": 12.6875
    },
    {
      "rank": 5,
      "token_id": 357,
      "text": " \\",
      "score": 12.0
    },
    {
      "rank": 6,
      "token_id": 58,
      "text": "[",
      "score": 11.375
    },
    {
      "rank": 7,
      "token_id": 17794,
      "text": " [\\",
      "score": 11.375
    },
    {
      "rank": 8,
      "token_id": 572,
      "text": " *",
      "score": 11.25
    },
    {
      "rank": 9,
      "token_id": 188,
      "text": "\r",
      "score": 10.5625
    },
    {
      "rank": 10,
      "token_id": 16603,
      "text": " [[",
      "score": 10.25
    },
    {
      "rank": 11,
      "token_id": 25234,
      "text": " [_",
      "score": 9.875
    },
    {
      "rank": 12,
      "token_id": 4052,
      "text": " expected",
      "score": 9.625
    },
    {
      "rank": 13,
      "token_id": 2220,
      "text": " `",
      "score": 9.5
    },
    {
      "rank": 14,
      "token_id": 5154,
      "text": "[\\",
      "score": 9.4375
    },
    {
      "rank": 15,
      "token_id": 1013,
      "text": " <",
      "score": 9.1875
    },
    {
      "rank": 16,
      "token_id": 17079,
      "text": " {{",
      "score": 8.9375
    },
    {
      "rank": 17,
      "token_id": 243,
      "text": "  ",
      "score": 8.8125
    },
    {
      "rank": 18,
      "token_id": 31421,
      "text": " \"[",
      "score": 8.625
    },
    {
      "rank": 19,
      "token_id": 8973,
      "text": "{{",
      "score": 8.5
    },
    {
      "rank": 20,
      "token_id": 12867,
      "text": " \\(",
      "score": 8.375
    }
  ]
}
```

```json
{
  "case_id": "12:2",
  "call": "sort_matrix([[5,8,9],[6,4,3],[2,1,4]])",
  "expected": "[[2, 1, 4], [6, 4, 3], [5, 8, 9]]",
  "answer": "[[6,4,3],[5,8,9],[2,1,4]]",
  "correct": false,
  "location": "prompt_end",
  "read_token_index": 72,
  "read_token_text": " ==",
  "a_cone": 2.1780789806829395,
  "ulens_score": 0.43041422409746405,
  "category_cosines": {
    "ambiguity": 0.01804935932159424,
    "incompleteness": 0.008046544156968594,
    "conflicting_evidence": -0.02489791251718998,
    "general_uncertainty": 0.027329182252287865
  },
  "top_uspace_tokens": [
    {
      "rank": 1,
      "token_id": 16503,
      "text": " weird",
      "score": 33.0
    },
    {
      "rank": 2,
      "token_id": 10796,
      "text": " surprise",
      "score": 27.75
    },
    {
      "rank": 3,
      "token_id": 5397,
      "text": " surpr",
      "score": 25.0
    },
    {
      "rank": 4,
      "token_id": 15250,
      "text": " surprising",
      "score": 25.0
    },
    {
      "rank": 5,
      "token_id": 14522,
      "text": " unexpected",
      "score": 24.125
    },
    {
      "rank": 6,
      "token_id": 12394,
      "text": " \ufffd",
      "score": 23.25
    },
    {
      "rank": 7,
      "token_id": 8392,
      "text": " strange",
      "score": 22.25
    },
    {
      "rank": 8,
      "token_id": 18971,
      "text": " mystery",
      "score": 21.875
    },
    {
      "rank": 9,
      "token_id": 22231,
      "text": " mysterious",
      "score": 21.875
    },
    {
      "rank": 10,
      "token_id": 12081,
      "text": " surprised",
      "score": 21.25
    },
    {
      "rank": 11,
      "token_id": 10047,
      "text": "\ufffd",
      "score": 20.75
    },
    {
      "rank": 12,
      "token_id": 22709,
      "text": " \ufffd",
      "score": 20.25
    },
    {
      "rank": 13,
      "token_id": 22900,
      "text": "\u610f\u5916",
      "score": 20.125
    },
    {
      "rank": 14,
      "token_id": 30499,
      "text": " tricky",
      "score": 20.0
    },
    {
      "rank": 15,
      "token_id": 6509,
      "text": " stuff",
      "score": 19.875
    },
    {
      "rank": 16,
      "token_id": 29437,
      "text": " suspicious",
      "score": 19.75
    },
    {
      "rank": 17,
      "token_id": 24331,
      "text": " fuck",
      "score": 19.5
    },
    {
      "rank": 18,
      "token_id": 17121,
      "text": " myster",
      "score": 19.375
    },
    {
      "rank": 19,
      "token_id": 12365,
      "text": " Maybe",
      "score": 19.0
    },
    {
      "rank": 20,
      "token_id": 28281,
      "text": " accidentally",
      "score": 18.875
    }
  ],
  "top_jlens_tokens": [
    {
      "rank": 1,
      "token_id": 19476,
      "text": " ([",
      "score": 13.0
    },
    {
      "rank": 2,
      "token_id": 821,
      "text": " [",
      "score": 12.5625
    },
    {
      "rank": 3,
      "token_id": 572,
      "text": " *",
      "score": 12.3125
    },
    {
      "rank": 4,
      "token_id": 32014,
      "text": "<\uff5cend\u2581of\u2581sentence\uff5c>",
      "score": 11.8125
    },
    {
      "rank": 5,
      "token_id": 185,
      "text": "\n",
      "score": 11.5
    },
    {
      "rank": 6,
      "token_id": 58,
      "text": "[",
      "score": 10.8125
    },
    {
      "rank": 7,
      "token_id": 17794,
      "text": " [\\",
      "score": 10.5625
    },
    {
      "rank": 8,
      "token_id": 4052,
      "text": " expected",
      "score": 10.25
    },
    {
      "rank": 9,
      "token_id": 16603,
      "text": " [[",
      "score": 10.25
    },
    {
      "rank": 10,
      "token_id": 25234,
      "text": " [_",
      "score": 10.0625
    },
    {
      "rank": 11,
      "token_id": 2220,
      "text": " `",
      "score": 10.0625
    },
    {
      "rank": 12,
      "token_id": 357,
      "text": " \\",
      "score": 9.875
    },
    {
      "rank": 13,
      "token_id": 1013,
      "text": " <",
      "score": 9.6875
    },
    {
      "rank": 14,
      "token_id": 31421,
      "text": " \"[",
      "score": 8.9375
    },
    {
      "rank": 15,
      "token_id": 1585,
      "text": " --",
      "score": 8.75
    },
    {
      "rank": 16,
      "token_id": 1938,
      "text": " >",
      "score": 8.375
    },
    {
      "rank": 17,
      "token_id": 17079,
      "text": " {{",
      "score": 8.3125
    },
    {
      "rank": 18,
      "token_id": 5154,
      "text": "[\\",
      "score": 8.125
    },
    {
      "rank": 19,
      "token_id": 1070,
      "text": " _",
      "score": 8.0625
    },
    {
      "rank": 20,
      "token_id": 6236,
      "text": " ``",
      "score": 8.0625
    }
  ]
}
```

```json
{
  "case_id": "13:0",
  "call": "count_common(['red','green','black','pink','black','white','black','eyes','white','black','orange','pink','pink','red','red','white','orange','white',\"black\",'pink','green','green','pink','green','pink','white','orange',\"orange\",'red'])",
  "expected": "[('pink', 6), ('black', 5), ('white', 5), ('red', 4)]",
  "answer": "[('pink', 8), ('black', 5), ('green', 4), ('red', 3)]",
  "correct": false,
  "location": "prompt_end",
  "read_token_index": 147,
  "read_token_text": " ==",
  "a_cone": 2.2687217017577974,
  "ulens_score": 0.586499542796539,
  "category_cosines": {
    "ambiguity": 0.012571495957672596,
    "incompleteness": 0.009258629754185677,
    "conflicting_evidence": -0.01780681498348713,
    "general_uncertainty": 0.025864416733384132
  },
  "top_uspace_tokens": [
    {
      "rank": 1,
      "token_id": 16503,
      "text": " weird",
      "score": 33.0
    },
    {
      "rank": 2,
      "token_id": 10796,
      "text": " surprise",
      "score": 30.375
    },
    {
      "rank": 3,
      "token_id": 5397,
      "text": " surpr",
      "score": 27.875
    },
    {
      "rank": 4,
      "token_id": 15250,
      "text": " surprising",
      "score": 27.875
    },
    {
      "rank": 5,
      "token_id": 18971,
      "text": " mystery",
      "score": 25.625
    },
    {
      "rank": 6,
      "token_id": 22231,
      "text": " mysterious",
      "score": 25.375
    },
    {
      "rank": 7,
      "token_id": 14522,
      "text": " unexpected",
      "score": 24.375
    },
    {
      "rank": 8,
      "token_id": 8392,
      "text": " strange",
      "score": 24.125
    },
    {
      "rank": 9,
      "token_id": 12933,
      "text": " suspect",
      "score": 23.25
    },
    {
      "rank": 10,
      "token_id": 17121,
      "text": " myster",
      "score": 23.0
    },
    {
      "rank": 11,
      "token_id": 26573,
      "text": " suspicion",
      "score": 22.75
    },
    {
      "rank": 12,
      "token_id": 12081,
      "text": " surprised",
      "score": 22.75
    },
    {
      "rank": 13,
      "token_id": 29437,
      "text": " suspicious",
      "score": 22.0
    },
    {
      "rank": 14,
      "token_id": 22900,
      "text": "\u610f\u5916",
      "score": 21.625
    },
    {
      "rank": 15,
      "token_id": 30499,
      "text": " tricky",
      "score": 21.25
    },
    {
      "rank": 16,
      "token_id": 10054,
      "text": "\u60ca",
      "score": 20.25
    },
    {
      "rank": 17,
      "token_id": 25196,
      "text": " suspected",
      "score": 19.875
    },
    {
      "rank": 18,
      "token_id": 21622,
      "text": "Maybe",
      "score": 19.75
    },
    {
      "rank": 19,
      "token_id": 12365,
      "text": " Maybe",
      "score": 19.375
    },
    {
      "rank": 20,
      "token_id": 28018,
      "text": "\u731c",
      "score": 19.25
    }
  ],
  "top_jlens_tokens": [
    {
      "rank": 1,
      "token_id": 19476,
      "text": " ([",
      "score": 16.75
    },
    {
      "rank": 2,
      "token_id": 821,
      "text": " [",
      "score": 15.8125
    },
    {
      "rank": 3,
      "token_id": 16603,
      "text": " [[",
      "score": 13.9375
    },
    {
      "rank": 4,
      "token_id": 572,
      "text": " *",
      "score": 13.5625
    },
    {
      "rank": 5,
      "token_id": 2220,
      "text": " `",
      "score": 13.0625
    },
    {
      "rank": 6,
      "token_id": 185,
      "text": "\n",
      "score": 12.75
    },
    {
      "rank": 7,
      "token_id": 17794,
      "text": " [\\",
      "score": 12.75
    },
    {
      "rank": 8,
      "token_id": 58,
      "text": "[",
      "score": 12.0625
    },
    {
      "rank": 9,
      "token_id": 32014,
      "text": "<\uff5cend\u2581of\u2581sentence\uff5c>",
      "score": 11.5
    },
    {
      "rank": 10,
      "token_id": 357,
      "text": " \\",
      "score": 10.75
    },
    {
      "rank": 11,
      "token_id": 1013,
      "text": " <",
      "score": 10.625
    },
    {
      "rank": 12,
      "token_id": 243,
      "text": "  ",
      "score": 10.5
    },
    {
      "rank": 13,
      "token_id": 1070,
      "text": " _",
      "score": 10.25
    },
    {
      "rank": 14,
      "token_id": 6236,
      "text": " ``",
      "score": 10.25
    },
    {
      "rank": 15,
      "token_id": 25234,
      "text": " [_",
      "score": 10.25
    },
    {
      "rank": 16,
      "token_id": 1938,
      "text": " >",
      "score": 10.0625
    },
    {
      "rank": 17,
      "token_id": 31421,
      "text": " \"[",
      "score": 9.8125
    },
    {
      "rank": 18,
      "token_id": 17079,
      "text": " {{",
      "score": 9.5
    },
    {
      "rank": 19,
      "token_id": 4052,
      "text": " expected",
      "score": 9.4375
    },
    {
      "rank": 20,
      "token_id": 27366,
      "text": "[[",
      "score": 9.4375
    }
  ]
}
```

```json
{
  "case_id": "13:1",
  "call": "count_common(['one', 'two', 'three', 'four', 'five', 'one', 'two', 'one', 'three', 'one'])",
  "expected": "[('one', 4), ('two', 2), ('three', 2), ('four', 1)]",
  "answer": "[('one', 4), ('two', 2), ('three', 2), ('one', 1)]",
  "correct": false,
  "location": "prompt_end",
  "read_token_index": 108,
  "read_token_text": " ==",
  "a_cone": 2.322993043017302,
  "ulens_score": 0.5124159695868746,
  "category_cosines": {
    "ambiguity": 0.01616625301539898,
    "incompleteness": 0.004801433999091387,
    "conflicting_evidence": -0.013194706290960312,
    "general_uncertainty": 0.021597370505332947
  },
  "top_uspace_tokens": [
    {
      "rank": 1,
      "token_id": 16503,
      "text": " weird",
      "score": 34.25
    },
    {
      "rank": 2,
      "token_id": 10796,
      "text": " surprise",
      "score": 29.625
    },
    {
      "rank": 3,
      "token_id": 15250,
      "text": " surprising",
      "score": 28.375
    },
    {
      "rank": 4,
      "token_id": 5397,
      "text": " surpr",
      "score": 27.125
    },
    {
      "rank": 5,
      "token_id": 14522,
      "text": " unexpected",
      "score": 26.625
    },
    {
      "rank": 6,
      "token_id": 12394,
      "text": " \ufffd",
      "score": 24.75
    },
    {
      "rank": 7,
      "token_id": 12081,
      "text": " surprised",
      "score": 23.75
    },
    {
      "rank": 8,
      "token_id": 22900,
      "text": "\u610f\u5916",
      "score": 23.0
    },
    {
      "rank": 9,
      "token_id": 10047,
      "text": "\ufffd",
      "score": 22.75
    },
    {
      "rank": 10,
      "token_id": 22709,
      "text": " \ufffd",
      "score": 22.75
    },
    {
      "rank": 11,
      "token_id": 8392,
      "text": " strange",
      "score": 22.5
    },
    {
      "rank": 12,
      "token_id": 30981,
      "text": " unclear",
      "score": 21.75
    },
    {
      "rank": 13,
      "token_id": 22231,
      "text": " mysterious",
      "score": 21.75
    },
    {
      "rank": 14,
      "token_id": 18971,
      "text": " mystery",
      "score": 21.625
    },
    {
      "rank": 15,
      "token_id": 29437,
      "text": " suspicious",
      "score": 21.5
    },
    {
      "rank": 16,
      "token_id": 30499,
      "text": " tricky",
      "score": 21.375
    },
    {
      "rank": 17,
      "token_id": 26573,
      "text": " suspicion",
      "score": 21.0
    },
    {
      "rank": 18,
      "token_id": 21622,
      "text": "Maybe",
      "score": 20.5
    },
    {
      "rank": 19,
      "token_id": 26754,
      "text": "\u65b0\u51a0",
      "score": 20.375
    },
    {
      "rank": 20,
      "token_id": 12933,
      "text": " suspect",
      "score": 20.125
    }
  ],
  "top_jlens_tokens": [
    {
      "rank": 1,
      "token_id": 32014,
      "text": "<\uff5cend\u2581of\u2581sentence\uff5c>",
      "score": 14.0
    },
    {
      "rank": 2,
      "token_id": 19476,
      "text": " ([",
      "score": 14.0
    },
    {
      "rank": 3,
      "token_id": 821,
      "text": " [",
      "score": 13.5625
    },
    {
      "rank": 4,
      "token_id": 185,
      "text": "\n",
      "score": 13.4375
    },
    {
      "rank": 5,
      "token_id": 16603,
      "text": " [[",
      "score": 11.8125
    },
    {
      "rank": 6,
      "token_id": 58,
      "text": "[",
      "score": 10.875
    },
    {
      "rank": 7,
      "token_id": 243,
      "text": "  ",
      "score": 10.6875
    },
    {
      "rank": 8,
      "token_id": 572,
      "text": " *",
      "score": 10.3125
    },
    {
      "rank": 9,
      "token_id": 1013,
      "text": " <",
      "score": 9.9375
    },
    {
      "rank": 10,
      "token_id": 17794,
      "text": " [\\",
      "score": 9.8125
    },
    {
      "rank": 11,
      "token_id": 8653,
      "text": "                                                                ",
      "score": 9.25
    },
    {
      "rank": 12,
      "token_id": 4052,
      "text": " expected",
      "score": 9.125
    },
    {
      "rank": 13,
      "token_id": 357,
      "text": " \\",
      "score": 9.0
    },
    {
      "rank": 14,
      "token_id": 4077,
      "text": " http",
      "score": 8.8125
    },
    {
      "rank": 15,
      "token_id": 1938,
      "text": " >",
      "score": 8.8125
    },
    {
      "rank": 16,
      "token_id": 2220,
      "text": " `",
      "score": 8.75
    },
    {
      "rank": 17,
      "token_id": 27366,
      "text": "[[",
      "score": 8.6875
    },
    {
      "rank": 18,
      "token_id": 17079,
      "text": " {{",
      "score": 8.625
    },
    {
      "rank": 19,
      "token_id": 188,
      "text": "\r",
      "score": 8.5
    },
    {
      "rank": 20,
      "token_id": 31421,
      "text": " \"[",
      "score": 8.3125
    }
  ]
}
```

```json
{
  "case_id": "13:2",
  "call": "count_common(['Facebook', 'Apple', 'Amazon', 'Netflix', 'Google', 'Apple', 'Netflix', 'Amazon'])",
  "expected": "[('Apple', 2), ('Amazon', 2), ('Netflix', 2), ('Facebook', 1)]",
  "answer": "[('Netflix', 2), ('Amazon', 2), ('Apple', 2), ('Facebook', 1)]",
  "correct": false,
  "location": "prompt_end",
  "read_token_index": 107,
  "read_token_text": " ==",
  "a_cone": 2.263259277967661,
  "ulens_score": 0.4001568329797863,
  "category_cosines": {
    "ambiguity": 0.016581928357481956,
    "incompleteness": 0.002334793098270893,
    "conflicting_evidence": -0.01432216726243496,
    "general_uncertainty": 0.019486287608742714
  },
  "top_uspace_tokens": [
    {
      "rank": 1,
      "token_id": 16503,
      "text": " weird",
      "score": 33.5
    },
    {
      "rank": 2,
      "token_id": 10796,
      "text": " surprise",
      "score": 27.5
    },
    {
      "rank": 3,
      "token_id": 12394,
      "text": " \ufffd",
      "score": 27.25
    },
    {
      "rank": 4,
      "token_id": 15250,
      "text": " surprising",
      "score": 25.875
    },
    {
      "rank": 5,
      "token_id": 14522,
      "text": " unexpected",
      "score": 25.75
    },
    {
      "rank": 6,
      "token_id": 10047,
      "text": "\ufffd",
      "score": 25.125
    },
    {
      "rank": 7,
      "token_id": 5397,
      "text": " surpr",
      "score": 24.875
    },
    {
      "rank": 8,
      "token_id": 22709,
      "text": " \ufffd",
      "score": 23.75
    },
    {
      "rank": 9,
      "token_id": 12081,
      "text": " surprised",
      "score": 22.25
    },
    {
      "rank": 10,
      "token_id": 22900,
      "text": "\u610f\u5916",
      "score": 21.375
    },
    {
      "rank": 11,
      "token_id": 24331,
      "text": " fuck",
      "score": 20.875
    },
    {
      "rank": 12,
      "token_id": 26754,
      "text": "\u65b0\u51a0",
      "score": 20.625
    },
    {
      "rank": 13,
      "token_id": 30981,
      "text": " unclear",
      "score": 20.375
    },
    {
      "rank": 14,
      "token_id": 8392,
      "text": " strange",
      "score": 20.25
    },
    {
      "rank": 15,
      "token_id": 30499,
      "text": " tricky",
      "score": 20.0
    },
    {
      "rank": 16,
      "token_id": 29437,
      "text": " suspicious",
      "score": 19.625
    },
    {
      "rank": 17,
      "token_id": 28281,
      "text": " accidentally",
      "score": 19.375
    },
    {
      "rank": 18,
      "token_id": 15428,
      "text": " crazy",
      "score": 19.25
    },
    {
      "rank": 19,
      "token_id": 12365,
      "text": " Maybe",
      "score": 19.25
    },
    {
      "rank": 20,
      "token_id": 24717,
      "text": " COVID",
      "score": 19.0
    }
  ],
  "top_jlens_tokens": [
    {
      "rank": 1,
      "token_id": 19476,
      "text": " ([",
      "score": 14.75
    },
    {
      "rank": 2,
      "token_id": 821,
      "text": " [",
      "score": 14.0625
    },
    {
      "rank": 3,
      "token_id": 185,
      "text": "\n",
      "score": 13.6875
    },
    {
      "rank": 4,
      "token_id": 32014,
      "text": "<\uff5cend\u2581of\u2581sentence\uff5c>",
      "score": 13.375
    },
    {
      "rank": 5,
      "token_id": 16603,
      "text": " [[",
      "score": 12.375
    },
    {
      "rank": 6,
      "token_id": 58,
      "text": "[",
      "score": 11.25
    },
    {
      "rank": 7,
      "token_id": 572,
      "text": " *",
      "score": 11.0625
    },
    {
      "rank": 8,
      "token_id": 243,
      "text": "  ",
      "score": 11.0625
    },
    {
      "rank": 9,
      "token_id": 17794,
      "text": " [\\",
      "score": 10.4375
    },
    {
      "rank": 10,
      "token_id": 1013,
      "text": " <",
      "score": 10.3125
    },
    {
      "rank": 11,
      "token_id": 8653,
      "text": "                                                                ",
      "score": 9.8125
    },
    {
      "rank": 12,
      "token_id": 4077,
      "text": " http",
      "score": 9.4375
    },
    {
      "rank": 13,
      "token_id": 357,
      "text": " \\",
      "score": 9.4375
    },
    {
      "rank": 14,
      "token_id": 1938,
      "text": " >",
      "score": 9.375
    },
    {
      "rank": 15,
      "token_id": 2220,
      "text": " `",
      "score": 9.0625
    },
    {
      "rank": 16,
      "token_id": 27366,
      "text": "[[",
      "score": 9.0625
    },
    {
      "rank": 17,
      "token_id": 1585,
      "text": " --",
      "score": 9.0
    },
    {
      "rank": 18,
      "token_id": 17079,
      "text": " {{",
      "score": 8.9375
    },
    {
      "rank": 19,
      "token_id": 31421,
      "text": " \"[",
      "score": 8.875
    },
    {
      "rank": 20,
      "token_id": 4052,
      "text": " expected",
      "score": 8.75
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
  "a_cone": 2.152277629903737,
  "ulens_score": 0.9731302664870007,
  "category_cosines": {
    "ambiguity": 0.009487709030508995,
    "incompleteness": 0.002279064152389765,
    "conflicting_evidence": -0.01537371426820755,
    "general_uncertainty": 0.018349267542362213
  },
  "top_uspace_tokens": [
    {
      "rank": 1,
      "token_id": 16503,
      "text": " weird",
      "score": 31.0
    },
    {
      "rank": 2,
      "token_id": 10796,
      "text": " surprise",
      "score": 27.25
    },
    {
      "rank": 3,
      "token_id": 12394,
      "text": " \ufffd",
      "score": 24.375
    },
    {
      "rank": 4,
      "token_id": 5397,
      "text": " surpr",
      "score": 24.125
    },
    {
      "rank": 5,
      "token_id": 15250,
      "text": " surprising",
      "score": 23.375
    },
    {
      "rank": 6,
      "token_id": 10047,
      "text": "\ufffd",
      "score": 21.75
    },
    {
      "rank": 7,
      "token_id": 14522,
      "text": " unexpected",
      "score": 21.625
    },
    {
      "rank": 8,
      "token_id": 22709,
      "text": " \ufffd",
      "score": 20.5
    },
    {
      "rank": 9,
      "token_id": 12081,
      "text": " surprised",
      "score": 20.375
    },
    {
      "rank": 10,
      "token_id": 12933,
      "text": " suspect",
      "score": 19.25
    },
    {
      "rank": 11,
      "token_id": 8392,
      "text": " strange",
      "score": 19.0
    },
    {
      "rank": 12,
      "token_id": 30499,
      "text": " tricky",
      "score": 18.875
    },
    {
      "rank": 13,
      "token_id": 6509,
      "text": " stuff",
      "score": 18.75
    },
    {
      "rank": 14,
      "token_id": 26573,
      "text": " suspicion",
      "score": 18.625
    },
    {
      "rank": 15,
      "token_id": 24331,
      "text": " fuck",
      "score": 18.625
    },
    {
      "rank": 16,
      "token_id": 29437,
      "text": " suspicious",
      "score": 18.625
    },
    {
      "rank": 17,
      "token_id": 22900,
      "text": "\u610f\u5916",
      "score": 18.5
    },
    {
      "rank": 18,
      "token_id": 18971,
      "text": " mystery",
      "score": 17.625
    },
    {
      "rank": 19,
      "token_id": 12365,
      "text": " Maybe",
      "score": 17.375
    },
    {
      "rank": 20,
      "token_id": 6643,
      "text": " maybe",
      "score": 17.125
    }
  ],
  "top_jlens_tokens": [
    {
      "rank": 1,
      "token_id": 19476,
      "text": " ([",
      "score": 11.125
    },
    {
      "rank": 2,
      "token_id": 572,
      "text": " *",
      "score": 10.5625
    },
    {
      "rank": 3,
      "token_id": 821,
      "text": " [",
      "score": 9.875
    },
    {
      "rank": 4,
      "token_id": 4052,
      "text": " expected",
      "score": 9.75
    },
    {
      "rank": 5,
      "token_id": 19118,
      "text": " attempting",
      "score": 8.75
    },
    {
      "rank": 6,
      "token_id": 2220,
      "text": " `",
      "score": 8.6875
    },
    {
      "rank": 7,
      "token_id": 23691,
      "text": " defaults",
      "score": 8.1875
    },
    {
      "rank": 8,
      "token_id": 1585,
      "text": " --",
      "score": 8.0
    },
    {
      "rank": 9,
      "token_id": 16603,
      "text": " [[",
      "score": 7.9375
    },
    {
      "rank": 10,
      "token_id": 357,
      "text": " \\",
      "score": 7.875
    },
    {
      "rank": 11,
      "token_id": 4129,
      "text": " likely",
      "score": 7.71875
    },
    {
      "rank": 12,
      "token_id": 58,
      "text": "[",
      "score": 7.6875
    },
    {
      "rank": 13,
      "token_id": 1013,
      "text": " <",
      "score": 7.65625
    },
    {
      "rank": 14,
      "token_id": 31421,
      "text": " \"[",
      "score": 7.59375
    },
    {
      "rank": 15,
      "token_id": 32014,
      "text": "<\uff5cend\u2581of\u2581sentence\uff5c>",
      "score": 7.53125
    },
    {
      "rank": 16,
      "token_id": 440,
      "text": " \"",
      "score": 7.46875
    },
    {
      "rank": 17,
      "token_id": 6236,
      "text": " ``",
      "score": 7.34375
    },
    {
      "rank": 18,
      "token_id": 1938,
      "text": " >",
      "score": 7.3125
    },
    {
      "rank": 19,
      "token_id": 17794,
      "text": " [\\",
      "score": 7.125
    },
    {
      "rank": 20,
      "token_id": 4077,
      "text": " http",
      "score": 7.0625
    }
  ]
}
```
