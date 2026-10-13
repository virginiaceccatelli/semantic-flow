# Execution prediction and U-Space

Model: `deepseek-coder-6.7b`. Block index: **6** (zero-based, post-block).
Read location: `prompt_end`. No fine-tuning or steering.

## Execution

Evaluated: 726 / 726 calls.
Program-macro accuracy: 0.33746556473829203; call accuracy: 0.33746556473829203; invalid answers: 55.

## Uncertainty

Higher scores indicate greater measured uncertainty.

| Score | Eligible calls | Error AUROC | Actual retained coverage | Retained error rate |
|---|---:|---:|---:|---:|
| ulens_score | 724 | 0.685612031869115 | 0.7997237569060773 | 0.6183074265975821 |
| a_cone | 726 | 0.43360346217489076 | 0.800275482093664 | 0.6764199655765921 |
| mean_answer_entropy | 724 | 0.6882280260747305 | 0.7997237569060773 | 0.616580310880829 |
| mean_answer_nll | 724 | 0.7163222700353629 | 0.7997237569060773 | 0.6096718480138169 |

U-Space alignment (`a_cone`) uses the published formula. `ulens_score` is `a_cone * mean_answer_entropy`: a short-answer adaptation, not the published end-of-thinking protocol. Entropy covers tokens producing the answer, excluding EOS and newline-only termination. It is not a probability of error.

`top_uspace_tokens` decodes the signed U-Space projection through the J-lens (an exploratory extension). `top_jlens_tokens` decodes the complete residual. Both lists contain literal tokenizer tokens, including whitespace/fragments; neither establishes the cause of a failure.

## First ten evaluated examples

```json
{
  "case_id": "sample_0",
  "call": "f([1, 1, 3, 1, 3, 1])",
  "expected": "[(4, 1), (4, 1), (4, 1), (4, 1), (2, 3), (2, 3)]",
  "answer": "[(3, 1), (2, 3), (1, 1)]",
  "correct": false,
  "location": "prompt_end",
  "read_token_index": 89,
  "read_token_text": " ==",
  "a_cone": 2.1937721621131527,
  "ulens_score": 0.4240509869116926,
  "category_cosines": {
    "ambiguity": 0.01429247297346592,
    "incompleteness": 0.013672983273863792,
    "conflicting_evidence": -0.02183360420167446,
    "general_uncertainty": 0.021760649979114532
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
      "token_id": 22231,
      "text": " mysterious",
      "score": 29.75
    },
    {
      "rank": 3,
      "token_id": 18971,
      "text": " mystery",
      "score": 28.625
    },
    {
      "rank": 4,
      "token_id": 10796,
      "text": " surprise",
      "score": 27.625
    },
    {
      "rank": 5,
      "token_id": 15250,
      "text": " surprising",
      "score": 26.0
    },
    {
      "rank": 6,
      "token_id": 8392,
      "text": " strange",
      "score": 26.0
    },
    {
      "rank": 7,
      "token_id": 5397,
      "text": " surpr",
      "score": 25.625
    },
    {
      "rank": 8,
      "token_id": 17121,
      "text": " myster",
      "score": 25.625
    },
    {
      "rank": 9,
      "token_id": 14522,
      "text": " unexpected",
      "score": 24.75
    },
    {
      "rank": 10,
      "token_id": 12081,
      "text": " surprised",
      "score": 20.875
    },
    {
      "rank": 11,
      "token_id": 6509,
      "text": " stuff",
      "score": 20.75
    },
    {
      "rank": 12,
      "token_id": 29437,
      "text": " suspicious",
      "score": 20.625
    },
    {
      "rank": 13,
      "token_id": 22900,
      "text": "\u610f\u5916",
      "score": 20.5
    },
    {
      "rank": 14,
      "token_id": 30499,
      "text": " tricky",
      "score": 20.375
    },
    {
      "rank": 15,
      "token_id": 14321,
      "text": "\u602a",
      "score": 20.125
    },
    {
      "rank": 16,
      "token_id": 28281,
      "text": " accidentally",
      "score": 20.0
    },
    {
      "rank": 17,
      "token_id": 16015,
      "text": " unusual",
      "score": 20.0
    },
    {
      "rank": 18,
      "token_id": 10054,
      "text": "\u60ca",
      "score": 19.75
    },
    {
      "rank": 19,
      "token_id": 12933,
      "text": " suspect",
      "score": 19.625
    },
    {
      "rank": 20,
      "token_id": 28580,
      "text": " strang",
      "score": 19.375
    }
  ],
  "top_jlens_tokens": [
    {
      "rank": 1,
      "token_id": 572,
      "text": " *",
      "score": 12.5625
    },
    {
      "rank": 2,
      "token_id": 2220,
      "text": " `",
      "score": 11.375
    },
    {
      "rank": 3,
      "token_id": 19476,
      "text": " ([",
      "score": 10.625
    },
    {
      "rank": 4,
      "token_id": 4052,
      "text": " expected",
      "score": 10.5625
    },
    {
      "rank": 5,
      "token_id": 821,
      "text": " [",
      "score": 10.0
    },
    {
      "rank": 6,
      "token_id": 58,
      "text": "[",
      "score": 8.75
    },
    {
      "rank": 7,
      "token_id": 32014,
      "text": "<\uff5cend\u2581of\u2581sentence\uff5c>",
      "score": 8.6875
    },
    {
      "rank": 8,
      "token_id": 16603,
      "text": " [[",
      "score": 8.375
    },
    {
      "rank": 9,
      "token_id": 6236,
      "text": " ``",
      "score": 8.375
    },
    {
      "rank": 10,
      "token_id": 1013,
      "text": " <",
      "score": 8.3125
    },
    {
      "rank": 11,
      "token_id": 25234,
      "text": " [_",
      "score": 8.0625
    },
    {
      "rank": 12,
      "token_id": 1070,
      "text": " _",
      "score": 7.96875
    },
    {
      "rank": 13,
      "token_id": 17079,
      "text": " {{",
      "score": 7.65625
    },
    {
      "rank": 14,
      "token_id": 3558,
      "text": " today",
      "score": 7.59375
    },
    {
      "rank": 15,
      "token_id": 6183,
      "text": " sex",
      "score": 7.53125
    },
    {
      "rank": 16,
      "token_id": 1585,
      "text": " --",
      "score": 7.46875
    },
    {
      "rank": 17,
      "token_id": 440,
      "text": " \"",
      "score": 7.46875
    },
    {
      "rank": 18,
      "token_id": 31421,
      "text": " \"[",
      "score": 7.3125
    },
    {
      "rank": 19,
      "token_id": 2974,
      "text": " code",
      "score": 7.25
    },
    {
      "rank": 20,
      "token_id": 23691,
      "text": " defaults",
      "score": 7.09375
    }
  ]
}
```

```json
{
  "case_id": "sample_2",
  "call": "f('hbtofdeiequ')",
  "expected": "'hbtofdeiequ'",
  "answer": "'hbtfdeiequ'",
  "correct": false,
  "location": "prompt_end",
  "read_token_index": 84,
  "read_token_text": " ==",
  "a_cone": 2.1536435968313117,
  "ulens_score": 2.46946507571074,
  "category_cosines": {
    "ambiguity": 0.007321914192289114,
    "incompleteness": 0.01852070726454258,
    "conflicting_evidence": -0.021623505279421806,
    "general_uncertainty": 0.018668368458747864
  },
  "top_uspace_tokens": [
    {
      "rank": 1,
      "token_id": 22231,
      "text": " mysterious",
      "score": 33.75
    },
    {
      "rank": 2,
      "token_id": 18971,
      "text": " mystery",
      "score": 32.25
    },
    {
      "rank": 3,
      "token_id": 17121,
      "text": " myster",
      "score": 28.75
    },
    {
      "rank": 4,
      "token_id": 16503,
      "text": " weird",
      "score": 27.75
    },
    {
      "rank": 5,
      "token_id": 8392,
      "text": " strange",
      "score": 26.375
    },
    {
      "rank": 6,
      "token_id": 10796,
      "text": " surprise",
      "score": 24.5
    },
    {
      "rank": 7,
      "token_id": 16015,
      "text": " unusual",
      "score": 23.375
    },
    {
      "rank": 8,
      "token_id": 5397,
      "text": " surpr",
      "score": 23.0
    },
    {
      "rank": 9,
      "token_id": 15250,
      "text": " surprising",
      "score": 22.625
    },
    {
      "rank": 10,
      "token_id": 28580,
      "text": " strang",
      "score": 20.375
    },
    {
      "rank": 11,
      "token_id": 14522,
      "text": " unexpected",
      "score": 19.875
    },
    {
      "rank": 12,
      "token_id": 6509,
      "text": " stuff",
      "score": 19.875
    },
    {
      "rank": 13,
      "token_id": 10054,
      "text": "\u60ca",
      "score": 19.75
    },
    {
      "rank": 14,
      "token_id": 12933,
      "text": " suspect",
      "score": 19.75
    },
    {
      "rank": 15,
      "token_id": 14321,
      "text": "\u602a",
      "score": 19.75
    },
    {
      "rank": 16,
      "token_id": 7103,
      "text": "\u5947",
      "score": 19.375
    },
    {
      "rank": 17,
      "token_id": 29437,
      "text": " suspicious",
      "score": 18.75
    },
    {
      "rank": 18,
      "token_id": 1782,
      "text": "thing",
      "score": 18.75
    },
    {
      "rank": 19,
      "token_id": 8965,
      "text": " suddenly",
      "score": 18.5
    },
    {
      "rank": 20,
      "token_id": 9257,
      "text": " unknown",
      "score": 18.5
    }
  ],
  "top_jlens_tokens": [
    {
      "rank": 1,
      "token_id": 572,
      "text": " *",
      "score": 12.4375
    },
    {
      "rank": 2,
      "token_id": 2220,
      "text": " `",
      "score": 10.0
    },
    {
      "rank": 3,
      "token_id": 6236,
      "text": " ``",
      "score": 9.3125
    },
    {
      "rank": 4,
      "token_id": 19476,
      "text": " ([",
      "score": 9.25
    },
    {
      "rank": 5,
      "token_id": 4052,
      "text": " expected",
      "score": 8.6875
    },
    {
      "rank": 6,
      "token_id": 821,
      "text": " [",
      "score": 8.5625
    },
    {
      "rank": 7,
      "token_id": 6183,
      "text": " sex",
      "score": 8.4375
    },
    {
      "rank": 8,
      "token_id": 1585,
      "text": " --",
      "score": 8.1875
    },
    {
      "rank": 9,
      "token_id": 23691,
      "text": " defaults",
      "score": 8.125
    },
    {
      "rank": 10,
      "token_id": 1070,
      "text": " _",
      "score": 7.875
    },
    {
      "rank": 11,
      "token_id": 31421,
      "text": " \"[",
      "score": 7.4375
    },
    {
      "rank": 12,
      "token_id": 58,
      "text": "[",
      "score": 7.28125
    },
    {
      "rank": 13,
      "token_id": 1013,
      "text": " <",
      "score": 7.28125
    },
    {
      "rank": 14,
      "token_id": 1938,
      "text": " >",
      "score": 7.25
    },
    {
      "rank": 15,
      "token_id": 4077,
      "text": " http",
      "score": 7.25
    },
    {
      "rank": 16,
      "token_id": 24066,
      "text": "\u4eca\u65e5",
      "score": 7.21875
    },
    {
      "rank": 17,
      "token_id": 440,
      "text": " \"",
      "score": 7.09375
    },
    {
      "rank": 18,
      "token_id": 3558,
      "text": " today",
      "score": 7.0625
    },
    {
      "rank": 19,
      "token_id": 9257,
      "text": " unknown",
      "score": 6.96875
    },
    {
      "rank": 20,
      "token_id": 2312,
      "text": " ==",
      "score": 6.90625
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
  "a_cone": 2.083570434243388,
  "ulens_score": 0.5828103759523228,
  "category_cosines": {
    "ambiguity": 0.003909777384251356,
    "incompleteness": 0.0161578506231308,
    "conflicting_evidence": -0.02302570268511772,
    "general_uncertainty": 0.019952936097979546
  },
  "top_uspace_tokens": [
    {
      "rank": 1,
      "token_id": 22231,
      "text": " mysterious",
      "score": 29.875
    },
    {
      "rank": 2,
      "token_id": 18971,
      "text": " mystery",
      "score": 29.375
    },
    {
      "rank": 3,
      "token_id": 17121,
      "text": " myster",
      "score": 25.875
    },
    {
      "rank": 4,
      "token_id": 16503,
      "text": " weird",
      "score": 25.5
    },
    {
      "rank": 5,
      "token_id": 8392,
      "text": " strange",
      "score": 23.875
    },
    {
      "rank": 6,
      "token_id": 10796,
      "text": " surprise",
      "score": 23.75
    },
    {
      "rank": 7,
      "token_id": 5397,
      "text": " surpr",
      "score": 21.875
    },
    {
      "rank": 8,
      "token_id": 16015,
      "text": " unusual",
      "score": 20.75
    },
    {
      "rank": 9,
      "token_id": 12933,
      "text": " suspect",
      "score": 20.375
    },
    {
      "rank": 10,
      "token_id": 15250,
      "text": " surprising",
      "score": 20.25
    },
    {
      "rank": 11,
      "token_id": 6509,
      "text": " stuff",
      "score": 19.75
    },
    {
      "rank": 12,
      "token_id": 10054,
      "text": "\u60ca",
      "score": 19.0
    },
    {
      "rank": 13,
      "token_id": 1782,
      "text": "thing",
      "score": 18.625
    },
    {
      "rank": 14,
      "token_id": 28580,
      "text": " strang",
      "score": 18.25
    },
    {
      "rank": 15,
      "token_id": 26573,
      "text": " suspicion",
      "score": 17.875
    },
    {
      "rank": 16,
      "token_id": 25196,
      "text": " suspected",
      "score": 17.75
    },
    {
      "rank": 17,
      "token_id": 14321,
      "text": "\u602a",
      "score": 17.5
    },
    {
      "rank": 18,
      "token_id": 29437,
      "text": " suspicious",
      "score": 17.5
    },
    {
      "rank": 19,
      "token_id": 7103,
      "text": "\u5947",
      "score": 17.5
    },
    {
      "rank": 20,
      "token_id": 9257,
      "text": " unknown",
      "score": 16.875
    }
  ],
  "top_jlens_tokens": [
    {
      "rank": 1,
      "token_id": 572,
      "text": " *",
      "score": 11.375
    },
    {
      "rank": 2,
      "token_id": 4052,
      "text": " expected",
      "score": 9.5
    },
    {
      "rank": 3,
      "token_id": 19476,
      "text": " ([",
      "score": 8.375
    },
    {
      "rank": 4,
      "token_id": 821,
      "text": " [",
      "score": 8.3125
    },
    {
      "rank": 5,
      "token_id": 1585,
      "text": " --",
      "score": 8.125
    },
    {
      "rank": 6,
      "token_id": 6236,
      "text": " ``",
      "score": 7.71875
    },
    {
      "rank": 7,
      "token_id": 2220,
      "text": " `",
      "score": 7.5
    },
    {
      "rank": 8,
      "token_id": 6183,
      "text": " sex",
      "score": 7.5
    },
    {
      "rank": 9,
      "token_id": 1938,
      "text": " >",
      "score": 7.4375
    },
    {
      "rank": 10,
      "token_id": 3558,
      "text": " today",
      "score": 7.28125
    },
    {
      "rank": 11,
      "token_id": 58,
      "text": "[",
      "score": 7.25
    },
    {
      "rank": 12,
      "token_id": 23691,
      "text": " defaults",
      "score": 7.25
    },
    {
      "rank": 13,
      "token_id": 4077,
      "text": " http",
      "score": 7.21875
    },
    {
      "rank": 14,
      "token_id": 2312,
      "text": " ==",
      "score": 7.0625
    },
    {
      "rank": 15,
      "token_id": 24066,
      "text": "\u4eca\u65e5",
      "score": 7.03125
    },
    {
      "rank": 16,
      "token_id": 9257,
      "text": " unknown",
      "score": 6.96875
    },
    {
      "rank": 17,
      "token_id": 440,
      "text": " \"",
      "score": 6.90625
    },
    {
      "rank": 18,
      "token_id": 357,
      "text": " \\",
      "score": 6.59375
    },
    {
      "rank": 19,
      "token_id": 545,
      "text": " but",
      "score": 6.5625
    },
    {
      "rank": 20,
      "token_id": 17237,
      "text": " >=",
      "score": 6.5625
    }
  ]
}
```

```json
{
  "case_id": "sample_4",
  "call": "f([' ', '  ', '    ', '   '])",
  "expected": "'           '",
  "answer": "'     '",
  "correct": false,
  "location": "prompt_end",
  "read_token_index": 59,
  "read_token_text": " ==",
  "a_cone": 2.278033471267145,
  "ulens_score": 2.5512942595756147,
  "category_cosines": {
    "ambiguity": 0.005932623520493507,
    "incompleteness": 0.010226215235888958,
    "conflicting_evidence": -0.013719341717660427,
    "general_uncertainty": 0.021345943212509155
  },
  "top_uspace_tokens": [
    {
      "rank": 1,
      "token_id": 10796,
      "text": " surprise",
      "score": 30.375
    },
    {
      "rank": 2,
      "token_id": 16503,
      "text": " weird",
      "score": 30.0
    },
    {
      "rank": 3,
      "token_id": 18971,
      "text": " mystery",
      "score": 28.875
    },
    {
      "rank": 4,
      "token_id": 22231,
      "text": " mysterious",
      "score": 28.625
    },
    {
      "rank": 5,
      "token_id": 5397,
      "text": " surpr",
      "score": 28.125
    },
    {
      "rank": 6,
      "token_id": 15250,
      "text": " surprising",
      "score": 27.5
    },
    {
      "rank": 7,
      "token_id": 17121,
      "text": " myster",
      "score": 26.125
    },
    {
      "rank": 8,
      "token_id": 12933,
      "text": " suspect",
      "score": 26.0
    },
    {
      "rank": 9,
      "token_id": 8392,
      "text": " strange",
      "score": 25.0
    },
    {
      "rank": 10,
      "token_id": 26573,
      "text": " suspicion",
      "score": 24.75
    },
    {
      "rank": 11,
      "token_id": 29437,
      "text": " suspicious",
      "score": 22.25
    },
    {
      "rank": 12,
      "token_id": 25196,
      "text": " suspected",
      "score": 21.875
    },
    {
      "rank": 13,
      "token_id": 16015,
      "text": " unusual",
      "score": 21.75
    },
    {
      "rank": 14,
      "token_id": 14522,
      "text": " unexpected",
      "score": 21.625
    },
    {
      "rank": 15,
      "token_id": 12081,
      "text": " surprised",
      "score": 21.625
    },
    {
      "rank": 16,
      "token_id": 10054,
      "text": "\u60ca",
      "score": 21.5
    },
    {
      "rank": 17,
      "token_id": 28018,
      "text": "\u731c",
      "score": 20.875
    },
    {
      "rank": 18,
      "token_id": 30499,
      "text": " tricky",
      "score": 20.5
    },
    {
      "rank": 19,
      "token_id": 22900,
      "text": "\u610f\u5916",
      "score": 20.25
    },
    {
      "rank": 20,
      "token_id": 28580,
      "text": " strang",
      "score": 20.125
    }
  ],
  "top_jlens_tokens": [
    {
      "rank": 1,
      "token_id": 572,
      "text": " *",
      "score": 11.375
    },
    {
      "rank": 2,
      "token_id": 821,
      "text": " [",
      "score": 10.4375
    },
    {
      "rank": 3,
      "token_id": 19476,
      "text": " ([",
      "score": 9.875
    },
    {
      "rank": 4,
      "token_id": 4052,
      "text": " expected",
      "score": 9.8125
    },
    {
      "rank": 5,
      "token_id": 2220,
      "text": " `",
      "score": 9.5
    },
    {
      "rank": 6,
      "token_id": 185,
      "text": "\n",
      "score": 9.375
    },
    {
      "rank": 7,
      "token_id": 32014,
      "text": "<\uff5cend\u2581of\u2581sentence\uff5c>",
      "score": 9.0625
    },
    {
      "rank": 8,
      "token_id": 4077,
      "text": " http",
      "score": 8.6875
    },
    {
      "rank": 9,
      "token_id": 58,
      "text": "[",
      "score": 8.5625
    },
    {
      "rank": 10,
      "token_id": 16603,
      "text": " [[",
      "score": 8.125
    },
    {
      "rank": 11,
      "token_id": 1585,
      "text": " --",
      "score": 7.84375
    },
    {
      "rank": 12,
      "token_id": 23691,
      "text": " defaults",
      "score": 7.375
    },
    {
      "rank": 13,
      "token_id": 1013,
      "text": " <",
      "score": 7.375
    },
    {
      "rank": 14,
      "token_id": 1938,
      "text": " >",
      "score": 7.34375
    },
    {
      "rank": 15,
      "token_id": 440,
      "text": " \"",
      "score": 7.34375
    },
    {
      "rank": 16,
      "token_id": 6236,
      "text": " ``",
      "score": 7.3125
    },
    {
      "rank": 17,
      "token_id": 1070,
      "text": " _",
      "score": 7.15625
    },
    {
      "rank": 18,
      "token_id": 3558,
      "text": " today",
      "score": 7.15625
    },
    {
      "rank": 19,
      "token_id": 2312,
      "text": " ==",
      "score": 7.09375
    },
    {
      "rank": 20,
      "token_id": 17079,
      "text": " {{",
      "score": 7.03125
    }
  ]
}
```

```json
{
  "case_id": "sample_5",
  "call": "f('DSUWeqExTQdCMGpqur', 'a', 'x')",
  "expected": "(0, 'xxxxxxxxxxxxxxxxxx')",
  "answer": "(3, 'DSUWexTxQdCMGpxqur')",
  "correct": false,
  "location": "prompt_end",
  "read_token_index": 126,
  "read_token_text": " ==",
  "a_cone": 2.111584315919997,
  "ulens_score": 2.2208640393067944,
  "category_cosines": {
    "ambiguity": 0.007051057182252407,
    "incompleteness": 0.02111062966287136,
    "conflicting_evidence": -0.02566891349852085,
    "general_uncertainty": 0.019875844940543175
  },
  "top_uspace_tokens": [
    {
      "rank": 1,
      "token_id": 22231,
      "text": " mysterious",
      "score": 33.0
    },
    {
      "rank": 2,
      "token_id": 18971,
      "text": " mystery",
      "score": 31.625
    },
    {
      "rank": 3,
      "token_id": 17121,
      "text": " myster",
      "score": 28.0
    },
    {
      "rank": 4,
      "token_id": 16503,
      "text": " weird",
      "score": 26.375
    },
    {
      "rank": 5,
      "token_id": 8392,
      "text": " strange",
      "score": 25.625
    },
    {
      "rank": 6,
      "token_id": 10796,
      "text": " surprise",
      "score": 23.0
    },
    {
      "rank": 7,
      "token_id": 16015,
      "text": " unusual",
      "score": 22.75
    },
    {
      "rank": 8,
      "token_id": 5397,
      "text": " surpr",
      "score": 21.625
    },
    {
      "rank": 9,
      "token_id": 15250,
      "text": " surprising",
      "score": 20.875
    },
    {
      "rank": 10,
      "token_id": 6509,
      "text": " stuff",
      "score": 19.875
    },
    {
      "rank": 11,
      "token_id": 28580,
      "text": " strang",
      "score": 19.625
    },
    {
      "rank": 12,
      "token_id": 10054,
      "text": "\u60ca",
      "score": 19.0
    },
    {
      "rank": 13,
      "token_id": 7103,
      "text": "\u5947",
      "score": 19.0
    },
    {
      "rank": 14,
      "token_id": 14321,
      "text": "\u602a",
      "score": 18.875
    },
    {
      "rank": 15,
      "token_id": 1782,
      "text": "thing",
      "score": 18.75
    },
    {
      "rank": 16,
      "token_id": 12933,
      "text": " suspect",
      "score": 18.625
    },
    {
      "rank": 17,
      "token_id": 9257,
      "text": " unknown",
      "score": 18.5
    },
    {
      "rank": 18,
      "token_id": 14522,
      "text": " unexpected",
      "score": 18.5
    },
    {
      "rank": 19,
      "token_id": 8965,
      "text": " suddenly",
      "score": 18.0
    },
    {
      "rank": 20,
      "token_id": 29437,
      "text": " suspicious",
      "score": 17.5
    }
  ],
  "top_jlens_tokens": [
    {
      "rank": 1,
      "token_id": 572,
      "text": " *",
      "score": 11.0625
    },
    {
      "rank": 2,
      "token_id": 821,
      "text": " [",
      "score": 10.5625
    },
    {
      "rank": 3,
      "token_id": 58,
      "text": "[",
      "score": 10.25
    },
    {
      "rank": 4,
      "token_id": 4052,
      "text": " expected",
      "score": 10.25
    },
    {
      "rank": 5,
      "token_id": 19476,
      "text": " ([",
      "score": 10.0
    },
    {
      "rank": 6,
      "token_id": 185,
      "text": "\n",
      "score": 10.0
    },
    {
      "rank": 7,
      "token_id": 32014,
      "text": "<\uff5cend\u2581of\u2581sentence\uff5c>",
      "score": 9.8125
    },
    {
      "rank": 8,
      "token_id": 1585,
      "text": " --",
      "score": 9.0
    },
    {
      "rank": 9,
      "token_id": 2220,
      "text": " `",
      "score": 9.0
    },
    {
      "rank": 10,
      "token_id": 357,
      "text": " \\",
      "score": 8.5625
    },
    {
      "rank": 11,
      "token_id": 6236,
      "text": " ``",
      "score": 8.5
    },
    {
      "rank": 12,
      "token_id": 1938,
      "text": " >",
      "score": 7.9375
    },
    {
      "rank": 13,
      "token_id": 1013,
      "text": " <",
      "score": 7.78125
    },
    {
      "rank": 14,
      "token_id": 3558,
      "text": " today",
      "score": 7.65625
    },
    {
      "rank": 15,
      "token_id": 440,
      "text": " \"",
      "score": 7.59375
    },
    {
      "rank": 16,
      "token_id": 31421,
      "text": " \"[",
      "score": 7.25
    },
    {
      "rank": 17,
      "token_id": 16603,
      "text": " [[",
      "score": 7.1875
    },
    {
      "rank": 18,
      "token_id": 4077,
      "text": " http",
      "score": 7.1875
    },
    {
      "rank": 19,
      "token_id": 17079,
      "text": " {{",
      "score": 7.15625
    },
    {
      "rank": 20,
      "token_id": 243,
      "text": "  ",
      "score": 7.125
    }
  ]
}
```

```json
{
  "case_id": "sample_6",
  "call": "f({'11': 52, '65': 34, 'a': 12, '4': 52, '74': 31})",
  "expected": "[('74', 31)]",
  "answer": "[('4', 52), ('74', 31)]",
  "correct": false,
  "location": "prompt_end",
  "read_token_index": 111,
  "read_token_text": " ==",
  "a_cone": 2.2222126483814995,
  "ulens_score": 0.8002742195481451,
  "category_cosines": {
    "ambiguity": 0.012245182879269123,
    "incompleteness": 0.018433038145303726,
    "conflicting_evidence": -0.02425827458500862,
    "general_uncertainty": 0.02785372920334339
  },
  "top_uspace_tokens": [
    {
      "rank": 1,
      "token_id": 22231,
      "text": " mysterious",
      "score": 31.375
    },
    {
      "rank": 2,
      "token_id": 16503,
      "text": " weird",
      "score": 31.125
    },
    {
      "rank": 3,
      "token_id": 18971,
      "text": " mystery",
      "score": 30.5
    },
    {
      "rank": 4,
      "token_id": 10796,
      "text": " surprise",
      "score": 28.375
    },
    {
      "rank": 5,
      "token_id": 17121,
      "text": " myster",
      "score": 27.375
    },
    {
      "rank": 6,
      "token_id": 8392,
      "text": " strange",
      "score": 26.375
    },
    {
      "rank": 7,
      "token_id": 5397,
      "text": " surpr",
      "score": 26.375
    },
    {
      "rank": 8,
      "token_id": 15250,
      "text": " surprising",
      "score": 26.25
    },
    {
      "rank": 9,
      "token_id": 14522,
      "text": " unexpected",
      "score": 22.875
    },
    {
      "rank": 10,
      "token_id": 12933,
      "text": " suspect",
      "score": 22.375
    },
    {
      "rank": 11,
      "token_id": 16015,
      "text": " unusual",
      "score": 22.25
    },
    {
      "rank": 12,
      "token_id": 26573,
      "text": " suspicion",
      "score": 21.375
    },
    {
      "rank": 13,
      "token_id": 29437,
      "text": " suspicious",
      "score": 21.125
    },
    {
      "rank": 14,
      "token_id": 10054,
      "text": "\u60ca",
      "score": 20.875
    },
    {
      "rank": 15,
      "token_id": 12081,
      "text": " surprised",
      "score": 20.5
    },
    {
      "rank": 16,
      "token_id": 28580,
      "text": " strang",
      "score": 20.5
    },
    {
      "rank": 17,
      "token_id": 14321,
      "text": "\u602a",
      "score": 20.375
    },
    {
      "rank": 18,
      "token_id": 30499,
      "text": " tricky",
      "score": 20.125
    },
    {
      "rank": 19,
      "token_id": 6509,
      "text": " stuff",
      "score": 20.0
    },
    {
      "rank": 20,
      "token_id": 25196,
      "text": " suspected",
      "score": 19.875
    }
  ],
  "top_jlens_tokens": [
    {
      "rank": 1,
      "token_id": 19476,
      "text": " ([",
      "score": 15.25
    },
    {
      "rank": 2,
      "token_id": 185,
      "text": "\n",
      "score": 14.1875
    },
    {
      "rank": 3,
      "token_id": 821,
      "text": " [",
      "score": 13.9375
    },
    {
      "rank": 4,
      "token_id": 572,
      "text": " *",
      "score": 13.0
    },
    {
      "rank": 5,
      "token_id": 17794,
      "text": " [\\",
      "score": 12.875
    },
    {
      "rank": 6,
      "token_id": 25234,
      "text": " [_",
      "score": 12.4375
    },
    {
      "rank": 7,
      "token_id": 16603,
      "text": " [[",
      "score": 12.375
    },
    {
      "rank": 8,
      "token_id": 32014,
      "text": "<\uff5cend\u2581of\u2581sentence\uff5c>",
      "score": 12.0625
    },
    {
      "rank": 9,
      "token_id": 58,
      "text": "[",
      "score": 12.0
    },
    {
      "rank": 10,
      "token_id": 2220,
      "text": " `",
      "score": 11.625
    },
    {
      "rank": 11,
      "token_id": 357,
      "text": " \\",
      "score": 11.0
    },
    {
      "rank": 12,
      "token_id": 17079,
      "text": " {{",
      "score": 10.875
    },
    {
      "rank": 13,
      "token_id": 1938,
      "text": " >",
      "score": 10.25
    },
    {
      "rank": 14,
      "token_id": 8973,
      "text": "{{",
      "score": 9.6875
    },
    {
      "rank": 15,
      "token_id": 1070,
      "text": " _",
      "score": 9.625
    },
    {
      "rank": 16,
      "token_id": 5154,
      "text": "[\\",
      "score": 9.5625
    },
    {
      "rank": 17,
      "token_id": 31421,
      "text": " \"[",
      "score": 9.5625
    },
    {
      "rank": 18,
      "token_id": 27366,
      "text": "[[",
      "score": 9.375
    },
    {
      "rank": 19,
      "token_id": 1013,
      "text": " <",
      "score": 9.1875
    },
    {
      "rank": 20,
      "token_id": 188,
      "text": "\r",
      "score": 9.0625
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
  "a_cone": 2.222046293582629,
  "ulens_score": 1.7470412406756386,
  "category_cosines": {
    "ambiguity": 0.011661306023597717,
    "incompleteness": 0.013675598427653313,
    "conflicting_evidence": -0.015437737107276917,
    "general_uncertainty": 0.012655769474804401
  },
  "top_uspace_tokens": [
    {
      "rank": 1,
      "token_id": 22231,
      "text": " mysterious",
      "score": 35.0
    },
    {
      "rank": 2,
      "token_id": 16503,
      "text": " weird",
      "score": 32.5
    },
    {
      "rank": 3,
      "token_id": 18971,
      "text": " mystery",
      "score": 32.5
    },
    {
      "rank": 4,
      "token_id": 17121,
      "text": " myster",
      "score": 29.25
    },
    {
      "rank": 5,
      "token_id": 8392,
      "text": " strange",
      "score": 28.25
    },
    {
      "rank": 6,
      "token_id": 15250,
      "text": " surprising",
      "score": 25.625
    },
    {
      "rank": 7,
      "token_id": 10796,
      "text": " surprise",
      "score": 25.5
    },
    {
      "rank": 8,
      "token_id": 14522,
      "text": " unexpected",
      "score": 25.5
    },
    {
      "rank": 9,
      "token_id": 5397,
      "text": " surpr",
      "score": 24.375
    },
    {
      "rank": 10,
      "token_id": 16015,
      "text": " unusual",
      "score": 23.5
    },
    {
      "rank": 11,
      "token_id": 14321,
      "text": "\u602a",
      "score": 21.75
    },
    {
      "rank": 12,
      "token_id": 28281,
      "text": " accidentally",
      "score": 21.375
    },
    {
      "rank": 13,
      "token_id": 28580,
      "text": " strang",
      "score": 21.375
    },
    {
      "rank": 14,
      "token_id": 8965,
      "text": " suddenly",
      "score": 20.875
    },
    {
      "rank": 15,
      "token_id": 6509,
      "text": " stuff",
      "score": 20.5
    },
    {
      "rank": 16,
      "token_id": 22900,
      "text": "\u610f\u5916",
      "score": 20.5
    },
    {
      "rank": 17,
      "token_id": 7103,
      "text": "\u5947",
      "score": 20.375
    },
    {
      "rank": 18,
      "token_id": 29437,
      "text": " suspicious",
      "score": 20.125
    },
    {
      "rank": 19,
      "token_id": 14744,
      "text": " funny",
      "score": 19.875
    },
    {
      "rank": 20,
      "token_id": 30499,
      "text": " tricky",
      "score": 19.75
    }
  ],
  "top_jlens_tokens": [
    {
      "rank": 1,
      "token_id": 572,
      "text": " *",
      "score": 10.8125
    },
    {
      "rank": 2,
      "token_id": 23691,
      "text": " defaults",
      "score": 9.5
    },
    {
      "rank": 3,
      "token_id": 6065,
      "text": "\\!",
      "score": 9.5
    },
    {
      "rank": 4,
      "token_id": 4052,
      "text": " expected",
      "score": 9.1875
    },
    {
      "rank": 5,
      "token_id": 19476,
      "text": " ([",
      "score": 8.8125
    },
    {
      "rank": 6,
      "token_id": 6236,
      "text": " ``",
      "score": 8.6875
    },
    {
      "rank": 7,
      "token_id": 2220,
      "text": " `",
      "score": 7.875
    },
    {
      "rank": 8,
      "token_id": 12867,
      "text": " \\(",
      "score": 7.78125
    },
    {
      "rank": 9,
      "token_id": 1585,
      "text": " --",
      "score": 7.5625
    },
    {
      "rank": 10,
      "token_id": 6044,
      "text": " assume",
      "score": 7.5625
    },
    {
      "rank": 11,
      "token_id": 4449,
      "text": " favor",
      "score": 7.40625
    },
    {
      "rank": 12,
      "token_id": 821,
      "text": " [",
      "score": 7.375
    },
    {
      "rank": 13,
      "token_id": 16603,
      "text": " [[",
      "score": 7.375
    },
    {
      "rank": 14,
      "token_id": 11666,
      "text": " assumed",
      "score": 7.21875
    },
    {
      "rank": 15,
      "token_id": 6183,
      "text": " sex",
      "score": 7.125
    },
    {
      "rank": 16,
      "token_id": 17237,
      "text": " >=",
      "score": 7.125
    },
    {
      "rank": 17,
      "token_id": 20493,
      "text": "{*",
      "score": 7.09375
    },
    {
      "rank": 18,
      "token_id": 1938,
      "text": " >",
      "score": 7.09375
    },
    {
      "rank": 19,
      "token_id": 17052,
      "text": "[:",
      "score": 7.0625
    },
    {
      "rank": 20,
      "token_id": 1013,
      "text": " <",
      "score": 7.03125
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
  "a_cone": 2.1064626092584926,
  "ulens_score": 1.176897759850336,
  "category_cosines": {
    "ambiguity": 0.006232768297195435,
    "incompleteness": 0.013115376234054565,
    "conflicting_evidence": -0.02406814508140087,
    "general_uncertainty": 0.024174001067876816
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
      "token_id": 18971,
      "text": " mystery",
      "score": 26.875
    },
    {
      "rank": 3,
      "token_id": 22231,
      "text": " mysterious",
      "score": 26.5
    },
    {
      "rank": 4,
      "token_id": 10796,
      "text": " surprise",
      "score": 26.375
    },
    {
      "rank": 5,
      "token_id": 5397,
      "text": " surpr",
      "score": 23.875
    },
    {
      "rank": 6,
      "token_id": 17121,
      "text": " myster",
      "score": 23.5
    },
    {
      "rank": 7,
      "token_id": 8392,
      "text": " strange",
      "score": 23.125
    },
    {
      "rank": 8,
      "token_id": 15250,
      "text": " surprising",
      "score": 22.375
    },
    {
      "rank": 9,
      "token_id": 12933,
      "text": " suspect",
      "score": 21.875
    },
    {
      "rank": 10,
      "token_id": 6509,
      "text": " stuff",
      "score": 20.125
    },
    {
      "rank": 11,
      "token_id": 26573,
      "text": " suspicion",
      "score": 19.625
    },
    {
      "rank": 12,
      "token_id": 10054,
      "text": "\u60ca",
      "score": 19.375
    },
    {
      "rank": 13,
      "token_id": 29437,
      "text": " suspicious",
      "score": 18.875
    },
    {
      "rank": 14,
      "token_id": 25196,
      "text": " suspected",
      "score": 18.625
    },
    {
      "rank": 15,
      "token_id": 14522,
      "text": " unexpected",
      "score": 18.625
    },
    {
      "rank": 16,
      "token_id": 16015,
      "text": " unusual",
      "score": 18.375
    },
    {
      "rank": 17,
      "token_id": 1782,
      "text": "thing",
      "score": 18.375
    },
    {
      "rank": 18,
      "token_id": 30499,
      "text": " tricky",
      "score": 18.125
    },
    {
      "rank": 19,
      "token_id": 12081,
      "text": " surprised",
      "score": 17.75
    },
    {
      "rank": 20,
      "token_id": 28580,
      "text": " strang",
      "score": 17.375
    }
  ],
  "top_jlens_tokens": [
    {
      "rank": 1,
      "token_id": 572,
      "text": " *",
      "score": 11.8125
    },
    {
      "rank": 2,
      "token_id": 4052,
      "text": " expected",
      "score": 9.6875
    },
    {
      "rank": 3,
      "token_id": 19476,
      "text": " ([",
      "score": 8.875
    },
    {
      "rank": 4,
      "token_id": 2220,
      "text": " `",
      "score": 8.6875
    },
    {
      "rank": 5,
      "token_id": 821,
      "text": " [",
      "score": 8.625
    },
    {
      "rank": 6,
      "token_id": 1585,
      "text": " --",
      "score": 8.5625
    },
    {
      "rank": 7,
      "token_id": 6236,
      "text": " ``",
      "score": 8.25
    },
    {
      "rank": 8,
      "token_id": 2312,
      "text": " ==",
      "score": 8.1875
    },
    {
      "rank": 9,
      "token_id": 9257,
      "text": " unknown",
      "score": 7.84375
    },
    {
      "rank": 10,
      "token_id": 6044,
      "text": " assume",
      "score": 7.46875
    },
    {
      "rank": 11,
      "token_id": 1938,
      "text": " >",
      "score": 7.34375
    },
    {
      "rank": 12,
      "token_id": 17237,
      "text": " >=",
      "score": 7.34375
    },
    {
      "rank": 13,
      "token_id": 440,
      "text": " \"",
      "score": 7.28125
    },
    {
      "rank": 14,
      "token_id": 23691,
      "text": " defaults",
      "score": 7.25
    },
    {
      "rank": 15,
      "token_id": 545,
      "text": " but",
      "score": 7.1875
    },
    {
      "rank": 16,
      "token_id": 6183,
      "text": " sex",
      "score": 7.09375
    },
    {
      "rank": 17,
      "token_id": 58,
      "text": "[",
      "score": 7.03125
    },
    {
      "rank": 18,
      "token_id": 4077,
      "text": " http",
      "score": 7.0
    },
    {
      "rank": 19,
      "token_id": 334,
      "text": " (",
      "score": 6.9375
    },
    {
      "rank": 20,
      "token_id": 4813,
      "text": " ~",
      "score": 6.9375
    }
  ]
}
```

```json
{
  "case_id": "sample_9",
  "call": "f('#284376598')",
  "expected": "False",
  "answer": "True",
  "correct": false,
  "location": "prompt_end",
  "read_token_index": 63,
  "read_token_text": " ==",
  "a_cone": 2.0939706733539825,
  "ulens_score": 1.8886900519867527,
  "category_cosines": {
    "ambiguity": 0.006739563308656216,
    "incompleteness": 0.013703101314604282,
    "conflicting_evidence": -0.019432993605732918,
    "general_uncertainty": 0.014959060586988926
  },
  "top_uspace_tokens": [
    {
      "rank": 1,
      "token_id": 22231,
      "text": " mysterious",
      "score": 31.625
    },
    {
      "rank": 2,
      "token_id": 18971,
      "text": " mystery",
      "score": 30.375
    },
    {
      "rank": 3,
      "token_id": 16503,
      "text": " weird",
      "score": 28.375
    },
    {
      "rank": 4,
      "token_id": 17121,
      "text": " myster",
      "score": 26.75
    },
    {
      "rank": 5,
      "token_id": 8392,
      "text": " strange",
      "score": 25.5
    },
    {
      "rank": 6,
      "token_id": 10796,
      "text": " surprise",
      "score": 24.0
    },
    {
      "rank": 7,
      "token_id": 5397,
      "text": " surpr",
      "score": 22.25
    },
    {
      "rank": 8,
      "token_id": 15250,
      "text": " surprising",
      "score": 21.625
    },
    {
      "rank": 9,
      "token_id": 16015,
      "text": " unusual",
      "score": 21.125
    },
    {
      "rank": 10,
      "token_id": 6509,
      "text": " stuff",
      "score": 20.75
    },
    {
      "rank": 11,
      "token_id": 14522,
      "text": " unexpected",
      "score": 19.875
    },
    {
      "rank": 12,
      "token_id": 28580,
      "text": " strang",
      "score": 19.125
    },
    {
      "rank": 13,
      "token_id": 10054,
      "text": "\u60ca",
      "score": 19.0
    },
    {
      "rank": 14,
      "token_id": 1782,
      "text": "thing",
      "score": 18.875
    },
    {
      "rank": 15,
      "token_id": 14321,
      "text": "\u602a",
      "score": 18.875
    },
    {
      "rank": 16,
      "token_id": 7103,
      "text": "\u5947",
      "score": 18.5
    },
    {
      "rank": 17,
      "token_id": 12933,
      "text": " suspect",
      "score": 18.375
    },
    {
      "rank": 18,
      "token_id": 8965,
      "text": " suddenly",
      "score": 18.0
    },
    {
      "rank": 19,
      "token_id": 29437,
      "text": " suspicious",
      "score": 18.0
    },
    {
      "rank": 20,
      "token_id": 9257,
      "text": " unknown",
      "score": 17.75
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
      "token_id": 2220,
      "text": " `",
      "score": 11.3125
    },
    {
      "rank": 3,
      "token_id": 19476,
      "text": " ([",
      "score": 10.25
    },
    {
      "rank": 4,
      "token_id": 6236,
      "text": " ``",
      "score": 9.75
    },
    {
      "rank": 5,
      "token_id": 4052,
      "text": " expected",
      "score": 9.5
    },
    {
      "rank": 6,
      "token_id": 58,
      "text": "[",
      "score": 8.875
    },
    {
      "rank": 7,
      "token_id": 821,
      "text": " [",
      "score": 8.875
    },
    {
      "rank": 8,
      "token_id": 4077,
      "text": " http",
      "score": 8.5625
    },
    {
      "rank": 9,
      "token_id": 1585,
      "text": " --",
      "score": 8.5
    },
    {
      "rank": 10,
      "token_id": 2312,
      "text": " ==",
      "score": 8.1875
    },
    {
      "rank": 11,
      "token_id": 1013,
      "text": " <",
      "score": 8.0625
    },
    {
      "rank": 12,
      "token_id": 1070,
      "text": " _",
      "score": 7.71875
    },
    {
      "rank": 13,
      "token_id": 31421,
      "text": " \"[",
      "score": 7.6875
    },
    {
      "rank": 14,
      "token_id": 440,
      "text": " \"",
      "score": 7.59375
    },
    {
      "rank": 15,
      "token_id": 2044,
      "text": " =>",
      "score": 7.59375
    },
    {
      "rank": 16,
      "token_id": 1938,
      "text": " >",
      "score": 7.53125
    },
    {
      "rank": 17,
      "token_id": 24066,
      "text": "\u4eca\u65e5",
      "score": 7.46875
    },
    {
      "rank": 18,
      "token_id": 17052,
      "text": "[:",
      "score": 7.46875
    },
    {
      "rank": 19,
      "token_id": 6183,
      "text": " sex",
      "score": 7.375
    },
    {
      "rank": 20,
      "token_id": 26001,
      "text": " cached",
      "score": 7.375
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
  "a_cone": 2.3105054060877714,
  "ulens_score": 1.6905049557642666,
  "category_cosines": {
    "ambiguity": 0.007871018722653389,
    "incompleteness": 0.01923476904630661,
    "conflicting_evidence": -0.013845412991940975,
    "general_uncertainty": 0.01577804982662201
  },
  "top_uspace_tokens": [
    {
      "rank": 1,
      "token_id": 22231,
      "text": " mysterious",
      "score": 37.75
    },
    {
      "rank": 2,
      "token_id": 18971,
      "text": " mystery",
      "score": 35.25
    },
    {
      "rank": 3,
      "token_id": 17121,
      "text": " myster",
      "score": 32.25
    },
    {
      "rank": 4,
      "token_id": 8392,
      "text": " strange",
      "score": 28.25
    },
    {
      "rank": 5,
      "token_id": 16503,
      "text": " weird",
      "score": 27.625
    },
    {
      "rank": 6,
      "token_id": 16015,
      "text": " unusual",
      "score": 27.625
    },
    {
      "rank": 7,
      "token_id": 15250,
      "text": " surprising",
      "score": 25.5
    },
    {
      "rank": 8,
      "token_id": 10796,
      "text": " surprise",
      "score": 25.125
    },
    {
      "rank": 9,
      "token_id": 5397,
      "text": " surpr",
      "score": 24.5
    },
    {
      "rank": 10,
      "token_id": 28580,
      "text": " strang",
      "score": 23.125
    },
    {
      "rank": 11,
      "token_id": 14522,
      "text": " unexpected",
      "score": 21.875
    },
    {
      "rank": 12,
      "token_id": 14321,
      "text": "\u602a",
      "score": 21.875
    },
    {
      "rank": 13,
      "token_id": 7103,
      "text": "\u5947",
      "score": 21.125
    },
    {
      "rank": 14,
      "token_id": 12933,
      "text": " suspect",
      "score": 20.625
    },
    {
      "rank": 15,
      "token_id": 10054,
      "text": "\u60ca",
      "score": 20.625
    },
    {
      "rank": 16,
      "token_id": 28018,
      "text": "\u731c",
      "score": 20.375
    },
    {
      "rank": 17,
      "token_id": 29437,
      "text": " suspicious",
      "score": 20.25
    },
    {
      "rank": 18,
      "token_id": 8965,
      "text": " suddenly",
      "score": 20.125
    },
    {
      "rank": 19,
      "token_id": 26573,
      "text": " suspicion",
      "score": 20.0
    },
    {
      "rank": 20,
      "token_id": 9257,
      "text": " unknown",
      "score": 19.875
    }
  ],
  "top_jlens_tokens": [
    {
      "rank": 1,
      "token_id": 4052,
      "text": " expected",
      "score": 9.875
    },
    {
      "rank": 2,
      "token_id": 23691,
      "text": " defaults",
      "score": 9.8125
    },
    {
      "rank": 3,
      "token_id": 572,
      "text": " *",
      "score": 9.375
    },
    {
      "rank": 4,
      "token_id": 6236,
      "text": " ``",
      "score": 8.5625
    },
    {
      "rank": 5,
      "token_id": 24066,
      "text": "\u4eca\u65e5",
      "score": 8.5
    },
    {
      "rank": 6,
      "token_id": 9257,
      "text": " unknown",
      "score": 8.3125
    },
    {
      "rank": 7,
      "token_id": 2220,
      "text": " `",
      "score": 8.25
    },
    {
      "rank": 8,
      "token_id": 30385,
      "text": " Unknown",
      "score": 7.8125
    },
    {
      "rank": 9,
      "token_id": 15973,
      "text": "\u6628",
      "score": 7.75
    },
    {
      "rank": 10,
      "token_id": 32014,
      "text": "<\uff5cend\u2581of\u2581sentence\uff5c>",
      "score": 7.71875
    },
    {
      "rank": 11,
      "token_id": 1938,
      "text": " >",
      "score": 7.3125
    },
    {
      "rank": 12,
      "token_id": 3558,
      "text": " today",
      "score": 7.3125
    },
    {
      "rank": 13,
      "token_id": 4813,
      "text": " ~",
      "score": 7.28125
    },
    {
      "rank": 14,
      "token_id": 6044,
      "text": " assume",
      "score": 7.25
    },
    {
      "rank": 15,
      "token_id": 19476,
      "text": " ([",
      "score": 7.15625
    },
    {
      "rank": 16,
      "token_id": 9146,
      "text": " {@",
      "score": 7.0625
    },
    {
      "rank": 17,
      "token_id": 17237,
      "text": " >=",
      "score": 7.03125
    },
    {
      "rank": 18,
      "token_id": 1070,
      "text": " _",
      "score": 6.9375
    },
    {
      "rank": 19,
      "token_id": 26001,
      "text": " cached",
      "score": 6.90625
    },
    {
      "rank": 20,
      "token_id": 17079,
      "text": " {{",
      "score": 6.84375
    }
  ]
}
```
