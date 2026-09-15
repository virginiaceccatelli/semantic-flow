# CruxEval preflight review

CPU-only pilot; stopped before activation extraction or probing.

| id       |   n_loads |   n_candidate_uses |   n_defuse_edges |   n_ambiguous_uses |   n_shadowing_uses |   n_legacy_edges_rejected | execution_passed   | alignment_passed   |
|:---------|----------:|-------------------:|-----------------:|-------------------:|-------------------:|--------------------------:|:-------------------|:-------------------|
| sample_0 |         7 |                  7 |                7 |                  0 |                  0 |                         0 | True               | True               |
| sample_1 |         7 |                  6 |                6 |                  0 |                  0 |                         0 | True               | True               |
| sample_2 |         7 |                  6 |                6 |                  0 |                  0 |                         0 | True               | True               |
| sample_3 |         5 |                  4 |                4 |                  0 |                  0 |                         0 | True               | True               |
| sample_4 |         2 |                  2 |                2 |                  0 |                  0 |                         0 | True               | True               |

Sample shadowing: 0 uses in 0 programs; binding adequate: False.

Measured defuse_edge surface floor: **0.602310**, shuffled-label accuracy 0.593429, selectivity 0.008881; train-majority control 0.738336.

## sample_0

```python
def f(nums):
    output = []
    for n in nums:
        output.append((nums.count(n), n))
    output.sort(reverse=True)
    return output
```

Input: `[1, 1, 3, 1, 3, 1]`

Recorded output: `[(4, 1), (4, 1), (4, 1), (4, 1), (2, 3), (2, 3)]`

| use (line:column) | reaching definitions | binding | last covering token | token text |
|---|---|---|---|---|
| nums@3:13 | nums@1:6 | unique | 17 | 's' |
| output@4:8 | output@2:4 | unique | 21 | ' output' |
| nums@4:23 | nums@1:6 | unique | 26 | 's' |
| n@4:34 | n@3:8 | unique | 30 | 'n' |
| n@4:38 | n@3:8 | unique | 32 | ' n' |
| output@5:4 | output@2:4 | unique | 36 | ' output' |
| output@6:11 | output@2:4 | unique | 48 | ' output' |

## sample_1

```python
def f(a, b, c):
    result = {}
    for d in a, b, c:
        result.update(dict.fromkeys(d))
    return result
```

Input: `(1, ), (1, ), (1, 2)`

Recorded output: `{1: None, 2: None}`

| use (line:column) | reaching definitions | binding | last covering token | token text |
|---|---|---|---|---|
| a@3:13 | a@1:6 | unique | 19 | ' a' |
| b@3:16 | b@1:9 | unique | 21 | ' b' |
| c@3:19 | c@1:12 | unique | 23 | ' c' |
| result@4:8 | result@2:4 | unique | 27 | ' result' |
| d@4:36 | d@3:8 | unique | 36 | 'd' |
| result@5:11 | result@2:4 | unique | 41 | ' result' |

## sample_2

```python
def f(text):
    new_text = list(text)
    for i in '+':
        if i in new_text:
            new_text.remove(i)
    return ''.join(new_text)
```

Input: `'hbtofdeiequ'`

Recorded output: `'hbtofdeiequ'`

| use (line:column) | reaching definitions | binding | last covering token | token text |
|---|---|---|---|---|
| text@2:20 | text@1:6 | unique | 13 | 'text' |
| i@4:11 | i@3:8 | unique | 26 | ' i' |
| new_text@4:16 | new_text@2:4 | unique | 30 | 'text' |
| new_text@5:12 | new_text@2:4 | unique | 36 | 'text' |
| i@5:28 | i@3:8 | unique | 40 | 'i' |
| new_text@6:19 | new_text@2:4 | unique | 51 | 'text' |

## sample_3

```python
def f(text, value):
    text_list = list(text)
    text_list.append(value)
    return ''.join(text_list)
```

Input: `'bcksrut', 'q'`

Recorded output: `'bcksrutq'`

| use (line:column) | reaching definitions | binding | last covering token | token text |
|---|---|---|---|---|
| text@2:21 | text@1:6 | unique | 15 | 'text' |
| text_list@3:4 | text_list@2:4 | unique | 21 | 'list' |
| value@3:21 | value@1:12 | unique | 25 | 'value' |
| text_list@4:19 | text_list@2:4 | unique | 36 | 'list' |

## sample_4

```python
def f(array):
    s = ' '
    s += ''.join(array)
    return s
```

Input: `[' ', '  ', '    ', '   ']`

Recorded output: `'           '`

| use (line:column) | reaching definitions | binding | last covering token | token text |
|---|---|---|---|---|
| array@3:17 | array@1:6 | unique | 19 | 'array' |
| s@4:11 | s@3:4 | unique | 24 | ' s' |

All listed anchors and input/output checks passed. Columns are zero-based character offsets.
May-reaching definitions are set-valued static approximations, not executed-path labels.
The pilot authorizes no representational conclusion; later stages require review and a floor on their exact population.
