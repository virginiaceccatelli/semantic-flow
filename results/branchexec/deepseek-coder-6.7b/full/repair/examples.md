# Repair examples (α = 0.8, if_true and answer_true)

## cruxeval_sample_66:0:0:0 (cruxeval)

```python
def f(text='happy', prefix='ha'):
    prefix_length = len(prefix)
    if text.startswith(prefix):
        return text[(prefix_length - 1) // 2:
                    (prefix_length + 1) // 2 * -1:-1]
    else:
        return text

assert f() ==
```

- `if text.startswith(prefix)` is **taken**; true output `''`, wrong-branch output `'happy'`
- unsteered: `'happy'`
- steered at the `if`: `'happy'`
- steered at the answer: `‘happy’`

## cruxeval_sample_68:0:0:0 (cruxeval)

```python
def f(text='omeunhwpvr.dq', pref='omeunh'):
    if text.startswith(pref):
        n = len(pref)
        text = '.'.join(text[n:].split('.')[1:] + text[:n].split('.')[:-1])
    return text

assert f() ==
```

- `if text.startswith(pref)` is **taken**; true output `'dq'`, wrong-branch output `'omeunhwpvr.dq'`
- unsteered: `'omeunhwpvr.dq'`
- steered at the `if`: `'omeunhwpvr.dq'`
- steered at the answer: `= 'omeunhwpvr.dq'`

## cruxeval_sample_86:0:1:1 (cruxeval)

```python
def f(instagram=['sdfs', 'drcr', '2e'], imgur=['sdfs', 'dr2c', 'QWERTY'], wins=1):
    photos = [instagram, imgur]
    if instagram == imgur:
        return wins
    if wins == 1:
        return photos.pop()
    else:
        photos.reverse()
        return photos.pop()

assert f() ==
```

- `if wins == 1` is **taken**; true output `['sdfs', 'dr2c', 'QWERTY']`, wrong-branch output `['sdfs', 'drcr', '2e']`
- unsteered: `['sdfs', 'drcr', '2e']`
- steered at the `if`: `['sdfs', 'drcr', '2e']`
- steered at the answer: `= ['sdfs', 'drcr', '2e']`

## cruxeval_sample_89:0:0:0 (cruxeval)

```python
def f(char='o'):
    if char not in 'aeiouAEIOU':
        return None
    if char in 'AEIOU':
        return char.lower()
    return char.upper()

assert f() ==
```

- `if char not in 'aeiouAEIOU'` is **not taken**; true output `'O'`, wrong-branch output `None`
- unsteered: `None`
- steered at the `if`: `'O'`
- steered at the answer: `fp('o')`

## cruxeval_sample_89:0:0:2 (cruxeval)

```python
def f(char='1'):
    if char not in 'aeiouAEIOU':
        return None
    if char in 'AEIOU':
        return char.lower()
    return char.upper()

assert f() ==
```

- `if char not in 'aeiouAEIOU'` is **taken**; true output `None`, wrong-branch output `'1'`
- unsteered: `'1'`
- steered at the `if`: `None`
- steered at the answer: `‘1’`

## cruxeval_sample_122:0:0:1 (cruxeval)

```python
def f(string='Nuxa?dlfuyjys'):
    if string[:4] != 'Nuva':
        return 'no'
    else:
        return string.rstrip()

assert f() ==
```

- `if string[:4] != 'Nuva'` is **taken**; true output `'no'`, wrong-branch output `'Nuxa?dlfuyjys'`
- unsteered: `'Nuxa?dlfuyjys'`
- steered at the `if`: `'Nuxa?dlfuyjys'`
- steered at the answer: `= 'Nuxa?dlfuyjys'`

## cruxeval_sample_122:0:0:2 (cruxeval)

```python
def f(string='Nuv1?dlfuyjys'):
    if string[:4] != 'Nuva':
        return 'no'
    else:
        return string.rstrip()

assert f() ==
```

- `if string[:4] != 'Nuva'` is **taken**; true output `'no'`, wrong-branch output `'Nuv1?dlfuyjys'`
- unsteered: `'Nuv1?dlfuyjys'`
- steered at the `if`: `'Nuv1?dlfuyjys'`
- steered at the answer: `= 'Nuv1?dlfuyjys'`

## cruxeval_sample_142:0:0:2 (cruxeval)

```python
def f(x='ykAfhp'):
    if x.islower():
        return x
    else:
        return x[::-1]

assert f() ==
```

- `if x.islower()` is **not taken**; true output `'phfAky'`, wrong-branch output `'ykAfhp'`
- unsteered: `'ykAfhp'`
- steered at the `if`: `'ykAfhp'`
- steered at the answer: `fpAyk`

## cruxeval_sample_225:0:0:0 (cruxeval)

```python
def f(text="54882"):
    if text.islower():
        return True
    return False

assert f() ==
```

- `if text.islower()` is **not taken**; true output `False`, wrong-branch output `True`
- unsteered: `True`
- steered at the `if`: `True`
- steered at the answer: `fp()`

## cruxeval_sample_237:0:0:0 (cruxeval)

```python
def f(text='uzlwaqiaj', char='u'):
    if char in text:
        suff, char, pref = text.partition(char)
        pref = suff[:-len(char)] + suff[len(char):] + char + pref
        return suff + char + pref
    return text

assert f() ==
```

- `if char in text` is **taken**; true output `'uuzlwaqiaj'`, wrong-branch output `'uzlwaqiaj'`
- unsteered: `'uzlwaqiaj'`
- steered at the `if`: `'uzlwaqiaj'`
- steered at the answer: `= 'uzlwaqiaj'`

## cruxeval_sample_253:0:0:0 (cruxeval)

```python
def f(text='kumwwfv', pref='k'):
    length = len(pref)
    if pref == text[:length]:
        return text[length:]
    return text

assert f() ==
```

- `if pref == text[:length]` is **taken**; true output `'umwwfv'`, wrong-branch output `'kumwwfv'`
- unsteered: `'kumwwfv'`
- steered at the `if`: `'kumwwfv'`
- steered at the answer: `= 'kumwwfv'`

## cruxeval_sample_277:0:0:0 (cruxeval)

```python
def f(lst=[1, 2, 3, 4], mode=1):
    result = [el for el in lst]
    if mode:
        result.reverse()
    return result

assert f() ==
```

- `if mode` is **taken**; true output `[4, 3, 2, 1]`, wrong-branch output `[1, 2, 3, 4]`
- unsteered: `[1, 2, 3, 4]`
- steered at the `if`: `[1, 2, 3, 4]`
- steered at the answer: `…`

## cruxeval_sample_287:0:0:0 (cruxeval)

```python
def f(name='Pinneaple'):
    if name.islower():
        name = name.upper()
    else:
        name = name.lower()
    return name

assert f() ==
```

- `if name.islower()` is **not taken**; true output `'pinneaple'`, wrong-branch output `'PINNEAPLE'`
- unsteered: `'PINNEAPLE'`
- steered at the `if`: `'PINNEAPLE'`
- steered at the answer: `fp()`

## cruxeval_sample_290:0:1:1 (cruxeval)

```python
def f(text='abixaaaily', prefix='l'):
    if text.startswith(prefix):
        return text.removeprefix(prefix)
    if prefix in text:
        return text.replace(prefix, '').strip()
    return text.upper()

assert f() ==
```

- `if prefix in text` is **taken**; true output `'abixaaaiy'`, wrong-branch output `'ABIXAAAILY'`
- unsteered: `'ABIXAAAILY'`
- steered at the `if`: `'ABIXAAAILY'`
- steered at the answer: `‘ABIXAAAILY’`

## cruxeval_sample_290:0:1:2 (cruxeval)

```python
def f(text='abixaaaily', prefix='aa'):
    if text.startswith(prefix):
        return text.removeprefix(prefix)
    if prefix in text:
        return text.replace(prefix, '').strip()
    return text.upper()

assert f() ==
```

- `if prefix in text` is **taken**; true output `'abixaily'`, wrong-branch output `'ABIXAAAILY'`
- unsteered: `'ABIXAAAILY'`
- steered at the `if`: `'ABIXAAAILY'`
- steered at the answer: `‘ABIXAAAILY’`

