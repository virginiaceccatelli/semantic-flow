# Steering examples (real, capable members, primary dose)

Dose 0.2; blocks [11, 12, 13, 14, 15].

## cruxeval_sample_225:0:0:0 (cruxeval)

```python
def f(text="54882"):
    if text.islower():
        return True
    return False

assert f() ==
```

- branch `if text.islower()` is **not taken**
- true output `False`; counterfactual (branch flipped) `True`
- unsteered greedy: `False`
- steered toward the other branch: `True`

## cruxeval_sample_287:0:0:1 (cruxeval)

```python
def f(name='ainneaple'):
    if name.islower():
        name = name.upper()
    else:
        name = name.lower()
    return name

assert f() ==
```

- branch `if name.islower()` is **taken**
- true output `'AINNEAPLE'`; counterfactual (branch flipped) `'ainneaple'`
- unsteered greedy: `'AINNEAPLE'`
- steered toward the other branch: `'ainneaple'`

## cruxeval_sample_287:0:0:2 (cruxeval)

```python
def f(name='einneaple'):
    if name.islower():
        name = name.upper()
    else:
        name = name.lower()
    return name

assert f() ==
```

- branch `if name.islower()` is **taken**
- true output `'EINNEAPLE'`; counterfactual (branch flipped) `'einneaple'`
- unsteered greedy: `'EINNEAPLE'`
- steered toward the other branch: `'einneaple'`

## cruxeval_sample_476:0:0:0 (cruxeval)

```python
def f(a="booty boot-boot bootclass", split_on='k'):
    t = a.split()
    a = []
    for i in t:
        for j in i:
            a.append(j)
    if split_on in a:
        return True
    else:
        return False

assert f() ==
```

- branch `if split_on in a` is **not taken**
- true output `False`; counterfactual (branch flipped) `True`
- unsteered greedy: `False`
- steered toward the other branch: `False`

## cruxeval_sample_789:0:0:0 (cruxeval)

```python
def f(text='bR', n=-1):
    if n < 0 or len(text) <= n:
        return text
    result = text[0 : n]
    i = len(result) - 1
    while i >= 0:
        if result[i] != text[i]:
            break
        i -= 1
    return text[0 : i + 1]

assert f() ==
```

- branch `if n < 0 or len(text) <= n` is **taken**
- true output `'bR'`; counterfactual (branch flipped) `''`
- unsteered greedy: `'bR'`
- steered toward the other branch: `'bR'`

## mbpp_650:0:0:0 (mbpp)

```python
def are_Equal(arr1=[1,2,3], arr2=[3,2,1], n=3, m=3):
    if (n != m):
        return False
    arr1.sort()
    arr2.sort()
    for i in range(0,n - 1):
        if (arr1[i] != arr2[i]):
            return False
    return True

assert are_Equal() ==
```

- branch `if n != m` is **not taken**
- true output `True`; counterfactual (branch flipped) `False`
- unsteered greedy: `True`
- steered toward the other branch: `True`

## mbpp_703:0:0:1 (mbpp)

```python
def is_key_present(d={1: 10, 2: 20, 3: 30, 4: 40, 7: 50, 6: 60}, x=5):
  if x in d:
    return True
  else:
     return False

assert is_key_present() ==
```

- branch `if x in d` is **not taken**
- true output `False`; counterfactual (branch flipped) `True`
- unsteered greedy: `False`
- steered toward the other branch: `False`

## mbpp_703:0:0:2 (mbpp)

```python
def is_key_present(d={1: 10, 2: 20, 3: 30, 4: 40, 4: 50, 6: 60}, x=5):
  if x in d:
    return True
  else:
     return False

assert is_key_present() ==
```

- branch `if x in d` is **not taken**
- true output `False`; counterfactual (branch flipped) `True`
- unsteered greedy: `False`
- steered toward the other branch: `False`

## mbpp_703:1:0:1 (mbpp)

```python
def is_key_present(d={1: 10, 2: 20, 3: 30, 4: 40, 5: 50, 6: 60}, x=7):
  if x in d:
    return True
  else:
     return False

assert is_key_present() ==
```

- branch `if x in d` is **not taken**
- true output `False`; counterfactual (branch flipped) `True`
- unsteered greedy: `False`
- steered toward the other branch: `False`

## mbpp_703:1:0:2 (mbpp)

```python
def is_key_present(d={1: 10, 2: 20, 3: 30, 4: 40, 5: 50, 8: 60}, x=6):
  if x in d:
    return True
  else:
     return False

assert is_key_present() ==
```

- branch `if x in d` is **not taken**
- true output `False`; counterfactual (branch flipped) `True`
- unsteered greedy: `False`
- steered toward the other branch: `False`

## mbpp_711:0:2:0 (mbpp)

```python
def product_Equal(n=2841):
    if n < 10: 
        return False
    prodOdd = 1; prodEven = 1
    while n > 0: 
        digit = n % 10
        prodOdd *= digit 
        n = n//10
        if n == 0: 
            break; 
        digit = n % 10
        prodEven *= digit 
        n = n//10
    if prodOdd == prodEven: 
        return True
    return False

assert product_Equal() ==
```

- branch `if prodOdd == prodEven` is **taken**
- true output `True`; counterfactual (branch flipped) `False`
- unsteered greedy: `True`
- steered toward the other branch: `False`

## mbpp_714:2:0:1 (mbpp)

```python
def count_Fac(n=6):
    m = n 
    count = 0
    i = 2
    while((i * i) <= m): 
        total = 0
        while (n % i == 0): 
            n /= i 
            total += 1 
        temp = 0
        j = 1
        while((temp + j) <= total): 
            temp += j 
            count += 1
            j += 1 
        i += 1
    if (n != 1): 
        count += 1 
    return count 

assert count_Fac() == 
```

- branch `if n != 1` is **taken**
- true output `2`; counterfactual (branch flipped) `1`
- unsteered greedy: `1`
- steered toward the other branch: `1`

## mbpp_719:0:0:2 (mbpp)

```python
import re
def text_match(text='xc'):
        patterns = 'ab*?'
        if re.search(patterns,  text):
                return 'Found a match!'
        else:
                return('Not matched!')

assert text_match() ==
```

- branch `if re.search(patterns,  text)` is **not taken**
- true output `'Not matched!'`; counterfactual (branch flipped) `'Found a match!'`
- unsteered greedy: `'Not matched!'`
- steered toward the other branch: `'Found a match!'`

## mbpp_756:0:0:2 (mbpp)

```python
import re
def text_match_zero_one(text='xc'):
        patterns = 'ab?'
        if re.search(patterns,  text):
                return 'Found a match!'
        else:
                return('Not matched!')

assert text_match_zero_one() ==
```

- branch `if re.search(patterns,  text)` is **not taken**
- true output `'Not matched!'`; counterfactual (branch flipped) `'Found a match!'`
- unsteered greedy: `'Not matched!'`
- steered toward the other branch: `'Not matched!'`

## mbpp_762:0:0:1 (mbpp)

```python
def check_monthnumber_number(monthnum3=5):
  if(monthnum3==4 or monthnum3==6 or monthnum3==9 or monthnum3==11):
    return True
  else:
    return False

assert check_monthnumber_number() ==
```

- branch `if monthnum3==4 or monthnum3==6 or monthnum3==9 or monthnum3==11` is **not taken**
- true output `False`; counterfactual (branch flipped) `True`
- unsteered greedy: `False`
- steered toward the other branch: `True`

## mbpp_762:0:0:2 (mbpp)

```python
def check_monthnumber_number(monthnum3=0):
  if(monthnum3==4 or monthnum3==6 or monthnum3==9 or monthnum3==11):
    return True
  else:
    return False

assert check_monthnumber_number() ==
```

- branch `if monthnum3==4 or monthnum3==6 or monthnum3==9 or monthnum3==11` is **not taken**
- true output `False`; counterfactual (branch flipped) `True`
- unsteered greedy: `False`
- steered toward the other branch: `True`

## mbpp_762:1:0:0 (mbpp)

```python
def check_monthnumber_number(monthnum3=2):
  if(monthnum3==4 or monthnum3==6 or monthnum3==9 or monthnum3==11):
    return True
  else:
    return False

assert check_monthnumber_number() ==
```

- branch `if monthnum3==4 or monthnum3==6 or monthnum3==9 or monthnum3==11` is **not taken**
- true output `False`; counterfactual (branch flipped) `True`
- unsteered greedy: `False`
- steered toward the other branch: `True`

## mbpp_762:2:0:0 (mbpp)

```python
def check_monthnumber_number(monthnum3=12):
  if(monthnum3==4 or monthnum3==6 or monthnum3==9 or monthnum3==11):
    return True
  else:
    return False

assert check_monthnumber_number() ==
```

- branch `if monthnum3==4 or monthnum3==6 or monthnum3==9 or monthnum3==11` is **not taken**
- true output `False`; counterfactual (branch flipped) `True`
- unsteered greedy: `False`
- steered toward the other branch: `True`

## mbpp_801:2:0:1 (mbpp)

```python
def test_three_equal(x=1, y=4, z=2):
  result= set([x,y,z])
  if len(result)==3:
    return 0
  else:
    return (4-len(result))

assert test_three_equal() == 
```

- branch `if len(result)==3` is **taken**
- true output `0`; counterfactual (branch flipped) `1`
- unsteered greedy: `0`
- steered toward the other branch: `1`

## mbpp_820:0:0:1 (mbpp)

```python
def check_monthnum_number(monthnum1=0):
  if monthnum1 == 2:
    return True
  else:
    return False

assert check_monthnum_number() ==
```

- branch `if monthnum1 == 2` is **not taken**
- true output `False`; counterfactual (branch flipped) `True`
- unsteered greedy: `False`
- steered toward the other branch: `True`

