# Steering examples (real, capable members, primary dose)

Dose 0.2; blocks [14, 15, 16, 17, 18].

## cruxeval_sample_89:0:0:1 (cruxeval)

```python
def f(char='x'):
    if char not in 'aeiouAEIOU':
        return None
    if char in 'AEIOU':
        return char.lower()
    return char.upper()

assert f() ==
```

- branch `if char not in 'aeiouAEIOU'` is **taken**
- true output `None`; counterfactual (branch flipped) `'X'`
- unsteered greedy: `None`
- steered toward the other branch: `None`

## cruxeval_sample_277:0:0:1 (cruxeval)

```python
def f(lst=[1, 2, 3, 4], mode=0):
    result = [el for el in lst]
    if mode:
        result.reverse()
    return result

assert f() ==
```

- branch `if mode` is **not taken**
- true output `[1, 2, 3, 4]`; counterfactual (branch flipped) `[4, 3, 2, 1]`
- unsteered greedy: `[1, 2, 3, 4]`
- steered toward the other branch: `[1, 2, 3, 4]`

## cruxeval_sample_290:0:1:0 (cruxeval)

```python
def f(text='abixaaaily', prefix='al'):
    if text.startswith(prefix):
        return text.removeprefix(prefix)
    if prefix in text:
        return text.replace(prefix, '').strip()
    return text.upper()

assert f() ==
```

- branch `if prefix in text` is **not taken**
- true output `'ABIXAAAILY'`; counterfactual (branch flipped) `'abixaaaily'`
- unsteered greedy: `'ABIXAAAILY'`
- steered toward the other branch: `'ABIXAAAILY'`

## cruxeval_sample_673:0:1:2 (cruxeval)

```python
def f(string='c1'):
    if string.isupper():
        return string.lower()
    elif string.islower():
        return string.upper()
    return string

assert f() ==
```

- branch `if string.islower()` is **taken**
- true output `'C1'`; counterfactual (branch flipped) `'c1'`
- unsteered greedy: `'C1'`
- steered toward the other branch: `'C1'`

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

## mbpp_677:1:0:0 (mbpp)

```python
def validity_triangle(a=45, b=75, c=60):
 total = a + b + c
 if total == 180:
    return True
 else:
    return False

assert validity_triangle() ==
```

- branch `if total == 180` is **taken**
- true output `True`; counterfactual (branch flipped) `False`
- unsteered greedy: `True`
- steered toward the other branch: `True`

## mbpp_681:1:0:0 (mbpp)

```python
def smallest_Divisor(n=25):
    if (n % 2 == 0): 
        return 2; 
    i = 3;  
    while (i*i <= n): 
        if (n % i == 0): 
            return i; 
        i += 2; 
    return n; 

assert smallest_Divisor() == 
```

- branch `if n % 2 == 0` is **not taken**
- true output `5`; counterfactual (branch flipped) `2`
- unsteered greedy: `5`
- steered toward the other branch: `5`

## mbpp_703:0:0:0 (mbpp)

```python
def is_key_present(d={1: 10, 2: 20, 3: 30, 4: 40, 5: 50, 6: 60}, x=5):
  if x in d:
    return True
  else:
     return False

assert is_key_present() ==
```

- branch `if x in d` is **taken**
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

## mbpp_703:1:0:0 (mbpp)

```python
def is_key_present(d={1: 10, 2: 20, 3: 30, 4: 40, 5: 50, 6: 60}, x=6):
  if x in d:
    return True
  else:
     return False

assert is_key_present() ==
```

- branch `if x in d` is **taken**
- true output `True`; counterfactual (branch flipped) `False`
- unsteered greedy: `True`
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
- steered toward the other branch: `False`

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
- steered toward the other branch: `False`

## mbpp_874:2:0:2 (mbpp)

```python
def check_Concat(str1='ab', str2="ab"):
    N = len(str1)
    M = len(str2)
    if (N % M != 0):
        return False
    for i in range(N):
        if (str1[i] != str2[i % M]):
            return False         
    return True

assert check_Concat() ==
```

- branch `if N % M != 0` is **not taken**
- true output `True`; counterfactual (branch flipped) `False`
- unsteered greedy: `True`
- steered toward the other branch: `True`

## mbpp_884:0:0:0 (mbpp)

```python
def all_Bits_Set_In_The_Given_Range(n=10, l=2, r=1):
    num = ((1 << r) - 1) ^ ((1 << (l - 1)) - 1) 
    new_num = n & num 
    if (num == new_num): 
        return True
    return False

assert all_Bits_Set_In_The_Given_Range() ==
```

- branch `if num == new_num` is **taken**
- true output `True`; counterfactual (branch flipped) `False`
- unsteered greedy: `True`
- steered toward the other branch: `True`

## mbpp_885:2:0:2 (mbpp)

```python
def is_Isomorphic(str1='aa', str2="aa"):
    dict_str1 = {}
    dict_str2 = {}
    for i, value in enumerate(str1):
        dict_str1[value] = dict_str1.get(value,[]) + [i]        
    for j, value in enumerate(str2):
        dict_str2[value] = dict_str2.get(value,[]) + [j]
    if sorted(dict_str1.values()) == sorted(dict_str2.values()):
        return True
    else:
        return False

assert is_Isomorphic() ==
```

- branch `if sorted(dict_str1.values()) == sorted(dict_str2.values())` is **taken**
- true output `True`; counterfactual (branch flipped) `False`
- unsteered greedy: `True`
- steered toward the other branch: `True`

## mbpp_891:0:0:1 (mbpp)

```python
def same_Length(A=12, B=3):
    while (A > 0 and B > 0): 
        A = A / 10; 
        B = B / 10; 
    if (A == 0 and B == 0): 
        return True; 
    return False; 

assert same_Length() ==
```

- branch `if A == 0 and B == 0` is **taken**
- true output `True`; counterfactual (branch flipped) `False`
- unsteered greedy: `True`
- steered toward the other branch: `True`

## mbpp_891:1:0:0 (mbpp)

```python
def same_Length(A=2, B=2):
    while (A > 0 and B > 0): 
        A = A / 10; 
        B = B / 10; 
    if (A == 0 and B == 0): 
        return True; 
    return False; 

assert same_Length() ==
```

- branch `if A == 0 and B == 0` is **taken**
- true output `True`; counterfactual (branch flipped) `False`
- unsteered greedy: `True`
- steered toward the other branch: `True`

## mbpp_891:2:0:0 (mbpp)

```python
def same_Length(A=10, B=20):
    while (A > 0 and B > 0): 
        A = A / 10; 
        B = B / 10; 
    if (A == 0 and B == 0): 
        return True; 
    return False; 

assert same_Length() ==
```

- branch `if A == 0 and B == 0` is **taken**
- true output `True`; counterfactual (branch flipped) `False`
- unsteered greedy: `True`
- steered toward the other branch: `True`

## mbpp_900:0:0:0 (mbpp)

```python
import re
def match_num(string='5-2345861'):
    text = re.compile(r"^5")
    if text.match(string):
        return True
    else:
        return False

assert match_num() ==
```

- branch `if text.match(string)` is **taken**
- true output `True`; counterfactual (branch flipped) `False`
- unsteered greedy: `True`
- steered toward the other branch: `True`

## mbpp_924:1:0:2 (mbpp)

```python
def max_of_two(x=19, y=30):
    if x > y:
        return x
    return y

assert max_of_two() == 
```

- branch `if x > y` is **not taken**
- true output `30`; counterfactual (branch flipped) `19`
- unsteered greedy: `30`
- steered toward the other branch: `30`

