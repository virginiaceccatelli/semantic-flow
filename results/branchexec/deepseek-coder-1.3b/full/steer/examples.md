# Steering examples (real, capable members, primary dose)

Dose 0.02; blocks [6, 7, 8, 9, 10].

## cruxeval_sample_673:0:0:1 (cruxeval)

```python
def f(string='1A'):
    if string.isupper():
        return string.lower()
    elif string.islower():
        return string.upper()
    return string

assert f() ==
```

- branch `if string.isupper()` is **taken**
- true output `'1a'`; counterfactual (branch flipped) `'1A'`
- unsteered greedy: `'1a'`
- steered toward the other branch: `'1a'`

## mbpp_637:1:0:0 (mbpp)

```python
def noprofit_noloss(actual_cost=100, sale_amount=100):
  if(sale_amount == actual_cost):
    return True
  else:
    return False

assert noprofit_noloss() ==
```

- branch `if sale_amount == actual_cost` is **taken**
- true output `True`; counterfactual (branch flipped) `False`
- unsteered greedy: `True`
- steered toward the other branch: `True`

## mbpp_677:1:0:1 (mbpp)

```python
def validity_triangle(a=45, b=75, c=70):
 total = a + b + c
 if total == 180:
    return True
 else:
    return False

assert validity_triangle() ==
```

- branch `if total == 180` is **not taken**
- true output `False`; counterfactual (branch flipped) `True`
- unsteered greedy: `False`
- steered toward the other branch: `False`

## mbpp_677:1:0:2 (mbpp)

```python
def validity_triangle(a=47, b=75, c=60):
 total = a + b + c
 if total == 180:
    return True
 else:
    return False

assert validity_triangle() ==
```

- branch `if total == 180` is **not taken**
- true output `False`; counterfactual (branch flipped) `True`
- unsteered greedy: `False`
- steered toward the other branch: `False`

## mbpp_677:2:0:1 (mbpp)

```python
def validity_triangle(a=30, b=49, c=100):
 total = a + b + c
 if total == 180:
    return True
 else:
    return False

assert validity_triangle() ==
```

- branch `if total == 180` is **not taken**
- true output `False`; counterfactual (branch flipped) `True`
- unsteered greedy: `False`
- steered toward the other branch: `False`

## mbpp_677:2:0:2 (mbpp)

```python
def validity_triangle(a=40, b=50, c=100):
 total = a + b + c
 if total == 180:
    return True
 else:
    return False

assert validity_triangle() ==
```

- branch `if total == 180` is **not taken**
- true output `False`; counterfactual (branch flipped) `True`
- unsteered greedy: `False`
- steered toward the other branch: `False`

## mbpp_681:0:0:1 (mbpp)

```python
def smallest_Divisor(n=11):
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
- true output `11`; counterfactual (branch flipped) `2`
- unsteered greedy: `11`
- steered toward the other branch: `11`

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
- steered toward the other branch: `False`

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
- steered toward the other branch: `False`

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

## mbpp_871:2:1:1 (mbpp)

```python
def are_Rotations(string1="abacd", string2='ceaba'):
    size1 = len(string1) 
    size2 = len(string2) 
    temp = '' 
    if size1 != size2: 
        return False
    temp = string1 + string1 
    if (temp.count(string2)> 0): 
        return True
    else: 
        return False

assert are_Rotations() ==
```

- branch `if temp.count(string2)> 0` is **not taken**
- true output `False`; counterfactual (branch flipped) `True`
- unsteered greedy: `False`
- steered toward the other branch: `False`

## mbpp_871:2:1:2 (mbpp)

```python
def are_Rotations(string1="abacd", string2='caaba'):
    size1 = len(string1) 
    size2 = len(string2) 
    temp = '' 
    if size1 != size2: 
        return False
    temp = string1 + string1 
    if (temp.count(string2)> 0): 
        return True
    else: 
        return False

assert are_Rotations() ==
```

- branch `if temp.count(string2)> 0` is **not taken**
- true output `False`; counterfactual (branch flipped) `True`
- unsteered greedy: `False`
- steered toward the other branch: `False`

## mbpp_879:2:0:2 (mbpp)

```python
import re
def text_match(text=' ccddbbjjjb'):
  patterns = 'a.*?b$'
  if re.search(patterns,  text):
    return ('Found a match!')
  else:
    return ('Not matched!')

assert text_match() ==
```

- branch `if re.search(patterns,  text)` is **not taken**
- true output `'Not matched!'`; counterfactual (branch flipped) `'Found a match!'`
- unsteered greedy: `'Not matched!'`
- steered toward the other branch: `'Found a match!'`

## mbpp_904:1:0:1 (mbpp)

```python
def even_num(x=1):
  if x%2==0:
     return True
  else:
    return False

assert even_num() ==
```

- branch `if x%2==0` is **not taken**
- true output `False`; counterfactual (branch flipped) `True`
- unsteered greedy: `False`
- steered toward the other branch: `False`

