# Steering examples (real, capable members, primary dose)

Dose 0.1; blocks [19, 20, 21, 22, 23].

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
- steered toward the other branch: `'AINNEAPLE'`

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
- steered toward the other branch: `'EINNEAPLE'`

## mbpp_605:0:1:0 (mbpp)

```python
def prime_num(num=13):
  if num >=1:
   for i in range(2, num//2):
     if (num % i) == 0:
                return False
     else:
                return True
  else:
          return False

assert prime_num() ==
```

- branch `if (num % i) == 0` is **not taken**
- true output `True`; counterfactual (branch flipped) `False`
- unsteered greedy: `True`
- steered toward the other branch: `True`

## mbpp_605:1:1:0 (mbpp)

```python
def prime_num(num=7):
  if num >=1:
   for i in range(2, num//2):
     if (num % i) == 0:
                return False
     else:
                return True
  else:
          return False

assert prime_num() ==
```

- branch `if (num % i) == 0` is **not taken**
- true output `True`; counterfactual (branch flipped) `False`
- unsteered greedy: `True`
- steered toward the other branch: `True`

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

## mbpp_670:1:0:0 (mbpp)

```python
def decreasing_trend(nums=[1,2,3]):
    if (sorted(nums)== nums):
        return True
    else:
        return False

assert decreasing_trend() ==
```

- branch `if sorted(nums)== nums` is **taken**
- true output `True`; counterfactual (branch flipped) `False`
- unsteered greedy: `True`
- steered toward the other branch: `True`

## mbpp_680:0:0:0 (mbpp)

```python
def increasing_trend(nums=[1,2,3,4]):
    if (sorted(nums)== nums):
        return True
    else:
        return False

assert increasing_trend() ==
```

- branch `if sorted(nums)== nums` is **taken**
- true output `True`; counterfactual (branch flipped) `False`
- unsteered greedy: `True`
- steered toward the other branch: `True`

## mbpp_680:2:0:0 (mbpp)

```python
def increasing_trend(nums=[0,1,4,9]):
    if (sorted(nums)== nums):
        return True
    else:
        return False

assert increasing_trend() ==
```

- branch `if sorted(nums)== nums` is **taken**
- true output `True`; counterfactual (branch flipped) `False`
- unsteered greedy: `True`
- steered toward the other branch: `True`

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

## mbpp_760:0:0:0 (mbpp)

```python
def unique_Element(arr=[1,1,1], n=3):
    s = set(arr)
    if (len(s) == 1):
        return ('YES')
    else:
        return ('NO')

assert unique_Element() ==
```

- branch `if len(s) == 1` is **taken**
- true output `'YES'`; counterfactual (branch flipped) `'NO'`
- unsteered greedy: `'YES'`
- steered toward the other branch: `'YES'`

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
- unsteered greedy: `True`
- steered toward the other branch: `False`

## mbpp_762:2:0:1 (mbpp)

```python
def check_monthnumber_number(monthnum3=11):
  if(monthnum3==4 or monthnum3==6 or monthnum3==9 or monthnum3==11):
    return True
  else:
    return False

assert check_monthnumber_number() ==
```

- branch `if monthnum3==4 or monthnum3==6 or monthnum3==9 or monthnum3==11` is **taken**
- true output `True`; counterfactual (branch flipped) `False`
- unsteered greedy: `True`
- steered toward the other branch: `True`

## mbpp_850:1:0:0 (mbpp)

```python
def is_triangleexists(a=90, b=45, c=45):
    if(a != 0 and b != 0 and c != 0 and (a + b + c)== 180): 
        if((a + b)>= c or (b + c)>= a or (a + c)>= b): 
            return True 
        else:
            return False
    else:
        return False

assert is_triangleexists() ==
```

- branch `if a != 0 and b != 0 and c != 0 and (a + b + c)== 180` is **taken**
- true output `True`; counterfactual (branch flipped) `False`
- unsteered greedy: `True`
- steered toward the other branch: `True`

