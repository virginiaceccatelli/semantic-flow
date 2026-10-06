# Steering examples (real, capable members, primary dose)

Dose 0.05; blocks [22, 23, 24, 25, 26].

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

