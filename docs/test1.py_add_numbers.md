# Add Two Numbers

**ID:** `test1.py_add_numbers`  
**Signature:** `add_numbers(a, b)`

**Summary:** Adds two numeric values and returns the result.

## Description

This function accepts two arguments, `a` and `b`, and returns their sum. It assumes both inputs support the `+` operator. The function does not perform type validation, so non-numeric or incompatible types may raise a TypeError at runtime.

## Example

```python
result = add_numbers(2, 3)
print(result)  # 5
```

## Edge cases & Notes

- Passing non-numeric types like strings or lists will raise a TypeError.
- Very large integers may still work, but floating-point addition can introduce rounding errors.

