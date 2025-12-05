# Calculator Class

**ID:** `test1.py_Calculator`  
**Signature:** `Calculator()`

**Summary:** A simple calculator class for basic arithmetic operations with history tracking.

## Description

This class provides basic arithmetic operations, specifically multiplication, with built-in history tracking. When initialized, it creates an empty history list. The multiply method performs multiplication of two numbers and records the operation in the history. The get_history method returns all recorded operations.

## Example

```python
calc = Calculator()
result = calc.multiply(4, 5)
print(result)  # 20
print(calc.get_history())  # ['4 * 5 = 20']
```

## Edge cases & Notes

- Multiplying very large numbers may lead to overflow in some systems.
- Non-numeric inputs will raise a TypeError during multiplication.
- History grows unbounded with each operation, which could consume significant memory over time.

