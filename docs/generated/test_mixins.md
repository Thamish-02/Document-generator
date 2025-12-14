## AI Summary

A file named test_mixins.py.


## Class: ArrayLike

### Function: wrap_array_like(result)

### Function: _assert_equal_type_and_value(result, expected, err_msg)

## Class: TestNDArrayOperatorsMixin

### Function: __init__(self, value)

### Function: __array_ufunc__(self, ufunc, method)

### Function: __repr__(self)

### Function: test_array_like_add(self)

### Function: test_inplace(self)

### Function: test_opt_out(self)

### Function: test_subclass(self)

### Function: test_object(self)

### Function: test_unary_methods(self)

### Function: test_forward_binary_methods(self)

### Function: test_reflected_binary_methods(self)

### Function: test_matmul(self)

### Function: test_ufunc_at(self)

### Function: test_ufunc_two_outputs(self)

### Function: check(result)

## Class: OptOut

**Description:** Object that opts out of __array_ufunc__.

## Class: SubArrayLike

**Description:** Should take precedence over ArrayLike.

### Function: __add__(self, other)

### Function: __radd__(self, other)
