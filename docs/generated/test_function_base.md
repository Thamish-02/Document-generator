## AI Summary

A file named test_function_base.py.


### Function: _is_armhf()

## Class: PhysicalQuantity

## Class: PhysicalQuantity2

## Class: TestLogspace

## Class: TestGeomspace

## Class: TestLinspace

## Class: TestAdd_newdoc

### Function: __new__(cls, value)

### Function: __add__(self, x)

### Function: __sub__(self, x)

### Function: __rsub__(self, x)

### Function: __mul__(self, x)

### Function: __div__(self, x)

### Function: __rdiv__(self, x)

### Function: test_basic(self)

### Function: test_start_stop_array(self)

### Function: test_base_array(self, axis)

### Function: test_stop_base_array(self, axis)

### Function: test_dtype(self)

### Function: test_physical_quantities(self)

### Function: test_subclass(self)

### Function: test_basic(self)

### Function: test_boundaries_match_start_and_stop_exactly(self)

### Function: test_nan_interior(self)

### Function: test_complex(self)

### Function: test_complex_shortest_path(self)

### Function: test_dtype(self)

### Function: test_start_stop_array_scalar(self)

### Function: test_start_stop_array(self)

### Function: test_physical_quantities(self)

### Function: test_subclass(self)

### Function: test_bounds(self)

### Function: test_basic(self)

### Function: test_corner(self)

### Function: test_type(self)

### Function: test_dtype(self)

### Function: test_start_stop_array_scalar(self)

### Function: test_start_stop_array(self)

### Function: test_complex(self)

### Function: test_physical_quantities(self)

### Function: test_subclass(self)

### Function: test_array_interface(self)

### Function: test_denormal_numbers(self)

### Function: test_equivalent_to_arange(self)

### Function: test_retstep(self)

### Function: test_object(self)

### Function: test_round_negative(self)

### Function: test_any_step_zero_and_not_mult_inplace(self)

### Function: test_add_doc(self)

### Function: test_errors_are_ignored(self)

## Class: Arrayish

**Description:** A generic object that supports the __array_interface__ and hence
can in principle be converted to a numeric scalar, but is not
otherwise recognized as numeric, but also happens to support
multiplication by floats.

Data should be an object that implements the buffer interface,
and contains at least 4 bytes.

### Function: __init__(self, data)

### Function: __array_interface__(self)

### Function: __mul__(self, other)
