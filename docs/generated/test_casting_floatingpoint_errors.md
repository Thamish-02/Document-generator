## AI Summary

A file named test_casting_floatingpoint_errors.py.


### Function: values_and_dtypes()

**Description:** Generate value+dtype pairs that generate floating point errors during
casts.  The invalid casts to integers will generate "invalid" value
warnings, the float casts all generate "overflow".

(The Python int/float paths don't need to get tested in all the same
situations, but it does not hurt.)

### Function: check_operations(dtype, value)

**Description:** There are many dedicated paths in NumPy which cast and should check for
floating point errors which occurred during those casts.

### Function: test_floatingpoint_errors_casting(dtype, value)

### Function: copyto_scalar()

### Function: copyto()

### Function: copyto_scalar_masked()

### Function: copyto_masked()

### Function: direct_cast()

### Function: direct_cast_nd_strided()

### Function: boolean_array_assignment()

### Function: integer_array_assignment()

### Function: integer_array_assignment_with_subspace()

### Function: flat_assignment()

### Function: assignment()

### Function: fill()
