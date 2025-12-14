## AI Summary

A file named test_custom_dtypes.py.


## Class: TestSFloat

### Function: test_type_pickle()

### Function: test_is_numeric()

### Function: _get_array(self, scaling, aligned)

### Function: test_sfloat_rescaled(self)

### Function: test_class_discovery(self)

### Function: test_scaled_float_from_floats(self, scaling)

### Function: test_repr(self)

### Function: test_dtype_name(self)

### Function: test_sfloat_structured_dtype_printing(self)

### Function: test_sfloat_from_float(self, scaling)

### Function: test_sfloat_getitem(self, aligned, scaling)

### Function: test_sfloat_casts(self, aligned)

### Function: test_sfloat_cast_internal_errors(self, aligned)

### Function: test_sfloat_promotion(self)

### Function: test_basic_multiply(self)

### Function: test_possible_and_impossible_reduce(self)

### Function: test_basic_ufunc_at(self)

### Function: test_basic_multiply_promotion(self)

### Function: test_basic_addition(self)

### Function: test_addition_cast_safety(self)

**Description:** The addition method is special for the scaled float, because it
includes the "cast" between different factors, thus cast-safety
is influenced by the implementation.

### Function: test_logical_ufuncs_casts_to_bool(self, ufunc)

### Function: test_wrapped_and_wrapped_reductions(self)

### Function: test_astype_class(self)

### Function: test_creation_class(self)

### Function: test_np_save_load(self)

### Function: test_flatiter(self)

### Function: test_flatiter_index(self, index)
