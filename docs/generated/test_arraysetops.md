## AI Summary

A file named test_arraysetops.py.


## Class: TestSetOps

## Class: TestUnique

### Function: test_intersect1d(self)

### Function: test_intersect1d_array_like(self)

### Function: test_intersect1d_indices(self)

### Function: test_setxor1d(self)

### Function: test_setxor1d_unique(self)

### Function: test_ediff1d(self)

### Function: test_ediff1d_forbidden_type_casts(self, ary, prepend, append, expected)

### Function: test_ediff1d_scalar_handling(self, ary, prepend, append, expected)

### Function: test_isin(self, kind)

### Function: test_isin_additional(self, kind)

### Function: test_isin_char_array(self)

### Function: test_isin_invert(self, kind)

**Description:** Test isin's invert parameter

### Function: test_isin_hit_alternate_algorithm(self)

**Description:** Hit the standard isin code with integers

### Function: test_isin_boolean(self, kind)

**Description:** Test that isin works for boolean input

### Function: test_isin_timedelta(self, kind)

**Description:** Test that isin works for timedelta input

### Function: test_isin_table_timedelta_fails(self)

### Function: test_isin_mixed_dtype(self, dtype1, dtype2, kind)

**Description:** Test that isin works as expected for mixed dtype input.

### Function: test_isin_mixed_huge_vals(self, kind, data)

**Description:** Test values outside intp range (negative ones if 32bit system)

### Function: test_isin_mixed_boolean(self, kind)

**Description:** Test that isin works as expected for bool/int input.

### Function: test_isin_first_array_is_object(self)

### Function: test_isin_second_array_is_object(self)

### Function: test_isin_both_arrays_are_object(self)

### Function: test_isin_both_arrays_have_structured_dtype(self)

### Function: test_isin_with_arrays_containing_tuples(self)

### Function: test_isin_errors(self)

**Description:** Test that isin raises expected errors.

### Function: test_union1d(self)

### Function: test_setdiff1d(self)

### Function: test_setdiff1d_unique(self)

### Function: test_setdiff1d_char_array(self)

### Function: test_manyways(self)

### Function: test_unique_1d(self)

### Function: test_unique_axis_errors(self)

### Function: test_unique_axis_list(self)

### Function: test_unique_axis(self)

### Function: test_unique_1d_with_axis(self, axis)

### Function: test_unique_inverse_with_axis(self, axis)

### Function: test_unique_axis_zeros(self)

### Function: test_unique_masked(self)

### Function: test_unique_sort_order_with_axis(self)

### Function: _run_axis_tests(self, dtype)

### Function: test_unique_nanequals(self)

### Function: test_unique_array_api_functions(self)

### Function: test_unique_inverse_shape(self)

## Class: Test

### Function: _isin_slow(a, b)

### Function: assert_isin_equal(a, b)

### Function: check_all(a, b, i1, i2, c, dt)

### Function: __array__(self, dtype, copy)
