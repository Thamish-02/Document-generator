## AI Summary

A file named test_half.py.


### Function: assert_raises_fpe(strmatch, callable)

## Class: TestHalf

### Function: setup_method(self)

### Function: test_half_conversions(self)

**Description:** Checks that all 16-bit values survive conversion
to/from 32-bit and 64-bit float

### Function: test_half_conversion_to_string(self, string_dt)

### Function: test_half_conversion_from_string(self, string_dt)

### Function: test_half_conversion_rounding(self, float_t, shift, offset)

### Function: test_half_conversion_denormal_round_even(self, float_t, uint_t, bits)

### Function: test_nans_infs(self)

### Function: test_half_values(self)

**Description:** Confirms a small number of known half values

### Function: test_half_rounding(self)

**Description:** Checks that rounding when converting to half is correct

### Function: test_half_correctness(self)

**Description:** Take every finite float16, and check the casting functions with
a manual conversion.

### Function: test_half_ordering(self)

**Description:** Make sure comparisons are working right

### Function: test_half_funcs(self)

**Description:** Test the various ArrFuncs

### Function: test_spacing_nextafter(self)

**Description:** Test np.spacing and np.nextafter

### Function: test_half_ufuncs(self)

**Description:** Test the various ufuncs

### Function: test_half_coercion(self)

**Description:** Test that half gets coerced properly with the other types

### Function: test_half_fpe(self)

### Function: test_half_array_interface(self)

**Description:** Test that half is compatible with __array_interface__

## Class: Dummy
