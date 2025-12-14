## AI Summary

A file named test_casting_unittests.py.


### Function: simple_dtype_instances()

### Function: get_expected_stringlength(dtype)

**Description:** Returns the string length when casting the basic dtypes to strings.
    

## Class: Casting

### Function: _get_cancast_table()

## Class: TestChanges

**Description:** These test cases exercise some behaviour changes

## Class: TestCasting

### Function: test_float_to_string(self, floating, string)

### Function: test_to_void(self)

### Function: get_data(self, dtype1, dtype2)

### Function: get_data_variation(self, arr1, arr2, aligned, contig)

**Description:** Returns a copy of arr1 that may be non-contiguous or unaligned, and a
matching array for arr2 (although not a copy).

### Function: test_simple_cancast(self, from_Dt)

### Function: test_simple_direct_casts(self, from_dt)

**Description:** This test checks numeric direct casts for dtypes supported also by the
struct module (plus complex).  It tries to be test a wide range of
inputs, but skips over possibly undefined behaviour (e.g. int rollover).
Longdouble and CLongdouble are tested, but only using double precision.

If this test creates issues, it should possibly just be simplified
or even removed (checking whether unaligned/non-contiguous casts give
the same results is useful, though).

### Function: test_numeric_to_times(self, from_Dt)

### Function: test_time_to_time(self, from_dt, to_dt, expected_casting, expected_view_off, nom, denom)

### Function: string_with_modified_length(self, dtype, change_length)

### Function: test_string_cancast(self, other_DT, string_char)

### Function: test_simple_string_casts_roundtrip(self, other_dt, string_char)

**Description:** Tests casts from and to string by checking the roundtripping property.

The test also covers some string to string casts (but not all).

If this test creates issues, it should possibly just be simplified
or even removed (checking whether unaligned/non-contiguous casts give
the same results is useful, though).

### Function: test_string_to_string_cancast(self, other_dt, string_char)

### Function: test_unicode_byteswapped_cast(self, order1, order2)

### Function: test_void_to_string_special_case(self)

### Function: test_object_to_parametric_internal_error(self)

### Function: test_object_and_simple_resolution(self, dtype)

### Function: test_simple_to_object_resolution(self, dtype)

### Function: test_void_and_structured_with_subarray(self, casting)

### Function: test_structured_field_offsets(self, to_dt, expected_off)

### Function: test_structured_view_offsets_parametric(self, from_dt, to_dt, expected_off)

### Function: test_object_casts_NULL_None_equivalence(self, dtype)

### Function: test_nonstandard_bool_to_other(self, dtype)
