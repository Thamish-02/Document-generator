## AI Summary

A file named test_arraypad.py.


## Class: TestAsPairs

## Class: TestConditionalShortcuts

## Class: TestStatistic

## Class: TestConstant

## Class: TestLinearRamp

## Class: TestReflect

## Class: TestEmptyArray

**Description:** Check how padding behaves on arrays with an empty dimension.

## Class: TestSymmetric

## Class: TestWrap

## Class: TestEdge

## Class: TestEmpty

### Function: test_legacy_vector_functionality()

### Function: test_unicode_mode()

### Function: test_object_input(mode)

## Class: TestPadWidth

### Function: test_kwargs(mode)

**Description:** Test behavior of pad's kwargs for the given mode.

### Function: test_constant_zero_default()

### Function: test_unsupported_mode(mode)

### Function: test_non_contiguous_array(mode)

### Function: test_memory_layout_persistence(mode)

**Description:** Test if C and F order is preserved for all pad modes.

### Function: test_dtype_persistence(dtype, mode)

### Function: test_single_value(self)

**Description:** Test casting for a single value.

### Function: test_two_values(self)

**Description:** Test proper casting for two different values.

### Function: test_with_none(self)

### Function: test_pass_through(self)

**Description:** Test if `x` already matching desired output are passed through.

### Function: test_as_index(self)

**Description:** Test results if `as_index=True`.

### Function: test_exceptions(self)

**Description:** Ensure faulty usage is discovered.

### Function: test_zero_padding_shortcuts(self, mode)

### Function: test_shallow_statistic_range(self, mode)

### Function: test_clip_statistic_range(self, mode)

### Function: test_check_mean_stat_length(self)

### Function: test_check_maximum_1(self)

### Function: test_check_maximum_2(self)

### Function: test_check_maximum_stat_length(self)

### Function: test_check_minimum_1(self)

### Function: test_check_minimum_2(self)

### Function: test_check_minimum_stat_length(self)

### Function: test_check_median(self)

### Function: test_check_median_01(self)

### Function: test_check_median_02(self)

### Function: test_check_median_stat_length(self)

### Function: test_check_mean_shape_one(self)

### Function: test_check_mean_2(self)

### Function: test_same_prepend_append(self, mode)

**Description:** Test that appended and prepended values are equal 

### Function: test_check_negative_stat_length(self, mode, stat_length)

### Function: test_simple_stat_length(self)

### Function: test_zero_stat_length_valid(self, mode)

### Function: test_zero_stat_length_invalid(self, mode)

### Function: test_check_constant(self)

### Function: test_check_constant_zeros(self)

### Function: test_check_constant_float(self)

### Function: test_check_constant_float2(self)

### Function: test_check_constant_float3(self)

### Function: test_check_constant_odd_pad_amount(self)

### Function: test_check_constant_pad_2d(self)

### Function: test_check_large_integers(self)

### Function: test_check_object_array(self)

### Function: test_pad_empty_dimension(self)

### Function: test_check_simple(self)

### Function: test_check_2d(self)

### Function: test_object_array(self)

### Function: test_end_values(self)

**Description:** Ensure that end values are exact.

### Function: test_negative_difference(self, dtype)

**Description:** Check correct behavior of unsigned dtypes if there is a negative
difference between the edge to pad and `end_values`. Check both cases
to be independent of implementation. Test behavior for all other dtypes
in case dtype casting interferes with complex dtypes. See gh-14191.

### Function: test_check_simple(self)

### Function: test_check_odd_method(self)

### Function: test_check_large_pad(self)

### Function: test_check_shape(self)

### Function: test_check_01(self)

### Function: test_check_02(self)

### Function: test_check_03(self)

### Function: test_check_04(self)

### Function: test_check_05(self)

### Function: test_check_06(self)

### Function: test_check_07(self)

### Function: test_pad_empty_dimension(self, mode)

### Function: test_pad_non_empty_dimension(self, mode)

### Function: test_check_simple(self)

### Function: test_check_odd_method(self)

### Function: test_check_large_pad(self)

### Function: test_check_large_pad_odd(self)

### Function: test_check_shape(self)

### Function: test_check_01(self)

### Function: test_check_02(self)

### Function: test_check_03(self)

### Function: test_check_simple(self)

### Function: test_check_large_pad(self)

### Function: test_check_01(self)

### Function: test_check_02(self)

### Function: test_pad_with_zero(self)

### Function: test_repeated_wrapping(self)

**Description:** Check wrapping on each side individually if the wrapped area is longer
than the original array.

### Function: test_repeated_wrapping_multiple_origin(self)

**Description:** Assert that 'wrap' pads only with multiples of the original area if
the pad width is larger than the original array.

### Function: test_check_simple(self)

### Function: test_check_width_shape_1_2(self)

### Function: test_simple(self)

### Function: test_pad_empty_dimension(self)

### Function: _padwithtens(vector, pad_width, iaxis, kwargs)

### Function: test_misshaped_pad_width(self, pad_width, mode)

### Function: test_misshaped_pad_width_2(self, mode)

### Function: test_negative_pad_width(self, pad_width, mode)

### Function: test_bad_type(self, pad_width, dtype, mode)

### Function: test_pad_width_as_ndarray(self)

### Function: test_zero_pad_width(self, pad_width, mode)
