## AI Summary

A file named test_datetime.py.


### Function: _assert_equal_hash(v1, v2)

## Class: TestDateTime

## Class: TestDateTimeData

### Function: test_comparisons_return_not_implemented()

### Function: test_string(self)

### Function: test_datetime(self)

### Function: test_datetime_dtype_creation(self)

### Function: test_datetime_casting_rules(self)

### Function: test_datetime_prefix_conversions(self)

### Function: test_prohibit_negative_datetime(self, unit)

### Function: test_compare_generic_nat(self)

### Function: test_datetime_nat_argsort_stability(self, size)

### Function: test_timedelta_nat_argsort_stability(self, size)

### Function: test_datetime_timedelta_sort_nat(self, arr, expected, dtype)

### Function: test_datetime_scalar_construction(self)

### Function: test_datetime_scalar_construction_timezone(self)

### Function: test_datetime_array_find_type(self)

### Function: test_timedelta_np_int_construction(self, unit)

### Function: test_timedelta_scalar_construction(self)

### Function: test_timedelta_object_array_conversion(self)

### Function: test_timedelta_0_dim_object_array_conversion(self)

### Function: test_timedelta_nat_format(self)

### Function: test_timedelta_scalar_construction_units(self)

### Function: test_datetime_nat_casting(self)

### Function: test_days_creation(self)

### Function: test_days_to_pydate(self)

### Function: test_dtype_comparison(self)

### Function: test_pydatetime_creation(self)

### Function: test_datetime_string_conversion(self)

### Function: test_time_byteswapping(self, time_dtype)

### Function: test_time_byteswapped_cast(self, time1, time2)

### Function: test_datetime_conversions_byteorders(self, str_dtype, time_dtype)

### Function: test_datetime_array_str(self)

### Function: test_timedelta_array_str(self)

### Function: test_pickle(self)

### Function: test_setstate(self)

**Description:** Verify that datetime dtype __setstate__ can handle bad arguments

### Function: test_dtype_promotion(self)

### Function: test_cast_overflow(self)

### Function: test_pyobject_roundtrip(self)

### Function: test_month_truncation(self)

### Function: test_different_unit_comparison(self)

### Function: test_datetime_like(self)

### Function: test_datetime_unary(self)

### Function: test_datetime_add(self)

### Function: test_datetime_subtract(self)

### Function: test_datetime_multiply(self)

### Function: test_timedelta_floor_divide(self, op1, op2, exp)

### Function: test_timedelta_floor_div_warnings(self, op1, op2)

### Function: test_timedelta_floor_div_precision(self, val1, val2)

### Function: test_timedelta_floor_div_error(self, val1, val2)

### Function: test_timedelta_divmod(self, op1, op2)

### Function: test_timedelta_divmod_typeerror(self, op1, op2)

### Function: test_timedelta_divmod_warnings(self, op1, op2)

### Function: test_datetime_divide(self)

### Function: test_datetime_compare(self)

### Function: test_datetime_compare_nat(self)

### Function: test_datetime_minmax(self)

### Function: test_hours(self)

### Function: test_divisor_conversion_year(self)

### Function: test_divisor_conversion_month(self)

### Function: test_divisor_conversion_week(self)

### Function: test_divisor_conversion_day(self)

### Function: test_divisor_conversion_hour(self)

### Function: test_divisor_conversion_minute(self)

### Function: test_divisor_conversion_second(self)

### Function: test_divisor_conversion_fs(self)

### Function: test_divisor_conversion_as(self)

### Function: test_string_parser_variants(self)

### Function: test_string_parser_error_check(self)

### Function: test_creation_overflow(self)

### Function: test_datetime_as_string(self)

### Function: test_datetime_as_string_timezone(self)

### Function: test_datetime_arange(self)

### Function: test_datetime_arange_no_dtype(self)

### Function: test_timedelta_arange(self)

### Function: test_timedelta_modulus(self, val1, val2, expected)

### Function: test_timedelta_modulus_error(self, val1, val2)

### Function: test_timedelta_modulus_div_by_zero(self)

### Function: test_timedelta_modulus_type_resolution(self, val1, val2)

### Function: test_timedelta_arange_no_dtype(self)

### Function: test_datetime_maximum_reduce(self)

### Function: test_timedelta_correct_mean(self)

### Function: test_datetime_no_subtract_reducelike(self)

### Function: test_datetime_busday_offset(self)

### Function: test_datetime_busdaycalendar(self)

### Function: test_datetime_busday_holidays_offset(self)

### Function: test_datetime_busday_holidays_count(self)

### Function: test_datetime_is_busday(self)

### Function: test_datetime_y2038(self)

### Function: test_isnat(self)

### Function: test_isnat_error(self)

### Function: test_isfinite_scalar(self)

### Function: test_isfinite_isinf_isnan_units(self, unit, dstr)

**Description:** check isfinite, isinf, isnan for all units of <M, >M, <m, >m dtypes
        

### Function: test_assert_equal(self)

### Function: test_corecursive_input(self)

### Function: test_discovery_from_object_array(self, shape)

### Function: test_limit_symmetry(self, time_unit)

**Description:** Dates should have symmetric limits around the unix epoch at +/-np.int64

### Function: test_limit_str_roundtrip(self, time_unit, sign)

**Description:** Limits should roundtrip when converted to strings.

This tests the conversion to and from npy_datetimestruct.

### Function: test_datetime_hash_nat(self)

### Function: test_datetime_hash_weeks(self, unit)

### Function: test_datetime_hash_weeks_vs_pydatetime(self, unit)

### Function: test_datetime_hash_big_negative(self, unit)

### Function: test_datetime_hash_minutes(self, unit)

### Function: test_datetime_hash_ns(self, unit)

### Function: test_datetime_hash_big_positive(self, wk, unit)

### Function: test_timedelta_hash_generic(self)

### Function: test_timedelta_hash_year_month(self, unit)

### Function: test_timedelta_hash_weeks(self, unit)

### Function: test_timedelta_hash_weeks_vs_pydelta(self, unit)

### Function: test_timedelta_hash_ms(self, unit)

### Function: test_timedelta_hash_big_positive(self, wk, unit)

### Function: test_basic(self)

### Function: test_bytes(self)

### Function: test_non_ascii(self)

## Class: custom

### Function: cast()

### Function: cast2()

### Function: check(a, b, res)
