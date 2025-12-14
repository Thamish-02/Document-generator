## AI Summary

A file named test_simd.py.


### Function: check_floatstatus(divbyzero, overflow, underflow, invalid, all)

## Class: _Test_Utility

## Class: _SIMD_BOOL

**Description:** To test all boolean vector types at once

## Class: _SIMD_INT

**Description:** To test all integer vector types at once

## Class: _SIMD_FP32

**Description:** To only test single precision

## Class: _SIMD_FP64

**Description:** To only test double precision

## Class: _SIMD_FP

**Description:** To test all float vector types at once

## Class: _SIMD_ALL

**Description:** To test all vector types at once

### Function: __getattr__(self, attr)

**Description:** To call NPV intrinsics without the attribute 'npyv' and
auto suffixing intrinsics according to class attribute 'sfx'

### Function: _x2(self, intrin_name)

### Function: _data(self, start, count, reverse)

**Description:** Create list of consecutive numbers according to number of vector's lanes.

### Function: _is_unsigned(self)

### Function: _is_signed(self)

### Function: _is_fp(self)

### Function: _scalar_size(self)

### Function: _int_clip(self, seq)

### Function: _int_max(self)

### Function: _int_min(self)

### Function: _true_mask(self)

### Function: _to_unsigned(self, vector)

### Function: _pinfinity(self)

### Function: _ninfinity(self)

### Function: _nan(self)

### Function: _cpu_features(self)

### Function: _nlanes(self)

### Function: _data(self, start, count, reverse)

### Function: _load_b(self, data)

### Function: test_operators_logical(self)

**Description:** Logical operations for boolean types.
Test intrinsics:
    npyv_xor_##SFX, npyv_and_##SFX, npyv_or_##SFX, npyv_not_##SFX,
    npyv_andc_b8, npvy_orc_b8, nvpy_xnor_b8

### Function: test_tobits(self)

### Function: test_pack(self)

**Description:** Pack multiple vectors into one
Test intrinsics:
    npyv_pack_b8_b16
    npyv_pack_b8_b32
    npyv_pack_b8_b64

### Function: test_operators_crosstest(self, intrin, data)

**Description:** Test intrinsics:
    npyv_any_##SFX
    npyv_all_##SFX

### Function: test_operators_shift(self)

### Function: test_arithmetic_subadd_saturated(self)

### Function: test_math_max_min(self)

### Function: test_reduce_max_min(self, start)

**Description:** Test intrinsics:
    npyv_reduce_max_##sfx
    npyv_reduce_min_##sfx

### Function: test_conversions(self)

**Description:** Round to nearest even integer, assume CPU control register is set to rounding.
Test intrinsics:
    npyv_round_s32_##SFX

### Function: test_conversions(self)

**Description:** Round to nearest even integer, assume CPU control register is set to rounding.
Test intrinsics:
    npyv_round_s32_##SFX

### Function: test_arithmetic_fused(self)

### Function: test_abs(self)

### Function: test_sqrt(self)

### Function: test_square(self)

### Function: test_rounding(self, intrin, func)

**Description:** Test intrinsics:
    npyv_rint_##SFX
    npyv_ceil_##SFX
    npyv_trunc_##SFX
    npyv_floor##SFX

### Function: test_max_min(self, intrin)

**Description:** Test intrinsics:
    npyv_max_##sfx
    npyv_maxp_##sfx
    npyv_maxn_##sfx
    npyv_min_##sfx
    npyv_minp_##sfx
    npyv_minn_##sfx
    npyv_reduce_max_##sfx
    npyv_reduce_maxp_##sfx
    npyv_reduce_maxn_##sfx
    npyv_reduce_min_##sfx
    npyv_reduce_minp_##sfx
    npyv_reduce_minn_##sfx

### Function: test_reciprocal(self)

### Function: test_special_cases(self)

**Description:** Compare Not NaN. Test intrinsics:
    npyv_notnan_##SFX

### Function: test_unary_invalid_fpexception(self, intrin_name)

### Function: test_comparison_with_nan(self, py_comp, np_comp)

### Function: test_operators_crosstest(self, intrin, data)

**Description:** Test intrinsics:
    npyv_any_##SFX
    npyv_all_##SFX

### Function: test_memory_load(self)

### Function: test_memory_store(self)

### Function: test_memory_partial_load(self, intrin, elsizes, scale, fill)

### Function: test_memory_partial_store(self, intrin, elsizes, scale)

### Function: test_memory_noncont_load(self, intrin, elsizes, scale)

### Function: test_memory_noncont_partial_load(self, intrin, elsizes, scale, fill)

### Function: test_memory_noncont_store(self, intrin, elsizes, scale)

### Function: test_memory_noncont_partial_store(self, intrin, elsizes, scale)

### Function: test_lut(self, intrin, table_size, elsize)

**Description:** Test lookup table intrinsics:
    npyv_lut32_##sfx
    npyv_lut16_##sfx

### Function: test_misc(self)

### Function: test_reorder(self)

### Function: test_reorder_rev64(self)

### Function: test_reorder_permi128(self)

**Description:** Test permuting elements for each 128-bit lane.
npyv_permi128_##sfx

### Function: test_operators_comparison(self, func, intrin)

### Function: test_operators_logical(self)

### Function: test_operators_crosstest(self, intrin, data)

**Description:** Test intrinsics:
    npyv_any_##SFX
    npyv_all_##SFX

### Function: test_conversion_boolean(self)

### Function: test_conversion_expand(self)

**Description:** Test expand intrinsics:
    npyv_expand_u16_u8
    npyv_expand_u32_u16

### Function: test_arithmetic_subadd(self)

### Function: test_arithmetic_mul(self)

### Function: test_arithmetic_div(self)

### Function: test_arithmetic_intdiv(self)

**Description:** Test integer division intrinsics:
    npyv_divisor_##sfx
    npyv_divc_##sfx

### Function: test_arithmetic_reduce_sum(self)

**Description:** Test reduce sum intrinsics:
    npyv_sum_##sfx

### Function: test_arithmetic_reduce_sumup(self)

**Description:** Test extend reduce sum intrinsics:
    npyv_sumup_##sfx

### Function: test_mask_conditional(self)

**Description:** Conditional addition and subtraction for all supported data types.
Test intrinsics:
    npyv_ifadd_##SFX, npyv_ifsub_##SFX

### Function: to_bool(vector)

### Function: to_bool(vector)

### Function: trunc_div(a, d)

**Description:** Divide towards zero works with large integers > 2^53,
and wrap around overflow similar to what C does.
