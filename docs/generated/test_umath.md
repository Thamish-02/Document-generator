## AI Summary

A file named test_umath.py.


### Function: interesting_binop_operands(val1, val2, dtype)

**Description:** Helper to create "interesting" operands to cover common code paths:
* scalar inputs
* only first "values" is an array (e.g. scalar division fast-paths)
* Longer array (SIMD) placing the value of interest at different positions
* Oddly strided arrays which may not be SIMD compatible

It does not attempt to cover unaligned access or mixed dtypes.
These are normally handled by the casting/buffering machinery.

This is not a fixture (currently), since I believe a fixture normally
only yields once?

### Function: on_powerpc()

**Description:** True if we are running on a Power PC platform.

### Function: bad_arcsinh()

**Description:** The blocklisted trig functions are not accurate on aarch64/PPC for
complex256. Rather than dig through the actual problem skip the
test. This should be fixed when we can move past glibc2.17
which is the version in manylinux2014

## Class: _FilterInvalids

## Class: TestConstants

## Class: TestOut

## Class: TestComparisons

## Class: TestAdd

## Class: TestDivision

### Function: floor_divide_and_remainder(x, y)

### Function: _signs(dt)

## Class: TestRemainder

## Class: TestDivisionIntegerOverflowsAndDivideByZero

## Class: TestCbrt

## Class: TestPower

## Class: TestFloat_power

## Class: TestLog2

## Class: TestExp2

## Class: TestLogAddExp2

## Class: TestLog

## Class: TestExp

## Class: TestSpecialFloats

## Class: TestFPClass

## Class: TestLDExp

## Class: TestFRExp

## Class: TestAVXUfuncs

## Class: TestAVXFloat32Transcendental

## Class: TestLogAddExp

## Class: TestLog1p

## Class: TestExpm1

## Class: TestHypot

### Function: assert_hypot_isnan(x, y)

### Function: assert_hypot_isinf(x, y)

## Class: TestHypotSpecialValues

### Function: assert_arctan2_isnan(x, y)

### Function: assert_arctan2_ispinf(x, y)

### Function: assert_arctan2_isninf(x, y)

### Function: assert_arctan2_ispzero(x, y)

### Function: assert_arctan2_isnzero(x, y)

## Class: TestArctan2SpecialValues

## Class: TestLdexp

## Class: TestMaximum

## Class: TestMinimum

## Class: TestFmax

## Class: TestFmin

## Class: TestBool

## Class: TestBitwiseUFuncs

## Class: TestInt

## Class: TestFloatingPoint

## Class: TestDegrees

## Class: TestRadians

## Class: TestHeavside

## Class: TestSign

## Class: TestMinMax

## Class: TestAbsoluteNegative

## Class: TestPositive

## Class: TestSpecialMethods

## Class: TestChoose

## Class: TestRationalFunctions

## Class: TestRoundingFunctions

## Class: TestComplexFunctions

## Class: TestAttributes

## Class: TestSubclass

## Class: TestFrompyfunc

### Function: _check_branch_cut(f, x0, dx, re_sign, im_sign, sig_zero_ok, dtype)

**Description:** Check for a branch cut in a function.

Assert that `x0` lies on a branch cut of function `f` and `f` is
continuous from the direction `dx`.

Parameters
----------
f : func
    Function to check
x0 : array-like
    Point on branch cut
dx : array-like
    Direction to check continuity in
re_sign, im_sign : {1, -1}
    Change of sign of the real or imaginary part expected
sig_zero_ok : bool
    Whether to check if the branch cut respects signed zero (if applicable)
dtype : dtype
    Dtype to check (should be complex)

### Function: test_copysign()

### Function: _test_nextafter(t)

### Function: test_nextafter()

### Function: test_nextafterf()

### Function: test_nextafterl()

### Function: test_nextafter_0()

### Function: _test_spacing(t)

### Function: test_spacing()

### Function: test_spacingf()

### Function: test_spacingl()

### Function: test_spacing_gfortran()

### Function: test_nextafter_vs_spacing()

### Function: test_pos_nan()

**Description:** Check np.nan is a positive nan.

### Function: test_reduceat()

**Description:** Test bug in reduceat when structured arrays are not copied.

### Function: test_reduceat_empty()

**Description:** Reduceat should work with empty arrays

### Function: test_complex_nan_comparisons()

### Function: test_rint_big_int()

### Function: test_memoverlap_accumulate(ftype)

### Function: test_memoverlap_accumulate_cmp(ufunc, dtype)

### Function: test_memoverlap_accumulate_symmetric(ufunc, dtype)

### Function: test_signaling_nan_exceptions()

### Function: test_outer_subclass_preserve(arr)

### Function: test_outer_bad_subclass()

### Function: test_outer_exceeds_maxdims()

### Function: test_bad_legacy_ufunc_silent_errors()

### Function: test_bad_legacy_gufunc_silent_errors(x1)

## Class: TestAddDocstring

## Class: TestAdd_newdoc_ufunc

### Function: setup_method(self)

### Function: teardown_method(self)

### Function: test_pi(self)

### Function: test_e(self)

### Function: test_euler_gamma(self)

### Function: test_out_subok(self)

### Function: test_out_wrap_subok(self)

### Function: test_out_wrap_no_leak(self)

### Function: test_comparison_functions(self, dtype, py_comp, np_comp)

### Function: test_ignore_object_identity_in_equal(self)

### Function: test_ignore_object_identity_in_not_equal(self)

### Function: test_error_in_equal_reduce(self)

### Function: test_object_dtype(self)

### Function: test_object_nonbool_dtype_error(self)

### Function: test_large_integer_direct_comparison(self, dtypes, py_comp, np_comp, vals)

### Function: test_unsigned_signed_direct_comparison(self, dtype, py_comp_func, np_comp_func, flip)

### Function: test_reduce_alignment(self)

### Function: test_division_int(self)

### Function: test_division_int_boundary(self, dtype, ex_val)

### Function: test_division_int_reduce(self, dtype, ex_val)

### Function: test_division_int_timedelta(self, dividend, divisor, quotient)

### Function: test_division_complex(self)

### Function: test_zero_division_complex(self)

### Function: test_floor_division_complex(self)

### Function: test_floor_division_signed_zero(self)

### Function: test_floor_division_errors(self, dtype)

### Function: test_floor_division_corner_cases(self, dtype)

### Function: test_remainder_basic(self)

### Function: test_float_remainder_exact(self)

### Function: test_float_remainder_roundoff(self)

### Function: test_float_divmod_errors(self, dtype)

### Function: test_float_remainder_errors(self, dtype, fn)

### Function: test_float_remainder_overflow(self)

### Function: test_float_divmod_corner_cases(self)

### Function: test_float_remainder_corner_cases(self)

### Function: test_signed_division_overflow(self, dtype)

### Function: test_divide_by_zero(self, dtype)

### Function: test_overflows(self, dividend_dtype, divisor_dtype, operation)

### Function: test_cbrt_scalar(self)

### Function: test_cbrt(self)

### Function: test_power_float(self)

### Function: test_power_complex(self)

### Function: test_power_zero(self)

### Function: test_zero_power_nonzero(self)

### Function: test_fast_power(self)

### Function: test_integer_power(self)

### Function: test_integer_power_with_integer_zero_exponent(self)

### Function: test_integer_power_of_1(self)

### Function: test_integer_power_of_zero(self)

### Function: test_integer_to_negative_power(self)

### Function: test_float_to_inf_power(self)

### Function: test_power_fast_paths(self)

### Function: test_type_conversion(self)

### Function: test_log2_values(self, dt)

### Function: test_log2_ints(self, i)

### Function: test_log2_special(self)

### Function: test_exp2_values(self)

### Function: test_logaddexp2_values(self)

### Function: test_logaddexp2_range(self)

### Function: test_inf(self)

### Function: test_nan(self)

### Function: test_reduce(self)

### Function: test_log_values(self)

### Function: test_log_values_maxofdtype(self)

### Function: test_log_strides(self)

### Function: test_log_precision_float64(self, z, wref)

### Function: test_log_precision_float32(self, z, wref)

### Function: test_exp_values(self)

### Function: test_exp_strides(self)

### Function: test_exp_values(self)

### Function: test_exp_exceptions(self)

### Function: test_log_values(self)

### Function: test_sincos_values(self, dtype)

### Function: test_sincos_underflow(self)

### Function: test_sincos_errors(self, callable, dtype, value)

### Function: test_sincos_overlaps(self, callable, dtype, stride)

### Function: test_sqrt_values(self, dt)

### Function: test_abs_values(self)

### Function: test_square_values(self)

### Function: test_reciprocal_values(self)

### Function: test_tan(self)

### Function: test_arcsincos(self)

### Function: test_arctan(self)

### Function: test_sinh(self)

### Function: test_cosh(self)

### Function: test_tanh(self)

### Function: test_arcsinh(self)

### Function: test_arccosh(self)

### Function: test_arctanh(self)

### Function: test_exp2(self)

### Function: test_expm1(self)

### Function: test_unary_spurious_fpexception(self, ufunc, dtype, data, escape)

### Function: test_divide_spurious_fpexception(self, dtype)

### Function: test_fpclass(self, stride)

### Function: test_fp_noncontiguous(self, dtype)

### Function: test_ldexp(self, dtype, stride)

### Function: test_frexp(self, dtype, stride)

### Function: test_avx_based_ufunc(self)

### Function: test_exp_float32(self)

### Function: test_log_float32(self)

### Function: test_sincos_float32(self)

### Function: test_strided_float32(self)

### Function: test_logaddexp_values(self)

### Function: test_logaddexp_range(self)

### Function: test_inf(self)

### Function: test_nan(self)

### Function: test_reduce(self)

### Function: test_log1p(self)

### Function: test_special(self)

### Function: test_expm1(self)

### Function: test_special(self)

### Function: test_complex(self)

### Function: test_simple(self)

### Function: test_reduce(self)

### Function: test_nan_outputs(self)

### Function: test_nan_outputs2(self)

### Function: test_no_fpe(self)

### Function: test_one_one(self)

### Function: test_zero_nzero(self)

### Function: test_zero_pzero(self)

### Function: test_zero_negative(self)

### Function: test_zero_positive(self)

### Function: test_positive_zero(self)

### Function: test_negative_zero(self)

### Function: test_any_ninf(self)

### Function: test_any_pinf(self)

### Function: test_inf_any(self)

### Function: test_inf_ninf(self)

### Function: test_inf_pinf(self)

### Function: test_nan_any(self)

### Function: _check_ldexp(self, tp)

### Function: test_ldexp(self)

### Function: test_ldexp_overflow(self)

### Function: test_reduce(self)

### Function: test_reduce_complex(self)

### Function: test_float_nans(self)

### Function: test_object_nans(self)

### Function: test_complex_nans(self)

### Function: test_object_array(self)

### Function: test_strided_array(self)

### Function: test_precision(self)

### Function: test_reduce(self)

### Function: test_reduce_complex(self)

### Function: test_float_nans(self)

### Function: test_object_nans(self)

### Function: test_complex_nans(self)

### Function: test_object_array(self)

### Function: test_strided_array(self)

### Function: test_precision(self)

### Function: test_reduce(self)

### Function: test_reduce_complex(self)

### Function: test_float_nans(self)

### Function: test_complex_nans(self)

### Function: test_precision(self)

### Function: test_reduce(self)

### Function: test_reduce_complex(self)

### Function: test_float_nans(self)

### Function: test_complex_nans(self)

### Function: test_precision(self)

### Function: test_exceptions(self)

### Function: test_truth_table_logical(self)

### Function: test_truth_table_bitwise(self)

### Function: test_reduce(self)

### Function: test_values(self)

### Function: test_types(self)

### Function: test_identity(self)

### Function: test_reduction(self)

### Function: test_bitwise_count(self, input_dtype_obj, bitsize)

### Function: test_logical_not(self)

### Function: test_floating_point(self)

### Function: test_degrees(self)

### Function: test_radians(self)

### Function: test_heaviside(self)

### Function: test_sign(self)

### Function: test_sign_complex(self)

### Function: test_sign_dtype_object(self)

### Function: test_sign_dtype_nan_object(self)

### Function: test_minmax_blocked(self)

### Function: test_lower_align(self)

### Function: test_reduce_reorder(self)

### Function: test_minimize_no_warns(self)

### Function: test_abs_neg_blocked(self)

### Function: test_lower_align(self)

### Function: test_noncontiguous(self, dtype, big)

### Function: test_valid(self)

### Function: test_invalid(self)

### Function: test_wrap(self)

### Function: test_wrap_out(self)

### Function: test_wrap_with_iterable(self)

### Function: test_priority_with_scalar(self)

### Function: test_priority(self)

### Function: test_failing_wrap(self)

### Function: test_failing_out_wrap(self)

### Function: test_none_wrap(self)

### Function: test_default_prepare(self)

### Function: test_array_too_many_args(self)

### Function: test_ufunc_override(self)

### Function: test_ufunc_override_mro(self)

### Function: test_ufunc_override_methods(self)

### Function: test_ufunc_override_out(self)

### Function: test_ufunc_override_where(self)

### Function: test_ufunc_override_exception(self)

### Function: test_ufunc_override_not_implemented(self)

### Function: test_ufunc_override_disabled(self)

### Function: test_gufunc_override(self)

### Function: test_ufunc_override_with_super(self)

### Function: test_array_ufunc_direct_call(self)

### Function: test_ufunc_docstring(self)

### Function: test_mixed(self)

### Function: test_lcm(self)

### Function: test_lcm_object(self)

### Function: test_gcd(self)

### Function: test_gcd_object(self)

### Function: _test_lcm_inner(self, dtype)

### Function: _test_gcd_inner(self, dtype)

### Function: test_lcm_overflow(self)

### Function: test_gcd_overflow(self)

### Function: test_decimal(self)

### Function: test_float(self)

### Function: test_huge_integers(self)

### Function: test_inf_and_nan(self)

### Function: test_object_direct(self)

**Description:** test direct implementation of these magic methods 

### Function: test_object_indirect(self)

**Description:** test implementations via __float__ 

### Function: test_fraction(self)

### Function: test_output_dtype(self, func, dtype)

### Function: test_it(self)

### Function: test_precisions_consistent(self)

### Function: test_branch_cuts(self)

### Function: test_branch_cuts_complex64(self)

### Function: test_against_cmath(self)

### Function: test_loss_of_precision(self, dtype)

**Description:** Check loss of precision in complex arc* functions

### Function: test_promotion_corner_cases(self)

### Function: test_attributes(self)

### Function: test_doc(self)

### Function: test_subclass_op(self)

### Function: test_identity(self)

## Class: foo

## Class: BadArr1

## Class: BadArr2

### Function: test_add_same_docstring(self)

### Function: test_different_docstring_fails(self)

### Function: test_ufunc_arg(self)

### Function: test_string_arg(self)

## Class: ArrayWrap

## Class: ArrSubclass

## Class: FunkyType

## Class: FunkyType

### Function: assert_complex_equal(x, y)

### Function: assert_complex_equal(x, y)

### Function: assert_complex_equal(x, y)

### Function: test_nan()

## Class: with_wrap

## Class: StoreArrayPrepareWrap

### Function: do_test(f_call, f_expected)

## Class: with_wrap

## Class: A

## Class: A

## Class: B

## Class: C

## Class: A

## Class: Ok

## Class: Bad

## Class: A

## Class: with_wrap

## Class: A

## Class: A

## Class: MyNDArray

### Function: tres_mul(a, b, c)

### Function: quatro_mul(a, b, c, d)

## Class: A

## Class: ASub

## Class: B

## Class: C

## Class: CSub

## Class: A

## Class: A

## Class: B

## Class: OverriddenArrayOld

## Class: OverriddenArrayNew

## Class: A

## Class: A

## Class: OptOut

## Class: GreedyArray

## Class: A

## Class: A

## Class: B

## Class: C

## Class: C

### Function: check(x, rtol)

### Function: check(func, z0, d)

## Class: simple

### Function: mul(a, b)

### Function: __array_finalize__(self, obj)

### Function: __array_finalize__(self, obj)

### Function: func()

**Description:** docstring

### Function: func()

**Description:** docstring

### Function: __new__(cls, arr)

### Function: __array_wrap__(self, arr, context, return_scalar)

### Function: __eq__(self, other)

### Function: __ne__(self, other)

### Function: __array__(self, dtype, copy)

### Function: __array_wrap__(self, arr, context, return_scalar)

### Function: __new__(cls)

### Function: __array_wrap__(self, obj, context, return_scalar)

### Function: args(self)

### Function: __repr__(self)

### Function: __new__(cls)

### Function: __array_wrap__(self, arr, context, return_scalar)

### Function: __new__(cls)

### Function: __array__(self, dtype, copy)

### Function: __array_wrap__(self, arr, context, return_scalar)

### Function: __array__(self, dtype, copy)

### Function: __array_wrap__(self, arr, context, return_scalar)

### Function: __array_wrap__(self, obj, context, return_scalar)

### Function: __array_wrap__(self, obj, context, return_scalar)

### Function: __array__(self, dtype, copy)

### Function: __array_wrap__(self, arr, context, return_scalar)

### Function: __array__(self, dtype, copy)

### Function: __array_wrap__(self, arr, context, return_scalar)

### Function: __array__(self, dtype, context, copy)

### Function: __array_ufunc__(self, func, method)

### Function: __array_ufunc__(self, func, method)

### Function: __array_ufunc__(self, func, method)

### Function: __array_ufunc__(self, func, method)

### Function: __init__(self)

### Function: __array_ufunc__(self, func, method)

### Function: __array_ufunc__(self, func, method)

### Function: __array_ufunc__(self, ufunc, method)

### Function: __array_ufunc__(self, ufunc, method)

### Function: __array_ufunc__(self, ufunc, method)

### Function: _unwrap(self, objs)

### Function: __array_ufunc__(self, ufunc, method)

### Function: __array_ufunc__(self, ufunc, method)

### Function: __array_ufunc__(self)

### Function: __array_ufunc__(self)

### Function: __array_ufunc__(self)

### Function: __array_ufunc__(self, ufunc, method)

### Function: __array_ufunc__(self, ufunc, method)

### Function: __array_ufunc__(self, ufunc, method)

### Function: __floor__(self)

### Function: __ceil__(self)

### Function: __trunc__(self)

### Function: __float__(self)

### Function: __new__(subtype, shape)
