## AI Summary

A file named test_numeric.py.


## Class: TestResize

## Class: TestNonarrayArgs

## Class: TestIsscalar

## Class: TestBoolScalar

## Class: TestBoolArray

## Class: TestBoolCmp

## Class: TestSeterr

## Class: TestFloatExceptions

## Class: TestTypes

## Class: NIterError

## Class: TestFromiter

## Class: TestNonzero

## Class: TestIndex

## Class: TestBinaryRepr

## Class: TestBaseRepr

### Function: _test_array_equal_parametrizations()

**Description:** we pre-create arrays as we sometime want to pass the same instance
and sometime not. Passing the same instances may not mean the array are
equal, especially when containing None

## Class: TestArrayComparisons

### Function: assert_array_strict_equal(x, y)

## Class: TestClip

## Class: TestAllclose

## Class: TestIsclose

## Class: TestStdVar

## Class: TestStdVarComplex

## Class: TestCreationFuncs

## Class: TestLikeFuncs

**Description:** Test ones_like, zeros_like, empty_like and full_like

## Class: TestCorrelate

## Class: TestConvolve

## Class: TestArgwhere

## Class: TestRoll

## Class: TestRollaxis

## Class: TestMoveaxis

## Class: TestCross

### Function: test_outer_out_param()

## Class: TestIndices

## Class: TestRequire

## Class: TestBroadcast

## Class: TestKeepdims

## Class: TestTensordot

## Class: TestAsType

### Function: test_copies(self)

### Function: test_repeats(self)

### Function: test_zeroresize(self)

### Function: test_reshape_from_zero(self)

### Function: test_negative_resize(self)

### Function: test_subclass(self)

### Function: test_choose(self)

### Function: test_clip(self)

### Function: test_compress(self)

### Function: test_count_nonzero(self)

### Function: test_diagonal(self)

### Function: test_mean(self)

### Function: test_ptp(self)

### Function: test_prod(self)

### Function: test_ravel(self)

### Function: test_repeat(self)

### Function: test_reshape(self)

### Function: test_reshape_shape_arg(self)

### Function: test_reshape_copy_arg(self)

### Function: test_round(self)

### Function: test_dunder_round(self, dtype)

### Function: test_dunder_round_edgecases(self, val, ndigits)

### Function: test_dunder_round_accuracy(self)

### Function: test_round_py_consistency(self)

### Function: test_searchsorted(self)

### Function: test_size(self)

### Function: test_squeeze(self)

### Function: test_std(self)

### Function: test_swapaxes(self)

### Function: test_sum(self)

### Function: test_take(self)

### Function: test_trace(self)

### Function: test_transpose(self)

### Function: test_var(self)

### Function: test_std_with_mean_keyword(self)

### Function: test_var_with_mean_keyword(self)

### Function: test_std_with_mean_keyword_keepdims_false(self)

### Function: test_var_with_mean_keyword_keepdims_false(self)

### Function: test_std_with_mean_keyword_where_nontrivial(self)

### Function: test_var_with_mean_keyword_where_nontrivial(self)

### Function: test_std_with_mean_keyword_multiple_axis(self)

### Function: test_std_with_mean_keyword_axis_None(self)

### Function: test_std_with_mean_keyword_keepdims_true_masked(self)

### Function: test_var_with_mean_keyword_keepdims_true_masked(self)

### Function: test_isscalar(self)

### Function: test_logical(self)

### Function: test_bitwise_or(self)

### Function: test_bitwise_and(self)

### Function: test_bitwise_xor(self)

### Function: setup_method(self)

### Function: test_all_any(self)

### Function: test_logical_not_abs(self)

### Function: test_logical_and_or_xor(self)

### Function: setup_method(self)

### Function: test_float(self)

### Function: test_double(self)

### Function: test_default(self)

### Function: test_set(self)

### Function: test_divide_err(self)

### Function: assert_raises_fpe(self, fpeerr, flop, x, y)

### Function: assert_op_raises_fpe(self, fpeerr, flop, sc1, sc2)

### Function: test_floating_exceptions(self, typecode)

### Function: test_warnings(self)

### Function: check_promotion_cases(self, promote_func)

### Function: test_coercion(self)

### Function: test_result_type(self)

### Function: test_promote_types_endian(self)

### Function: test_can_cast_and_promote_usertypes(self)

### Function: test_promote_types_strings(self, swap, string_dtype)

### Function: test_invalid_void_promotion(self, dtype1, dtype2)

### Function: test_valid_void_promotion(self, dtype1, dtype2)

### Function: test_promote_identical_types_metadata(self, dtype)

### Function: test_promote_types_metadata(self, dtype1, dtype2)

**Description:** Metadata handling in promotion does not appear formalized
right now in NumPy. This test should thus be considered to
document behaviour, rather than test the correct definition of it.

This test is very ugly, it was useful for rewriting part of the
promotion, but probably should eventually be replaced/deleted
(i.e. when metadata handling in promotion is better defined).

### Function: test_can_cast(self)

### Function: test_can_cast_simple_to_structured(self)

### Function: test_can_cast_structured_to_simple(self)

### Function: test_can_cast_values(self)

### Function: test_can_cast_scalars(self, dtype)

### Function: makegen(self)

### Function: test_types(self)

### Function: test_lengths(self)

### Function: test_values(self)

### Function: load_data(self, n, eindex)

### Function: test_2592(self, count, error_index, dtype)

### Function: test_empty_not_structured(self, dtype)

### Function: test_growth_and_complicated_dtypes(self, dtype, data, length_hint)

### Function: test_empty_result(self)

### Function: test_too_few_items(self)

### Function: test_failed_itemsetting(self)

### Function: test_nonzero_trivial(self)

### Function: test_nonzero_zerodim(self)

### Function: test_nonzero_onedim(self)

### Function: test_nonzero_twodim(self)

### Function: test_sparse(self)

### Function: test_nonzero_float_dtypes(self, dtype)

### Function: test_nonzero_integer_dtypes(self, dtype)

### Function: test_return_type(self)

### Function: test_count_nonzero_axis(self)

### Function: test_count_nonzero_axis_all_dtypes(self)

### Function: test_count_nonzero_axis_consistent(self)

### Function: test_countnonzero_axis_empty(self)

### Function: test_countnonzero_keepdims(self)

### Function: test_array_method(self)

### Function: test_nonzero_invalid_object(self)

### Function: test_nonzero_sideeffect_safety(self)

### Function: test_nonzero_sideffects_structured_void(self)

### Function: test_nonzero_exception_safe(self)

### Function: test_structured_threadsafety(self)

### Function: test_boolean(self)

### Function: test_boolean_edgecase(self)

### Function: test_zero(self)

### Function: test_positive(self)

### Function: test_negative(self)

### Function: test_sufficient_width(self)

### Function: test_neg_width_boundaries(self)

### Function: test_large_neg_int64(self)

### Function: test_base3(self)

### Function: test_positive(self)

### Function: test_negative(self)

### Function: test_base_range(self)

### Function: test_minimal_signed_int(self)

### Function: test_array_equal_equal_nan(self, bx, by, equal_nan, expected)

**Description:** This test array_equal for a few combinations:

- are the two inputs the same object or not (same object may not
  be equal if contains NaNs)
- Whether we should consider or not, NaNs, being equal.

### Function: test_array_equal_different_scalar_types(self)

### Function: test_none_compares_elementwise(self)

### Function: test_array_equiv(self)

### Function: test_compare_unstructured_voids(self, dtype)

### Function: setup_method(self)

### Function: fastclip(self, a, m, M, out)

### Function: clip(self, a, m, M, out)

### Function: _generate_data(self, n, m)

### Function: _generate_data_complex(self, n, m)

### Function: _generate_flt_data(self, n, m)

### Function: _neg_byteorder(self, a)

### Function: _generate_non_native_data(self, n, m)

### Function: _generate_int_data(self, n, m)

### Function: _generate_int32_data(self, n, m)

### Function: test_ones_pathological(self, dtype)

### Function: test_simple_double(self)

### Function: test_simple_int(self)

### Function: test_array_double(self)

### Function: test_simple_nonnative(self)

### Function: test_simple_complex(self)

### Function: test_clip_complex(self)

### Function: test_clip_non_contig(self)

### Function: test_simple_out(self)

### Function: test_simple_int32_inout(self, casting)

### Function: test_simple_int64_out(self)

### Function: test_simple_int64_inout(self)

### Function: test_simple_int32_out(self)

### Function: test_simple_inplace_01(self)

### Function: test_simple_inplace_02(self)

### Function: test_noncontig_inplace(self)

### Function: test_type_cast_01(self)

### Function: test_type_cast_02(self)

### Function: test_type_cast_03(self)

### Function: test_type_cast_04(self)

### Function: test_type_cast_05(self)

### Function: test_type_cast_06(self)

### Function: test_type_cast_07(self)

### Function: test_type_cast_08(self)

### Function: test_type_cast_09(self)

### Function: test_type_cast_10(self)

### Function: test_type_cast_11(self)

### Function: test_type_cast_12(self)

### Function: test_clip_with_out_simple(self)

### Function: test_clip_with_out_simple2(self)

### Function: test_clip_with_out_simple_int32(self)

### Function: test_clip_with_out_array_int32(self)

### Function: test_clip_with_out_array_outint32(self)

### Function: test_clip_with_out_transposed(self)

### Function: test_clip_with_out_memory_overlap(self)

### Function: test_clip_inplace_array(self)

### Function: test_clip_inplace_simple(self)

### Function: test_clip_func_takes_out(self)

### Function: test_clip_nan(self)

### Function: test_object_clip(self)

### Function: test_clip_all_none(self)

### Function: test_clip_invalid_casting(self)

### Function: test_clip_value_min_max_flip(self, amin, amax)

### Function: test_clip_problem_cases(self, arr, amin, amax, exp)

### Function: test_clip_scalar_nan_propagation(self, arr, amin, amax)

### Function: test_NaT_propagation(self, arr, amin, amax)

### Function: test_clip_property(self, data, arr)

**Description:** A property-based test using Hypothesis.

This aims for maximum generality: it could in principle generate *any*
valid inputs to np.clip, and in practice generates much more varied
inputs than human testers come up with.

Because many of the inputs have tricky dependencies - compatible dtypes
and mutually-broadcastable shapes - we use `st.data()` strategy draw
values *inside* the test function, from strategies we construct based
on previous values.  An alternative would be to define a custom strategy
with `@st.composite`, but until we have duplicated code inline is fine.

That accounts for most of the function; the actual test is just three
lines to calculate and compare actual vs expected results!

### Function: test_clip_min_max_args(self)

### Function: test_out_of_bound_pyints(self, dtype, min, max)

### Function: setup_method(self)

### Function: teardown_method(self)

### Function: tst_allclose(self, x, y)

### Function: tst_not_allclose(self, x, y)

### Function: test_ip_allclose(self)

### Function: test_ip_not_allclose(self)

### Function: test_no_parameter_modification(self)

### Function: test_min_int(self)

### Function: test_equalnan(self)

### Function: test_return_class_is_ndarray(self)

### Function: _setup(self)

### Function: test_ip_isclose(self)

### Function: test_nep50_isclose(self)

### Function: tst_all_isclose(self, x, y)

### Function: tst_none_isclose(self, x, y)

### Function: tst_isclose_allclose(self, x, y)

### Function: test_ip_all_isclose(self)

### Function: test_ip_none_isclose(self)

### Function: test_ip_isclose_allclose(self)

### Function: test_equal_nan(self)

### Function: test_masked_arrays(self)

### Function: test_scalar_return(self)

### Function: test_no_parameter_modification(self)

### Function: test_non_finite_scalar(self)

### Function: test_timedelta(self)

### Function: setup_method(self)

### Function: test_basic(self)

### Function: test_scalars(self)

### Function: test_ddof1(self)

### Function: test_ddof2(self)

### Function: test_correction(self)

### Function: test_out_scalar(self)

### Function: test_basic(self)

### Function: test_scalars(self)

### Function: setup_method(self)

### Function: check_function(self, func, fill_value)

### Function: test_zeros(self)

### Function: test_ones(self)

### Function: test_empty(self)

### Function: test_full(self)

### Function: test_for_reference_leak(self)

### Function: setup_method(self)

### Function: compare_array_value(self, dz, value, fill_value)

### Function: check_like_function(self, like_function, value, fill_value)

### Function: test_ones_like(self)

### Function: test_zeros_like(self)

### Function: test_empty_like(self)

### Function: test_filled_like(self)

### Function: test_dtype_str_bytes(self, likefunc, dtype)

### Function: _setup(self, dt)

### Function: test_float(self)

### Function: test_object(self)

### Function: test_no_overwrite(self)

### Function: test_complex(self)

### Function: test_zero_size(self)

### Function: test_mode(self)

### Function: test_object(self)

### Function: test_no_overwrite(self)

### Function: test_mode(self)

### Function: test_nd(self, nd)

### Function: test_2D(self)

### Function: test_list(self)

### Function: test_roll1d(self)

### Function: test_roll2d(self)

### Function: test_roll_empty(self)

### Function: test_roll_unsigned_shift(self)

### Function: test_roll_big_int(self)

### Function: test_exceptions(self)

### Function: test_results(self)

### Function: test_move_to_end(self)

### Function: test_move_new_position(self)

### Function: test_preserve_order(self)

### Function: test_move_multiples(self)

### Function: test_errors(self)

### Function: test_array_likes(self)

### Function: test_2x2(self)

### Function: test_2x3(self)

### Function: test_3x3(self)

### Function: test_broadcasting(self)

### Function: test_broadcasting_shapes(self)

### Function: test_uint8_int32_mixed_dtypes(self)

### Function: test_zero_dimension(self, a, b)

### Function: test_simple(self)

### Function: test_single_input(self)

### Function: test_scalar_input(self)

### Function: test_sparse(self)

### Function: test_return_type(self, dtype, dims)

### Function: generate_all_false(self, dtype)

### Function: set_and_check_flag(self, flag, dtype, arr)

### Function: test_require_each(self)

### Function: test_unknown_requirement(self)

### Function: test_non_array_input(self)

### Function: test_C_and_F_simul(self)

### Function: test_ensure_array(self)

### Function: test_preserve_subtype(self)

### Function: test_broadcast_in_args(self)

### Function: test_broadcast_single_arg(self)

### Function: test_number_of_arguments(self)

### Function: test_broadcast_error_kwargs(self)

### Function: test_shape_mismatch_error_message(self)

## Class: sub_array

### Function: test_raise(self)

### Function: test_zero_dimension(self)

### Function: test_zero_dimensional(self)

### Function: test_astype(self)

## Class: MyArray

### Function: res_type(a, b)

## Class: MyIter

## Class: MyIter

## Class: C

### Function: assert_equal_w_dt(a, b, err_msg)

## Class: BoolErrors

## Class: FalseThenTrue

## Class: TrueThenFalse

## Class: ThrowsAfter

### Function: func(arr)

## Class: Foo

## Class: MyNDArray

## Class: ArraySubclass

## Class: ArraySubclass

### Function: sum(self, axis, dtype, out)

### Function: __length_hint__(self)

### Function: __iter__(self)

### Function: __length_hint__(self)

### Function: __iter__(self)

### Function: __bool__(self)

### Function: __bool__(self)

### Function: __bool__(self)

### Function: __init__(self, iters)

### Function: __bool__(self)

### Function: __new__(cls)
