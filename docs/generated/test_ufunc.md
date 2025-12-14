## AI Summary

A file named test_ufunc.py.


## Class: TestUfuncKwargs

## Class: TestUfuncGenericLoops

**Description:** Test generic loops.

The loops to be tested are:

    PyUFunc_ff_f_As_dd_d
    PyUFunc_ff_f
    PyUFunc_dd_d
    PyUFunc_gg_g
    PyUFunc_FF_F_As_DD_D
    PyUFunc_DD_D
    PyUFunc_FF_F
    PyUFunc_GG_G
    PyUFunc_OO_O
    PyUFunc_OO_O_method
    PyUFunc_f_f_As_d_d
    PyUFunc_d_d
    PyUFunc_f_f
    PyUFunc_g_g
    PyUFunc_F_F_As_D_D
    PyUFunc_F_F
    PyUFunc_D_D
    PyUFunc_G_G
    PyUFunc_O_O
    PyUFunc_O_O_method
    PyUFunc_On_Om

Where:

    f -- float
    d -- double
    g -- long double
    F -- complex float
    D -- complex double
    G -- complex long double
    O -- python object

It is difficult to assure that each of these loops is entered from the
Python level as the special cased loops are a moving target and the
corresponding types are architecture dependent. We probably need to
define C level testing ufuncs to get at them. For the time being, I've
just looked at the signatures registered in the build directory to find
relevant functions.

### Function: _pickleable_module_global()

## Class: TestUfunc

## Class: TestGUFuncProcessCoreDims

### Function: test_ufunc_types(ufunc)

**Description:** Check all ufuncs that the correct type is returned. Avoid
object and boolean types since many operations are not defined for
for them.

Choose the shape so even dot and matmul will succeed

### Function: test_ufunc_noncontiguous(ufunc)

**Description:** Check that contiguous and non-contiguous calls to ufuncs
have the same results for values in range(9)

### Function: test_ufunc_warn_with_nan(ufunc)

### Function: test_ufunc_out_casterrors()

### Function: test_ufunc_input_casterrors(bad_offset)

### Function: test_ufunc_input_floatingpoint_error(bad_offset)

### Function: test_trivial_loop_invalid_cast()

### Function: test_reduce_casterrors(offset)

### Function: test_object_reduce_cleanup_on_failure()

### Function: test_ufunc_methods_floaterrors(method)

### Function: _check_neg_zero(value)

### Function: test_addition_negative_zero(dtype)

### Function: test_addition_reduce_negative_zero(dtype, use_initial)

### Function: test_addition_string_types(dt1, dt2)

### Function: test_addition_unicode_inverse_byte_order(order1, order2)

### Function: test_find_non_long_args(dtype)

### Function: test_find_access_past_buffer()

## Class: TestLowlevelAPIAccess

### Function: test_kwarg_exact(self)

### Function: test_sig_signature(self)

### Function: test_sig_dtype(self)

### Function: test_extobj_removed(self)

### Function: test_unary_PyUFunc(self, input_dtype, output_dtype, f, x, y)

### Function: f2(x, y)

### Function: test_binary_PyUFunc(self, input_dtype, output_dtype, f, x, y)

## Class: foo

### Function: test_unary_PyUFunc_O_O(self)

### Function: test_unary_PyUFunc_O_O_method_simple(self, foo)

### Function: test_binary_PyUFunc_OO_O(self)

### Function: test_binary_PyUFunc_OO_O_method(self, foo)

### Function: test_binary_PyUFunc_On_Om_method(self, foo)

### Function: test_python_complex_conjugate(self)

### Function: test_unary_PyUFunc_O_O_method_full(self, ufunc)

**Description:** Compare the result of the object loop with non-object one

### Function: test_pickle(self)

### Function: test_pickle_withstring(self)

### Function: test_pickle_name_is_qualname(self)

### Function: test_reduceat_shifting_sum(self)

### Function: test_all_ufunc(self)

**Description:** Try to check presence and results of all ufuncs.

The list of ufuncs comes from generate_umath.py and is as follows:

=====  ====  =============  ===============  ========================
done   args   function        types                notes
=====  ====  =============  ===============  ========================
n      1     conjugate      nums + O
n      1     absolute       nums + O         complex -> real
n      1     negative       nums + O
n      1     sign           nums + O         -> int
n      1     invert         bool + ints + O  flts raise an error
n      1     degrees        real + M         cmplx raise an error
n      1     radians        real + M         cmplx raise an error
n      1     arccos         flts + M
n      1     arccosh        flts + M
n      1     arcsin         flts + M
n      1     arcsinh        flts + M
n      1     arctan         flts + M
n      1     arctanh        flts + M
n      1     cos            flts + M
n      1     sin            flts + M
n      1     tan            flts + M
n      1     cosh           flts + M
n      1     sinh           flts + M
n      1     tanh           flts + M
n      1     exp            flts + M
n      1     expm1          flts + M
n      1     log            flts + M
n      1     log10          flts + M
n      1     log1p          flts + M
n      1     sqrt           flts + M         real x < 0 raises error
n      1     ceil           real + M
n      1     trunc          real + M
n      1     floor          real + M
n      1     fabs           real + M
n      1     rint           flts + M
n      1     isnan          flts             -> bool
n      1     isinf          flts             -> bool
n      1     isfinite       flts             -> bool
n      1     signbit        real             -> bool
n      1     modf           real             -> (frac, int)
n      1     logical_not    bool + nums + M  -> bool
n      2     left_shift     ints + O         flts raise an error
n      2     right_shift    ints + O         flts raise an error
n      2     add            bool + nums + O  boolean + is ||
n      2     subtract       bool + nums + O  boolean - is ^
n      2     multiply       bool + nums + O  boolean * is &
n      2     divide         nums + O
n      2     floor_divide   nums + O
n      2     true_divide    nums + O         bBhH -> f, iIlLqQ -> d
n      2     fmod           nums + M
n      2     power          nums + O
n      2     greater        bool + nums + O  -> bool
n      2     greater_equal  bool + nums + O  -> bool
n      2     less           bool + nums + O  -> bool
n      2     less_equal     bool + nums + O  -> bool
n      2     equal          bool + nums + O  -> bool
n      2     not_equal      bool + nums + O  -> bool
n      2     logical_and    bool + nums + M  -> bool
n      2     logical_or     bool + nums + M  -> bool
n      2     logical_xor    bool + nums + M  -> bool
n      2     maximum        bool + nums + O
n      2     minimum        bool + nums + O
n      2     bitwise_and    bool + ints + O  flts raise an error
n      2     bitwise_or     bool + ints + O  flts raise an error
n      2     bitwise_xor    bool + ints + O  flts raise an error
n      2     arctan2        real + M
n      2     remainder      ints + real + O
n      2     hypot          real + M
=====  ====  =============  ===============  ========================

Types other than those listed will be accepted, but they are cast to
the smallest compatible type for which the function is defined. The
casting rules are:

bool -> int8 -> float32
ints -> double

### Function: test_signature0(self)

### Function: test_signature1(self)

### Function: test_signature2(self)

### Function: test_signature3(self)

### Function: test_signature4(self)

### Function: test_signature5(self)

### Function: test_signature6(self)

### Function: test_signature7(self)

### Function: test_signature8(self)

### Function: test_signature9(self)

### Function: test_signature10(self)

### Function: test_signature_failure_extra_parenthesis(self)

### Function: test_signature_failure_mismatching_parenthesis(self)

### Function: test_signature_failure_signature_missing_input_arg(self)

### Function: test_signature_failure_signature_missing_output_arg(self)

### Function: test_get_signature(self)

### Function: test_forced_sig(self)

### Function: test_signature_all_None(self)

### Function: test_signature_dtype_type(self)

### Function: test_signature_dtype_instances_allowed(self, get_kwarg)

### Function: test_signature_dtype_instances_allowed(self, get_kwarg)

### Function: test_partial_signature_mismatch(self, casting)

### Function: test_partial_signature_mismatch_with_cache(self)

### Function: test_use_output_signature_for_all_arguments(self)

### Function: test_signature_errors(self)

### Function: test_forced_dtype_times(self)

### Function: test_cast_safety(self, ufunc)

**Description:** Basic test for the safest casts, because ufuncs inner loops can
indicate a cast-safety as well (which is normally always "no").

### Function: test_cast_safety_scalar(self, ufunc)

### Function: test_cast_safety_scalar_special(self)

### Function: test_true_divide(self)

### Function: test_sum_stability(self)

### Function: test_sum(self)

### Function: test_sum_complex(self)

### Function: test_sum_initial(self)

### Function: test_sum_where(self)

### Function: test_vecdot(self)

### Function: test_matvec(self)

### Function: test_vecmatvec_identity(self, matrix, vec)

**Description:** Check that (x†A)x equals x†(Ax).

### Function: test_vecdot_matvec_vecmat_complex(self, ufunc, shape1, shape2, conj)

### Function: test_vecdot_subclass(self)

### Function: test_vecdot_object_no_conjugate(self)

### Function: test_vecdot_object_breaks_outer_loop_on_error(self)

### Function: test_broadcast(self)

### Function: test_out_broadcasts(self)

### Function: test_out_broadcast_errors(self, arr, out)

### Function: test_type_cast(self)

### Function: test_endian(self)

### Function: test_incontiguous_array(self)

### Function: test_output_argument(self)

### Function: test_axes_argument(self)

### Function: test_axis_argument(self)

### Function: test_keepdims_argument(self)

### Function: test_innerwt(self)

### Function: test_innerwt_empty(self)

**Description:** Test generalized ufunc with zero-sized operands

### Function: test_cross1d(self)

**Description:** Test with fixed-sized signature.

### Function: test_can_ignore_signature(self)

### Function: test_matrix_multiply(self)

### Function: test_matrix_multiply_umath_empty(self)

### Function: compare_matrix_multiply_results(self, tp)

### Function: test_euclidean_pdist(self)

### Function: test_cumsum(self)

### Function: test_object_logical(self)

### Function: test_object_comparison(self)

### Function: test_object_array_reduction(self)

### Function: test_object_array_accumulate_inplace(self)

### Function: test_object_array_accumulate_failure(self)

### Function: test_object_array_reduceat_inplace(self)

### Function: test_object_array_reduceat_failure(self)

### Function: test_zerosize_reduction(self)

### Function: test_axis_out_of_bounds(self)

### Function: test_scalar_reduction(self)

### Function: test_casting_out_param(self)

### Function: test_where_param(self)

### Function: test_where_param_buffer_output(self)

### Function: test_where_param_alloc(self)

### Function: test_where_with_broadcasting(self)

### Function: identityless_reduce_arrs()

### Function: test_identityless_reduction(self, a, pos)

### Function: test_identityless_reduction_huge_array(self)

### Function: test_reduce_identity_depends_on_loop(self)

**Description:** The type of the result should always depend on the selected loop, not
necessarily the output (only relevant for object arrays).

### Function: test_initial_reduction(self)

### Function: test_empty_reduction_and_identity(self)

### Function: test_reduction_with_where(self, axis, where)

### Function: test_reduction_with_where_and_initial(self, axis, where, initial)

### Function: test_reduction_where_initial_needed(self)

### Function: test_identityless_reduction_nonreorderable(self)

### Function: test_reduce_zero_axis(self)

### Function: test_safe_casting(self)

### Function: test_ufunc_custom_out(self)

### Function: test_operand_flags(self)

### Function: test_struct_ufunc(self)

### Function: test_custom_ufunc(self)

### Function: test_custom_ufunc_forced_sig(self)

### Function: test_custom_array_like(self)

### Function: test_ufunc_at_basic(self, a)

### Function: test_ufunc_at_inner_loops(self, typecode, ufunc)

### Function: test_ufunc_at_inner_loops_complex(self, typecode, ufunc)

### Function: test_ufunc_at_ellipsis(self)

### Function: test_ufunc_at_negative(self)

### Function: test_ufunc_at_large(self)

### Function: test_cast_index_fastpath(self)

### Function: test_ufunc_at_scalar_value_fastpath(self, value)

### Function: test_ufunc_at_multiD(self)

### Function: test_ufunc_at_0D(self)

### Function: test_ufunc_at_dtypes(self)

### Function: test_ufunc_at_boolean(self)

### Function: test_ufunc_at_advanced(self)

### Function: test_at_negative_indexes(self, dtype, ufunc)

### Function: test_at_not_none_signature(self)

### Function: test_at_no_loop_for_op(self)

### Function: test_at_output_casting(self)

### Function: test_at_broadcast_failure(self)

### Function: test_reduce_arguments(self)

### Function: test_structured_equal(self)

### Function: test_scalar_equal(self)

### Function: test_NotImplemented_not_returned(self)

### Function: test_logical_ufuncs_object_signatures(self, ufunc, signature)

### Function: test_logical_ufuncs_mixed_object_signatures(self, ufunc, signature)

### Function: test_logical_ufuncs_support_anything(self, ufunc)

### Function: test_logical_ufuncs_supports_string(self, ufunc, dtype, values)

### Function: test_logical_ufuncs_out_cast_check(self, ufunc)

### Function: test_reducelike_byteorder_resolution(self)

### Function: test_reducelike_out_promotes(self)

### Function: test_reducelike_output_needs_identical_cast(self)

### Function: test_reduce_noncontig_output(self)

### Function: test_reduceat_and_accumulate_out_shape_mismatch(self, with_cast)

### Function: test_reduce_wrong_dimension_output(self, f_reduce, keepdims, out_shape)

### Function: test_reduce_output_does_not_broadcast_input(self)

### Function: test_reduce_output_subclass_ok(self)

### Function: test_no_doc_string(self)

### Function: test_invalid_args(self)

### Function: test_nat_is_not_finite(self, nat)

### Function: test_nat_is_nan(self, nat)

### Function: test_nat_is_not_inf(self, nat)

### Function: test_conv1d_full_without_out(self)

### Function: test_conv1d_full_with_out(self)

### Function: test_conv1d_full_basic_broadcast(self)

### Function: test_bad_out_shape(self)

### Function: test_bad_input_both_inputs_length_zero(self)

### Function: test_resolve_dtypes_basic(self)

### Function: test_resolve_dtypes_comparison(self)

### Function: test_weird_dtypes(self)

### Function: test_resolve_dtypes_reduction(self)

### Function: test_resolve_dtypes_reduction_no_output(self)

### Function: test_resolve_dtypes_errors(self, dtypes)

### Function: test_resolve_dtypes_reduction_errors(self)

### Function: test_loop_access(self)

### Function: test__get_strided_loop_errors_bad_strides(self, strides)

### Function: test__get_strided_loop_errors_bad_call_info(self)

### Function: test_long_arrays(self)

### Function: conjugate(self)

### Function: logical_xor(self, obj)

## Class: MyFloat

### Function: call_ufunc(arr)

## Class: MySubclass

### Function: permute_n(n)

### Function: slice_n(n)

### Function: broadcastable(s1, s2)

## Class: HasComparisons

## Class: MyArray

### Function: ok(f)

### Function: err(f)

### Function: t(expect, func, n, m)

### Function: add_inplace(a, b)

## Class: MyThing

## Class: MyA

## Class: MyArr

## Class: call_info_t

### Function: __getattr__(self, attr)

### Function: __eq__(self, other)

### Function: __init__(self, shape)

### Function: __len__(self)

### Function: __getitem__(self, i)

### Function: __rmul__(self, other)

### Function: __array_ufunc__(self, ufunc, method)
