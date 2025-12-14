## AI Summary

A file named test_core.py.


## Class: TestMaskedArray

## Class: TestMaskedArrayArithmetic

## Class: TestMaskedArrayAttributes

## Class: TestFillingValues

## Class: TestUfuncs

## Class: TestMaskedArrayInPlaceArithmetic

## Class: TestMaskedArrayMethods

## Class: TestMaskedArrayMathMethods

## Class: TestMaskedArrayMathMethodsComplex

## Class: TestMaskedArrayFunctions

## Class: TestMaskedFields

## Class: TestMaskedObjectArray

## Class: TestMaskedView

## Class: TestOptionalArgs

## Class: TestMaskedConstant

## Class: TestMaskedWhereAliases

### Function: test_masked_array()

### Function: test_masked_array_no_copy()

### Function: test_append_masked_array()

### Function: test_append_masked_array_along_axis()

### Function: test_default_fill_value_complex()

### Function: test_ufunc_with_output()

### Function: test_ufunc_with_out_varied()

**Description:** Test that masked arrays are immune to gh-10459 

### Function: test_astype_mask_ordering()

### Function: test_astype_basic(dt1, dt2)

### Function: test_fieldless_void()

### Function: test_mask_shape_assignment_does_not_break_masked()

### Function: test_doc_note()

### Function: test_gh_22556()

### Function: test_gh_21022()

### Function: test_deepcopy_2d_obj()

### Function: test_deepcopy_0d_obj()

### Function: test_uint_fill_value_and_filled()

### Function: setup_method(self)

### Function: test_basicattributes(self)

### Function: test_basic0d(self)

### Function: test_basic1d(self)

### Function: test_basic2d(self)

### Function: test_concatenate_basic(self)

### Function: test_concatenate_alongaxis(self)

### Function: test_concatenate_flexible(self)

### Function: test_creation_ndmin(self)

### Function: test_creation_ndmin_from_maskedarray(self)

### Function: test_creation_maskcreation(self)

### Function: test_masked_singleton_array_creation_warns(self)

### Function: test_creation_with_list_of_maskedarrays(self)

### Function: test_creation_with_list_of_maskedarrays_no_bool_cast(self)

### Function: test_creation_from_ndarray_with_padding(self)

### Function: test_unknown_keyword_parameter(self)

### Function: test_asarray(self)

### Function: test_asarray_default_order(self)

### Function: test_asarray_enforce_order(self)

### Function: test_fix_invalid(self)

### Function: test_maskedelement(self)

### Function: test_set_element_as_object(self)

### Function: test_indexing(self)

### Function: test_setitem_no_warning(self)

### Function: test_copy(self)

### Function: test_copy_0d(self)

### Function: test_copy_on_python_builtins(self)

### Function: test_copy_immutable(self)

### Function: test_deepcopy(self)

### Function: test_format(self)

### Function: test_str_repr(self)

### Function: test_str_repr_legacy(self)

### Function: test_0d_unicode(self)

### Function: test_pickling(self)

### Function: test_pickling_subbaseclass(self)

### Function: test_pickling_maskedconstant(self)

### Function: test_pickling_wstructured(self)

### Function: test_pickling_keepalignment(self)

### Function: test_single_element_subscript(self)

### Function: test_topython(self)

### Function: test_oddfeatures_1(self)

### Function: test_oddfeatures_2(self)

### Function: test_oddfeatures_3(self)

### Function: test_filled_with_object_dtype(self)

### Function: test_filled_with_flexible_dtype(self)

### Function: test_filled_with_mvoid(self)

### Function: test_filled_with_nested_dtype(self)

### Function: test_filled_with_f_order(self)

### Function: test_optinfo_propagation(self)

### Function: test_optinfo_forward_propagation(self)

### Function: test_fancy_printoptions(self)

### Function: test_flatten_structured_array(self)

### Function: test_void0d(self)

### Function: test_mvoid_getitem(self)

### Function: test_mvoid_iter(self)

### Function: test_mvoid_print(self)

### Function: test_mvoid_multidim_print(self)

### Function: test_object_with_array(self)

### Function: test_maskedarray_tofile_raises_notimplementederror(self)

### Function: setup_method(self)

### Function: teardown_method(self)

### Function: test_basic_arithmetic(self)

### Function: test_divide_on_different_shapes(self)

### Function: test_mixed_arithmetic(self)

### Function: test_limits_arithmetic(self)

### Function: test_masked_singleton_arithmetic(self)

### Function: test_masked_singleton_equality(self)

### Function: test_arithmetic_with_masked_singleton(self)

### Function: test_arithmetic_with_masked_singleton_on_1d_singleton(self)

### Function: test_scalar_arithmetic(self)

### Function: test_basic_ufuncs(self)

### Function: test_basic_ufuncs_masked(self)

### Function: test_count_func(self)

### Function: test_count_on_python_builtins(self)

### Function: test_minmax_func(self)

### Function: test_minimummaximum_func(self)

### Function: test_minmax_reduce(self)

### Function: test_minmax_funcs_with_output(self)

### Function: test_minmax_methods(self)

### Function: test_minmax_dtypes(self)

### Function: test_minmax_ints(self, dtype, mask, axis)

### Function: test_minmax_time_dtypes(self, time_type)

### Function: test_addsumprod(self)

### Function: test_binops_d2D(self)

### Function: test_domained_binops_d2D(self)

### Function: test_noshrinking(self)

### Function: test_ufunc_nomask(self)

### Function: test_noshink_on_creation(self)

### Function: test_mod(self)

### Function: test_TakeTransposeInnerOuter(self)

### Function: test_imag_real(self)

### Function: test_methods_with_output(self)

### Function: test_eq_on_structured(self)

### Function: test_ne_on_structured(self)

### Function: test_eq_ne_structured_with_non_masked(self)

### Function: test_eq_ne_structured_extra(self)

### Function: test_eq_for_strings(self, dt, fill)

### Function: test_ne_for_strings(self, dt, fill)

### Function: test_eq_for_numeric(self, dt1, dt2, fill)

### Function: test_eq_broadcast_with_unmasked(self, op)

### Function: test_comp_no_mask_not_broadcast(self, op)

### Function: test_ne_for_numeric(self, dt1, dt2, fill)

### Function: test_comparisons_for_numeric(self, op, dt1, dt2, fill)

### Function: test_comparisons_strings(self, op, fill)

### Function: test_eq_with_None(self)

### Function: test_eq_with_scalar(self)

### Function: test_eq_different_dimensions(self)

### Function: test_numpyarithmetic(self)

### Function: test_keepmask(self)

### Function: test_hardmask(self)

### Function: test_hardmask_again(self)

### Function: test_hardmask_oncemore_yay(self)

### Function: test_smallmask(self)

### Function: test_shrink_mask(self)

### Function: test_flat(self)

### Function: test_assign_dtype(self)

### Function: test_check_on_scalar(self)

### Function: test_check_on_fields(self)

### Function: test_fillvalue_conversion(self)

### Function: test_default_fill_value(self)

### Function: test_default_fill_value_structured(self)

### Function: test_default_fill_value_void(self)

### Function: test_fillvalue(self)

### Function: test_subarray_fillvalue(self)

### Function: test_fillvalue_exotic_dtype(self)

### Function: test_fillvalue_datetime_timedelta(self)

### Function: test_extremum_fill_value(self)

### Function: test_extremum_fill_value_subdtype(self)

### Function: test_fillvalue_individual_fields(self)

### Function: test_fillvalue_implicit_structured_array(self)

### Function: test_fillvalue_as_arguments(self)

### Function: test_shape_argument(self)

### Function: test_fillvalue_in_view(self)

### Function: test_fillvalue_bytes_or_str(self)

### Function: setup_method(self)

### Function: teardown_method(self)

### Function: test_testUfuncRegression(self)

### Function: test_reduce(self)

### Function: test_minmax(self)

### Function: test_ndarray_mask(self)

### Function: test_treatment_of_NotImplemented(self)

### Function: test_no_masked_nan_warnings(self)

### Function: test_masked_array_underflow(self)

### Function: setup_method(self)

### Function: test_inplace_addition_scalar(self)

### Function: test_inplace_addition_array(self)

### Function: test_inplace_subtraction_scalar(self)

### Function: test_inplace_subtraction_array(self)

### Function: test_inplace_multiplication_scalar(self)

### Function: test_inplace_multiplication_array(self)

### Function: test_inplace_division_scalar_int(self)

### Function: test_inplace_division_scalar_float(self)

### Function: test_inplace_division_array_float(self)

### Function: test_inplace_division_misc(self)

### Function: test_datafriendly_add(self)

### Function: test_datafriendly_sub(self)

### Function: test_datafriendly_mul(self)

### Function: test_datafriendly_div(self)

### Function: test_datafriendly_pow(self)

### Function: test_datafriendly_add_arrays(self)

### Function: test_datafriendly_sub_arrays(self)

### Function: test_datafriendly_mul_arrays(self)

### Function: test_inplace_addition_scalar_type(self)

### Function: test_inplace_addition_array_type(self)

### Function: test_inplace_subtraction_scalar_type(self)

### Function: test_inplace_subtraction_array_type(self)

### Function: test_inplace_multiplication_scalar_type(self)

### Function: test_inplace_multiplication_array_type(self)

### Function: test_inplace_floor_division_scalar_type(self)

### Function: test_inplace_floor_division_array_type(self)

### Function: test_inplace_division_scalar_type(self)

### Function: test_inplace_division_array_type(self)

### Function: test_inplace_pow_type(self)

### Function: setup_method(self)

### Function: test_generic_methods(self)

### Function: test_allclose(self)

### Function: test_allclose_timedelta(self)

### Function: test_allany(self)

### Function: test_allany_oddities(self)

### Function: test_argmax_argmin(self)

### Function: test_clip(self)

### Function: test_clip_out(self)

### Function: test_compress(self)

### Function: test_compressed(self)

### Function: test_empty(self)

### Function: test_zeros(self)

### Function: test_ones(self)

### Function: test_put(self)

### Function: test_put_nomask(self)

### Function: test_put_hardmask(self)

### Function: test_putmask(self)

### Function: test_ravel(self)

### Function: test_ravel_order(self, order, data_order)

### Function: test_reshape(self)

### Function: test_sort(self)

### Function: test_stable_sort(self)

### Function: test_argsort_matches_sort(self)

### Function: test_sort_2d(self)

### Function: test_sort_flexible(self)

### Function: test_argsort(self)

### Function: test_squeeze(self)

### Function: test_swapaxes(self)

### Function: test_take(self)

### Function: test_take_masked_indices(self)

### Function: test_tolist(self)

### Function: test_tolist_specialcase(self)

### Function: test_toflex(self)

### Function: test_fromflex(self)

### Function: test_arraymethod(self)

### Function: test_arraymethod_0d(self)

### Function: test_transpose_view(self)

### Function: test_diagonal_view(self)

### Function: setup_method(self)

### Function: test_cumsumprod(self)

### Function: test_cumsumprod_with_output(self)

### Function: test_ptp(self)

### Function: test_add_object(self)

### Function: test_sum_object(self)

### Function: test_prod_object(self)

### Function: test_meananom_object(self)

### Function: test_anom_shape(self)

### Function: test_anom(self)

### Function: test_trace(self)

### Function: test_dot(self)

### Function: test_dot_shape_mismatch(self)

### Function: test_varmean_nomask(self)

### Function: test_varstd(self)

### Function: test_varstd_specialcases(self)

### Function: test_varstd_ddof(self)

### Function: test_diag(self)

### Function: test_axis_methods_nomask(self)

### Function: test_mean_overflow(self)

### Function: test_diff_with_prepend(self)

### Function: test_diff_with_append(self)

### Function: test_diff_with_dim_0(self)

### Function: test_diff_with_n_0(self)

### Function: setup_method(self)

### Function: test_varstd(self)

### Function: setup_method(self)

### Function: test_masked_where_bool(self)

### Function: test_masked_equal_wlist(self)

### Function: test_masked_equal_fill_value(self)

### Function: test_masked_where_condition(self)

### Function: test_masked_where_oddities(self)

### Function: test_masked_where_shape_constraint(self)

### Function: test_masked_where_structured(self)

### Function: test_masked_where_mismatch(self)

### Function: test_masked_otherfunctions(self)

### Function: test_round(self)

### Function: test_round_with_output(self)

### Function: test_round_with_scalar(self)

### Function: test_identity(self)

### Function: test_power(self)

### Function: test_power_with_broadcasting(self)

### Function: test_where(self)

### Function: test_where_object(self)

### Function: test_where_with_masked_choice(self)

### Function: test_where_with_masked_condition(self)

### Function: test_where_type(self)

### Function: test_where_broadcast(self)

### Function: test_where_structured(self)

### Function: test_where_structured_masked(self)

### Function: test_masked_invalid_error(self)

### Function: test_masked_invalid_pandas(self)

### Function: test_masked_invalid_full_mask(self, copy)

### Function: test_choose(self)

### Function: test_choose_with_out(self)

### Function: test_reshape(self)

### Function: test_make_mask_descr(self)

### Function: test_make_mask(self)

### Function: test_mask_or(self)

### Function: test_allequal(self)

### Function: test_flatten_mask(self)

### Function: test_on_ndarray(self)

### Function: test_compress(self)

### Function: test_compressed(self)

### Function: test_convolve(self)

### Function: setup_method(self)

### Function: test_set_records_masks(self)

### Function: test_set_record_element(self)

### Function: test_set_record_slice(self)

### Function: test_mask_element(self)

**Description:** Check record access

### Function: test_getmaskarray(self)

### Function: test_view(self)

### Function: test_getitem(self)

### Function: test_setitem(self)

### Function: test_setitem_scalar(self)

### Function: test_element_len(self)

### Function: test_getitem(self)

### Function: test_nested_ma(self)

### Function: setup_method(self)

### Function: test_view_to_nothing(self)

### Function: test_view_to_type(self)

### Function: test_view_to_simple_dtype(self)

### Function: test_view_to_flexible_dtype(self)

### Function: test_view_to_subdtype(self)

### Function: test_view_to_dtype_and_type(self)

### Function: test_ndarrayfuncs(self)

### Function: test_count(self)

### Function: _do_add_test(self, add)

### Function: test_ufunc(self)

### Function: test_operator(self)

### Function: test_ctor(self)

### Function: test_repr(self)

### Function: test_pickle(self)

### Function: test_copy(self)

### Function: test__copy(self)

### Function: test_deepcopy(self)

### Function: test_immutable(self)

### Function: test_coercion_int(self)

### Function: test_coercion_float(self)

### Function: test_coercion_unicode(self)

### Function: test_coercion_bytes(self)

### Function: test_subclass(self)

### Function: test_attributes_readonly(self)

### Function: test_masked_values(self)

### Function: method(self)

**Description:** This docstring

Has multiple lines

And notes

Notes
-----
original note

## Class: NotBool

### Function: minmax_with_mask(arr, mask)

### Function: assign()

## Class: MyClass

## Class: MyClass2

## Class: Series

## Class: A

## Class: M

## Class: M

### Function: _test_index(i)

### Function: testaxis(f, a, d)

### Function: testkeepdims(f, a, d)

## Class: Sub

### Function: __bool__(self)

### Function: __mul__(self, other)

### Function: __rmul__(self, other)

### Function: __mul__(self, other)

### Function: __rmul__(self, other)

### Function: __rdiv__(self, other)

### Function: __array__(self, dtype, copy)

### Function: compressed(self)
