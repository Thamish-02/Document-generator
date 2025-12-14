## AI Summary

A file named test_multiarray.py.


### Function: assert_arg_sorted(arr, arg)

### Function: assert_arr_partitioned(kth, k, arr_part)

### Function: _aligned_zeros(shape, dtype, order, align)

**Description:** Allocate a new ndarray with aligned memory.

The ndarray is guaranteed *not* aligned to twice the requested alignment.
Eg, if align=4, guarantees it is not aligned to 8. If align=None uses
dtype.alignment.

## Class: TestFlags

## Class: TestHash

## Class: TestAttributes

## Class: TestArrayConstruction

## Class: TestAssignment

## Class: TestDtypedescr

## Class: TestZeroRank

## Class: TestScalarIndexing

## Class: TestCreation

**Description:** Test the np.array constructor

## Class: TestStructured

## Class: TestBool

## Class: TestZeroSizeFlexible

## Class: TestMethods

## Class: TestCequenceMethods

## Class: TestBinop

## Class: TestTemporaryElide

## Class: TestCAPI

## Class: TestSubscripting

## Class: TestPickling

## Class: TestFancyIndexing

## Class: TestStringCompare

## Class: TestArgmaxArgminCommon

## Class: TestArgmax

## Class: TestArgmin

## Class: TestMinMax

## Class: TestNewaxis

## Class: TestClip

## Class: TestCompress

## Class: TestPutmask

## Class: TestTake

## Class: TestLexsort

## Class: TestIO

**Description:** Test tofile, fromfile, tobytes, and fromstring

## Class: TestFromBuffer

## Class: TestFlat

## Class: TestResize

## Class: TestRecord

## Class: TestView

### Function: _mean(a)

### Function: _var(a)

### Function: _std(a)

## Class: TestStats

## Class: TestVdot

## Class: TestDot

## Class: MatmulCommon

**Description:** Common tests for '@' operator and numpy.matmul.

    

## Class: TestMatmul

## Class: TestMatmulOperator

## Class: TestMatmulInplace

### Function: test_matmul_axes()

## Class: TestInner

## Class: TestChoose

## Class: TestRepeat

## Class: TestNeighborhoodIter

## Class: TestStackedNeighborhoodIter

## Class: TestWarnings

## Class: TestMinScalarType

## Class: TestPEP3118Dtype

## Class: TestNewBufferProtocol

**Description:** Test PEP3118 buffers 

## Class: TestArrayCreationCopyArgument

## Class: TestArrayAttributeDeletion

## Class: TestArrayInterface

### Function: test_interface_no_shape()

### Function: test_array_interface_itemsize()

### Function: test_array_interface_empty_shape()

### Function: test_array_interface_offset()

### Function: test_array_interface_unicode_typestr()

### Function: test_flat_element_deletion()

### Function: test_scalar_element_deletion()

## Class: TestAsCArray

## Class: TestConversion

## Class: TestWhere

## Class: TestHashing

## Class: TestArrayPriority

## Class: TestBytestringArrayNonzero

## Class: TestUnicodeEncoding

**Description:** Tests for encoding related bugs, such as UCS2 vs UCS4, round-tripping
issues, etc

## Class: TestUnicodeArrayNonzero

## Class: TestFormat

## Class: TestCTypes

## Class: TestWritebackIfCopy

## Class: TestArange

## Class: TestArrayFinalize

**Description:** Tests __array_finalize__ 

### Function: test_orderconverter_with_nonASCII_unicode_ordering()

### Function: test_equal_override()

### Function: test_equal_subclass_no_override(op, dt1, dt2)

### Function: test_no_loop_gives_all_true_or_false(dt1, dt2)

### Function: test_comparisons_forwards_error(op)

### Function: test_richcompare_scalar_boolean_singleton_return()

### Function: test_ragged_comparison_fails(op)

### Function: test_npymath_complex(fun, npfun, x, y, test_dtype)

### Function: test_npymath_real()

### Function: test_uintalignment_and_alignment()

## Class: TestAlignment

### Function: test_getfield()

## Class: TestViewDtype

**Description:** Verify that making a view of a non-contiguous array works as expected.

### Function: test_sort_float(N, dtype)

### Function: test_sort_float16()

### Function: test_sort_int(N, dtype)

### Function: test_sort_uint()

### Function: test_private_get_ndarray_c_version()

### Function: test_argsort_float(N, dtype)

### Function: test_argsort_int(N, dtype)

### Function: test_gh_22683()

### Function: test_gh_24459()

### Function: test_partition_int(N, dtype)

### Function: test_partition_fp(N, dtype)

### Function: test_cannot_assign_data()

### Function: test_insufficient_width()

**Description:** If a 'width' parameter is passed into ``binary_repr`` that is insufficient
to represent the number in base 2 (positive) or 2's complement (negative)
form, the function used to silently ignore the parameter and return a
representation using the minimal number of bits needed for the form in
question. Such behavior is now considered unsafe from a user perspective
and will raise an error.

### Function: test_npy_char_raises()

## Class: TestDevice

**Description:** Test arr.device attribute and arr.to_device() method.

### Function: test_array_interface_excess_dimensions_raises()

**Description:** Regression test for gh-27949: ensure too many dims raises ValueError instead of segfault.

### Function: setup_method(self)

### Function: test_writeable(self)

### Function: test_writeable_any_base(self)

### Function: test_writeable_from_readonly(self)

### Function: test_writeable_from_buffer(self)

### Function: test_writeable_pickle(self)

### Function: test_writeable_from_c_data(self)

### Function: test_warnonwrite(self)

### Function: test_readonly_flag_protocols(self, flag, flag_value, writeable)

### Function: test_otherflags(self)

### Function: test_string_align(self)

### Function: test_void_align(self)

### Function: test_xcontiguous_load_txt(self, row_size, row_count, ndmin)

### Function: test_int(self)

### Function: setup_method(self)

### Function: test_attributes(self)

### Function: test_dtypeattr(self)

### Function: test_int_subclassing(self)

### Function: test_stridesattr(self)

### Function: test_set_stridesattr(self)

### Function: test_fill(self)

### Function: test_fill_max_uint64(self)

### Function: test_fill_struct_array(self)

### Function: test_fill_readonly(self)

### Function: test_fill_subarrays(self)

### Function: test_array(self)

### Function: test_array_empty(self)

### Function: test_0d_array_shape(self)

### Function: test_array_copy_false(self)

### Function: test_array_copy_if_needed(self)

### Function: test_array_copy_true(self)

### Function: test_array_copy_str(self)

### Function: test_array_cont(self)

### Function: test_bad_arguments_error(self, func)

### Function: test_array_as_keyword(self, func)

### Function: test_assignment_broadcasting(self)

### Function: test_assignment_errors(self)

### Function: test_unicode_assignment(self)

### Function: test_stringlike_empty_list(self)

### Function: test_longdouble_assignment(self)

### Function: test_cast_to_string(self)

### Function: test_construction(self)

### Function: test_byteorders(self)

### Function: test_structured_non_void(self)

### Function: setup_method(self)

### Function: test_ellipsis_subscript(self)

### Function: test_empty_subscript(self)

### Function: test_invalid_subscript(self)

### Function: test_ellipsis_subscript_assignment(self)

### Function: test_empty_subscript_assignment(self)

### Function: test_invalid_subscript_assignment(self)

### Function: test_newaxis(self)

### Function: test_invalid_newaxis(self)

### Function: test_constructor(self)

### Function: test_output(self)

### Function: test_real_imag(self)

### Function: setup_method(self)

### Function: test_ellipsis_subscript(self)

### Function: test_empty_subscript(self)

### Function: test_invalid_subscript(self)

### Function: test_invalid_subscript_assignment(self)

### Function: test_newaxis(self)

### Function: test_invalid_newaxis(self)

### Function: test_overlapping_assignment(self)

### Function: test_from_attribute(self)

### Function: test_from_string(self)

### Function: test_void(self)

### Function: test_structured_void_promotion(self, idx)

### Function: test_too_big_error(self)

### Function: test_malloc_fails(self)

### Function: test_zeros(self)

### Function: test_zeros_big(self)

### Function: test_zeros_obj(self)

### Function: test_zeros_obj_obj(self)

### Function: test_zeros_like_like_zeros(self)

### Function: test_empty_unicode(self)

### Function: test_sequence_non_homogeneous(self)

### Function: test_non_sequence_sequence(self)

**Description:** Should not segfault.

Class Fail breaks the sequence protocol for new style classes, i.e.,
those derived from object. Class Map is a mapping type indicated by
raising a ValueError. At some point we may raise a warning instead
of an error in the Fail case.

### Function: test_no_len_object_type(self)

### Function: test_false_len_sequence(self)

### Function: test_false_len_iterable(self)

### Function: test_failed_len_sequence(self)

### Function: test_array_too_big(self)

### Function: _ragged_creation(self, seq)

### Function: test_ragged_ndim_object(self)

### Function: test_ragged_shape_object(self)

### Function: test_array_of_ragged_array(self)

### Function: test_deep_nonragged_object(self)

### Function: test_object_initialized_to_None(self, function, dtype)

### Function: test_creation_from_dtypemeta(self, func)

### Function: test_subarray_field_access(self)

### Function: test_subarray_comparison(self)

### Function: test_empty_structured_array_comparison(self)

### Function: test_structured_array_comparison_bad_broadcasts(self, op)

### Function: test_structured_comparisons_with_promotion(self)

### Function: test_void_comparison_failures(self, op)

### Function: test_casting(self)

### Function: test_objview(self)

### Function: test_setfield(self)

### Function: test_setfield_object(self)

### Function: test_zero_width_string(self)

### Function: test_base_attr(self)

### Function: test_assignment(self)

### Function: test_structuredscalar_indexing(self)

### Function: test_multiindex_titles(self)

### Function: test_structured_cast_promotion_fieldorder(self)

### Function: test_structured_promotion_packs(self, dtype_dict, align)

### Function: test_structured_asarray_is_view(self)

### Function: test_test_interning(self)

### Function: test_sum(self)

### Function: check_count_nonzero(self, power, length)

### Function: test_count_nonzero(self)

### Function: test_count_nonzero_all(self)

### Function: test_count_nonzero_unaligned(self)

### Function: _test_cast_from_flexible(self, dtype)

### Function: test_cast_from_void(self)

### Function: test_cast_from_unicode(self)

### Function: test_cast_from_bytes(self)

### Function: _zeros(shape, dtype)

### Function: test_create(self)

### Function: _test_sort_partition(self, name, kinds)

### Function: test_sort(self)

### Function: test_argsort(self)

### Function: test_partition(self)

### Function: test_argpartition(self)

### Function: test_resize(self)

### Function: test_view(self)

### Function: test_dumps(self)

### Function: test_pickle(self)

### Function: test_pickle_empty(self)

**Description:** Checking if an empty array pickled and un-pickled will not cause a
segmentation fault

### Function: test_pickle_with_buffercallback(self)

### Function: test_all_where(self)

### Function: test_any_where(self)

### Function: test_any_and_all_result_dtype(self, dtype)

### Function: test_any_and_all_object_dtype(self)

### Function: test_compress(self)

### Function: test_choose(self)

### Function: test_prod(self)

### Function: test_repeat(self, dtype)

### Function: test_reshape(self)

### Function: test_round(self)

### Function: test_squeeze(self)

### Function: test_transpose(self)

### Function: test_sort(self)

### Function: test_sort_unsigned(self, dtype)

### Function: test_sort_signed(self, dtype)

### Function: test_sort_complex(self, part, dtype)

### Function: test_sort_complex_byte_swapping(self)

### Function: test_sort_string(self, dtype)

### Function: test_sort_object(self)

### Function: test_sort_structured(self, dt, step)

### Function: test_sort_time(self, dtype)

### Function: test_sort_axis(self)

### Function: test_sort_size_0(self)

### Function: test_sort_bad_ordering(self)

### Function: test_void_sort(self)

### Function: test_sort_raises(self)

### Function: test_sort_degraded(self)

### Function: test_copy(self)

### Function: test__deepcopy__(self, dtype)

### Function: test__deepcopy__catches_failure(self)

### Function: test_sort_order(self)

### Function: test_argsort(self)

### Function: test_sort_unicode_kind(self)

### Function: test_searchsorted_floats(self, a)

### Function: test_searchsorted_complex(self)

### Function: test_searchsorted_n_elements(self)

### Function: test_searchsorted_unaligned_array(self)

### Function: test_searchsorted_resetting(self)

### Function: test_searchsorted_type_specific(self)

### Function: test_searchsorted_unicode(self)

### Function: test_searchsorted_with_invalid_sorter(self)

### Function: test_searchsorted_with_sorter(self)

### Function: test_searchsorted_return_type(self)

### Function: test_argpartition_out_of_range(self, dtype)

### Function: test_partition_out_of_range(self, dtype)

### Function: test_argpartition_integer(self)

### Function: test_partition_integer(self)

### Function: test_partition_empty_array(self, kth_dtype)

### Function: test_argpartition_empty_array(self, kth_dtype)

### Function: test_partition(self)

### Function: assert_partitioned(self, d, kth)

### Function: test_partition_iterative(self)

### Function: test_partition_cdtype(self)

### Function: test_partition_unicode_kind(self)

### Function: test_partition_fuzz(self)

### Function: test_argpartition_gh5524(self, kth_dtype)

### Function: test_flatten(self)

### Function: test_arr_mult(self, func)

### Function: test_no_dgemv(self, func, dtype)

### Function: test_dot(self)

### Function: test_dot_type_mismatch(self)

### Function: test_dot_out_mem_overlap(self)

### Function: test_dot_matmul_out(self)

### Function: test_dot_matmul_inner_array_casting_fails(self)

### Function: test_matmul_out(self)

### Function: test_diagonal(self)

### Function: test_diagonal_view_notwriteable(self)

### Function: test_diagonal_memleak(self)

### Function: test_size_zero_memleak(self)

### Function: test_trace(self)

### Function: test_trace_subclass(self)

### Function: test_put(self)

### Function: test_ravel(self)

### Function: test_ravel_subclass(self)

### Function: test_swapaxes(self)

### Function: test_conjugate(self)

### Function: test_conjugate_out(self)

### Function: test__complex__(self)

### Function: test__complex__should_not_work(self)

### Function: test_array_contains(self)

### Function: test_inplace(self)

### Function: test_ufunc_binop_interaction(self)

### Function: test_ufunc_binop_bad_array_priority(self, priority)

### Function: test_scalar_binop_guarantees_ufunc(self, scalar, op)

### Function: test_ufunc_override_normalize_signature(self)

### Function: test_array_ufunc_index(self)

### Function: test_out_override(self)

### Function: test_pow_override_with_errors(self)

### Function: test_pow_array_object_dtype(self)

### Function: test_pos_array_ufunc_override(self)

### Function: test_extension_incref_elide(self)

### Function: test_extension_incref_elide_stack(self)

### Function: test_temporary_with_cast(self)

### Function: test_elide_broadcast(self)

### Function: test_elide_scalar(self)

### Function: test_elide_scalar_readonly(self)

### Function: test_elide_readonly(self)

### Function: test_elide_updateifcopy(self)

### Function: test_IsPythonScalar(self)

### Function: test_intp_sequence_converters(self, converter)

### Function: test_intp_sequence_converters_errors(self, converter)

### Function: test_test_zero_rank(self)

### Function: test_correct_protocol5_error_message(self)

### Function: test_record_array_with_object_dtype(self)

### Function: test_f_contiguous_array(self)

### Function: test_non_contiguous_array(self)

### Function: test_roundtrip(self)

### Function: _loads(self, obj)

### Function: test_version0_int8(self)

### Function: test_version0_float32(self)

### Function: test_version0_object(self)

### Function: test_version1_int8(self)

### Function: test_version1_float32(self)

### Function: test_version1_object(self)

### Function: test_subarray_int_shape(self)

### Function: test_datetime64_byteorder(self)

### Function: test_list(self)

### Function: test_tuple(self)

### Function: test_mask(self)

### Function: test_mask2(self)

### Function: test_assign_mask(self)

### Function: test_assign_mask2(self)

### Function: test_string(self)

### Function: test_mixed(self)

### Function: test_unicode(self)

### Function: test_np_argmin_argmax_keepdims(self, size, axis, method)

### Function: test_all(self, method)

### Function: test_output_shape(self, method)

### Function: test_ret_is_out(self, ndim, method)

### Function: test_unicode(self, np_array, method, idx, val)

### Function: test_np_vs_ndarray(self, arr_method, np_method)

### Function: test_object_with_NULLs(self, method, vals)

### Function: test_combinations(self, data)

### Function: test_maximum_signed_integers(self)

### Function: test_combinations(self, data)

### Function: test_minimum_signed_integers(self)

### Function: test_scalar(self)

### Function: test_axis(self)

### Function: test_datetime(self)

### Function: test_basic(self)

### Function: _check_range(self, x, cmin, cmax)

### Function: _clip_type(self, type_group, array_max, clip_min, clip_max, inplace, expected_min, expected_max)

### Function: test_basic(self)

### Function: test_int_out_of_range(self, inplace)

### Function: test_record_array(self)

### Function: test_max_or_min(self)

### Function: test_nan(self)

### Function: test_axis(self)

### Function: test_truncate(self)

### Function: test_flatten(self)

### Function: tst_basic(self, x, T, mask, val)

### Function: test_ip_types(self)

### Function: test_mask_size(self)

### Function: test_byteorder(self, dtype)

### Function: test_record_array(self)

### Function: test_overlaps(self)

### Function: test_writeable(self)

### Function: test_kwargs(self)

### Function: tst_basic(self, x)

### Function: test_ip_types(self)

### Function: test_raise(self)

### Function: test_clip(self)

### Function: test_wrap(self)

### Function: test_byteorder(self, dtype)

### Function: test_record_array(self)

### Function: test_out_overlap(self)

### Function: test_ret_is_out(self, shape)

### Function: test_basic(self, dtype)

### Function: test_mixed(self)

### Function: test_datetime(self)

### Function: test_object(self)

### Function: test_strings(self)

### Function: test_invalid_axis(self)

### Function: x(self)

### Function: tmp_filename(self, tmp_path, request)

### Function: test_nofile(self)

### Function: test_bool_fromstring(self)

### Function: test_uint64_fromstring(self)

### Function: test_int64_fromstring(self)

### Function: test_fromstring_count0(self)

### Function: test_empty_files_text(self, tmp_filename)

### Function: test_empty_files_binary(self, tmp_filename)

### Function: test_roundtrip_file(self, x, tmp_filename)

### Function: test_roundtrip(self, x, tmp_filename)

### Function: test_roundtrip_dump_pathlib(self, x, tmp_filename)

### Function: test_roundtrip_binary_str(self, x)

### Function: test_roundtrip_str(self, x)

### Function: test_roundtrip_repr(self, x)

### Function: test_unseekable_fromfile(self, x, tmp_filename)

### Function: test_io_open_unbuffered_fromfile(self, x, tmp_filename)

### Function: test_largish_file(self, tmp_filename)

### Function: test_io_open_buffered_fromfile(self, x, tmp_filename)

### Function: test_file_position_after_fromfile(self, tmp_filename)

### Function: test_file_position_after_tofile(self, tmp_filename)

### Function: test_load_object_array_fromfile(self, tmp_filename)

### Function: test_fromfile_offset(self, x, tmp_filename)

### Function: test_fromfile_bad_dup(self, x, tmp_filename)

### Function: _check_from(self, s, value, filename)

### Function: decimal_sep_localization(self, request)

**Description:** Including this fixture in a test will automatically
execute it with both types of decimal separator.

So::

    def test_decimal(decimal_sep_localization):
        pass

is equivalent to the following two tests::

    def test_decimal_period_separator():
        pass

    def test_decimal_comma_separator():
        with CommaDecimalPointLocale():
            pass

### Function: test_nan(self, tmp_filename, decimal_sep_localization)

### Function: test_inf(self, tmp_filename, decimal_sep_localization)

### Function: test_numbers(self, tmp_filename, decimal_sep_localization)

### Function: test_binary(self, tmp_filename)

### Function: test_string(self, tmp_filename)

### Function: test_counted_string(self, tmp_filename, decimal_sep_localization)

### Function: test_string_with_ws(self, tmp_filename)

### Function: test_counted_string_with_ws(self, tmp_filename)

### Function: test_ascii(self, tmp_filename, decimal_sep_localization)

### Function: test_malformed(self, tmp_filename, decimal_sep_localization)

### Function: test_long_sep(self, tmp_filename)

### Function: test_dtype(self, tmp_filename)

### Function: test_dtype_bool(self, tmp_filename)

### Function: test_tofile_sep(self, tmp_filename, decimal_sep_localization)

### Function: test_tofile_format(self, tmp_filename, decimal_sep_localization)

### Function: test_tofile_cleanup(self, tmp_filename)

### Function: test_fromfile_subarray_binary(self, tmp_filename)

### Function: test_parsing_subarray_unsupported(self, tmp_filename)

### Function: test_read_shorter_than_count_subarray(self, tmp_filename)

### Function: test_basic(self, byteorder, dtype)

### Function: test_array_base(self, obj)

### Function: test_empty(self)

### Function: test_mmap_close(self)

### Function: setup_method(self)

### Function: test_contiguous(self)

### Function: test_discontiguous(self)

### Function: test___array__(self)

### Function: test_refcount(self)

### Function: test_index_getset(self)

### Function: test_maxdims(self)

### Function: test_basic(self)

### Function: test_check_reference(self)

### Function: test_int_shape(self)

### Function: test_none_shape(self)

### Function: test_0d_shape(self)

### Function: test_invalid_arguments(self)

### Function: test_freeform_shape(self)

### Function: test_zeros_appended(self)

### Function: test_obj_obj(self)

### Function: test_empty_view(self)

### Function: test_check_weakref(self)

### Function: test_field_rename(self)

### Function: test_multiple_field_name_occurrence(self)

### Function: test_bytes_fields(self)

### Function: test_multiple_field_name_unicode(self)

### Function: test_fromarrays_unicode(self)

### Function: test_unicode_order(self)

### Function: test_field_names(self)

### Function: test_record_hash(self)

### Function: test_record_no_hash(self)

### Function: test_empty_structure_creation(self)

### Function: test_multifield_indexing_view(self)

### Function: test_basic(self)

### Function: setup_method(self)

### Function: test_python_type(self)

### Function: test_keepdims(self)

### Function: test_out(self)

### Function: test_dtype_from_input(self)

### Function: test_dtype_from_dtype(self)

### Function: test_ddof(self)

### Function: test_ddof_too_big(self)

### Function: test_empty(self)

### Function: test_mean_values(self)

### Function: test_mean_float16(self)

### Function: test_mean_axis_error(self)

### Function: test_mean_where(self)

### Function: test_var_values(self)

### Function: test_var_complex_values(self, complex_dtype, ndec)

### Function: test_var_dimensions(self)

### Function: test_var_complex_byteorder(self)

### Function: test_var_axis_error(self)

### Function: test_var_where(self)

### Function: test_std_values(self)

### Function: test_std_where(self)

### Function: test_subclass(self)

### Function: test_basic(self)

### Function: test_vdot_array_order(self)

### Function: test_vdot_uncontiguous(self)

### Function: setup_method(self)

### Function: test_dotmatmat(self)

### Function: test_dotmatvec(self)

### Function: test_dotmatvec2(self)

### Function: test_dotvecmat(self)

### Function: test_dotvecmat2(self)

### Function: test_dotvecmat3(self)

### Function: test_dotvecvecouter(self)

### Function: test_dotvecvecinner(self)

### Function: test_dotcolumnvect1(self)

### Function: test_dotcolumnvect2(self)

### Function: test_dotvecscalar(self)

### Function: test_dotvecscalar2(self)

### Function: test_all(self)

### Function: test_vecobject(self)

### Function: test_dot_2args(self)

### Function: test_dot_3args(self)

### Function: test_dot_3args_errors(self)

### Function: test_dot_out_result(self)

### Function: test_dot_out_aliasing(self)

### Function: test_dot_array_order(self)

### Function: test_accelerate_framework_sgemv_fix(self)

### Function: test_huge_vectordot(self, dtype)

### Function: test_dtype_discovery_fails(self)

### Function: test_exceptions(self)

### Function: test_shapes(self)

### Function: test_result_types(self)

### Function: test_scalar_output(self)

### Function: test_vector_vector_values(self)

### Function: test_vector_matrix_values(self)

### Function: test_matrix_vector_values(self)

### Function: test_matrix_matrix_values(self)

### Function: test_out_arg(self)

### Function: test_empty_out(self)

### Function: test_out_contiguous(self)

### Function: test_dot_equivalent(self, args)

### Function: test_matmul_object(self)

### Function: test_matmul_object_type_scalar(self)

### Function: test_matmul_empty(self)

### Function: test_matmul_exception_multiply(self)

### Function: test_matmul_exception_add(self)

### Function: test_matmul_bool(self)

### Function: test_array_priority_override(self)

### Function: test_matmul_raises(self)

### Function: test_basic(self, dtype1, dtype2)

### Function: test_shapes(self, a_shape, b_shape)

### Function: test_inner_type_mismatch(self)

### Function: test_inner_scalar_and_vector(self)

### Function: test_vecself(self)

### Function: test_inner_product_with_various_contiguities(self)

### Function: test_3d_tensor(self)

### Function: setup_method(self)

### Function: test_basic(self)

### Function: test_broadcast1(self)

### Function: test_broadcast2(self)

### Function: test_output_dtype(self, ops)

### Function: test_dimension_and_args_limit(self)

### Function: setup_method(self)

### Function: test_basic(self)

### Function: test_broadcast1(self)

### Function: test_axis_spec(self)

### Function: test_broadcast2(self)

### Function: test_simple2d(self, dt)

### Function: test_mirror2d(self, dt)

### Function: test_simple(self, dt)

### Function: test_mirror(self, dt)

### Function: test_circular(self, dt)

### Function: test_simple_const(self)

### Function: test_simple_mirror(self)

### Function: test_simple_circular(self)

### Function: test_simple_strict_within(self)

### Function: test_complex_warning(self)

### Function: test_usigned_shortshort(self)

### Function: test_usigned_short(self)

### Function: test_usigned_int(self)

### Function: test_usigned_longlong(self)

### Function: test_object(self)

### Function: _check(self, spec, wanted)

### Function: test_native_padding(self)

### Function: test_native_padding_2(self)

### Function: test_trailing_padding(self)

### Function: test_native_padding_3(self)

### Function: test_padding_with_array_inside_struct(self)

### Function: test_byteorder_inside_struct(self)

### Function: test_intra_padding(self)

### Function: test_char_vs_string(self)

### Function: test_field_order(self)

### Function: test_unnamed_fields(self)

### Function: _check_roundtrip(self, obj)

### Function: test_roundtrip(self)

### Function: test_roundtrip_half(self)

### Function: test_roundtrip_single_types(self)

### Function: test_roundtrip_scalar(self)

### Function: test_invalid_buffer_format(self)

### Function: test_export_simple_1d(self)

### Function: test_export_simple_nd(self)

### Function: test_export_discontiguous(self)

### Function: test_export_record(self)

### Function: test_export_subarray(self)

### Function: test_export_endian(self)

### Function: test_export_flags(self)

### Function: test_export_and_pickle_user_dtype(self, obj, error)

### Function: test_padding(self)

### Function: test_reference_leak(self)

### Function: test_padded_struct_array(self)

### Function: test_relaxed_strides(self, c)

### Function: test_out_of_order_fields(self)

### Function: test_max_dims(self)

### Function: test_error_pointer_type(self)

### Function: test_error_message_unsupported(self)

### Function: test_ctypes_integer_via_memoryview(self)

### Function: test_ctypes_struct_via_memoryview(self)

### Function: test_error_if_stored_buffer_info_is_corrupted(self, obj)

**Description:** If a user extends a NumPy array before 1.20 and then runs it
on NumPy 1.20+. A C-subclassed array might in theory modify
the new buffer-info field. This checks that an error is raised
if this happens (for buffer export), an error is written on delete.
This is a sanity check to help users transition to safe code, it
may be deleted at any point.

### Function: test_no_suboffsets(self)

## Class: RaiseOnBool

### Function: test_scalars(self)

### Function: test_compatible_cast(self)

### Function: test_buffer_interface(self)

### Function: test_array_interfaces(self)

### Function: test___array__(self)

### Function: test___array__copy_arg(self)

### Function: test___array__copy_once(self)

### Function: test__array__reference_leak(self)

### Function: test_order_mismatch(self, arr, order1, order2)

### Function: test_striding_not_ok(self)

### Function: test_multiarray_writable_attributes_deletion(self)

### Function: test_multiarray_not_writable_attributes_deletion(self)

### Function: test_multiarray_flags_writable_attribute_deletion(self)

### Function: test_multiarray_flags_not_writable_attribute_deletion(self)

## Class: Foo

### Function: test_scalar_interface(self, val, iface, expected)

## Class: ArrayLike

## Class: DummyArray1

## Class: DummyArray2

## Class: DummyArray

## Class: DummyArray

### Function: test_1darray(self)

### Function: test_2darray(self)

### Function: test_3darray(self)

### Function: test_array_scalar_relational_operation(self)

### Function: test_to_bool_scalar(self)

### Function: test_to_bool_scalar_not_convertible(self)

### Function: test_to_bool_scalar_size_errors(self)

### Function: test_to_int_scalar(self)

### Function: test_basic(self)

### Function: test_exotic(self)

### Function: test_ndim(self)

### Function: test_dtype_mix(self)

### Function: test_foreign(self)

### Function: test_error(self)

### Function: test_string(self)

### Function: test_empty_result(self)

### Function: test_largedim(self)

### Function: test_kwargs(self)

## Class: TestSizeOf

### Function: test_arrays_not_hashable(self)

### Function: test_collections_hashable(self)

## Class: Foo

## Class: Bar

## Class: Other

### Function: test_ndarray_subclass(self)

### Function: test_ndarray_other(self)

### Function: test_subclass_subclass(self)

### Function: test_subclass_other(self)

### Function: test_empty_bstring_array_is_falsey(self)

### Function: test_whitespace_bstring_array_is_truthy(self)

### Function: test_all_null_bstring_array_is_falsey(self)

### Function: test_null_inside_bstring_array_is_truthy(self)

### Function: test_round_trip(self)

**Description:** Tests that GETITEM, SETITEM, and PyArray_Scalar roundtrip 

### Function: test_assign_scalar(self)

### Function: test_fill_scalar(self)

### Function: test_empty_ustring_array_is_falsey(self)

### Function: test_whitespace_ustring_array_is_truthy(self)

### Function: test_all_null_ustring_array_is_falsey(self)

### Function: test_null_inside_ustring_array_is_truthy(self)

### Function: test_0d(self)

### Function: test_1d_no_format(self)

### Function: test_1d_format(self)

### Function: test_ctypes_is_available(self)

### Function: test_ctypes_is_not_available(self)

### Function: _make_readonly(x)

### Function: test_ctypes_data_as_holds_reference(self, arr)

### Function: test_ctypes_as_parameter_holds_reference(self)

### Function: test_argmax_with_out(self)

### Function: test_argmin_with_out(self)

### Function: test_insert_noncontiguous(self)

### Function: test_put_noncontiguous(self)

### Function: test_putmask_noncontiguous(self)

### Function: test_take_mode_raise(self)

### Function: test_choose_mod_raise(self)

### Function: test_flatiter__array__(self)

### Function: test_dot_out(self)

### Function: test_view_assign(self)

### Function: test_dealloc_warning(self)

### Function: test_view_discard_refcount(self)

### Function: test_infinite(self)

### Function: test_nan_step(self)

### Function: test_zero_step(self)

### Function: test_require_range(self)

### Function: test_start_stop_kwarg(self)

### Function: test_arange_booleans(self)

### Function: test_rejects_bad_dtypes(self, dtype)

### Function: test_rejects_strings(self)

### Function: test_byteswapped(self)

### Function: test_error_paths_and_promotion(self, which)

### Function: test_receives_base(self)

### Function: test_bad_finalize1(self)

### Function: test_bad_finalize2(self)

### Function: test_bad_finalize3(self)

### Function: test_lifetime_on_error(self)

### Function: test_can_use_super(self)

## Class: MyAlwaysEqual

## Class: MyAlwaysEqualOld

## Class: MyAlwaysEqualNew

## Class: MyArr

## Class: NotArray

### Function: check(self, shape, dtype, order, align)

### Function: test_various_alignments(self)

### Function: test_strided_loop_alignments(self)

### Function: test_smaller_dtype_multiple(self)

### Function: test_smaller_dtype_not_multiple(self)

### Function: test_larger_dtype_multiple(self)

### Function: test_larger_dtype_not_multiple(self)

### Function: test_f_contiguous(self)

### Function: test_non_c_contiguous(self)

### Function: test_device(self, func, arg)

### Function: test_to_device(self)

## Class: DummyArray

## Class: subclass

## Class: frominterface

## Class: MyArr

### Function: make_array(size, offset, strides)

### Function: make_array(size, offset, strides)

### Function: set_strides(arr, strides)

### Function: assign(a, b)

## Class: C

### Function: assign(v)

### Function: inject_str(s)

**Description:** replace ndarray.__str__ temporarily 

## Class: bad_sequence

### Function: assign(x, i, v)

### Function: subscript(x, i)

### Function: assign(x, i, v)

### Function: subscript(x, i)

## Class: x

## Class: Fail

## Class: Map

## Class: Point2

## Class: C

## Class: C

## Class: A

### Function: testassign()

### Function: testassign(arr, v)

### Function: check_round(arr, expected)

## Class: Boom

## Class: Raiser

### Function: assert_fortran(arr)

### Function: assert_c(arr)

## Class: MyObj

## Class: A

## Class: Sub

## Class: A

## Class: MyArray

## Class: ArraySubclass

## Class: Coerced

### Function: array_impl(self)

### Function: op_impl(self, other)

### Function: rop_impl(self, other)

### Function: iop_impl(self, other)

### Function: array_ufunc_impl(self, ufunc, method)

### Function: make_obj(base, array_priority, array_ufunc, alleged_module)

### Function: check(obj, binop_override_expected, ufunc_override_expected, inplace_override_expected, check_scalar)

## Class: BadPriority

## Class: LowPriority

## Class: SomeClass

## Class: SomeClass

## Class: CheckIndex

## Class: OutClass

## Class: PowerOnly

## Class: SomeClass

### Function: pow_for(exp, arr)

## Class: A

### Function: fail()

### Function: dup_str(fd)

### Function: dup_bigint(fd)

### Function: test_dtype_init()

### Function: test_dtype_unicode()

## Class: TestArray

## Class: Vec

### Function: aligned_array(shape, align, dtype, order)

### Function: as_aligned(arr, align, dtype, order)

### Function: assert_dot_close(A, X, desired)

## Class: BadObject

### Function: random_ints()

## Class: add_not_multiply

## Class: multiply_not_add

## Class: A

### Function: aligned(n)

### Function: aligned(n)

## Class: foo

### Function: __bool__(self)

### Function: int_types(byteswap)

## Class: ArrayLike

## Class: ArrayLike

## Class: ArrayLikeNoCopy

## Class: ArrayRandom

## Class: NotAnArray

### Function: __init__(self, value)

### Function: __float__(self)

### Function: __array_interface__(self)

## Class: NotConvertible

### Function: test_empty_array(self)

### Function: check_array(self, dtype)

### Function: test_array_int32(self)

### Function: test_array_int64(self)

### Function: test_array_float32(self)

### Function: test_array_float64(self)

### Function: test_view(self)

### Function: test_reshape(self)

### Function: test_resize(self)

### Function: test_resize_structured(self, dtype)

### Function: test_error(self)

### Function: __new__(cls)

### Function: __new__(cls)

### Function: _all(self, other)

## Class: SavesBase

## Class: BadAttributeArray

## Class: BadAttributeArray

## Class: BadAttributeArray

## Class: RaisesInFinalize

## Class: Dummy

## Class: SuperFinalize

### Function: __eq__(self, other)

### Function: __ne__(self, other)

### Function: __array_wrap__(self, new, context, return_scalar)

### Function: __array__(self, dtype, copy)

### Function: __init__(self, interface)

### Function: __init__(self, arr)

### Function: __getitem__(self)

### Function: __len__(self)

### Function: __array__(self, dtype, copy)

### Function: __len__(self)

### Function: __getitem__(self, index)

### Function: __len__(self)

### Function: __getitem__(self, index)

### Function: __init__(self)

### Function: __getitem__(self, ind)

### Function: __getitem__(self, i)

### Function: __len__(self)

### Function: __getitem__(self, x)

### Function: __iter__(self)

### Function: __len__(self)

### Function: __init__(self, data)

### Function: __getitem__(self, item)

### Function: __len__(self)

### Function: __lt__(self, other)

### Function: raises_anything()

### Function: __deepcopy__(self)

### Function: __array__(self)

### Function: __array_priority__(self)

### Function: __radd__(self, other)

### Function: __array_ufunc__(self, ufunc, method)

### Function: __array_ufunc__(self, ufunc, method)

### Function: __array_ufunc__(self, ufunc, method)

### Function: __array_ufunc__(self, ufunc, method)

### Function: __array_ufunc__(self, ufunc, method)

### Function: __init__(self, num)

### Function: __mul__(self, other)

### Function: __div__(self, other)

### Function: __pow__(self, exp)

### Function: __eq__(self, other)

### Function: __array_ufunc__(self, ufunc, method)

### Function: __new__(cls, data, info)

### Function: __array_finalize__(self, obj)

### Function: __init__(self, sequence)

### Function: __add__(self, other)

### Function: __sub__(self, other)

### Function: __mul__(self, other)

### Function: __rmul__(self, other)

### Function: __array__(self, dtype, copy)

### Function: __add__(self, other)

### Function: __mul__(self, other)

### Function: __matmul__(self, other)

### Function: __rmatmul__(self, other)

### Function: __array__(self, dtype, copy)

### Function: __array__(self, dtype)

### Function: __init__(self)

### Function: __array__(self, dtype, copy)

### Function: __array__(self, dtype, copy)

### Function: __bool__(self)

## Class: NotConvertible

### Function: __array_finalize__(self, obj)

### Function: __array_finalize__(self)

### Function: __array_finalize__(self)

### Function: __array_finalize__(self, obj)

### Function: __array_finalize__(self, obj)

### Function: __array_finalize__(self, obj)

## Class: HasTrunc

### Function: __int__(self)

### Function: first_out_arg(result)

### Function: __trunc__(self)
