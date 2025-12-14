## AI Summary

A file named test_nditer.py.


### Function: iter_multi_index(i)

### Function: iter_indices(i)

### Function: iter_iterindices(i)

### Function: test_iter_refcount()

### Function: test_iter_best_order()

### Function: test_iter_c_order()

### Function: test_iter_f_order()

### Function: test_iter_c_or_f_order()

### Function: test_nditer_multi_index_set()

### Function: test_nditer_multi_index_set_refcount()

### Function: test_iter_best_order_multi_index_1d()

### Function: test_iter_best_order_multi_index_2d()

### Function: test_iter_best_order_multi_index_3d()

### Function: test_iter_best_order_c_index_1d()

### Function: test_iter_best_order_c_index_2d()

### Function: test_iter_best_order_c_index_3d()

### Function: test_iter_best_order_f_index_1d()

### Function: test_iter_best_order_f_index_2d()

### Function: test_iter_best_order_f_index_3d()

### Function: test_iter_no_inner_full_coalesce()

### Function: test_iter_no_inner_dim_coalescing()

### Function: test_iter_dim_coalescing()

### Function: test_iter_broadcasting()

### Function: test_iter_itershape()

### Function: test_iter_broadcasting_errors()

### Function: test_iter_flags_errors()

### Function: test_iter_slice()

### Function: test_iter_assign_mapping()

### Function: test_iter_nbo_align_contig()

### Function: test_iter_array_cast()

### Function: test_iter_array_cast_errors()

### Function: test_iter_scalar_cast()

### Function: test_iter_scalar_cast_errors()

### Function: test_iter_object_arrays_basic()

### Function: test_iter_object_arrays_conversions()

### Function: test_iter_common_dtype()

### Function: test_iter_copy_if_overlap()

### Function: test_iter_op_axes()

### Function: test_iter_op_axes_errors()

### Function: test_iter_copy()

### Function: test_iter_copy_casts(dtype, loop_dtype)

### Function: test_iter_copy_casts_structured()

### Function: test_iter_copy_casts_structured2()

### Function: test_iter_allocate_output_simple()

### Function: test_iter_allocate_output_buffered_readwrite()

### Function: test_iter_allocate_output_itorder()

### Function: test_iter_allocate_output_opaxes()

### Function: test_iter_allocate_output_types_promotion()

### Function: test_iter_allocate_output_types_byte_order()

### Function: test_iter_allocate_output_types_scalar()

### Function: test_iter_allocate_output_subtype()

### Function: test_iter_allocate_output_errors()

### Function: test_all_allocated()

### Function: test_iter_remove_axis()

### Function: test_iter_remove_multi_index_inner_loop()

### Function: test_iter_iterindex()

### Function: test_iter_iterrange()

### Function: test_iter_buffering()

### Function: test_iter_write_buffering()

### Function: test_iter_buffering_delayed_alloc()

### Function: test_iter_buffered_cast_simple()

### Function: test_iter_buffered_cast_byteswapped()

### Function: test_iter_buffered_cast_byteswapped_complex()

### Function: test_iter_buffered_cast_structured_type()

### Function: test_iter_buffered_cast_structured_type_failure_with_cleanup()

### Function: test_buffered_cast_error_paths()

### Function: test_buffered_cast_error_paths_unraisable()

### Function: test_iter_buffered_cast_subarray()

### Function: test_iter_buffering_badwriteback()

### Function: test_iter_buffering_string()

### Function: test_iter_buffering_growinner()

### Function: test_iter_buffered_reduce_reuse()

### Function: test_iter_no_broadcast()

## Class: TestIterNested

### Function: test_iter_reduction_error()

### Function: test_iter_reduction()

### Function: test_iter_buffering_reduction()

### Function: test_iter_buffering_reduction_reuse_reduce_loops()

### Function: test_iter_writemasked_badinput()

### Function: _is_buffered(iterator)

### Function: test_iter_writemasked(a)

### Function: test_iter_writemasked_broadcast_error(mask, mask_axes)

### Function: test_iter_writemasked_decref()

### Function: test_iter_non_writable_attribute_deletion()

### Function: test_iter_writable_attribute_deletion()

### Function: test_iter_element_deletion()

### Function: test_iter_allocated_array_dtypes()

### Function: test_0d_iter()

### Function: test_object_iter_cleanup()

### Function: test_object_iter_cleanup_reduce()

### Function: test_object_iter_cleanup_large_reduce(arr)

### Function: test_iter_too_large()

### Function: test_iter_too_large_with_multiindex()

### Function: test_writebacks()

### Function: test_close_equivalent()

**Description:** using a context amanger and using nditer.close are equivalent
    

### Function: test_close_raises()

### Function: test_close_parameters()

### Function: test_warn_noclose()

### Function: test_partial_iteration_cleanup(in_dtype, buf_dtype, steps)

**Description:** Checks for reference counting leaks during cleanup.  Using explicit
reference counts lead to occasional false positives (at least in parallel
test setups).  This test now should still test leaks correctly when
run e.g. with pytest-valgrind or pytest-leaks

### Function: test_partial_iteration_error(in_dtype, buf_dtype)

### Function: test_debug_print(capfd)

**Description:** Matches the expected output of a debug print with the actual output.
Note that the iterator dump should not be considered stable API,
this test is mainly to ensure the print does not crash.

Currently uses a subprocess to avoid dealing with the C level `printf`s.

### Function: assign_multi_index(i)

### Function: assign_index(i)

### Function: assign_iterindex(i)

### Function: assign_iterrange(i)

## Class: MyNDArray

### Function: get_array(i)

### Function: assign_iter(i)

### Function: get_params()

### Function: test_basic(self)

### Function: test_reorder(self)

### Function: test_flip_axes(self)

### Function: test_broadcast(self)

### Function: test_dtype_copy(self)

### Function: test_dtype_buffered(self)

### Function: test_0d(self)

### Function: test_iter_nested_iters_dtype_buffered(self)

## Class: T

### Function: add_close(x, y, out)

### Function: add_context(x, y, out)

### Function: __bool__(self)
