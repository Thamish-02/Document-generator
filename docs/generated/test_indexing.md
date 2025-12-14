## AI Summary

A file named test_indexing.py.


## Class: TestIndexing

## Class: TestFieldIndexing

## Class: TestBroadcastedAssignments

## Class: TestSubclasses

## Class: TestFancyIndexingCast

## Class: TestFancyIndexingEquivalence

## Class: TestMultiIndexingAutomated

**Description:** These tests use code to mimic the C-Code indexing for selection.

NOTE:

    * This still lacks tests for complex item setting.
    * If you change behavior of indexing, you might want to modify
      these tests to try more combinations.
    * Behavior was written to match numpy version 1.8. (though a
      first version matched 1.7.)
    * Only tuple indices are supported by the mimicking code.
      (and tested as of writing this)
    * Error types should match most of the time as long as there
      is only one error. For multiple errors, what gets raised
      will usually not be the same one. They are *not* tested.

Update 2016-11-30: It is probably not worth maintaining this test
indefinitely and it can be dropped if maintenance becomes a burden.

## Class: TestFloatNonIntegerArgument

**Description:** These test that ``TypeError`` is raised when you try to use
non-integers as arguments to for indexing and slicing e.g. ``a[0.0:5]``
and ``a[0.5]``, or other functions like ``array.reshape(1., -1)``.

## Class: TestBooleanIndexing

## Class: TestArrayToIndexDeprecation

**Description:** Creating an index from array not 0-D is an error.

    

## Class: TestNonIntegerArrayLike

**Description:** Tests that array_likes only valid if can safely cast to integer.

For instance, lists give IndexError when they cannot be safely cast to
an integer.

## Class: TestMultipleEllipsisError

**Description:** An index can only have a single ellipsis.

    

## Class: TestCApiAccess

### Function: test_index_no_floats(self)

### Function: test_slicing_no_floats(self)

### Function: test_index_no_array_to_index(self)

### Function: test_none_index(self)

### Function: test_empty_tuple_index(self)

### Function: test_void_scalar_empty_tuple(self)

### Function: test_same_kind_index_casting(self)

### Function: test_empty_fancy_index(self)

### Function: test_gh_26542(self)

### Function: test_gh_26542_2d(self)

### Function: test_gh_26542_index_overlap(self)

### Function: test_ellipsis_index(self)

### Function: test_single_int_index(self)

### Function: test_single_bool_index(self)

### Function: test_boolean_shape_mismatch(self)

### Function: test_boolean_indexing_onedim(self)

### Function: test_boolean_assignment_value_mismatch(self)

### Function: test_boolean_assignment_needs_api(self)

### Function: test_boolean_indexing_twodim(self)

### Function: test_boolean_indexing_list(self)

### Function: test_reverse_strides_and_subspace_bufferinit(self)

### Function: test_reversed_strides_result_allocation(self)

### Function: test_uncontiguous_subspace_assignment(self)

### Function: test_too_many_fancy_indices_special_case(self)

### Function: test_scalar_array_bool(self)

### Function: test_everything_returns_views(self)

### Function: test_broaderrors_indexing(self)

### Function: test_trivial_fancy_out_of_bounds(self)

### Function: test_trivial_fancy_not_possible(self)

### Function: test_nonbaseclass_values(self)

### Function: test_array_like_values(self)

### Function: test_subclass_writeable(self, writeable)

### Function: test_memory_order(self)

### Function: test_scalar_return_type(self)

### Function: test_small_regressions(self)

### Function: test_unaligned(self)

### Function: test_tuple_subclass(self)

### Function: test_broken_sequence_not_nd_index(self)

### Function: test_indexing_array_weird_strides(self)

### Function: test_indexing_array_negative_strides(self)

### Function: test_character_assignment(self)

### Function: test_too_many_advanced_indices(self, index, num, original_ndim)

### Function: test_structured_advanced_indexing(self)

### Function: test_nontuple_ndindex(self)

### Function: test_scalar_return_type(self)

### Function: assign(self, a, ind, val)

### Function: test_prepending_ones(self)

### Function: test_prepend_not_one(self)

### Function: test_simple_broadcasting_errors(self)

### Function: test_broadcast_error_reports_correct_shape(self, index)

### Function: test_index_is_larger(self)

### Function: test_broadcast_subspace(self)

### Function: test_basic(self)

### Function: test_fancy_on_read_only(self)

### Function: test_finalize_gets_full_info(self)

### Function: test_boolean_index_cast_assign(self)

### Function: test_object_assign(self)

### Function: test_cast_equivalence(self)

### Function: setup_method(self)

### Function: _get_multi_index(self, arr, indices)

**Description:** Mimic multi dimensional indexing.

Parameters
----------
arr : ndarray
    Array to be indexed.
indices : tuple of index objects

Returns
-------
out : ndarray
    An array equivalent to the indexing operation (but always a copy).
    `arr[indices]` should be identical.
no_copy : bool
    Whether the indexing operation requires a copy. If this is `True`,
    `np.may_share_memory(arr, arr[indices])` should be `True` (with
    some exceptions for scalars and possibly 0-d arrays).

Notes
-----
While the function may mostly match the errors of normal indexing this
is generally not the case.

### Function: _check_multi_index(self, arr, index)

**Description:** Check a multi index item getting and simple setting.

Parameters
----------
arr : ndarray
    Array to be indexed, must be a reshaped arange.
index : tuple of indexing objects
    Index being tested.

### Function: _check_single_index(self, arr, index)

**Description:** Check a single index item getting and simple setting.

Parameters
----------
arr : ndarray
    Array to be indexed, must be an arange.
index : indexing object
    Index being tested. Must be a single index and not a tuple
    of indexing objects (see also `_check_multi_index`).

### Function: _compare_index_result(self, arr, index, mimic_get, no_copy)

**Description:** Compare mimicked result to indexing result.
        

### Function: test_boolean(self)

### Function: test_multidim(self)

### Function: test_1d(self)

### Function: test_valid_indexing(self)

### Function: test_valid_slicing(self)

### Function: test_non_integer_argument_errors(self)

### Function: test_non_integer_sequence_multiplication(self)

### Function: test_reduce_axis_float_index(self)

### Function: test_bool_as_int_argument_errors(self)

### Function: test_boolean_indexing_weirdness(self)

### Function: test_boolean_indexing_fast_path(self)

### Function: test_array_to_index_error(self)

### Function: test_basic(self)

### Function: test_basic(self)

### Function: test_getitem(self)

### Function: test_setitem(self)

### Function: f(a, v)

## Class: SubClass

## Class: Zero

## Class: ArrayLike

## Class: TupleSubclass

## Class: SequenceLike

### Function: func(arr)

## Class: SubClass

## Class: SubClass

## Class: SubClass

### Function: mult(a, b)

### Function: __array_finalize__(self, old)

### Function: __index__(self)

### Function: __array__(self, dtype, copy)

### Function: __index__(self)

### Function: __len__(self)

### Function: __getitem__(self, item)

### Function: __array_finalize__(self, old)

### Function: isskip(idx)
