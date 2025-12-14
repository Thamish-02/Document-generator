## AI Summary

A file named test_dtype.py.


### Function: assert_dtype_equal(a, b)

### Function: assert_dtype_not_equal(a, b)

## Class: TestBuiltin

## Class: TestRecord

## Class: TestSubarray

### Function: iter_struct_object_dtypes()

**Description:** Iterates over a few complex dtypes and object pattern which
fill the array with a given object (defaults to a singleton).

Yields
------
dtype : dtype
pattern : tuple
    Structured tuple for use with `np.array`.
count : int
    Number of objects stored in the dtype.
singleton : object
    A singleton object. The returned pattern is constructed so that
    all objects inside the datatype are set to the singleton.

## Class: TestStructuredObjectRefcounting

**Description:** These tests cover various uses of complicated structured types which
include objects and thus require reference counting.

## Class: TestStructuredDtypeSparseFields

**Description:** Tests subarray fields which contain sparse dtypes so that
not all memory is used by the dtype work. Such dtype's should
leave the underlying memory unchanged.

## Class: TestMonsterType

**Description:** Test deeply nested subtypes.

## Class: TestMetadata

## Class: TestString

## Class: TestDtypeAttributeDeletion

## Class: TestDtypeAttributes

## Class: TestDTypeMakeCanonical

## Class: TestPickling

## Class: TestPromotion

**Description:** Test cases related to more complex DType promotions.  Further promotion
tests are defined in `test_numeric.py`

### Function: test_rational_dtype()

### Function: test_dtypes_are_true()

### Function: test_invalid_dtype_string()

### Function: test_keyword_argument()

## Class: TestFromDTypeAttribute

## Class: TestDTypeClasses

## Class: TestFromCTypes

## Class: TestUserDType

## Class: TestClassGetItem

### Function: test_result_type_integers_and_unitless_timedelta64()

### Function: test_creating_dtype_with_dtype_class_errors()

### Function: test_run(self, t)

**Description:** Only test hash runs at all.

### Function: test_dtype(self, t)

### Function: test_equivalent_dtype_hashing(self)

### Function: test_invalid_types(self)

### Function: test_richcompare_invalid_dtype_equality(self)

### Function: test_richcompare_invalid_dtype_comparison(self, operation)

### Function: test_numeric_style_types_are_invalid(self, dtype)

### Function: test_expired_dtypes_with_bad_bytesize(self)

### Function: test_dtype_bytes_str_equivalence(self, value)

### Function: test_dtype_from_bytes(self)

### Function: test_bad_param(self)

### Function: test_field_order_equality(self)

### Function: test_create_string_dtypes_directly(self, type_char, char_size, scalar_type)

### Function: test_create_invalid_string_errors(self)

### Function: test_leading_zero_parsing(self)

### Function: test_equivalent_record(self)

**Description:** Test whether equivalent record dtypes hash the same.

### Function: test_different_names(self)

### Function: test_different_titles(self)

### Function: test_refcount_dictionary_setting(self)

### Function: test_mutate(self)

### Function: test_init_simple_structured(self)

### Function: test_mutate_error(self)

### Function: test_not_lists(self)

**Description:** Test if an appropriate exception is raised when passing bad values to
the dtype constructor.

### Function: test_aligned_size(self)

### Function: test_empty_struct_alignment(self)

### Function: test_union_struct(self)

### Function: test_subarray_list(self, obj, dtype, expected)

### Function: test_parenthesized_single_number(self)

### Function: test_comma_datetime(self)

### Function: test_from_dictproxy(self)

### Function: test_from_dict_with_zero_width_field(self)

### Function: test_bool_commastring(self)

### Function: test_nonint_offsets(self)

### Function: test_fields_by_index(self)

### Function: test_multifield_index(self, align_flag)

### Function: test_partial_dict(self)

### Function: test_fieldless_views(self)

### Function: test_nonstructured_with_object(self)

### Function: test_single_subarray(self)

### Function: test_equivalent_record(self)

**Description:** Test whether equivalent subarray dtypes hash the same.

### Function: test_nonequivalent_record(self)

**Description:** Test whether different subarray dtypes hash differently.

### Function: test_shape_equal(self)

**Description:** Test some data types that are equal

### Function: test_shape_simple(self)

**Description:** Test some simple cases that shouldn't be equal

### Function: test_shape_monster(self)

**Description:** Test some more complicated cases that shouldn't be equal

### Function: test_shape_sequence(self)

### Function: test_shape_matches_ndim(self)

### Function: test_shape_invalid(self)

### Function: test_alignment(self)

### Function: test_aligned_empty(self)

### Function: test_subarray_base_item(self)

### Function: test_subarray_cast_copies(self)

### Function: test_structured_object_create_delete(self, dt, pat, count, singleton, creation_func, creation_obj)

**Description:** Structured object reference counting in creation and deletion

### Function: test_structured_object_item_setting(self, dt, pat, count, singleton)

**Description:** Structured object reference counting for simple item setting

### Function: test_structured_object_indexing(self, shape, index, items_changed, dt, pat, count, singleton)

**Description:** Structured object reference counting for advanced indexing.

### Function: test_structured_object_take_and_repeat(self, dt, pat, count, singleton)

**Description:** Structured object reference counting for specialized functions.
The older functions such as take and repeat use different code paths
then item setting (when writing this).

### Function: test_sparse_field_assignment(self)

### Function: test_sparse_field_assignment_fancy(self)

### Function: test1(self)

### Function: test_list_recursion(self)

### Function: test_tuple_recursion(self)

### Function: test_dict_recursion(self)

### Function: test_no_metadata(self)

### Function: test_metadata_takes_dict(self)

### Function: test_metadata_rejects_nondict(self)

### Function: test_nested_metadata(self)

### Function: test_base_metadata_copied(self)

### Function: test_complex_dtype_str(self)

### Function: test_repr_structured(self)

### Function: test_repr_structured_not_packed(self)

### Function: test_repr_structured_datetime(self)

### Function: test_repr_str_subarray(self)

### Function: test_base_dtype_with_object_type(self)

### Function: test_empty_string_to_object(self)

### Function: test_void_subclass_unsized(self)

### Function: test_void_subclass_sized(self)

### Function: test_void_subclass_fields(self)

### Function: test_dtype_non_writable_attributes_deletion(self)

### Function: test_dtype_writable_attributes_deletion(self)

### Function: test_descr_has_trailing_void(self)

### Function: test_name_dtype_subclass(self)

### Function: test_zero_stride(self)

### Function: check_canonical(self, dtype, canonical)

**Description:** Check most properties relevant to "canonical" versions of a dtype,
which is mainly native byte order for datatypes supporting this.

The main work is checking structured dtypes with fields, where we
reproduce most the actual logic used in the C-code.

### Function: test_simple(self)

### Function: test_object_flag_not_inherited(self)

### Function: test_make_canonical_hypothesis(self, dtype)

### Function: test_structured(self, dtype)

### Function: check_pickling(self, dtype)

### Function: test_builtin(self, t)

### Function: test_structured(self)

### Function: test_structured_aligned(self)

### Function: test_structured_unaligned(self)

### Function: test_structured_padded(self)

### Function: test_structured_titles(self)

### Function: test_datetime(self, base, unit)

### Function: test_metadata(self)

### Function: test_pickle_dtype_class(self, DType)

### Function: test_pickle_dtype(self, dt)

### Function: test_complex_other_value_based(self, other, expected)

### Function: test_complex_scalar_value_based(self, other, expected)

### Function: test_complex_pyscalar_promote_rational(self)

### Function: test_python_integer_promotion(self, val)

### Function: test_float_int_pyscalar_promote_rational(self, other, expected)

### Function: test_permutations_do_not_influence_result(self, dtypes, expected)

### Function: test_simple(self)

### Function: test_recursion(self)

### Function: test_void_subtype(self)

### Function: test_void_subtype_recursion(self)

### Function: test_basic_dtypes_subclass_properties(self, dtype)

### Function: test_dtype_superclass(self)

### Function: test_is_numeric(self)

### Function: test_integer_alias_names(self, int_, size)

### Function: test_float_alias_names(self, name)

### Function: check(ctype, dtype)

### Function: test_array(self)

### Function: test_padded_structure(self)

### Function: test_bit_fields(self)

### Function: test_pointer(self)

### Function: test_size_t(self)

### Function: test_void_pointer(self)

### Function: test_union(self)

### Function: test_union_with_struct_packed(self)

### Function: test_union_packed(self)

### Function: test_packed_structure(self)

### Function: test_large_packed_structure(self)

### Function: test_big_endian_structure_packed(self)

### Function: test_little_endian_structure_packed(self)

### Function: test_little_endian_structure(self)

### Function: test_big_endian_structure(self)

### Function: test_simple_endian_types(self)

### Function: test_pairs(self, pair)

**Description:** Check that np.dtype('x,y') matches [np.dtype('x'), np.dtype('y')]
Example: np.dtype('d,I') -> dtype([('f0', '<f8'), ('f1', '<u4')])

### Function: test_custom_structured_dtype(self)

### Function: test_custom_structured_dtype_errors(self)

### Function: test_dtype(self)

### Function: test_dtype_subclass(self, code)

### Function: test_subscript_tuple(self, arg_len)

### Function: test_subscript_scalar(self)

### Function: make_dtype(off)

## Class: IntLike

## Class: user_def_subcls

### Function: aligned_offset(offset, alignment)

## Class: dt

## Class: dt

## Class: dt

## Class: vdt

## Class: PaddedStruct

## Class: BitfieldStruct

## Class: Union

## Class: Struct

## Class: Union

## Class: Struct

## Class: Union

## Class: PackedStructure

## Class: PackedStructure

## Class: BigEndStruct

## Class: LittleEndStruct

## Class: PaddedStruct

## Class: PaddedStruct

## Class: mytype

## Class: mytype

### Function: __index__(self)

### Function: __int__(self)
