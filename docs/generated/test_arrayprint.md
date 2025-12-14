## AI Summary

A file named test_arrayprint.py.


## Class: TestArrayRepr

## Class: TestComplexArray

## Class: TestArray2String

## Class: TestPrintOptions

**Description:** Test getting and setting global print options.

### Function: test_unicode_object_array()

## Class: TestContextManager

### Function: test_scalar_repr_numbers(dtype, value)

### Function: test_scalar_repr_special(scalar, legacy_repr, representation)

### Function: test_scalar_void_float_str()

### Function: test_printoptions_asyncio_safe()

### Function: test_multithreaded_array_printing()

### Function: test_nan_inf(self)

### Function: test_subclass(self)

### Function: test_object_subclass(self)

### Function: test_0d_object_subclass(self)

### Function: test_self_containing(self)

### Function: test_containing_list(self)

### Function: test_void_scalar_recursion(self)

### Function: test_fieldless_structured(self)

### Function: test_str(self)

### Function: test_basic(self)

**Description:** Basic test of array2string.

### Function: test_unexpected_kwarg(self)

### Function: test_format_function(self)

**Description:** Test custom format function for each element in array.

### Function: test_structure_format_mixed(self)

### Function: test_structure_format_int(self)

### Function: test_structure_format_float(self)

### Function: test_unstructured_void_repr(self)

### Function: test_edgeitems_kwarg(self)

### Function: test_summarize_1d(self)

### Function: test_summarize_2d(self)

### Function: test_summarize_2d_dtype(self)

### Function: test_summarize_structure(self)

### Function: test_linewidth(self)

### Function: test_wide_element(self)

### Function: test_multiline_repr(self)

### Function: test_nested_array_repr(self)

### Function: test_any_text(self, text)

### Function: test_refcount(self)

### Function: test_with_sign(self)

### Function: setup_method(self)

### Function: teardown_method(self)

### Function: test_basic(self)

### Function: test_precision_zero(self)

### Function: test_formatter(self)

### Function: test_formatter_reset(self)

### Function: test_override_repr(self)

### Function: test_0d_arrays(self)

### Function: test_float_spacing(self)

### Function: test_bool_spacing(self)

### Function: test_sign_spacing(self)

### Function: test_float_overflow_nowarn(self)

### Function: test_sign_spacing_structured(self)

### Function: test_floatmode(self)

### Function: test_legacy_mode_scalars(self)

### Function: test_legacy_stray_comma(self)

### Function: test_dtype_linewidth_wrapping(self)

### Function: test_dtype_endianness_repr(self, native)

**Description:** there was an issue where
repr(array([0], dtype='<u2')) and repr(array([0], dtype='>u2'))
both returned the same thing:
array([0], dtype=uint16)
even though their dtypes have different endianness.

### Function: test_linewidth_repr(self)

### Function: test_linewidth_str(self)

### Function: test_edgeitems(self)

### Function: test_edgeitems_structured(self)

### Function: test_bad_args(self)

### Function: test_ctx_mgr(self)

### Function: test_ctx_mgr_restores(self)

### Function: test_ctx_mgr_exceptions(self)

### Function: test_ctx_mgr_as_smth(self)

## Class: sub

## Class: sub

## Class: sub

## Class: DuckCounter

### Function: _format_function(x)

### Function: make_str(a, width)

## Class: MultiLine

## Class: MultiLineLong

### Function: __new__(cls, inp)

### Function: __getitem__(self, ind)

### Function: __new__(cls, inp)

### Function: __getitem__(self, ind)

### Function: __getitem__(self, item)

### Function: to_string(self)

### Function: __str__(self)

### Function: __repr__(self)

### Function: __repr__(self)
