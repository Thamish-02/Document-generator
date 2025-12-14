## AI Summary

A file named test_deprecations.py.


## Class: _DeprecationTestCase

## Class: _VisibleDeprecationTestCase

## Class: TestDTypeAttributeIsDTypeDeprecation

## Class: TestTestDeprecated

## Class: TestNonNumericConjugate

**Description:** Deprecate no-op behavior of ndarray.conjugate on non-numeric dtypes,
which conflicts with the error behavior of np.conjugate.

## Class: TestDatetimeEvent

## Class: TestBincount

## Class: TestGeneratorSum

## Class: TestFromstring

## Class: TestFromStringAndFileInvalidData

## Class: TestToString

## Class: TestDTypeCoercion

## Class: BuiltInRoundComplexDType

## Class: TestIncorrectAdvancedIndexWithEmptyResult

## Class: TestNonExactMatchDeprecation

## Class: TestMatrixInOuter

## Class: FlatteningConcatenateUnsafeCast

## Class: TestDeprecatedUnpickleObjectScalar

**Description:** Technically, it should be impossible to create numpy object scalars,
but there was an unpickle path that would in theory allow it. That
path is invalid and must lead to the warning.

## Class: TestSingleElementSignature

## Class: TestCtypesGetter

## Class: TestPartitionBoolIndex

## Class: TestMachAr

## Class: TestQuantileInterpolationDeprecation

## Class: TestArrayFinalizeNone

## Class: TestLoadtxtParseIntsViaFloat

## Class: TestScalarConversion

## Class: TestPyIntConversion

### Function: test_future_scalar_attributes(name)

## Class: TestRemovedGlobals

## Class: TestDeprecatedFinfo

## Class: TestMathAlias

## Class: TestLibImports

## Class: TestDeprecatedDTypeAliases

## Class: TestDeprecatedArrayWrap

## Class: TestDeprecatedDTypeParenthesizedRepeatCount

## Class: TestDeprecatedSaveFixImports

## Class: TestAddNewdocUFunc

### Function: setup_method(self)

### Function: teardown_method(self)

### Function: assert_deprecated(self, function, num, ignore_others, function_fails, exceptions, args, kwargs)

**Description:** Test if DeprecationWarnings are given and raised.

This first checks if the function when called gives `num`
DeprecationWarnings, after that it tries to raise these
DeprecationWarnings and compares them with `exceptions`.
The exceptions can be different for cases where this code path
is simply not anticipated and the exception is replaced.

Parameters
----------
function : callable
    The function to test
num : int
    Number of DeprecationWarnings to expect. This should normally be 1.
ignore_others : bool
    Whether warnings of the wrong type should be ignored (note that
    the message is not checked)
function_fails : bool
    If the function would normally fail, setting this will check for
    warnings inside a try/except block.
exceptions : Exception or tuple of Exceptions
    Exception to expect when turning the warnings into an error.
    The default checks for DeprecationWarnings. If exceptions is
    empty the function is expected to run successfully.
args : tuple
    Arguments for `function`
kwargs : dict
    Keyword arguments for `function`

### Function: assert_not_deprecated(self, function, args, kwargs)

**Description:** Test that warnings are not raised.

This is just a shorthand for:

self.assert_deprecated(function, num=0, ignore_others=True,
                exceptions=tuple(), args=args, kwargs=kwargs)

### Function: test_deprecation_dtype_attribute_is_dtype(self)

### Function: test_assert_deprecated(self)

### Function: test_conjugate(self)

### Function: test_3_tuple(self)

### Function: test_bincount_minlength(self)

### Function: test_bincount_bad_list(self, badlist)

### Function: test_generator_sum(self)

### Function: test_fromstring(self)

### Function: test_deprecate_unparsable_data_file(self, invalid_str)

### Function: test_deprecate_unparsable_string(self, invalid_str)

### Function: test_tostring(self)

### Function: test_tostring_matches_tobytes(self)

### Function: test_dtype_coercion(self)

### Function: test_array_construction(self)

### Function: test_not_deprecated(self)

### Function: test_deprecated(self)

### Function: test_not_deprecated(self)

### Function: test_empty_subspace(self, index)

### Function: test_empty_index_broadcast_not_deprecated(self)

### Function: test_non_exact_match(self)

### Function: test_deprecated(self)

### Function: test_deprecated(self)

### Function: test_not_deprecated(self)

### Function: test_deprecated(self)

### Function: test_deprecated(self)

### Function: test_deprecated(self, name)

### Function: test_not_deprecated(self, name)

### Function: test_deprecated(self, func)

### Function: test_not_deprecated(self, func)

### Function: test_deprecated_module(self)

### Function: test_deprecated(self, func)

### Function: test_both_passed(self, func)

### Function: test_use_none_is_deprecated(self)

### Function: test_deprecated_warning(self, dtype)

### Function: test_deprecated_raised(self, dtype)

### Function: test_float_conversion(self)

### Function: test_behaviour(self)

### Function: test_deprecated_scalar(self, dtype)

### Function: test_attributeerror_includes_info(self, name)

### Function: test_deprecated_none(self)

### Function: test_deprecated_np_lib_math(self)

### Function: test_lib_functions_deprecation_call(self)

### Function: _check_for_warning(self, func)

### Function: test_a_dtype_alias(self)

### Function: test_deprecated(self)

### Function: test_parenthesized_repeat_count(self, string)

### Function: test_deprecated(self)

### Function: test_deprecated(self)

## Class: dt

## Class: vdt

### Function: foo()

## Class: NoFinalize

### Function: scalar(value, dtype)

### Function: assign(value, dtype)

### Function: create(value, dtype)

## Class: Test1

## Class: Test2

### Function: __array__(self, dtype, copy)

### Function: __array_wrap__(self, arr, context)

### Function: __array_wrap__(self, arr)
