## AI Summary

A file named test_array_coercion.py.


### Function: arraylikes()

**Description:** Generator for functions converting an array into various array-likes.
If full is True (default) it includes array-likes not capable of handling
all dtypes.

### Function: scalar_instances(times, extended_precision, user_dtype)

### Function: is_parametric_dtype(dtype)

**Description:** Returns True if the dtype is a parametric legacy dtype (itemsize
is 0, or a datetime without units)

## Class: TestStringDiscovery

## Class: TestScalarDiscovery

## Class: TestTimeScalars

## Class: TestNested

## Class: TestBadSequences

## Class: TestArrayLikes

## Class: TestAsArray

**Description:** Test expected behaviors of ``asarray``.

## Class: TestSpecialAttributeLookupFailure

### Function: test_subarray_from_array_construction()

### Function: test_empty_string()

### Function: ndarray(a)

## Class: MyArr

### Function: subclass(a)

## Class: _SequenceLike

## Class: ArrayDunder

## Class: ArrayInterface

## Class: ArrayStruct

### Function: test_basic_stringlength(self, obj)

### Function: test_nested_arrays_stringlength(self, obj)

### Function: test_unpack_first_level(self, arraylike)

### Function: test_void_special_case(self)

### Function: test_char_special_case(self)

### Function: test_char_special_case_deep(self)

### Function: test_unknown_object(self)

### Function: test_scalar(self, scalar)

### Function: test_scalar_promotion(self)

### Function: test_scalar_coercion(self, scalar)

### Function: test_scalar_coercion_same_as_cast_and_assignment(self, cast_to)

**Description:** Test that in most cases:
   * `np.array(scalar, dtype=dtype)`
   * `np.empty((), dtype=dtype)[()] = scalar`
   * `np.array(scalar).astype(dtype)`
should behave the same.  The only exceptions are parametric dtypes
(mainly datetime/timedelta without unit) and void without fields.

### Function: test_pyscalar_subclasses(self, pyscalar)

**Description:** NumPy arrays are read/write which means that anything but invariant
behaviour is on thin ice.  However, we currently are happy to discover
subclasses of Python float, int, complex the same as the base classes.
This should potentially be deprecated.

### Function: test_default_dtype_instance(self, dtype_char)

### Function: test_scalar_to_int_coerce_does_not_cast(self, dtype, scalar, error)

**Description:** Signed integers are currently different in that they do not cast other
NumPy scalar, but instead use scalar.__int__(). The hardcoded
exception to this rule is `np.array(scalar, dtype=integer)`.

### Function: test_coercion_basic(self, dtype, scalar)

### Function: test_coercion_timedelta_convert_to_number(self, dtype, scalar)

### Function: test_coercion_assignment_datetime(self, val, unit, dtype)

### Function: test_coercion_assignment_timedelta(self, val, unit)

### Function: test_nested_simple(self)

### Function: test_pathological_self_containing(self)

### Function: test_nested_arraylikes(self, arraylike)

### Function: test_uneven_depth_ragged(self, arraylike)

### Function: test_empty_sequence(self)

### Function: test_array_of_different_depths(self)

### Function: test_growing_list(self)

### Function: test_mutated_list(self)

### Function: test_replace_0d_array(self)

### Function: test_0d_object_special_case(self, arraylike)

### Function: test_object_assignment_special_case(self, arraylike, arr)

### Function: test_0d_generic_special_case(self)

### Function: test_arraylike_classes(self)

### Function: test_too_large_array_error_paths(self)

**Description:** Test the error paths, including for memory leaks

### Function: test_bad_array_like_attributes(self, attribute, error)

### Function: test_bad_array_like_bad_length(self, error)

### Function: test_array_interface_descr_optional(self)

### Function: test_dtype_identity(self)

**Description:** Confirm the intended behavior for *dtype* kwarg.

The result of ``asarray()`` should have the dtype provided through the
keyword argument, when used. This forces unique array handles to be
produced for unique np.dtype objects, but (for equivalent dtypes), the
underlying data (the base object) is shared with the original array
object.

Ref https://github.com/numpy/numpy/issues/1468

## Class: WeirdArrayLike

## Class: WeirdArrayInterface

### Function: test_deprecated(self)

### Function: __len__(self)

### Function: __getitem__(self)

### Function: __init__(self, a)

### Function: __array__(self, dtype, copy)

### Function: __init__(self, a)

### Function: __init__(self, a)

## Class: MyScalar

## Class: mylist

## Class: mylist

## Class: baditem

## Class: ArraySubclass

## Class: ArrayLike

## Class: BadInterface

## Class: BadSequence

## Class: MyClass

### Function: __array__(self, dtype, copy)

### Function: __array_interface__(self)

### Function: __len__(self)

### Function: __len__(self)

### Function: __len__(self)

### Function: __getitem__(self)

### Function: __float__(self)

### Function: __array_interface__(self)

### Function: __array_struct__(self)

### Function: __array__(self, dtype, copy)

### Function: __getattr__(self, attr)

### Function: __len__(self)

### Function: __getitem__(self)
