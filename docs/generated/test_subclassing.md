## AI Summary

A file named test_subclassing.py.


### Function: assert_startswith(a, b)

## Class: SubArray

## Class: SubMaskedArray

**Description:** Pure subclass of MaskedArray, keeping some info on subclass.

## Class: MSubArray

## Class: CSAIterator

**Description:** Flat iterator object that uses its own setter/getter
(works around ndarray.flat not propagating subclass setters/getters
see https://github.com/numpy/numpy/issues/4564)
roughly following MaskedIterator

## Class: ComplicatedSubArray

## Class: WrappedArray

**Description:** Wrapping a MaskedArray rather than subclassing to test that
ufunc deferrals are commutative.
See: https://github.com/numpy/numpy/issues/15200)

## Class: TestSubclassing

## Class: ArrayNoInheritance

**Description:** Quantity-like class that does not inherit from ndarray

### Function: test_array_no_inheritance()

## Class: TestClassWrapping

### Function: __new__(cls, arr, info)

### Function: __array_finalize__(self, obj)

### Function: __add__(self, other)

### Function: __iadd__(self, other)

### Function: __new__(cls, info)

### Function: __new__(cls, data, info, mask)

### Function: _series(self)

### Function: __init__(self, a)

### Function: __iter__(self)

### Function: __getitem__(self, indx)

### Function: __setitem__(self, index, value)

### Function: __next__(self)

### Function: __str__(self)

### Function: __repr__(self)

### Function: _validate_input(self, value)

### Function: __setitem__(self, item, value)

### Function: __getitem__(self, item)

### Function: flat(self)

### Function: flat(self, value)

### Function: __array_wrap__(self, obj, context, return_scalar)

### Function: __init__(self, array)

### Function: __repr__(self)

### Function: __array__(self, dtype, copy)

### Function: __array_ufunc__(self, ufunc, method)

### Function: setup_method(self)

### Function: test_data_subclassing(self)

### Function: test_maskedarray_subclassing(self)

### Function: test_masked_unary_operations(self)

### Function: test_masked_binary_operations(self)

### Function: test_masked_binary_operations2(self)

### Function: test_attributepropagation(self)

### Function: test_subclasspreservation(self)

### Function: test_subclass_items(self)

**Description:** test that getter and setter go via baseclass

### Function: test_subclass_nomask_items(self)

### Function: test_subclass_repr(self)

**Description:** test that repr uses the name of the subclass
and 'array' for np.ndarray

### Function: test_subclass_str(self)

**Description:** test str with subclass that has overridden str, setitem

### Function: test_pure_subclass_info_preservation(self)

### Function: __init__(self, data, units)

### Function: __getattr__(self, attr)

### Function: setup_method(self)

### Function: test_masked_unary_operations(self)

### Function: test_masked_binary_operations(self)

### Function: test_mixins_have_slots(self)
