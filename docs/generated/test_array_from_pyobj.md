## AI Summary

A file named test_array_from_pyobj.py.


### Function: get_testdir()

### Function: setup_module()

**Description:** Build the required testing extension module

### Function: flags_info(arr)

### Function: flags2names(flags)

## Class: Intent

## Class: Type

## Class: Array

## Class: TestIntent

## Class: TestSharedMemory

### Function: __init__(self, intent_list)

### Function: __getattr__(self, name)

### Function: __str__(self)

### Function: __repr__(self)

### Function: is_intent(self)

### Function: is_intent_exact(self)

### Function: __new__(cls, name)

### Function: _init(self, name)

### Function: __repr__(self)

### Function: cast_types(self)

### Function: all_types(self)

### Function: smaller_types(self)

### Function: equal_types(self)

### Function: larger_types(self)

### Function: __repr__(self)

### Function: __init__(self, typ, dims, intent, obj)

### Function: arr_equal(self, arr1, arr2)

### Function: __str__(self)

### Function: has_shared_memory(self)

**Description:** Check that created array shares data with input array.

### Function: test_in_out(self)

### Function: setup_type(self, request)

### Function: num2seq(self)

### Function: num23seq(self)

### Function: test_in_from_2seq(self)

### Function: test_in_from_2casttype(self)

### Function: test_in_nocopy(self, write, order, inp)

**Description:** Test if intent(in) array can be passed without copies

### Function: test_inout_2seq(self)

### Function: test_f_inout_23seq(self)

### Function: test_c_inout_23seq(self)

### Function: test_in_copy_from_2casttype(self)

### Function: test_c_in_from_23seq(self)

### Function: test_in_from_23casttype(self)

### Function: test_f_in_from_23casttype(self)

### Function: test_c_in_from_23casttype(self)

### Function: test_f_copy_in_from_23casttype(self)

### Function: test_c_copy_in_from_23casttype(self)

### Function: test_in_cache_from_2casttype(self)

### Function: test_in_cache_from_2casttype_failure(self)

### Function: test_cache_hidden(self)

### Function: test_hidden(self)

### Function: test_optional_none(self)

### Function: test_optional_from_2seq(self)

### Function: test_optional_from_23seq(self)

### Function: test_inplace(self)

### Function: test_inplace_from_casttype(self)
