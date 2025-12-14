## AI Summary

A file named dynamic_arrays.py.


### Function: check_array_additions(context, sequence)

**Description:** Just a mapper function for the internal _internal_check_array_additions 

### Function: _internal_check_array_additions(context, sequence)

**Description:** Checks if a `Array` has "add" (append, insert, extend) statements:

>>> a = [""]
>>> a.append(1)

### Function: get_dynamic_array_instance(instance, arguments)

**Description:** Used for set() and list() instances.

## Class: _DynamicArrayAdditions

**Description:** Used for the usage of set() and list().
This is definitely a hack, but a good one :-)
It makes it possible to use set/list conversions.

This is not a proper context, because it doesn't have to be. It's not used
in the wild, it's just used within typeshed as an argument to `__init__`
for set/list and never used in any other place.

## Class: _Modification

## Class: DictModification

## Class: ListModification

### Function: find_additions(context, arglist, add_name)

### Function: __init__(self, instance, arguments)

### Function: py__class__(self)

### Function: py__iter__(self, contextualized_node)

### Function: iterate(self, contextualized_node, is_async)

### Function: __init__(self, wrapped_value, assigned_values, contextualized_key)

### Function: py__getitem__(self)

### Function: py__simple_getitem__(self, index)

### Function: py__iter__(self, contextualized_node)

### Function: get_key_values(self)

### Function: py__iter__(self, contextualized_node)
