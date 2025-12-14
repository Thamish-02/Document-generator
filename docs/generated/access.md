## AI Summary

A file named access.py.


### Function: safe_getattr(obj, name, default)

### Function: shorten_repr(func)

### Function: create_access(inference_state, obj)

### Function: load_module(inference_state, dotted_name, sys_path)

## Class: AccessPath

### Function: create_access_path(inference_state, obj)

### Function: get_api_type(obj)

## Class: DirectObjectAccess

### Function: _is_class_instance(obj)

**Description:** Like inspect.* methods.

### Function: wrapper(self)

### Function: __init__(self, accesses)

### Function: __init__(self, inference_state, obj)

### Function: __repr__(self)

### Function: _create_access(self, obj)

### Function: _create_access_path(self, obj)

### Function: py__bool__(self)

### Function: py__file__(self)

### Function: py__doc__(self)

### Function: py__name__(self)

### Function: py__mro__accesses(self)

### Function: py__getitem__all_values(self)

### Function: py__simple_getitem__(self, index)

### Function: py__iter__list(self)

### Function: py__class__(self)

### Function: py__bases__(self)

### Function: py__path__(self)

### Function: get_repr(self)

### Function: is_class(self)

### Function: is_function(self)

### Function: is_module(self)

### Function: is_instance(self)

### Function: ismethoddescriptor(self)

### Function: get_qualified_names(self)

### Function: dir(self)

### Function: has_iter(self)

### Function: is_allowed_getattr(self, name, safe)

### Function: getattr_paths(self, name, default)

### Function: get_safe_value(self)

### Function: get_api_type(self)

### Function: get_array_type(self)

### Function: get_key_paths(self)

### Function: get_access_path_tuples(self)

### Function: _get_objects_path(self)

### Function: execute_operation(self, other_access_handle, operator)

### Function: get_annotation_name_and_args(self)

**Description:** Returns Tuple[Optional[str], Tuple[AccessPath, ...]]

### Function: needs_type_completions(self)

### Function: _annotation_to_str(self, annotation)

### Function: get_signature_params(self)

### Function: _get_signature(self)

### Function: get_return_annotation(self)

### Function: negate(self)

### Function: get_dir_infos(self)

**Description:** Used to return a couple of infos that are needed when accessing the sub
objects of an objects

### Function: try_to_get_name(obj)

### Function: iter_partial_keys()

### Function: get()
