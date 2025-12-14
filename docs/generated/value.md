## AI Summary

A file named value.py.


## Class: CheckAttribute

**Description:** Raises :exc:`AttributeError` if the attribute X is not available.

## Class: CompiledValue

## Class: CompiledModule

## Class: CompiledName

## Class: SignatureParamName

## Class: UnresolvableParamName

## Class: CompiledValueName

## Class: EmptyCompiledName

**Description:** Accessing some names will raise an exception. To avoid not having any
completions, just give Jedi the option to return this object. It infers to
nothing.

## Class: CompiledValueFilter

### Function: _parse_function_doc(doc)

**Description:** Takes a function and returns the params and return value as a tuple.
This is nothing more than a docstring parser.

TODO docstrings like utime(path, (atime, mtime)) and a(b [, b]) -> None
TODO docstrings like 'tuple of integers'

### Function: create_from_name(inference_state, compiled_value, name)

### Function: _normalize_create_args(func)

**Description:** The cache doesn't care about keyword vs. normal args.

### Function: create_from_access_path(inference_state, access_path)

### Function: create_cached_compiled_value(inference_state, access_handle, parent_context)

### Function: __init__(self, check_name)

### Function: __call__(self, func)

### Function: __get__(self, instance, owner)

### Function: __init__(self, inference_state, access_handle, parent_context)

### Function: py__call__(self, arguments)

### Function: py__class__(self)

### Function: py__mro__(self)

### Function: py__bases__(self)

### Function: get_qualified_names(self)

### Function: py__bool__(self)

### Function: is_class(self)

### Function: is_function(self)

### Function: is_module(self)

### Function: is_compiled(self)

### Function: is_stub(self)

### Function: is_instance(self)

### Function: py__doc__(self)

### Function: get_param_names(self)

### Function: get_signatures(self)

### Function: __repr__(self)

### Function: _parse_function_doc(self)

### Function: api_type(self)

### Function: get_filters(self, is_instance, origin_scope)

### Function: _ensure_one_filter(self, is_instance)

### Function: py__simple_getitem__(self, index)

### Function: py__getitem__(self, index_value_set, contextualized_node)

### Function: py__iter__(self, contextualized_node)

### Function: py__name__(self)

### Function: name(self)

### Function: _execute_function(self, params)

### Function: get_safe_value(self, default)

### Function: execute_operation(self, other, operator)

### Function: execute_annotation(self)

### Function: negate(self)

### Function: get_metaclasses(self)

### Function: _as_context(self)

### Function: array_type(self)

### Function: get_key_values(self)

### Function: get_type_hint(self, add_class_info)

### Function: _as_context(self)

### Function: py__path__(self)

### Function: is_package(self)

### Function: string_names(self)

### Function: py__file__(self)

### Function: __init__(self, inference_state, parent_value, name, is_descriptor)

### Function: py__doc__(self)

### Function: _get_qualified_names(self)

### Function: get_defining_qualified_value(self)

### Function: __repr__(self)

### Function: api_type(self)

### Function: infer(self)

### Function: infer_compiled_value(self)

### Function: __init__(self, compiled_value, signature_param)

### Function: string_name(self)

### Function: to_string(self)

### Function: get_kind(self)

### Function: infer(self)

### Function: __init__(self, compiled_value, name, default)

### Function: get_kind(self)

### Function: to_string(self)

### Function: infer(self)

### Function: __init__(self, value, name)

### Function: __init__(self, inference_state, name)

### Function: infer(self)

### Function: __init__(self, inference_state, compiled_value, is_instance)

### Function: get(self, name)

### Function: _get(self, name, allowed_getattr_callback, in_dir_callback, check_has_attribute)

**Description:** To remove quite a few access calls we introduced the callback here.

### Function: _get_cached_name(self, name, is_empty)

### Function: values(self)

### Function: _create_name(self, name, is_descriptor)

### Function: __repr__(self)

### Function: wrapper(inference_state, obj, parent_context)

### Function: change_options(m)
