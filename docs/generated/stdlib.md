## AI Summary

A file named stdlib.py.


### Function: execute(callback)

### Function: _follow_param(inference_state, arguments, index)

### Function: argument_clinic(clinic_string, want_value, want_context, want_arguments, want_inference_state, want_callback)

**Description:** Works like Argument Clinic (PEP 436), to validate function params.

### Function: builtins_next(iterators, defaults, inference_state)

### Function: builtins_iter(iterators_or_callables, defaults)

### Function: builtins_getattr(objects, names, defaults)

### Function: builtins_type(objects, bases, dicts)

## Class: SuperInstance

**Description:** To be used like the object ``super`` returns.

### Function: builtins_super(types, objects, context)

## Class: ReversedObject

### Function: builtins_reversed(sequences, value, arguments)

### Function: builtins_isinstance(objects, types, arguments, inference_state)

## Class: StaticMethodObject

### Function: builtins_staticmethod(functions)

## Class: ClassMethodObject

## Class: ClassMethodGet

## Class: ClassMethodArguments

### Function: builtins_classmethod(functions, value, arguments)

## Class: PropertyObject

### Function: builtins_property(functions, callback)

### Function: collections_namedtuple(value, arguments, callback)

**Description:** Implementation of the namedtuple function.

This has to be done by processing the namedtuple class template and
inferring the result.

## Class: PartialObject

## Class: PartialMethodObject

## Class: PartialSignature

## Class: MergedPartialArguments

### Function: functools_partial(value, arguments, callback)

### Function: functools_partialmethod(value, arguments, callback)

### Function: _return_first_param(firsts)

### Function: _random_choice(sequences)

### Function: _dataclass(value, arguments, callback)

## Class: DataclassWrapper

## Class: DataclassSignature

## Class: DataclassParamName

## Class: ItemGetterCallable

### Function: _functools_wraps(funcs)

## Class: WrapsCallable

## Class: Wrapped

### Function: _operator_itemgetter(args_value_set, value, arguments)

### Function: _create_string_input_function(func)

### Function: _os_path_join(args_set, callback)

### Function: get_metaclass_filters(func)

## Class: EnumInstance

### Function: tree_name_to_values(func)

### Function: wrapper(value, arguments)

### Function: f(func)

### Function: __init__(self, inference_state, instance)

### Function: _get_bases(self)

### Function: _get_wrapped_value(self)

### Function: get_filters(self, origin_scope)

### Function: __init__(self, reversed_obj, iter_list)

### Function: py__iter__(self, contextualized_node)

### Function: _next(self, arguments)

### Function: py__get__(self, instance, class_value)

### Function: __init__(self, class_method_obj, function)

### Function: py__get__(self, instance, class_value)

### Function: __init__(self, get_method, klass, function)

### Function: get_signatures(self)

### Function: py__call__(self, arguments)

### Function: __init__(self, klass, arguments)

### Function: unpack(self, func)

### Function: __init__(self, property_obj, function)

### Function: py__get__(self, instance, class_value)

### Function: _return_self(self, arguments)

### Function: __init__(self, actual_value, arguments, instance)

### Function: _get_functions(self, unpacked_arguments)

### Function: get_signatures(self)

### Function: py__call__(self, arguments)

### Function: py__doc__(self)

**Description:** In CPython partial does not replace the docstring. However we are still
imitating it here, because we want this docstring to be worth something
for the user.

### Function: py__get__(self, instance, class_value)

### Function: py__get__(self, instance, class_value)

### Function: __init__(self, wrapped_signature, skipped_arg_count, skipped_arg_set)

### Function: get_param_names(self, resolve_stars)

### Function: __init__(self, partial_arguments, call_arguments, instance)

### Function: unpack(self, funcdef)

### Function: get_signatures(self)

### Function: __init__(self, value, param_names)

### Function: get_param_names(self, resolve_stars)

### Function: __init__(self, parent_context, tree_name, annotation_node, default_node)

### Function: get_kind(self)

### Function: infer(self)

### Function: __init__(self, instance, args_value_set)

### Function: py__call__(self, item_value_set)

### Function: py__call__(self, funcs)

### Function: __init__(self, func, original_function)

### Function: name(self)

### Function: get_signature_functions(self)

### Function: wrapper(strings, value, arguments)

### Function: wrapper(cls, metaclasses, is_instance)

### Function: __init__(self, cls, name)

### Function: name(self)

### Function: _get_wrapped_value(self)

### Function: get_filters(self, origin_scope)

### Function: wrapper(inference_state, context, tree_name)

### Function: call()

### Function: wrapper(value, arguments, callback)

### Function: iterate()
