## AI Summary

A file named instance.py.


## Class: InstanceExecutedParamName

## Class: AnonymousMethodExecutionFilter

## Class: AnonymousMethodExecutionContext

## Class: MethodExecutionContext

## Class: AbstractInstanceValue

## Class: CompiledInstance

## Class: _BaseTreeInstance

## Class: TreeInstance

## Class: AnonymousInstance

## Class: CompiledInstanceName

## Class: CompiledInstanceClassFilter

## Class: BoundMethod

## Class: CompiledBoundMethod

## Class: SelfName

**Description:** This name calculates the parent_context lazily.

## Class: LazyInstanceClassName

## Class: InstanceClassFilter

**Description:** This filter is special in that it uses the class filter and wraps the
resulting names in LazyInstanceClassName. The idea is that the class name
filtering can be very flexible and always be reflected in instances.

## Class: SelfAttributeFilter

**Description:** This class basically filters all the use cases where `self.*` was assigned.

## Class: InstanceArguments

### Function: __init__(self, instance, function_value, tree_name)

### Function: infer(self)

### Function: matches_signature(self)

### Function: __init__(self, instance)

### Function: _convert_param(self, param, name)

### Function: __init__(self, instance, value)

### Function: get_filters(self, until_position, origin_scope)

### Function: get_param_names(self)

### Function: __init__(self, instance)

### Function: __init__(self, inference_state, parent_context, class_value)

### Function: is_instance(self)

### Function: get_qualified_names(self)

### Function: get_annotated_class_object(self)

### Function: py__class__(self)

### Function: py__bool__(self)

### Function: name(self)

### Function: get_signatures(self)

### Function: get_function_slot_names(self, name)

### Function: execute_function_slots(self, names)

### Function: get_type_hint(self, add_class_info)

### Function: py__getitem__(self, index_value_set, contextualized_node)

### Function: py__iter__(self, contextualized_node)

### Function: __repr__(self)

### Function: __init__(self, inference_state, parent_context, class_value, arguments)

### Function: get_filters(self, origin_scope, include_self_names)

### Function: name(self)

### Function: is_stub(self)

### Function: array_type(self)

### Function: name(self)

### Function: get_filters(self, origin_scope, include_self_names)

### Function: create_instance_context(self, class_context, node)

### Function: py__getattribute__alternatives(self, string_name)

**Description:** Since nothing was inferred, now check the __getattr__ and
__getattribute__ methods. Stubs don't need to be checked, because
they don't contain any logic.

### Function: py__next__(self, contextualized_node)

### Function: py__call__(self, arguments)

### Function: py__get__(self, instance, class_value)

**Description:** obj may be None.

### Function: __init__(self, inference_state, parent_context, class_value, arguments)

### Function: _get_annotated_class_object(self)

### Function: get_annotated_class_object(self)

### Function: get_key_values(self)

### Function: py__simple_getitem__(self, index)

### Function: __repr__(self)

### Function: infer(self)

### Function: __init__(self, instance, f)

### Function: get(self, name)

### Function: values(self)

### Function: _convert(self, names)

### Function: __init__(self, instance, class_context, function)

### Function: is_bound_method(self)

### Function: name(self)

### Function: py__class__(self)

### Function: _get_arguments(self, arguments)

### Function: _as_context(self, arguments)

### Function: py__call__(self, arguments)

### Function: get_signature_functions(self)

### Function: get_signatures(self)

### Function: __repr__(self)

### Function: is_bound_method(self)

### Function: get_signatures(self)

### Function: __init__(self, instance, class_context, tree_name)

### Function: parent_context(self)

### Function: get_defining_qualified_value(self)

### Function: infer(self)

### Function: __init__(self, instance, class_member_name)

### Function: infer(self)

### Function: get_signatures(self)

### Function: get_defining_qualified_value(self)

### Function: __init__(self, instance, class_filter)

### Function: get(self, name)

### Function: values(self)

### Function: _convert(self, names)

### Function: __repr__(self)

### Function: __init__(self, instance, instance_class, node_context, origin_scope)

### Function: _filter(self, names)

### Function: _filter_self_names(self, names)

### Function: _is_in_right_scope(self, self_name, name)

### Function: _convert_names(self, names)

### Function: _check_flows(self, names)

### Function: __init__(self, instance, arguments)

### Function: unpack(self, func)

### Function: iterate()
