## AI Summary

A file named base_value.py.


## Class: HasNoContext

## Class: HelperValueMixin

## Class: Value

**Description:** To be implemented by subclasses.

### Function: iterate_values(values, contextualized_node, is_async)

**Description:** Calls `iterate`, on all values but ignores the ordering and just returns
all values that the iterate functions yield.

## Class: _ValueWrapperBase

## Class: LazyValueWrapper

## Class: ValueWrapper

## Class: TreeValue

## Class: ContextualizedNode

### Function: _getitem(value, index_values, contextualized_node)

## Class: ValueSet

### Function: iterator_to_value_set(func)

### Function: get_root_context(self)

### Function: execute(self, arguments)

### Function: execute_with_values(self)

### Function: execute_annotation(self)

### Function: gather_annotation_classes(self)

### Function: merge_types_of_iterate(self, contextualized_node, is_async)

### Function: _get_value_filters(self, name_or_str)

### Function: goto(self, name_or_str, name_context, analysis_errors)

### Function: py__getattribute__(self, name_or_str, name_context, position, analysis_errors)

**Description:** :param position: Position of the last statement -> tuple of line, column

### Function: py__await__(self)

### Function: py__name__(self)

### Function: iterate(self, contextualized_node, is_async)

### Function: is_sub_class_of(self, class_value)

### Function: is_same_class(self, class2)

### Function: as_context(self)

### Function: __init__(self, inference_state, parent_context)

### Function: py__getitem__(self, index_value_set, contextualized_node)

### Function: py__simple_getitem__(self, index)

### Function: py__iter__(self, contextualized_node)

### Function: py__next__(self, contextualized_node)

### Function: get_signatures(self)

### Function: is_class(self)

### Function: is_class_mixin(self)

### Function: is_instance(self)

### Function: is_function(self)

### Function: is_module(self)

### Function: is_namespace(self)

### Function: is_compiled(self)

### Function: is_bound_method(self)

### Function: is_builtins_module(self)

### Function: py__bool__(self)

**Description:** Since Wrapper is a super class for classes, functions and modules,
the return value will always be true.

### Function: py__doc__(self)

### Function: get_safe_value(self, default)

### Function: execute_operation(self, other, operator)

### Function: py__call__(self, arguments)

### Function: py__stop_iteration_returns(self)

### Function: py__getattribute__alternatives(self, name_or_str)

**Description:** For now a way to add values in cases like __getattr__.

### Function: py__get__(self, instance, class_value)

### Function: py__get__on_class(self, calling_instance, instance, class_value)

### Function: get_qualified_names(self)

### Function: is_stub(self)

### Function: _as_context(self)

### Function: name(self)

### Function: get_type_hint(self, add_class_info)

### Function: infer_type_vars(self, value_set)

**Description:** When the current instance represents a type annotation, this method
tries to find information about undefined type vars and returns a dict
from type var name to value set.

This is for example important to understand what `iter([1])` returns.
According to typeshed, `iter` returns an `Iterator[_T]`:

    def iter(iterable: Iterable[_T]) -> Iterator[_T]: ...

This functions would generate `int` for `_T` in this case, because it
unpacks the `Iterable`.

Parameters
----------

`self`: represents the annotation of the current parameter to infer the
    value for. In the above example, this would initially be the
    `Iterable[_T]` of the `iterable` parameter and then, when recursing,
    just the `_T` generic parameter.

`value_set`: represents the actual argument passed to the parameter
    we're inferred for, or (for recursive calls) their types. In the
    above example this would first be the representation of the list
    `[1]` and then, when recursing, just of `1`.

### Function: name(self)

### Function: create_cached(cls, inference_state)

### Function: __getattr__(self, name)

### Function: _wrapped_value(self)

### Function: __repr__(self)

### Function: _get_wrapped_value(self)

### Function: __init__(self, wrapped_value)

### Function: __repr__(self)

### Function: __init__(self, inference_state, parent_context, tree_node)

### Function: __repr__(self)

### Function: __init__(self, context, node)

### Function: get_root_context(self)

### Function: infer(self)

### Function: __repr__(self)

### Function: __init__(self, iterable)

### Function: _from_frozen_set(cls, frozenset_)

### Function: from_sets(cls, sets)

**Description:** Used to work with an iterable of set.

### Function: __or__(self, other)

### Function: __and__(self, other)

### Function: __iter__(self)

### Function: __bool__(self)

### Function: __len__(self)

### Function: __repr__(self)

### Function: filter(self, filter_func)

### Function: __getattr__(self, name)

### Function: __eq__(self, other)

### Function: __ne__(self, other)

### Function: __hash__(self)

### Function: py__class__(self)

### Function: iterate(self, contextualized_node, is_async)

### Function: execute(self, arguments)

### Function: execute_with_values(self)

### Function: goto(self)

### Function: py__getattribute__(self)

### Function: get_item(self)

### Function: try_merge(self, function_name)

### Function: gather_annotation_classes(self)

### Function: get_signatures(self)

### Function: get_type_hint(self, add_class_info)

### Function: infer_type_vars(self, value_set)

### Function: wrapper()

### Function: mapper()
