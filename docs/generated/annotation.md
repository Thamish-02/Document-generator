## AI Summary

A file named annotation.py.


### Function: infer_annotation(context, annotation)

**Description:** Inferes an annotation node. This means that it inferes the part of
`int` here:

    foo: int = 3

Also checks for forward references (strings)

### Function: _infer_annotation_string(context, string, index)

### Function: _get_forward_reference_node(context, string)

### Function: _split_comment_param_declaration(decl_text)

**Description:** Split decl_text on commas, but group generic expressions
together.

For example, given "foo, Bar[baz, biz]" we return
['foo', 'Bar[baz, biz]'].

### Function: infer_param(function_value, param, ignore_stars)

### Function: _infer_param(function_value, param)

**Description:** Infers the type of a function parameter, using type annotations.

### Function: py__annotations__(funcdef)

### Function: resolve_forward_references(context, all_annotations)

### Function: infer_return_types(function, arguments)

**Description:** Infers the type of a function's return value,
according to type annotations.

### Function: infer_type_vars_for_execution(function, arguments, annotation_dict)

**Description:** Some functions use type vars that are not defined by the class, but rather
only defined in the function. See for example `iter`. In those cases we
want to:

1. Search for undefined type vars.
2. Infer type vars with the execution state we have.
3. Return the union of all type vars that have been found.

### Function: infer_return_for_callable(arguments, param_values, result_values)

### Function: _infer_type_vars_for_callable(arguments, lazy_params)

**Description:** Infers type vars for the Calllable class:

    def x() -> Callable[[Callable[..., _T]], _T]: ...

### Function: merge_type_var_dicts(base_dict, new_dict)

### Function: merge_pairwise_generics(annotation_value, annotated_argument_class)

**Description:** Match up the generic parameters from the given argument class to the
target annotation.

This walks the generic parameters immediately within the annotation and
argument's type, in order to determine the concrete values of the
annotation's parameters for the current case.

For example, given the following code:

    def values(mapping: Mapping[K, V]) -> List[V]: ...

    for val in values({1: 'a'}):
        val

Then this function should be given representations of `Mapping[K, V]`
and `Mapping[int, str]`, so that it can determine that `K` is `int and
`V` is `str`.

Note that it is responsibility of the caller to traverse the MRO of the
argument type as needed in order to find the type matching the
annotation (in this case finding `Mapping[int, str]` as a parent of
`Dict[int, str]`).

Parameters
----------

`annotation_value`: represents the annotation to infer the concrete
    parameter types of.

`annotated_argument_class`: represents the annotated class of the
    argument being passed to the object annotated by `annotation_value`.

### Function: find_type_from_comment_hint_for(context, node, name)

### Function: find_type_from_comment_hint_with(context, node, name)

### Function: find_type_from_comment_hint_assign(context, node, name)

### Function: _find_type_from_comment_hint(context, node, varlist, name)

### Function: find_unknown_type_vars(context, node)

### Function: _filter_type_vars(value_set, found)

### Function: _unpack_subscriptlist(subscriptlist)

### Function: resolve(node)

### Function: check_node(node)
