## AI Summary

A file named iterable.py.


## Class: IterableMixin

## Class: GeneratorBase

## Class: Generator

**Description:** Handling of `yield` functions.

### Function: comprehension_from_atom(inference_state, value, atom)

## Class: ComprehensionMixin

## Class: _DictMixin

## Class: Sequence

## Class: _BaseComprehension

## Class: ListComprehension

## Class: SetComprehension

## Class: GeneratorComprehension

## Class: _DictKeyMixin

## Class: DictComprehension

## Class: SequenceLiteralValue

## Class: DictLiteralValue

## Class: _FakeSequence

## Class: FakeTuple

## Class: FakeList

## Class: FakeDict

## Class: MergedArray

### Function: unpack_tuple_to_dict(context, types, exprlist)

**Description:** Unpacking tuple assignments in for statements and expr_stmts.

## Class: Slice

### Function: py__next__(self, contextualized_node)

### Function: py__stop_iteration_returns(self)

### Function: _get_wrapped_value(self)

### Function: _get_cls(self)

### Function: py__bool__(self)

### Function: _iter(self, arguments)

### Function: _next(self, arguments)

### Function: py__stop_iteration_returns(self)

### Function: name(self)

### Function: get_annotated_class_object(self)

### Function: __init__(self, inference_state, func_execution_context)

### Function: py__iter__(self, contextualized_node)

### Function: py__stop_iteration_returns(self)

### Function: __repr__(self)

### Function: _get_comp_for_context(self, parent_context, comp_for)

### Function: _nested(self, comp_fors, parent_context)

### Function: _iterate(self)

### Function: py__iter__(self, contextualized_node)

### Function: __repr__(self)

### Function: _get_generics(self)

### Function: name(self)

### Function: _get_generics(self)

### Function: _cached_generics(self)

### Function: _get_wrapped_value(self)

### Function: py__bool__(self)

### Function: parent(self)

### Function: py__getitem__(self, index_value_set, contextualized_node)

### Function: __init__(self, inference_state, defining_context, sync_comp_for_node, entry_node)

### Function: py__simple_getitem__(self, index)

### Function: get_mapping_item_values(self)

### Function: get_key_values(self)

### Function: __init__(self, inference_state, defining_context, sync_comp_for_node, key_node, value_node)

### Function: py__iter__(self, contextualized_node)

### Function: py__simple_getitem__(self, index)

### Function: _dict_keys(self)

### Function: _dict_values(self)

### Function: _imitate_values(self, arguments)

### Function: _imitate_items(self, arguments)

### Function: exact_key_items(self)

### Function: __init__(self, inference_state, defining_context, atom)

### Function: _get_generics(self)

### Function: py__simple_getitem__(self, index)

**Description:** Here the index is an int/str. Raises IndexError/KeyError.

### Function: py__iter__(self, contextualized_node)

**Description:** While values returns the possible values for any array field, this
function returns the value for a certain index.

### Function: py__len__(self)

### Function: get_tree_entries(self)

### Function: __repr__(self)

### Function: __init__(self, inference_state, defining_context, atom)

### Function: py__simple_getitem__(self, index)

**Description:** Here the index is an int/str. Raises IndexError/KeyError.

### Function: py__iter__(self, contextualized_node)

**Description:** While values returns the possible values for any array field, this
function returns the value for a certain index.

### Function: _imitate_values(self, arguments)

### Function: _imitate_items(self, arguments)

### Function: exact_key_items(self)

**Description:** Returns a generator of tuples like dict.items(), where the key is
resolved (as a string) and the values are still lazy values.

### Function: _dict_values(self)

### Function: _dict_keys(self)

### Function: __init__(self, inference_state, lazy_value_list)

**Description:** type should be one of "tuple", "list"

### Function: py__simple_getitem__(self, index)

### Function: py__iter__(self, contextualized_node)

### Function: py__bool__(self)

### Function: __repr__(self)

### Function: __init__(self, inference_state, dct)

### Function: py__iter__(self, contextualized_node)

### Function: py__simple_getitem__(self, index)

### Function: _values(self, arguments)

### Function: _dict_values(self)

### Function: _dict_keys(self)

### Function: exact_key_items(self)

### Function: __repr__(self)

### Function: __init__(self, inference_state, arrays)

### Function: py__iter__(self, contextualized_node)

### Function: py__simple_getitem__(self, index)

### Function: __init__(self, python_context, start, stop, step)

### Function: _get_wrapped_value(self)

### Function: get_safe_value(self, default)

**Description:** Imitate CompiledValue.obj behavior and return a ``builtin.slice()``
object.

### Function: get(element)
