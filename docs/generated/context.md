## AI Summary

A file named context.py.


## Class: AbstractContext

## Class: ValueContext

**Description:** Should be defined, otherwise the API returns empty types.

## Class: TreeContextMixin

## Class: FunctionContext

## Class: ModuleContext

## Class: NamespaceContext

## Class: ClassContext

## Class: CompForContext

## Class: CompiledContext

## Class: CompiledModuleContext

### Function: _get_global_filters_for_name(context, name_or_none, position)

### Function: get_global_filters(context, until_position, origin_scope)

**Description:** Returns all filters in order of priority for name resolution.

For global name lookups. The filters will handle name resolution
themselves, but here we gather possible filters downwards.

>>> from jedi import Script
>>> script = Script('''
... x = ['a', 'b', 'c']
... def func():
...     y = None
... ''')
>>> module_node = script._module_node
>>> scope = next(module_node.iter_funcdefs())
>>> scope
<Function: func@3-5>
>>> context = script._get_module_context().create_context(scope)
>>> filters = list(get_global_filters(context, (4, 0), None))

First we get the names from the function scope.

>>> print(filters[0])  # doctest: +ELLIPSIS
MergedFilter(<ParserTreeFilter: ...>, <GlobalNameFilter: ...>)
>>> sorted(str(n) for n in filters[0].values())  # doctest: +NORMALIZE_WHITESPACE
['<TreeNameDefinition: string_name=func start_pos=(3, 4)>',
 '<TreeNameDefinition: string_name=x start_pos=(2, 0)>']
>>> filters[0]._filters[0]._until_position
(4, 0)
>>> filters[0]._filters[1]._until_position

Then it yields the names from one level "lower". In this example, this is
the module scope (including globals).
As a side note, you can see, that the position in the filter is None on the
globals filter, because there the whole module is searched.

>>> list(filters[1].values())  # package modules -> Also empty.
[]
>>> sorted(name.string_name for name in filters[2].values())  # Module attributes
['__doc__', '__name__', '__package__']

Finally, it yields the builtin filter, if `include_builtin` is
true (default).

>>> list(filters[3].values())  # doctest: +ELLIPSIS
[...]

### Function: __init__(self, inference_state)

### Function: get_filters(self, until_position, origin_scope)

### Function: goto(self, name_or_str, position)

### Function: py__getattribute__(self, name_or_str, name_context, position, analysis_errors)

**Description:** :param position: Position of the last statement -> tuple of line, column

### Function: _check_for_additional_knowledge(self, name_or_str, name_context, position)

### Function: get_root_context(self)

### Function: is_module(self)

### Function: is_builtins_module(self)

### Function: is_class(self)

### Function: is_stub(self)

### Function: is_instance(self)

### Function: is_compiled(self)

### Function: is_bound_method(self)

### Function: py__name__(self)

### Function: get_value(self)

### Function: name(self)

### Function: get_qualified_names(self)

### Function: py__doc__(self)

### Function: predefine_names(self, flow_scope, dct)

### Function: __init__(self, value)

### Function: tree_node(self)

### Function: parent_context(self)

### Function: is_module(self)

### Function: is_builtins_module(self)

### Function: is_class(self)

### Function: is_stub(self)

### Function: is_instance(self)

### Function: is_compiled(self)

### Function: is_bound_method(self)

### Function: py__name__(self)

### Function: name(self)

### Function: get_qualified_names(self)

### Function: py__doc__(self)

### Function: get_value(self)

### Function: __repr__(self)

### Function: infer_node(self, node)

### Function: create_value(self, node)

### Function: create_context(self, node)

### Function: create_name(self, tree_name)

### Function: get_filters(self, until_position, origin_scope)

### Function: py__file__(self)

### Function: get_filters(self, until_position, origin_scope)

### Function: get_global_filter(self)

### Function: string_names(self)

### Function: code_lines(self)

### Function: get_value(self)

**Description:** This is the only function that converts a context back to a value.
This is necessary for stub -> python conversion and vice versa. However
this method shouldn't be moved to AbstractContext.

### Function: get_filters(self, until_position, origin_scope)

### Function: get_value(self)

### Function: string_names(self)

### Function: py__file__(self)

### Function: get_filters(self, until_position, origin_scope)

### Function: get_global_filter(self, until_position, origin_scope)

### Function: __init__(self, parent_context, comp_for)

### Function: get_filters(self, until_position, origin_scope)

### Function: get_value(self)

### Function: py__name__(self)

### Function: __repr__(self)

### Function: get_filters(self, until_position, origin_scope)

### Function: get_value(self)

### Function: string_names(self)

### Function: py__file__(self)

### Function: from_scope_node(scope_node, is_nested)

### Function: parent_scope(node)
