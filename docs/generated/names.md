## AI Summary

A file named names.py.


### Function: _merge_name_docs(names)

## Class: AbstractNameDefinition

## Class: AbstractArbitraryName

**Description:** When you e.g. want to complete dicts keys, you probably want to complete
string literals, which is not really a name, but for Jedi we use this
concept of Name for completions as well.

## Class: AbstractTreeName

## Class: ValueNameMixin

## Class: ValueName

## Class: TreeNameDefinition

## Class: _ParamMixin

## Class: ParamNameInterface

## Class: BaseTreeParamName

## Class: _ActualTreeParamName

## Class: AnonymousParamName

## Class: ParamName

## Class: ParamNameWrapper

## Class: ImportName

## Class: SubModuleName

## Class: NameWrapper

## Class: StubNameMixin

## Class: StubName

## Class: ModuleName

## Class: StubModuleName

### Function: infer(self)

### Function: goto(self)

### Function: get_qualified_names(self, include_module_names)

### Function: _get_qualified_names(self)

### Function: get_root_context(self)

### Function: get_public_name(self)

### Function: __repr__(self)

### Function: is_import(self)

### Function: py__doc__(self)

### Function: api_type(self)

### Function: get_defining_qualified_value(self)

**Description:** Returns either None or the value that is public and qualified. Won't
return a function, because a name in a function is never public.

### Function: __init__(self, inference_state, string)

### Function: infer(self)

### Function: __init__(self, parent_context, tree_name)

### Function: get_qualified_names(self, include_module_names)

### Function: _get_qualified_names(self)

### Function: get_defining_qualified_value(self)

### Function: goto(self)

### Function: is_import(self)

### Function: string_name(self)

### Function: start_pos(self)

### Function: infer(self)

### Function: py__doc__(self)

### Function: _get_qualified_names(self)

### Function: get_root_context(self)

### Function: get_defining_qualified_value(self)

### Function: api_type(self)

### Function: __init__(self, value, tree_name)

### Function: goto(self)

### Function: infer(self)

### Function: api_type(self)

### Function: assignment_indexes(self)

**Description:** Returns an array of tuple(int, node) of the indexes that are used in
tuple assignments.

For example if the name is ``y`` in the following code::

    x, (y, z) = 2, ''

would result in ``[(1, xyz_node), (0, yz_node)]``.

When searching for b in the case ``a, *b, c = [...]`` it will return::

    [(slice(1, -1), abc_node)]

### Function: inference_state(self)

### Function: py__doc__(self)

### Function: maybe_positional_argument(self, include_star)

### Function: maybe_keyword_argument(self, include_stars)

### Function: _kind_string(self)

### Function: get_qualified_names(self, include_module_names)

### Function: get_kind(self)

### Function: to_string(self)

### Function: get_executed_param_name(self)

**Description:** For dealing with type inference and working around the graph, we
sometimes want to have the param name of the execution. This feels a
bit strange and we might have to refactor at some point.

For now however it exists to avoid infering params when we don't really
need them (e.g. when we can just instead use annotations.

### Function: star_count(self)

### Function: infer_default(self)

### Function: to_string(self)

### Function: get_public_name(self)

### Function: goto(self)

### Function: __init__(self, function_value, tree_name)

### Function: _get_param_node(self)

### Function: annotation_node(self)

### Function: infer_annotation(self, execute_annotation, ignore_stars)

### Function: infer_default(self)

### Function: default_node(self)

### Function: get_kind(self)

### Function: infer(self)

### Function: goto(self)

### Function: infer(self)

### Function: __init__(self, function_value, tree_name, arguments)

### Function: infer(self)

### Function: get_executed_param_name(self)

### Function: __init__(self, param_name)

### Function: __getattr__(self, name)

### Function: __repr__(self)

### Function: __init__(self, parent_context, string_name)

### Function: get_qualified_names(self, include_module_names)

### Function: parent_context(self)

### Function: infer(self)

### Function: goto(self)

### Function: api_type(self)

### Function: py__doc__(self)

### Function: __init__(self, wrapped_name)

### Function: __getattr__(self, name)

### Function: __repr__(self)

### Function: py__doc__(self)

### Function: infer(self)

### Function: __init__(self, value, name)

### Function: string_name(self)
