## AI Summary

A file named parser_utils.py.


### Function: get_executable_nodes(node, last_added)

**Description:** For static analysis.

### Function: get_sync_comp_fors(comp_for)

### Function: for_stmt_defines_one_name(for_stmt)

**Description:** Returns True if only one name is returned: ``for x in y``.
Returns False if the for loop is more complicated: ``for x, z in y``.

:returns: bool

### Function: get_flow_branch_keyword(flow_node, node)

### Function: clean_scope_docstring(scope_node)

**Description:** Returns a cleaned version of the docstring token. 

### Function: find_statement_documentation(tree_node)

### Function: safe_literal_eval(value)

### Function: get_signature(funcdef, width, call_string, omit_first_param, omit_return_annotation)

**Description:** Generate a string signature of a function.

:param width: Fold lines if a line is longer than this value.
:type width: int
:arg func_name: Override function name when given.
:type func_name: str

:rtype: str

### Function: move(node, line_offset)

**Description:** Move the `Node` start_pos.

### Function: get_following_comment_same_line(node)

**Description:** returns (as string) any comment that appears on the same line,
after the node, including the #

### Function: is_scope(node)

### Function: _get_parent_scope_cache(func)

### Function: get_parent_scope(node, include_flows)

**Description:** Returns the underlying scope.

### Function: get_cached_code_lines(grammar, path)

**Description:** Basically access the cached code lines in parso. This is not the nicest way
to do this, but we avoid splitting all the lines again.

### Function: get_parso_cache_node(grammar, path)

**Description:** This is of course not public. But as long as I control parso, this
shouldn't be a problem. ~ Dave

The reason for this is mostly caching. This is obviously also a sign of a
broken caching architecture.

### Function: cut_value_at_position(leaf, position)

**Description:** Cuts of the value of the leaf at position

### Function: expr_is_dotted(node)

**Description:** Checks if a path looks like `name` or `name.foo.bar` and not `name()`.

### Function: _function_is_x_method(decorator_checker)

### Function: wrapper(parso_cache_node, node, include_flows)

### Function: wrapper(function_node)

**Description:** This is a heuristic. It will not hold ALL the times, but it will be
correct pretty much for anyone that doesn't try to beat it.
staticmethod/classmethod are builtins and unless overwritten, this will
be correct.
