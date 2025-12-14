## AI Summary

A file named extract.py.


### Function: extract_variable(inference_state, path, module_node, name, pos, until_pos)

### Function: _is_expression_with_error(nodes)

**Description:** Returns a tuple (is_expression, error_string).

### Function: _find_nodes(module_node, pos, until_pos)

**Description:** Looks up a module and tries to find the appropriate amount of nodes that
are in there.

### Function: _replace(nodes, expression_replacement, extracted, pos, insert_before_leaf, remaining_prefix)

### Function: _expression_nodes_to_string(nodes)

### Function: _suite_nodes_to_string(nodes, pos)

### Function: _split_prefix_at(leaf, until_line)

**Description:** Returns a tuple of the leaf's prefix, split at the until_line
position.

### Function: _get_indentation(node)

### Function: _get_parent_definition(node)

**Description:** Returns the statement where a node is defined.

### Function: _remove_unwanted_expression_nodes(parent_node, pos, until_pos)

**Description:** This function makes it so for `1 * 2 + 3` you can extract `2 + 3`, even
though it is not part of the expression.

### Function: _is_not_extractable_syntax(node)

### Function: extract_function(inference_state, path, module_context, name, pos, until_pos)

### Function: _check_for_non_extractables(nodes)

### Function: _is_name_input(module_context, names, first, last)

### Function: _find_inputs_and_outputs(module_context, context, nodes)

### Function: _find_non_global_names(nodes)

### Function: _get_code_insertion_node(node, is_bound_method)

### Function: _find_needed_output_variables(context, search_node, at_least_pos, return_variables)

**Description:** Searches everything after at_least_pos in a node and checks if any of the
return_variables are used in there and returns those.

### Function: _is_node_ending_return_stmt(node)
