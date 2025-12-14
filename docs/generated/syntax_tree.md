## AI Summary

A file named syntax_tree.py.


### Function: _limit_value_infers(func)

**Description:** This is for now the way how we limit type inference going wild. There are
other ways to ensure recursion limits as well. This is mostly necessary
because of instance (self) access that can be quite tricky to limit.

I'm still not sure this is the way to go, but it looks okay for now and we
can still go anther way in the future. Tests are there. ~ dave

### Function: infer_node(context, element)

### Function: _infer_node_if_inferred(context, element)

**Description:** TODO This function is temporary: Merge with infer_node.

### Function: _infer_node_cached(context, element)

### Function: _infer_node(context, element)

### Function: infer_trailer(context, atom_values, trailer)

### Function: infer_atom(context, atom)

**Description:** Basically to process ``atom`` nodes. The parser sometimes doesn't
generate the node (because it has just one child). In that case an atom
might be a name or a literal as well.

### Function: infer_expr_stmt(context, stmt, seek_name)

### Function: _infer_expr_stmt(context, stmt, seek_name)

**Description:** The starting point of the completion. A statement always owns a call
list, which are the calls, that a statement does. In case multiple
names are defined in the statement, `seek_name` returns the result for
this name.

expr_stmt: testlist_star_expr (annassign | augassign (yield_expr|testlist) |
                 ('=' (yield_expr|testlist_star_expr))*)
annassign: ':' test ['=' test]
augassign: ('+=' | '-=' | '*=' | '@=' | '/=' | '%=' | '&=' | '|=' | '^=' |
            '<<=' | '>>=' | '**=' | '//=')

:param stmt: A `tree.ExprStmt`.

### Function: infer_or_test(context, or_test)

### Function: infer_factor(value_set, operator)

**Description:** Calculates `+`, `-`, `~` and `not` prefixes.

### Function: _literals_to_types(inference_state, result)

### Function: _infer_comparison(context, left_values, operator, right_values)

### Function: _is_annotation_name(name)

### Function: _is_list(value)

### Function: _is_tuple(value)

### Function: _bool_to_value(inference_state, bool_)

### Function: _get_tuple_ints(value)

### Function: _infer_comparison_part(inference_state, context, left, operator, right)

### Function: tree_name_to_values(inference_state, context, tree_name)

### Function: _apply_decorators(context, node)

**Description:** Returns the function, that should to be executed in the end.
This is also the places where the decorators are processed.

### Function: check_tuple_assignments(name, value_set)

**Description:** Checks if tuples are assigned.

## Class: ContextualizedSubscriptListNode

### Function: _infer_subscript_list(context, index)

**Description:** Handles slices in subscript nodes.

### Function: wrapper(context)

### Function: check_setitem(stmt)

### Function: check(obj)

**Description:** Checks if a Jedi object is either a float or an int.

### Function: infer(self)

### Function: to_mod(v)
