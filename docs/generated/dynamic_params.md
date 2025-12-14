## AI Summary

A file named dynamic_params.py.


### Function: _avoid_recursions(func)

### Function: dynamic_param_lookup(function_value, param_index)

**Description:** A dynamic search for param values. If you try to complete a type:

>>> def func(foo):
...     foo
>>> func(1)
>>> func("")

It is not known what the type ``foo`` without analysing the whole code. You
have to look for all calls to ``func`` to find out what ``foo`` possibly
is.

### Function: _search_function_arguments(module_context, funcdef, string_name)

**Description:** Returns a list of param names.

### Function: _get_lambda_name(node)

### Function: _get_potential_nodes(module_value, func_string_name)

### Function: _check_name_for_execution(inference_state, context, compare_node, name, trailer)

### Function: wrapper(function_value, param_index)

### Function: create_args(value)
