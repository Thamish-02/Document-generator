## AI Summary

A file named param.py.


### Function: _add_argument_issue(error_name, lazy_value, message)

## Class: ExecutedParamName

### Function: get_executed_param_names_and_issues(function_value, arguments)

**Description:** Return a tuple of:
  - a list of `ExecutedParamName`s corresponding to the arguments of the
    function execution `function_value`, containing the inferred value of
    those arguments (whether explicit or default)
  - a list of the issues encountered while building that list

For example, given:
```
def foo(a, b, c=None, d='d'): ...

foo(42, c='c')
```

Then for the execution of `foo`, this will return a tuple containing:
  - a list with entries for each parameter a, b, c & d; the entries for a,
    c, & d will have their values (42, 'c' and 'd' respectively) included.
  - a list with a single entry about the lack of a value for `b`

### Function: get_executed_param_names(function_value, arguments)

**Description:** Return a list of `ExecutedParamName`s corresponding to the arguments of the
function execution `function_value`, containing the inferred value of those
arguments (whether explicit or default). Any issues building this list (for
example required arguments which are missing in the invocation) are ignored.

For example, given:
```
def foo(a, b, c=None, d='d'): ...

foo(42, c='c')
```

Then for the execution of `foo`, this will return a list containing entries
for each parameter a, b, c & d; the entries for a, c, & d will have their
values (42, 'c' and 'd' respectively) included.

### Function: _error_argument_count(funcdef, actual_count)

### Function: __init__(self, function_value, arguments, param_node, lazy_value, is_default)

### Function: infer(self)

### Function: matches_signature(self)

### Function: __repr__(self)

### Function: too_many_args(argument)
