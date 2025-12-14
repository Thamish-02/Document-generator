## AI Summary

A file named analysis.py.


## Class: Error

## Class: Warning

### Function: add(node_context, error_name, node, message, typ, payload)

### Function: _check_for_setattr(instance)

**Description:** Check if there's any setattr method inside an instance. If so, return True.

### Function: add_attribute_error(name_context, lookup_value, name)

### Function: _check_for_exception_catch(node_context, jedi_name, exception, payload)

**Description:** Checks if a jedi object (e.g. `Statement`) sits inside a try/catch and
doesn't count as an error (if equal to `exception`).
Also checks `hasattr` for AttributeErrors and uses the `payload` to compare
it.
Returns True if the exception was catched.

### Function: __init__(self, name, module_path, start_pos, message)

### Function: line(self)

### Function: column(self)

### Function: code(self)

### Function: __str__(self)

### Function: __eq__(self, other)

### Function: __ne__(self, other)

### Function: __hash__(self)

### Function: __repr__(self)

### Function: check_match(cls, exception)

### Function: check_try_for_except(obj, exception)

### Function: check_hasattr(node, suite)
