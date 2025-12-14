## AI Summary

A file named call_context.py.


## Class: CallContext

**Description:** CallContext essentially acts as a namespace for managing context variables.

Although not required, it is recommended that any "file-spanning" context variable
names (i.e., variables that will be set or retrieved from multiple files or services) be
added as constants to this class definition.

### Function: get(cls, name)

**Description:** Returns the value corresponding the named variable relative to this context.

If the named variable doesn't exist, None will be returned.

Parameters
----------
name : str
    The name of the variable to get from the call context

Returns
-------
value: Any
    The value associated with the named variable for this call context

### Function: set(cls, name, value)

**Description:** Sets the named variable to the specified value in the current call context.

Parameters
----------
name : str
    The name of the variable to store into the call context
value : Any
    The value of the variable to store into the call context

Returns
-------
None

### Function: context_variable_names(cls)

**Description:** Returns a list of variable names set for this call context.

Returns
-------
names: List[str]
    A list of variable names set for this call context.

### Function: _get_map(cls)

**Description:** Get the map of names to their values from the _NAME_VALUE_MAP context var.

If the map does not exist in the current context, an empty map is created and returned.
